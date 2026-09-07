"""LangGraph-узлы пайплайна построения и уточнения дневного рациона.

Клиенты (``openai.OpenAI``, ``QdrantClient``) и ``embed_fn`` не хранятся в
``MealPlanState`` — они инъецируются через замыкание при сборке графа (см.
``graph.py``). Фабрики узлов (``make_*_node``) принимают клиентов и
возвращают функцию узла, которая принимает только ``state`` — тот же
dependency inversion принцип, что и у ``embed_fn``/``client`` во всём
остальном проекте, адаптированный под LangGraph-контракт "узел = функция от
state", возвращающая partial update, а не весь state целиком.

``classify_intent`` — единственный узел с "гибридной" фабрикой: сама
функция узла, возвращаемая ``make_classify_intent_node``, сначала
механически проверяет ``len(state["messages"])`` и только при непустой
истории идёт в LLM — отдельной публичной "чистой" версии узла нет, чтобы не
плодить два способа собрать один и тот же узел графа (это единственная
точка входа, которую вызывает ``graph.py``).
"""

from __future__ import annotations

import json
from collections.abc import Callable

from .state import MealPlanState
from ..nlp.parse_refinement import parse_refinement
from ..nlp.parse_request import parse_meal_request, to_meal_plan_constraints
from ..planning.optimizer import (
    MealPlanVariant,
    _choose_search_function,
    _is_within_tolerance,
    build_meal_plan_variants,
)
from ..preprocessing.exclusions import ingredient_matches_terms
from ..retrieval.search import SearchConstraints

STARTING_POOL_SIZE = 30
MAX_POOL_SIZE = 100

INTENT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": ["new_request", "refine_slot"]},
    },
    "required": ["intent"],
    "additionalProperties": False,
}

INTENT_SYSTEM_PROMPT = (
    "The user already has a current meal plan for today. Given the "
    "conversation so far, decide whether the user's latest message is "
    "asking for a completely new day of meals (a new day, different "
    "goals, starting over) or asking to refine/adjust one specific meal "
    "in their existing plan (swap a dish, remove an ingredient they "
    "dislike, etc.). Respond with a JSON object matching the schema — no "
    "extra commentary."
)


def _latest_user_message(state: MealPlanState) -> str:
    """Найти текст последнего сообщения с ``role == "user"`` в ``state["messages"]``."""
    for message in reversed(state["messages"]):
        if message.get("role") == "user":
            return message["content"]
    return ""


def make_classify_intent_node(openai_client) -> Callable[[MealPlanState], dict]:
    """Собрать узел, определяющий intent последнего сообщения пользователя.

    Первое сообщение в диалоге (``len(messages) <= 1`` — только что
    отправленное сообщение пользователя, без предыдущего обмена репликами)
    механически классифицируется как ``"new_request"`` без похода в LLM:
    уточнять точно нечего, рабочего плана ещё не существует. При непустой
    истории классификация идёт через structured output.
    """

    def classify_intent_node(state: MealPlanState) -> dict:
        if len(state["messages"]) <= 1:
            return {"intent": "new_request"}

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                *state["messages"],
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "intent_classification",
                    "strict": True,
                    "schema": INTENT_JSON_SCHEMA,
                },
            },
        )

        data = json.loads(response.choices[0].message.content)
        return {"intent": data["intent"]}

    return classify_intent_node


def make_parse_request_node(openai_client) -> Callable[[MealPlanState], dict]:
    """Собрать узел, парсящий последнее сообщение пользователя в constraints через OpenAI."""

    def parse_request_node(state: MealPlanState) -> dict:
        parsed_request = parse_meal_request(openai_client, _latest_user_message(state))
        constraints = to_meal_plan_constraints(parsed_request)
        return {
            "parsed_request": parsed_request,
            "constraints": constraints,
            "pool_size": STARTING_POOL_SIZE,
        }

    return parse_request_node


def make_build_plan_node(
    qdrant_client, embed_fn: Callable[[list[str]], list[list[float]]]
) -> Callable[[MealPlanState], dict]:
    """Собрать узел, строящий три варианта плана поверх ``state["constraints"]``.

    ``build_meal_plan_variants`` может бросить ``ValueError``, если пул
    кандидатов исчерпан для какого-то слота — это ожидаемый путь, не баг
    (см. докстроку ``planning/optimizer.py``), поэтому он перехватывается
    здесь и превращается в ``error_message`` для роутера, а не всплывает
    из узла наружу. При успехе первый вариант (``"balanced"``, по порядку
    из ``build_meal_plan_variants``) становится дефолтным ``working_variant``
    — планом, который дальше точечно правит узел ``replace_slot``.
    """

    def build_plan_node(state: MealPlanState) -> dict:
        try:
            variants = build_meal_plan_variants(
                qdrant_client,
                embed_fn,
                state["constraints"],
                pool_size=state["pool_size"],
            )
        except ValueError as exception:
            return {"variants": None, "error_message": str(exception)}

        return {
            "variants": variants,
            "working_variant": variants[0],
            "error_message": None,
        }

    return build_plan_node


def relax_pool_size_node(state: MealPlanState) -> dict:
    """Удвоить ``pool_size`` для повторной попытки, не превышая ``MAX_POOL_SIZE``.

    Чистая функция — не ходит ни в какого клиента.
    """
    return {"pool_size": min(state["pool_size"] * 2, MAX_POOL_SIZE)}


def route_after_build_plan(state: MealPlanState) -> str:
    """Conditional edge после ``build_plan``: сравнивать варианты или ещё раз пробовать шире.

    Не узел, а роутер для ``add_conditional_edges``. ``pool_size < MAX_POOL_SIZE``
    используется как признак "максимальный пул ещё не пробовали" вместо
    отдельного счётчика попыток — как только пул дошёл до потолка, дальнейший
    retry бессмыслен, и граф идёт объяснять (с ошибкой вместо сравнения планов).
    """
    if state["variants"] is not None:
        return "explain"

    if state["error_message"] is not None and state["pool_size"] < MAX_POOL_SIZE:
        return "relax"

    return "explain"


def route_after_classify_intent(state: MealPlanState) -> str:
    """Conditional edge после ``classify_intent``: полный replan или точечное уточнение."""
    return state["intent"]


def make_parse_refinement_node(openai_client) -> Callable[[MealPlanState], dict]:
    """Собрать узел, парсящий уточняющее сообщение в ``slot_index``/``avoid_terms``."""

    def parse_refinement_node(state: MealPlanState) -> dict:
        parsed = parse_refinement(
            openai_client, state["working_variant"], _latest_user_message(state)
        )
        return {
            "refine_slot_index": parsed.slot_index,
            "refine_avoid_terms": parsed.avoid_terms,
        }

    return parse_refinement_node


def make_replace_slot_node(
    qdrant_client, embed_fn: Callable[[list[str]], list[list[float]]]
) -> Callable[[MealPlanState], dict]:
    """Собрать детерминированный узел, точечно заменяющий один слот ``working_variant``.

    ``meal_type`` заменяемого слота берётся как первый элемент
    ``slots[slot_index]["meal_types"]`` — это multi-label список из
    curated-схемы (payload не меняется), но для целей повторного retrieval
    нужен ровно один ``meal_type``, поэтому здесь он упрощается до первого
    элемента; сам payload с полным списком остаётся как был.

    Retrieval-путь выбирается той же ``_choose_search_function``, что и в
    ``_fetch_candidate_pools`` (не дублируем routing-условие). Из
    полученных кандидатов исключаются: (а) ``recipe_id``, уже занятые
    другими слотами ``working_variant`` — не предлагать то же блюдо на
    двух слотах; (б) кандидаты, хотя бы один ingredient которых матчит
    хотя бы один термин из ``avoid_terms`` (``ingredient_matches_terms``).
    Из оставшихся выбирается кандидат, ближайший по ``calories`` к
    калорийности самого заменяемого слота (``target_slot["calories"]``) —
    тот же принцип, что в ``_select_balanced``, но целью служит калорийность
    конкретного слота, а не остаточный таргет всего дня. Порядок
    ранжирования retrieval-функции (semantic score или rating) для выбора
    не используется: он ничем не связан с калорийностью заменяемого блюда,
    и "первый по этому порядку" мог увести план далеко от исходного
    баланса калорий даже без явных avoid_terms — например, точечная замена
    без конкретных ограничений заменяла блюдо на другое, отличающееся по
    калорийности вдвое, выводя весь план за пределы ``calorie_tolerance``.

    Если после фильтрации кандидатов не осталось, ``working_variant`` не
    трогается — узел возвращает только ``error_message``, чтобы граф мог
    показать это через ``explain``, а не падать.

    ``slot_index`` из ``parse_refinement`` — LLM-ответ, провалидированный
    strict JSON schema только по типу (``integer``), не по диапазону
    значений, поэтому здесь он явно проверяется на попадание в
    ``[0, len(working_variant.slots))`` до обращения к
    ``working_variant.slots[slot_index]`` — тот же паттерн, что и для
    других ожидаемых сбоев в проекте (исчерпание пула и т.п.):
    превращается в ``error_message``, а не всплывает как необработанный
    ``IndexError``.
    """

    def replace_slot_node(state: MealPlanState) -> dict:
        working_variant = state["working_variant"]
        slot_index = state["refine_slot_index"]
        avoid_terms = state["refine_avoid_terms"]
        constraints = state["constraints"]

        if not (0 <= slot_index < len(working_variant.slots)):
            return {
                "error_message": (
                    f"Model returned an out-of-range slot_index ({slot_index}) "
                    f"for a plan with {len(working_variant.slots)} slots."
                )
            }

        target_slot = working_variant.slots[slot_index]
        meal_type = target_slot["meal_types"][0]

        search_constraints = SearchConstraints(
            query_text=constraints.query_text,
            meal_type=meal_type,
            protein_min=constraints.protein_min,
            excluded_categories=constraints.excluded_categories,
            max_total_time_minutes=constraints.max_total_time_minutes,
            limit=state["pool_size"],
        )

        search_fn = _choose_search_function(meal_type, constraints.query_text)
        candidates = search_fn(qdrant_client, embed_fn, search_constraints)

        existing_ids = {slot["recipe_id"] for slot in working_variant.slots}
        filtered = [
            candidate
            for candidate in candidates
            if candidate["recipe_id"] not in existing_ids
            and not any(
                ingredient_matches_terms(ingredient, avoid_terms)
                for ingredient in candidate["ingredients"]
            )
        ]

        if not filtered:
            return {
                "error_message": (
                    f"Couldn't find a replacement for slot {slot_index} "
                    f"avoiding {avoid_terms}"
                )
            }

        target_calories = target_slot["calories"]
        best_match = min(filtered, key=lambda c: abs(c["calories"] - target_calories))

        new_slots = list(working_variant.slots)
        new_slots[slot_index] = best_match
        total_calories = sum(slot["calories"] for slot in new_slots)
        within_tolerance = _is_within_tolerance(
            total_calories, constraints.target_calories, constraints.calorie_tolerance
        )

        updated_variant = MealPlanVariant(
            label=working_variant.label,
            slots=new_slots,
            total_calories=total_calories,
            within_tolerance=within_tolerance,
        )
        return {"working_variant": updated_variant, "error_message": None}

    return replace_slot_node


def _format_variants_for_prompt(variants) -> str:
    lines = []
    for variant in variants:
        lines.append(f"{variant.label} (total {variant.total_calories:.0f} kcal):")
        for slot in variant.slots:
            lines.append(
                f"  - {slot['name']}: {slot['calories']:.0f} kcal, "
                f"{slot['total_time_minutes']} min"
            )
    return "\n".join(lines)


LANGUAGE_INSTRUCTION = " Respond in the same language the user has been writing in."

NEW_REQUEST_EXPLAIN_SYSTEM_PROMPT = (
    "You briefly compare meal plan variants for a user in plain, friendly "
    "language. Focus on how the variants differ in spirit (e.g. balanced, "
    "quick to cook, or more varied) — don't list individual dishes or "
    "calorie numbers line by line, those are already shown to the user in a "
    "table." + LANGUAGE_INSTRUCTION
)

REFINE_SLOT_EXPLAIN_SYSTEM_PROMPT = (
    "You briefly tell the user what just changed in their meal plan, in "
    "plain, friendly language." + LANGUAGE_INSTRUCTION
)


def _format_refinement_change_for_prompt(state: MealPlanState) -> str:
    working_variant = state["working_variant"]
    slot_index = state["refine_slot_index"]
    avoid_terms = state["refine_avoid_terms"]
    slot = working_variant.slots[slot_index]
    return (
        f"The user asked to avoid: {', '.join(avoid_terms)}.\n"
        f"Slot {slot_index} was replaced with: {slot['name']} "
        f"({slot['calories']:.0f} kcal, {slot['total_time_minutes']} min)."
    )


def make_explain_node(openai_client) -> Callable[[MealPlanState], dict]:
    """Собрать узел, объясняющий пользователю итог на человеческом языке.

    Ветвится по ``state["intent"]``:

    - ``"new_request"`` (или ``None`` — обратная совместимость со старым
      поведением до появления intent-классификации): если ``variants`` —
      ``None`` (пул исчерпан даже после retry), объяснять нечего — LLM не
      вызывается, узел возвращает статичное сообщение об ошибке. Иначе —
      просит модель кратко сравнить три варианта.
    - ``"refine_slot"``: если ``error_message`` не пуст (замена слота не
      удалась) — тот же паттерн без LLM-вызова, сообщение об ошибке
      становится объяснением напрямую. Иначе — отдельный текстовый
      LLM-вызов, объясняющий, что именно изменилось в ``working_variant``
      (какой слот, на что заменили), а не сравнение трёх вариантов заново.

    Оба LLM-вызова получают последнее сообщение пользователя
    (``_latest_user_message``) как отдельное сообщение в контексте, а
    system-промпт явно просит отвечать на языке пользователя — без этого
    модель видела только сгенерированный нами английский текст
    (``_format_variants_for_prompt``/``_format_refinement_change_for_prompt``)
    и не имела сигнала о том, на каком языке писал пользователь.
    """

    def explain_node(state: MealPlanState) -> dict:
        if state.get("intent") == "refine_slot":
            if state["error_message"]:
                return {"explanation": state["error_message"]}

            change_text = _format_refinement_change_for_prompt(state)
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": REFINE_SLOT_EXPLAIN_SYSTEM_PROMPT},
                    {"role": "user", "content": _latest_user_message(state)},
                    {"role": "user", "content": change_text},
                ],
            )
            return {"explanation": response.choices[0].message.content}

        if state["variants"] is None:
            return {
                "explanation": (
                    f"Couldn't build a meal plan with these constraints: "
                    f"{state['error_message']}. Try relaxing calorie range, "
                    f"excluded categories, or cooking time."
                )
            }

        variants_text = _format_variants_for_prompt(state["variants"])
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": NEW_REQUEST_EXPLAIN_SYSTEM_PROMPT},
                {"role": "user", "content": _latest_user_message(state)},
                {"role": "user", "content": variants_text},
            ],
        )

        return {"explanation": response.choices[0].message.content}

    return explain_node
