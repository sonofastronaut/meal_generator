"""Парсинг natural language запроса пользователя в structured constraints.

Использует OpenAI structured outputs (``response_format={"type": "json_schema", ...,
"strict": True}``) — модель обязана вернуть JSON, соответствующий схеме из
``build_json_schema()``, поэтому после парсинга не нужна ручная валидация
enum-значений (``meal_slots``/``excluded_categories``): strict-режим OpenAI
API уже это гарантирует на своей стороне.

``client`` инъецируется в ``parse_meal_request``, а не создаётся внутри неё —
та же причина тестируемости без реального похода в сеть/API-ключа, что и у
``embed_fn`` в ``retrieval/search.py`` и ``indexing/qdrant_index.py``: логику
парсинга ответа и сборки ``ParsedMealRequest`` можно проверить с фейковым
клиентом.

Допустимые значения ``meal_slots``/``excluded_categories`` в JSON-схеме
строятся программно из ``meal_generator.preprocessing.meal_types`` и
``meal_generator.preprocessing.exclusions`` — той же логики, что уже
используется при индексации curated-рецептов (``add_meal_types``,
``add_allergen_categories``). Так enum в схеме не может рассинхронизироваться
с реальным набором категорий, если тот изменится.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ..planning.optimizer import MealPlanConstraints
from ..preprocessing.exclusions import EXCLUSION_CATEGORY_SIGNALS
from ..preprocessing.meal_types import _MEAL_TYPE_SIGNALS

MEAL_TYPES: list[str] = [meal_type for meal_type, _ in _MEAL_TYPE_SIGNALS]
EXCLUDED_CATEGORIES: list[str] = list(EXCLUSION_CATEGORY_SIGNALS.keys())


@dataclass
class ParsedMealRequest:
    meal_slots: list[str]
    target_calories: float
    protein_min: float | None
    excluded_categories: list[str]
    query_text: str
    max_total_time_minutes: int | None


def build_json_schema() -> dict:
    """Построить JSON-схему ``ParsedMealRequest`` для OpenAI structured outputs.

    Чистая функция — не ходит в OpenAI. ``meal_slots``/``excluded_categories``
    — массивы с ``enum`` из допустимых значений, построенных программно из
    ``MEAL_TYPES``/``EXCLUDED_CATEGORIES``. ``protein_min`` и
    ``max_total_time_minutes`` — nullable-поля (``type: [..., "null"]``), как
    того требует strict-режим OpenAI, а не Python ``Optional`` в схеме.
    """
    return {
        "type": "object",
        "properties": {
            "meal_slots": {
                "type": "array",
                "items": {"type": "string", "enum": MEAL_TYPES},
            },
            "target_calories": {"type": "number"},
            "protein_min": {"type": ["number", "null"]},
            "excluded_categories": {
                "type": "array",
                "items": {"type": "string", "enum": EXCLUDED_CATEGORIES},
            },
            "query_text": {"type": "string"},
            "max_total_time_minutes": {"type": ["integer", "null"]},
        },
        "required": [
            "meal_slots",
            "target_calories",
            "protein_min",
            "excluded_categories",
            "query_text",
            "max_total_time_minutes",
        ],
        "additionalProperties": False,
    }


def _ensure_breakfast_first(meal_slots: list[str]) -> list[str]:
    """Гарантировать, что первый элемент ``meal_slots`` — ``"breakfast"``.

    Чистая функция — не мутирует переданный список.

    - Если ``"breakfast"`` уже на первой позиции — список возвращается как
      есть.
    - Если ``"breakfast"`` есть, но не первым — он переставляется на
      позицию 0, остальные элементы сохраняют относительный порядок между
      собой.
    - Если ``"breakfast"`` в списке нет — первый элемент заменяется на
      ``"breakfast"`` (не добавляется), остальные элементы не меняются;
      общее количество слотов сохраняется.
    - Пустой список — defensive edge case, на практике не должен
      происходить, так как strict JSON schema гарантирует непустой массив;
      возвращает ``["breakfast"]`` как единственный разумный fallback.
    """
    if not meal_slots:
        return ["breakfast"]

    if meal_slots[0] == "breakfast":
        return meal_slots

    if "breakfast" in meal_slots:
        rest = [slot for slot in meal_slots if slot != "breakfast"]
        return ["breakfast", *rest]

    return ["breakfast", *meal_slots[1:]]


def build_system_prompt() -> str:
    """Построить system-промпт для парсинга запроса пользователя в structured constraints.

    ``target_calories`` обязателен в схеме (``required``), но пользователь
    может не назвать конкретное число — на этот случай промпт просит модель
    сделать разумную оценку по контексту (тип приёма пищи, количество слотов,
    формулировки вроде "плотный" / "лёгкий"). Это временное MVP-поведение:
    оценка "на глаз" от LLM должна быть в будущем заменена либо явным
    уточняющим вопросом пользователю (переспросить целевые калории), либо
    расчётом на основе профиля пользователя (вес/рост/цель), если такой
    профиль появится в проекте.
    """
    return f"""You turn a user's free-form meal request into a structured request \
that matches the given JSON schema.

Field-by-field guidance:

- meal_slots: a list of meal types for the day, using only these values: \
{MEAL_TYPES}.

  Step 1 — decide the COUNT: if the user explicitly states how many meals \
  they want (e.g. "4 meals", "3 приёма пищи"), meal_slots must have exactly \
  that many entries — never more, never fewer, even if the user also \
  mentions a snack, dessert, or other specific meal within that count. If \
  the user does not state a count, use the number of distinct meals \
  reasonably implied by their message (default to 3 if genuinely \
  unspecified).

  Step 2 — decide the TYPES: use this table as a default composition for a \
  given count, then adjust individual slot types (not the count) to match \
  anything more specific the user said:
    - 2 meals -> ["breakfast", "main_meal"]
    - 3 meals -> ["breakfast", "main_meal", "main_meal"]
    - 4 meals -> ["breakfast", "main_meal", "main_meal", "dessert"]
    - 5 meals -> ["breakfast", "main_meal", "main_meal", "main_meal", "lunch_snack"]

  Example: "4 meals, with a snack between lunch and dinner" means 4 total \
  slots where one of the existing non-breakfast slots becomes lunch_snack \
  instead of dessert/main_meal — NOT 4 default slots plus a 5th added for \
  the snack. The user's stated count is a hard constraint; the table only \
  decides what fills each position.

  Note: the first slot will be forced to "breakfast" automatically after \
this response regardless of what you put here, so there's no need to \
double-check that specifically.

- target_calories: the total calorie target for the day. If the user did not \
state a number explicitly, make a reasonable estimate from context (number \
of meals, phrases like "light"/"heavy", cuisine, etc.) — this field is \
required and must always be filled with your best estimate, never left as a \
placeholder value.

- protein_min: minimum total protein for the day in grams, only if the user \
asked for it (e.g. "high protein", "at least 100g protein"); otherwise null.

- excluded_categories: map any allergy/restriction the user mentions (fish, \
nuts, dairy, etc.) strictly onto one of these predefined categories: \
{EXCLUDED_CATEGORIES}. If something the user mentions does not clearly map \
onto any of these categories, do not include it — leave it out rather than \
guessing a category for it.

- query_text: the taste/vibe/cuisine part of the user's request (cuisine, \
flavor, how light or heavy it should feel), kept in or close to the user's \
own words. Do not restate structured constraints (meal count, calories, \
allergies, cooking time, snack placement) here — those already live in the \
other fields. If the user's message contains no taste/cuisine/vibe \
preference at all (e.g. it only specifies meal count, calories, and \
macros), query_text must be an empty string "" — do not fall back to \
repeating the full original message.

- max_total_time_minutes: maximum cooking time per meal in minutes, only if \
the user mentioned a time constraint; otherwise null.

Respond with a JSON object matching the schema — no extra commentary."""


def parse_meal_request(
    client, user_message: str, model: str = "gpt-4o-mini"
) -> ParsedMealRequest:
    """Распарсить свободный текст пользователя в ``ParsedMealRequest`` через OpenAI.

    ``client`` — экземпляр ``openai.OpenAI``, создаётся и передаётся
    вызывающим кодом (см. докстроку модуля). Strict JSON schema гарантирует
    валидность enum-полей на стороне OpenAI API, поэтому дополнительная
    валидация здесь не нужна. Единственная поправка поверх сырого ответа
    модели — ``meal_slots[0]`` гарантированно ``"breakfast"``: это
    жёсткое code-level правило (``_ensure_breakfast_first``), применяемое
    после ответа и не полагающееся на то, что модель следует дефолтной схеме
    из system-промпта.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": user_message},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "meal_request",
                "strict": True,
                "schema": build_json_schema(),
            },
        },
    )

    data = json.loads(response.choices[0].message.content)

    return ParsedMealRequest(
        meal_slots=_ensure_breakfast_first(data["meal_slots"]),
        target_calories=data["target_calories"],
        protein_min=data["protein_min"],
        excluded_categories=data["excluded_categories"],
        query_text=data["query_text"],
        max_total_time_minutes=data["max_total_time_minutes"],
    )


def to_meal_plan_constraints(parsed: ParsedMealRequest) -> MealPlanConstraints:
    """Отобразить ``ParsedMealRequest`` на ``MealPlanConstraints`` для optimizer-слоя.

    ``calorie_tolerance`` не переопределяется — используется дефолт из
    ``MealPlanConstraints``, у ``ParsedMealRequest`` нет соответствующего поля.
    """
    return MealPlanConstraints(
        meal_slots=parsed.meal_slots,
        target_calories=parsed.target_calories,
        protein_min=parsed.protein_min,
        excluded_categories=parsed.excluded_categories,
        max_total_time_minutes=parsed.max_total_time_minutes,
        query_text=parsed.query_text,
    )
