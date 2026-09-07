"""Optimizer-слой: сборка дневного рациона из трёх вариантов жадными эвристиками.

Это жадные эвристики поверх пула кандидатов, а не ILP-солвер. Для MVP это
осознанный trade-off: жадный проход по слотам по порядку с локальным выбором
"лучшего сейчас" кандидата на порядки проще реализовать, отладить и
объяснить пользователю, чем формулировать задачу как ILP (OR-Tools/PuLP) с
глобальной оптимизацией по калориям/времени/разнообразию сразу. Апгрейд до
ILP-солвера — будущий шаг, если на реальных данных жадный подход даст
заметно неоптимальные планы (например, застревание в локально хорошем, но
глобально плохом наборе слотов из-за порядка обхода).

Функции ``_select_*`` бросают ``ValueError``, если пул кандидатов для
какого-то типа слота исчерпан (например, два слота ``main_meal``, а в пуле
после фильтрации остался только один непровыбранный кандидат) — вместо
молчаливого повтора одного и того же рецепта в двух слотах. План с
одинаковым блюдом на обед и ужин хуже, чем явная ошибка: она может быть
показана пользователю напрямую или обработана релаксацией constraints (шире
``pool_size``, мягче фильтры) на уровне выше — в LangGraph-узле, который
вызывает этот модуль, а не здесь. Здесь это не наша забота — оптимизатор
работает с тем пулом, что ему дали.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

from qdrant_client import QdrantClient

from ..retrieval.search import SearchConstraints, search_recipes, search_recipes_by_rating


@dataclass
class MealPlanConstraints:
    meal_slots: list[str]
    target_calories: float
    calorie_tolerance: float = 0.07
    protein_min: float | None = None
    excluded_categories: list[str] = field(default_factory=list)
    max_total_time_minutes: int | None = None
    query_text: str = ""


@dataclass
class MealPlanVariant:
    label: str
    slots: list[dict]
    total_calories: float
    within_tolerance: bool


def _is_within_tolerance(
    total_calories: float, target_calories: float, tolerance: float
) -> bool:
    """Проверить, укладывается ли ``total_calories`` в допуск от ``target_calories``.

    ``tolerance`` — доля от ``target_calories``, не абсолютное число.
    """
    return abs(total_calories - target_calories) <= target_calories * tolerance


def _choose_search_function(
    meal_type: str, query_text: str
) -> Callable[
    [QdrantClient, Callable[[list[str]], list[list[float]]], SearchConstraints],
    list[dict],
]:
    """Выбрать retrieval-путь (semantic vs rating-based) для одного ``meal_type``.

    Общая routing-логика для ``_fetch_candidate_pools`` и для узла
    ``replace_slot`` в LangGraph-графе (см. ``graph/nodes.py``) — вынесена
    сюда, чтобы условие не дублировалось в двух местах.

    - ``search_recipes`` (семантический поиск по ``query_text``) — когда
      слот не ``breakfast`` и ``query_text`` непустой.
    - ``search_recipes_by_rating`` (non-semantic: фильтр + сортировка по
      популярности, см. докстроку ``retrieval/search.py``) — во всех
      остальных случаях: (а) ``breakfast`` всегда идёт этим путём — это
      осознанный point-fix для bleed-проблемы, когда vibe-запрос вроде
      "spicy asian dinner" семантически "перетягивал" завтрак к
      несвойственным ему блюдам; (б) любой другой тип слота при пустом
      ``query_text`` — эмбеддинг пустой строки вырожденный, и семантический
      поиск по нему это не нейтральный, а фактически случайный ранкинг,
      поэтому лучше явно деградировать до честной фильтрации.

    Возвращаемая функция всегда имеет единую сигнатуру
    ``(client, embed_fn, search_constraints)`` — ``search_recipes_by_rating``
    не использует ``embed_fn``, но единая сигнатура на вызывающей стороне
    избавляет оба места использования от необходимости самим ветвиться по
    тому, какой путь был выбран.
    """
    use_semantic_search = meal_type != "breakfast" and query_text != ""
    if use_semantic_search:
        return lambda client, embed_fn, search_constraints: search_recipes(
            client, embed_fn, search_constraints
        )

    return lambda client, embed_fn, search_constraints: search_recipes_by_rating(
        client, search_constraints
    )


def _fetch_candidate_pools(
    client: QdrantClient,
    embed_fn: Callable[[list[str]], list[list[float]]],
    constraints: MealPlanConstraints,
    pool_size: int = 30,
) -> dict[str, list[dict]]:
    """Получить по одному пулу кандидатов на каждый уникальный тип слота.

    Один запрос на уникальный ``meal_type`` — если ``main_meal`` встречается
    в ``meal_slots`` дважды, пул запрашивается один раз и переиспользуется
    для обоих слотов. Retrieval-путь для каждого типа слота выбирается через
    ``_choose_search_function`` (см. её докстроку за правилом и обоснованием).

    ``constraints.protein_min`` намеренно не передаётся в ``SearchConstraints``
    здесь. По смыслу (LLM-схема в ``nlp/parse_request.py``) это агрегатная
    цель на весь день ("минимум N г белка за день"), а не порционный минимум.
    Ранее он применялся как жёсткий Qdrant-фильтр "белок на порцию >= N"
    одинаково для каждого ``meal_type`` — это давало сломанный retrieval:
    для десертов такой фильтр почти всегда обнулял пул (пул размера 0, план
    не строится вовсе), а для остальных типов слотов пропускал только редкие
    выбросы (например, вяленое мясо), а не осмысленную выборку.

    Известное ограничение MVP: белковая цель сейчас вообще не влияет на отбор
    кандидатов — в отличие от ``target_calories``, который участвует в выборе
    через residual_target-логику в ``_select_*``-функциях. Protein-aware выбор
    по аналогии с калориями — кандидат на будущую доработку, не реализован
    здесь. Поле ``protein_min`` остаётся в ``MealPlanConstraints``/
    ``SearchConstraints``/JSON-схеме парсинга — контракт не меняется, меняется
    только то, что этот конкретный retrieval-путь его не использует.
    """
    unique_meal_types = dict.fromkeys(constraints.meal_slots)

    pools: dict[str, list[dict]] = {}
    for meal_type in unique_meal_types:
        search_constraints = SearchConstraints(
            query_text=constraints.query_text,
            meal_type=meal_type,
            excluded_categories=constraints.excluded_categories,
            max_total_time_minutes=constraints.max_total_time_minutes,
            limit=pool_size,
        )

        search_fn = _choose_search_function(meal_type, constraints.query_text)
        pools[meal_type] = search_fn(client, embed_fn, search_constraints)

    return pools


def _require_candidates(
    candidates: list[dict], pool: list[dict], meal_type: str, slot_count: int
) -> None:
    """Бросить ValueError, если для слота не осталось непровыбранных кандидатов."""
    if not candidates:
        raise ValueError(
            f"Not enough distinct candidates for meal_type '{meal_type}': "
            f"pool size {len(pool)}, needed at least {slot_count} for "
            f"{slot_count} slot(s)"
        )


def _select_balanced(
    meal_slots: list[str], pools: dict[str, list[dict]], target_calories: float
) -> list[dict]:
    """Жадно выбрать по одному рецепту на слот, приближаясь к остаточному таргету калорий.

    На каждом шаге остаточный таргет — это то, сколько калорий в среднем
    должно приходиться на каждый из ещё не заполненных слотов (включая
    текущий), чтобы суммарно попасть в ``target_calories``.
    """
    slot_counts = Counter(meal_slots)
    selected: list[dict] = []
    selected_ids: set = set()
    total = 0.0
    n = len(meal_slots)

    for index, meal_type in enumerate(meal_slots):
        remaining_slots = n - index
        residual_target = (target_calories - total) / remaining_slots

        pool = pools[meal_type]
        candidates = [c for c in pool if c["recipe_id"] not in selected_ids]
        _require_candidates(candidates, pool, meal_type, slot_counts[meal_type])

        best = min(candidates, key=lambda c: abs(c["calories"] - residual_target))
        selected.append(best)
        selected_ids.add(best["recipe_id"])
        total += best["calories"]

    return selected


def _select_quick_cooking(
    meal_slots: list[str],
    pools: dict[str, list[dict]],
    target_calories: float,
    local_tolerance: float = 0.15,
) -> list[dict]:
    """Жадно выбрать по одному рецепту на слот, предпочитая быстрые в готовке.

    На каждом шаге сначала сужает кандидатов до тех, чьи ``calories``
    попадают в ``residual_target * (1 ± local_tolerance)``, и среди них
    выбирает с минимальным ``total_time_minutes`` (``None`` считается
    "неизвестно" и сортируется в конец, как ``float("inf")``).

    Если после такого сужения кандидатов не осталось (например, слишком
    узкий ``local_tolerance`` или пул мал), функция не падает — она
    деградирует к логике ``_select_balanced`` для этого конкретного слота:
    берёт кандидата, ближайшего по calories к остаточному таргету, из всего
    оставшегося пула без фильтра по времени.
    """
    slot_counts = Counter(meal_slots)
    selected: list[dict] = []
    selected_ids: set = set()
    total = 0.0
    n = len(meal_slots)

    for index, meal_type in enumerate(meal_slots):
        remaining_slots = n - index
        residual_target = (target_calories - total) / remaining_slots

        pool = pools[meal_type]
        candidates = [c for c in pool if c["recipe_id"] not in selected_ids]
        _require_candidates(candidates, pool, meal_type, slot_counts[meal_type])

        low = residual_target * (1 - local_tolerance)
        high = residual_target * (1 + local_tolerance)
        within_range = [c for c in candidates if low <= c["calories"] <= high]

        if within_range:
            best = min(
                within_range,
                key=lambda c: c["total_time_minutes"]
                if c["total_time_minutes"] is not None
                else float("inf"),
            )
        else:
            best = min(candidates, key=lambda c: abs(c["calories"] - residual_target))

        selected.append(best)
        selected_ids.add(best["recipe_id"])
        total += best["calories"]

    return selected


def _select_variety(
    meal_slots: list[str], pools: dict[str, list[dict]], target_calories: float
) -> list[dict]:
    """Жадно выбрать по одному рецепту на слот, минимизируя повтор блюда/ингредиентов.

    Штраф кандидата — трёхуровневый лексикографический ключ:

    1. ``category_repeat_penalty`` — 1, если ``candidate["source_category"]``
       уже встречался среди выбранных в этом варианте плана, иначе 0.
       ``None`` не считается совпадением категории ни с чем, даже с другим
       ``None`` — у curated-схемы ``source_category`` может отсутствовать, и
       два рецепта без категории не должны штрафоваться как "одна и та же
       категория".
    2. ``ingredient_jaccard_penalty`` — Jaccard-коэффициент между
       нормализованными (``casefold()``, точное совпадение значения)
       ``ingredients`` кандидата и объединённым множеством ингредиентов уже
       выбранных рецептов: ``|пересечение| / |объединение|``, 0.0 если оба
       множества пусты.
    3. ``calorie_diff`` — ``abs(candidate["calories"] - residual_target)``,
       тот же tie-break, что и в ``_select_balanced``.

    Кортеж сравнивается лексикографически через ``min(candidates, key=...)``
    без каких-либо весовых коэффициентов между уровнями — сам порядок
    элементов в кортеже уже даёт строгое доминирование первого критерия над
    вторым и второго над третьим, вводить константы вроде
    ``category_weight`` незачем.

    Почему не сырое количество пересекающихся ingredients (как было раньше):
    эмпирически на реальных данных Food.com пересечения ingredients между
    парами кандидатов редки — 77.9% пар вообще не пересекаются, поэтому
    raw-count штраф почти всегда был одинаковым нулём у большинства
    кандидатов, и выбор схлопывался к тому же tie-break по calories, что и
    в ``_select_balanced`` — variety-вариант вырождался в balanced.
    ``source_category`` грубее (это категория рецепта, а не состав), но
    гораздо денсер как сигнал повтора "типа блюда".

    Это ситуативный сигнал, а не гарантия полного разнообразия плана: он
    предотвращает повтор конкретно тогда, когда калорийно-оптимальный выбор
    совпал бы по категории (или сильно пересёкся по ингредиентам) с уже
    выбранным. Когда естественно выбранные по calories кандидаты и так
    разных категорий и не пересекаются, штраф не меняет исход — это
    ожидаемое поведение, не баг.
    """
    slot_counts = Counter(meal_slots)
    selected: list[dict] = []
    selected_ids: set = set()
    selected_ingredients: set[str] = set()
    selected_categories: set[str] = set()
    total = 0.0
    n = len(meal_slots)

    for index, meal_type in enumerate(meal_slots):
        remaining_slots = n - index
        residual_target = (target_calories - total) / remaining_slots

        pool = pools[meal_type]
        candidates = [c for c in pool if c["recipe_id"] not in selected_ids]
        _require_candidates(candidates, pool, meal_type, slot_counts[meal_type])

        def penalty(candidate: dict) -> tuple[int, float, float]:
            category = candidate["source_category"]
            category_repeat_penalty = int(
                category is not None and category in selected_categories
            )

            candidate_ingredients = {
                ingredient.casefold() for ingredient in candidate["ingredients"]
            }
            union = candidate_ingredients | selected_ingredients
            intersection = candidate_ingredients & selected_ingredients
            ingredient_jaccard_penalty = (
                len(intersection) / len(union) if union else 0.0
            )

            calorie_diff = abs(candidate["calories"] - residual_target)

            return (category_repeat_penalty, ingredient_jaccard_penalty, calorie_diff)

        best = min(candidates, key=penalty)
        selected.append(best)
        selected_ids.add(best["recipe_id"])
        selected_ingredients |= {
            ingredient.casefold() for ingredient in best["ingredients"]
        }
        if best["source_category"] is not None:
            selected_categories.add(best["source_category"])
        total += best["calories"]

    return selected


def build_meal_plan_variants(
    client: QdrantClient,
    embed_fn: Callable[[list[str]], list[list[float]]],
    constraints: MealPlanConstraints,
    pool_size: int = 30,
) -> list[MealPlanVariant]:
    """Построить три варианта дневного рациона поверх одного общего пула кандидатов.

    Пулы кандидатов запрашиваются один раз через ``_fetch_candidate_pools``
    и переиспользуются между всеми тремя вариантами. ``pool_size`` пробрасывается
    в ``_fetch_candidate_pools`` как есть (дефолт ``30`` сохраняет прежнее
    поведение) — это точка релаксации для вызывающего кода (например,
    LangGraph-узла с retry), который может повторить вызов с более широким
    пулом, если предыдущая попытка упёрлась в ``ValueError`` из-за нехватки
    кандидатов.
    """
    pools = _fetch_candidate_pools(client, embed_fn, constraints, pool_size)

    variant_builders = (
        ("balanced", _select_balanced),
        ("quick_cooking", _select_quick_cooking),
        ("variety", _select_variety),
    )

    variants = []
    for label, select_fn in variant_builders:
        slots = select_fn(constraints.meal_slots, pools, constraints.target_calories)
        total_calories = sum(slot["calories"] for slot in slots)
        within_tolerance = _is_within_tolerance(
            total_calories, constraints.target_calories, constraints.calorie_tolerance
        )
        variants.append(
            MealPlanVariant(
                label=label,
                slots=slots,
                total_calories=total_calories,
                within_tolerance=within_tolerance,
            )
        )

    return variants
