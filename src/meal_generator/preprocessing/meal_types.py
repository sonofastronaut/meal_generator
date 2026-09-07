"""Классификация типа приёма пищи для curated-рецептов."""

from __future__ import annotations

from collections.abc import Iterable


BREAKFAST_SIGNALS = frozenset(
    {
        "breakfast",
        "brunch",
    }
)

MAIN_MEAL_SIGNALS = frozenset(
    {
        "one dish meal",
        "chicken",
        "chicken breast",
        "meat",
        "pork",
        "stew",
        "poultry",
        "steak",
    }
)

DESSERT_SIGNALS = frozenset(
    {
        "dessert",
        "frozen desserts",
    }
)

LUNCH_SNACK_SIGNALS = frozenset(
    {
        "lunch/snacks",
    }
)

_MEAL_TYPE_SIGNALS = (
    ("breakfast", BREAKFAST_SIGNALS),
    ("main_meal", MAIN_MEAL_SIGNALS),
    ("dessert", DESSERT_SIGNALS),
    ("lunch_snack", LUNCH_SNACK_SIGNALS),
)


def _normalize_signal(value: object) -> str | None:
    """Нормализовать строковый сигнал для точного сравнения."""
    if not isinstance(value, str):
        return None

    normalized = value.strip().casefold()
    return normalized or None


def classify_meal_types(
    source_category: object,
    keywords: Iterable[object] | str | None,
) -> list[str]:
    """Классифицировать один curated-рецепт по категории и ключевым словам.

    Сравнение выполняется по нормализованным значениям целиком, без поиска
    подстрок. Классификация мульти-лейбл: возвращаются все типы, для
    которых нашёлся хотя бы один совпавший сигнал. Порядок элементов в
    результате фиксированный (breakfast, main_meal, dessert, lunch_snack)
    и не отражает приоритет.
    """
    normalized_values: set[str] = set()

    normalized_category = _normalize_signal(source_category)
    if normalized_category is not None:
        normalized_values.add(normalized_category)

    if isinstance(keywords, str):
        keyword_values: Iterable[object] = (keywords,)
    elif keywords is None:
        keyword_values = ()
    else:
        keyword_values = keywords

    for keyword in keyword_values:
        normalized_keyword = _normalize_signal(keyword)
        if normalized_keyword is not None:
            normalized_values.add(normalized_keyword)

    return [
        meal_type
        for meal_type, signals in _MEAL_TYPE_SIGNALS
        if normalized_values & signals
    ]
