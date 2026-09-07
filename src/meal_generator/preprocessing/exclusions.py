"""Exclusion-фильтрация рецептов по категориям продуктов (аллергии/непереносимости).

В отличие от ``classify_meal_types``, здесь используется token-based substring
match, а не точное совпадение целой нормализованной строки — потому что
``ingredients`` это произвольные фразы, а не стандартизированные Food.com
категории.

Известное ограничение: сигнал "flour" матчит категорию ``gluten_wheat`` даже
в ingredient-строках вроде "almond flour" или "rice flour", которые
фактически безглютеновые — это осознанное упрощение v1, а не баг.
"""

from __future__ import annotations

import re

from collections.abc import Iterable


FISH_SIGNALS = frozenset(
    {
        "fish",
        "salmon",
        "tuna",
        "cod",
        "tilapia",
        "anchovy",
        "anchovies",
        "sardine",
        "sardines",
        "halibut",
        "trout",
        "mahi",
        "snapper",
        "catfish",
        "haddock",
        "sole",
        "herring",
        "swordfish",
    }
)

SHELLFISH_SIGNALS = frozenset(
    {
        "shrimp",
        "prawn",
        "prawns",
        "crab",
        "lobster",
        "clam",
        "clams",
        "mussel",
        "mussels",
        "oyster",
        "oysters",
        "scallop",
        "scallops",
        "crawfish",
    }
)

PEANUT_SIGNALS = frozenset(
    {
        "peanut",
        "peanuts",
        "groundnut",
        "groundnuts",
    }
)

TREE_NUTS_SIGNALS = frozenset(
    {
        "almond",
        "almonds",
        "walnut",
        "walnuts",
        "pecan",
        "pecans",
        "cashew",
        "cashews",
        "pistachio",
        "pistachios",
        "hazelnut",
        "hazelnuts",
        "macadamia",
    }
)

DAIRY_SIGNALS = frozenset(
    {
        "milk",
        "cream",
        "butter",
        "cheese",
        "yogurt",
        "yoghurt",
        "margarine",
    }
)

EGG_SIGNALS = frozenset(
    {
        "egg",
        "eggs",
    }
)

GLUTEN_WHEAT_SIGNALS = frozenset(
    {
        "flour",
        "wheat",
        "bread",
        "breadcrumbs",
        "pasta",
        "spaghetti",
        "macaroni",
        "noodles",
        "barley",
        "rye",
    }
)

SOY_SIGNALS = frozenset(
    {
        "soy",
        "soya",
        "tofu",
        "edamame",
    }
)

EXCLUSION_CATEGORY_SIGNALS: dict[str, frozenset[str]] = {
    "fish": FISH_SIGNALS,
    "shellfish": SHELLFISH_SIGNALS,
    "peanut": PEANUT_SIGNALS,
    "tree_nuts": TREE_NUTS_SIGNALS,
    "dairy": DAIRY_SIGNALS,
    "egg": EGG_SIGNALS,
    "gluten_wheat": GLUTEN_WHEAT_SIGNALS,
    "soy": SOY_SIGNALS,
}


def _normalize_signal(value: object) -> str | None:
    """Нормализовать строковый сигнал для сравнения."""
    if not isinstance(value, str):
        return None

    normalized = value.strip().casefold()
    return normalized or None


def _tokenize_ingredient(ingredient: str) -> list[str]:
    """Разбить строку ингредиента на нормализованные буквенные токены."""
    normalized = ingredient.strip().casefold()
    return re.findall(r"[a-z]+", normalized)


def ingredient_matches_exclusion(
    ingredient: str, excluded_categories: Iterable[str]
) -> bool:
    """Проверить, содержит ли ингредиент сигнал хотя бы одной excluded-категории.

    Неизвестные категории в ``excluded_categories`` молча игнорируются
    (fail-open).
    """
    tokens = set(_tokenize_ingredient(ingredient))
    if not tokens:
        return False

    for category in excluded_categories:
        signals = EXCLUSION_CATEGORY_SIGNALS.get(category)
        if signals is not None and tokens & signals:
            return True

    return False


def ingredient_matches_terms(ingredient: str, avoid_terms: list[str]) -> bool:
    """Проверить, содержит ли ингредиент хотя бы один из произвольных ``avoid_terms``.

    В отличие от ``ingredient_matches_exclusion``, здесь нет фиксированного
    словаря ``EXCLUSION_CATEGORY_SIGNALS`` — эта функция для динамических,
    введённых пользователем на лету уточнений ("не люблю овсянку", "без
    грибов"), а не для заранее закодированного набора allergen-категорий.
    Токенизация переиспользует ``_tokenize_ingredient`` — тот же
    token-based подход и то же поведение на границах слов (scallion не
    матчит scallop), но термины сравниваются напрямую, без промежуточного
    маппинга на категорию.

    Пустой ``avoid_terms`` → ``False``.
    """
    if not avoid_terms:
        return False

    tokens = set(_tokenize_ingredient(ingredient))
    if not tokens:
        return False

    normalized_terms = {term.casefold().strip() for term in avoid_terms}
    return bool(tokens & normalized_terms)


def recipe_is_excluded(
    ingredients: list[str], excluded_categories: Iterable[str]
) -> bool:
    """Проверить, должен ли рецепт быть исключён по списку категорий.

    Категории материализуются один раз, чтобы не исчерпать переданный
    итератор при повторной проверке на каждом ингредиенте.
    """
    categories = list(excluded_categories)
    return any(
        ingredient_matches_exclusion(ingredient, categories)
        for ingredient in ingredients
    )


def classify_allergen_categories(ingredients: list[str]) -> list[str]:
    """Определить все allergen-категории, присутствующие в составе рецепта.

    Это денормализация для индексации в Qdrant (payload-поле
    ``allergen_categories``): результат не привязан к какому-либо
    пользовательскому запросу и вычисляется один раз при индексации, чтобы
    exclusion-фильтрация на retrieval-шаге работала как обычный payload-фильтр
    в Qdrant, без Python post-filter. Не путать с ``recipe_is_excluded``,
    которая принимает конкретный список категорий из пользовательского
    запроса и возвращает bool.

    Порядок вывода фиксирован и соответствует порядку ключей в
    ``EXCLUSION_CATEGORY_SIGNALS`` (fish, shellfish, peanut, tree_nuts,
    dairy, egg, gluten_wheat, soy), а не порядку появления сигналов в
    ``ingredients``.
    """
    return [
        category
        for category in EXCLUSION_CATEGORY_SIGNALS
        if any(
            ingredient_matches_exclusion(ingredient, [category])
            for ingredient in ingredients
        )
    ]
