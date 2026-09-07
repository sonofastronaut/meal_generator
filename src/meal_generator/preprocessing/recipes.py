"""Вспомогательные функции для предобработки сырых рецептов Food.com."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sized

import numpy as np
import pandas as pd
from pandas.api.types import is_scalar

from .exclusions import classify_allergen_categories
from .meal_types import classify_meal_types

REQUIRED_COLUMNS = (
    "RecipeId",
    "Name",
    "RecipeIngredientParts",
    "RecipeInstructions",
    "RecipeServings",
    "Calories",
    "ProteinContent",
    "FatContent",
    "CarbohydrateContent",
)

CURATED_COLUMN_MAP = {
    "RecipeId": "recipe_id",
    "Name": "name",
    "RecipeCategory": "source_category",
    "Keywords": "keywords",
    "Calories": "calories",
    "ProteinContent": "protein",
    "FatContent": "fat",
    "CarbohydrateContent": "carbs",
    "RecipeServings": "servings",
    "RecipeIngredientParts": "ingredients",
    "RecipeInstructions": "instructions",
    "AggregatedRating": "rating",
    "ReviewCount": "review_count",
}

CURATED_RECIPE_COLUMNS = (
    "recipe_id",
    "name",
    "source_category",
    "keywords",
    "calories",
    "protein",
    "fat",
    "carbs",
    "servings",
    "total_time_minutes",
    "ingredients",
    "instructions",
    "rating",
    "review_count",
    "source",
)

OPTIONAL_CURATED_SOURCE_COLUMNS = {
    "AggregatedRating",
    "ReviewCount",
}

_TOTAL_TIME_PATTERN = re.compile(r"PT(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?")


def _parse_total_time_minutes(value: object) -> int | None:
    """Преобразовать длительность Food.com в формате PT[hours]H[minutes]M в целые минуты."""
    if not isinstance(value, str):
        return None

    match = _TOTAL_TIME_PATTERN.fullmatch(value.strip())
    if match is None or not any(match.group("hours", "minutes")):
        return None

    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes") or 0)
    return hours * 60 + minutes


def _is_missing_or_empty(value: object) -> bool:
    """Проверить, что скалярное значение или коллекция не содержат данных."""
    if value is None:
        return True

    if isinstance(value, str):
        normalized = value.strip()
        return not normalized or normalized in {"[]", "c()"}

    missing = pd.isna(value)
    if is_scalar(missing) and bool(missing):
        return True

    return isinstance(value, Sized) and len(value) == 0


def _to_string_list(value: object) -> list[str]:
    """Канонизировать скаляр или коллекцию в обычный список строк."""
    if _is_missing_or_empty(value):
        return []

    if isinstance(value, str):
        return [value.strip()]

    if isinstance(value, Iterable):
        normalized_items = []
        for item in value:
            if _is_missing_or_empty(item):
                continue
            normalized_item = str(item).strip()
            if normalized_item:
                normalized_items.append(normalized_item)
        return normalized_items

    normalized_value = str(value).strip()
    return [normalized_value] if normalized_value else []


def _strip_if_string(value: object) -> object:
    """Удалить крайние пробелы у строки, не изменяя значения других типов."""
    return value.strip() if isinstance(value, str) else value


def filter_technically_valid_recipes(recipes: pd.DataFrame) -> pd.DataFrame:
    """Вернуть сырые рецепты Food.com, пригодные для технического использования.

    Валидация намеренно ограничена данными, необходимыми для идентификации
    рецепта, отображения ингредиентов и инструкций, а также расчёта пищевой
    ценности порции. Продуктовые ограничения — рейтинг, количество отзывов,
    максимальная калорийность, предел порций и количества ингредиентов — здесь
    не применяются.

    Входной DataFrame не изменяется. В возвращаемой копии сохраняются все его
    столбцы и исходный индекс.

    Исключения:
        ValueError: отсутствует столбец, обязательный для технической валидации.
    """
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in recipes.columns
    ]
    if missing_columns:
        columns = ", ".join(missing_columns)
        raise ValueError(f"Missing required recipe columns: {columns}")

    recipe_ids = pd.to_numeric(recipes["RecipeId"], errors="coerce")
    finite_recipe_ids = np.isfinite(recipe_ids)
    integer_recipe_ids = recipe_ids.where(finite_recipe_ids).mod(1).eq(0)
    recipe_id_valid = finite_recipe_ids & recipe_ids.gt(0) & integer_recipe_ids
    name_valid = ~recipes["Name"].map(_is_missing_or_empty)
    ingredients_valid = recipes["RecipeIngredientParts"].map(_to_string_list).map(bool)
    instructions_valid = recipes["RecipeInstructions"].map(_to_string_list).map(bool)

    servings = pd.to_numeric(recipes["RecipeServings"], errors="coerce")
    calories = pd.to_numeric(recipes["Calories"], errors="coerce")
    macros = recipes[["ProteinContent", "FatContent", "CarbohydrateContent"]].apply(
        pd.to_numeric, errors="coerce"
    )
    finite_nutrition = (
        np.isfinite(servings)
        & np.isfinite(calories)
        & np.isfinite(macros).all(axis="columns")
    )

    valid_mask = (
        recipe_id_valid
        & name_valid
        & ingredients_valid
        & instructions_valid
        & finite_nutrition
        & servings.gt(0)
        & calories.gt(0)
        & macros.ge(0).all(axis="columns")
    )

    return recipes.loc[valid_mask].copy()


def build_curated_recipes(recipes: pd.DataFrame) -> pd.DataFrame:
    """Преобразовать технически валидные рецепты Food.com во внутреннюю схему.

    Функция не выполняет повторную техническую или продуктовую валидацию.
    Она выбирает и переименовывает поля исходной схемы, а также преобразует
    ``TotalTime`` в количество минут. Отсутствующие или некорректные
    ``TotalTime``, ``AggregatedRating`` и ``ReviewCount`` допустимы и не приводят
    к удалению рецепта. Исходный DataFrame не изменяется.

    ``calories``/``protein``/``fat``/``carbs`` в исходной Food.com-схеме — это
    значения на весь рецепт целиком (``RecipeServings`` порций), а не на одну
    порцию. Весь downstream-код (retrieval-фильтры по калориям, optimizer,
    объяснения пользователю) молчаливо считает эти поля значениями на один
    приём пищи, поэтому здесь они делятся на ``servings`` — с этого момента
    ``calories``/``protein``/``fat``/``carbs`` в curated-схеме это
    ``recipe_total / servings``, то есть на одну порцию. Деление безопасно:
    ``filter_technically_valid_recipes`` (обязательный шаг пайплайна перед
    этой функцией) уже гарантирует, что ``servings`` конечен и строго
    положителен. Реалистичный потолок калорийности на порцию — отдельная
    продуктовая эвристика, не часть этой трансформации схемы (см.
    ``filter_plausible_calories``).

    Исключения:
        ValueError: отсутствует столбец, необходимый для преобразования схемы.
    """
    missing_columns = [
        column
        for column in CURATED_COLUMN_MAP
        if column not in OPTIONAL_CURATED_SOURCE_COLUMNS
        and column not in recipes.columns
    ]
    if missing_columns:
        columns = ", ".join(missing_columns)
        raise ValueError(
            f"Missing columns required to build curated recipes: {columns}"
        )

    available_columns = [
        column for column in CURATED_COLUMN_MAP if column in recipes.columns
    ]
    curated = (
        recipes.loc[:, available_columns].rename(columns=CURATED_COLUMN_MAP).copy()
    )

    if "TotalTime" in recipes.columns:
        total_time_minutes = recipes["TotalTime"].map(_parse_total_time_minutes)
        total_time_minutes = total_time_minutes.astype("Int64").astype("object")
        total_time_minutes = total_time_minutes.where(total_time_minutes.notna(), None)
    else:
        total_time_minutes = pd.Series(None, index=recipes.index, dtype="object")
    curated["total_time_minutes"] = total_time_minutes
    curated["total_time_minutes"] = curated["total_time_minutes"].where(
        curated["total_time_minutes"].notna(), None
    )

    for source_column in ("AggregatedRating", "ReviewCount"):
        curated_column = CURATED_COLUMN_MAP[source_column]
        if curated_column not in curated.columns:
            curated[curated_column] = None

    curated["recipe_id"] = pd.to_numeric(curated["recipe_id"], errors="coerce").astype(
        "int64"
    )
    for column in ("calories", "protein", "fat", "carbs", "servings"):
        curated[column] = pd.to_numeric(curated[column], errors="coerce")
    for column in ("rating", "review_count"):
        curated[column] = pd.to_numeric(curated[column], errors="coerce")

    for column in ("calories", "protein", "fat", "carbs"):
        curated[column] = curated[column] / curated["servings"]

    for column in ("name", "source_category"):
        curated[column] = curated[column].map(_strip_if_string)

    for column in ("ingredients", "instructions", "keywords"):
        curated[column] = curated[column].map(_to_string_list)

    curated["source"] = "food.com"
    return curated.loc[:, CURATED_RECIPE_COLUMNS]


def filter_plausible_calories(
    recipes: pd.DataFrame, max_calories_per_serving: float = 2000.0
) -> pd.DataFrame:
    """Отсечь curated-рецепты с неправдоподобной калорийностью на порцию.

    Продуктовая эвристика против выбросов/ошибок ввода в исходных данных —
    отдельный шаг пайплайна, применяемый после ``build_curated_recipes``
    (когда ``calories`` уже нормализована на порцию, см. её докстроку) и до
    ``add_meal_types``/``add_allergen_categories``/индексации.

    Порог ``2000.0`` подобран эмпирически: на реальном датасете из ~336к
    curated-рецептов калорийность на порцию выше 2000 составляет ~0.41%
    записей, и выборочная проверка показала, что это преимущественно
    data-quality выбросы (целые выпечки/торты/партии с некорректным
    ``RecipeServings`` в исходных данных, например ``servings=1`` для целого
    торта), а не легитимные плотные блюда. Деление на ``servings`` в
    ``build_curated_recipes`` такие записи не спасает — при некорректном
    ``servings`` результат деления остаётся таким же неправдоподобным, как и
    исходное значение.

    Входной DataFrame не изменяется. В возвращаемой копии сохраняются исходный
    индекс и все столбцы у оставшихся строк.

    Исключения:
        ValueError: отсутствует столбец ``calories``.
    """
    if "calories" not in recipes.columns:
        raise ValueError(
            "Missing columns required to filter plausible calories: calories"
        )

    return recipes.loc[recipes["calories"] <= max_calories_per_serving].copy()


def add_meal_types(recipes: pd.DataFrame) -> pd.DataFrame:
    """Добавить столбец ``meal_types`` к curated-рецептам.

    Для каждой строки тип приёма пищи определяется функцией
    ``classify_meal_types`` по столбцам ``source_category`` и ``keywords``.
    Исходный DataFrame не изменяется.

    Исключения:
        ValueError: отсутствует столбец, необходимый для классификации.
    """
    missing_columns = [
        column
        for column in ("source_category", "keywords")
        if column not in recipes.columns
    ]
    if missing_columns:
        columns = ", ".join(missing_columns)
        raise ValueError(f"Missing columns required to add meal types: {columns}")

    result = recipes.copy()
    result["meal_types"] = [
        classify_meal_types(source_category, keywords)
        for source_category, keywords in zip(
            result["source_category"], result["keywords"]
        )
    ]
    return result


def add_allergen_categories(recipes: pd.DataFrame) -> pd.DataFrame:
    """Добавить столбец ``allergen_categories`` к curated-рецептам.

    Для каждой строки набор allergen-категорий определяется функцией
    ``classify_allergen_categories`` по столбцу ``ingredients``. Исходный
    DataFrame не изменяется.

    Исключения:
        ValueError: отсутствует столбец, необходимый для классификации.
    """
    missing_columns = [
        column for column in ("ingredients",) if column not in recipes.columns
    ]
    if missing_columns:
        columns = ", ".join(missing_columns)
        raise ValueError(
            f"Missing columns required to add allergen categories: {columns}"
        )

    result = recipes.copy()
    result["allergen_categories"] = [
        classify_allergen_categories(ingredients)
        for ingredients in result["ingredients"]
    ]
    return result
