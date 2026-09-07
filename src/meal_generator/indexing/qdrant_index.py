"""Индексация curated-рецептов в Qdrant.

Построение payload/точек отделено от похода в реальную модель эмбеддингов:
``index_recipes`` принимает ``embed_fn`` как параметр (инверсия зависимости),
чтобы логику батчинга и апсерта можно было протестировать без загрузки
sentence-transformers и без реального Qdrant-контейнера — в тестах
используется ``QdrantClient(":memory:")`` и детерминированный фейковый
эмбеддер, а реальная модель и реальный клиент живут в
``scripts/build_index.py``.

Версия модели эмбеддингов зашита в ``COLLECTION_NAME`` намеренно: при смене
модели эмбеддинги окажутся в другой коллекции и не смешаются со старыми
векторами другой размерности/пространства.

Решение по колонкам ``meal_types``/``allergen_categories``: ``index_recipes``
ожидает, что они уже посчитаны заранее (``add_meal_types``,
``add_allergen_categories`` из ``meal_generator.preprocessing.recipes``) и
требует их присутствия, а не считает сама. Индексация — самостоятельный шаг
пайплайна, ей не нужно знать, как классифицируются типы приёма пищи или
аллергены; она лишь проверяет, что curated-контракт соблюдён — тот же
fail-fast паттерн, что уже используется в ``add_meal_types`` и
``add_allergen_categories`` при отсутствии обязательных колонок.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pandas as pd
from qdrant_client import QdrantClient, models

from ..preprocessing.embedding_text import build_embedding_text

COLLECTION_NAME = "recipes_minilm_v1"
EMBEDDING_DIMENSION = 384

REQUIRED_COLUMNS = (
    "recipe_id",
    "name",
    "source_category",
    "keywords",
    "meal_types",
    "calories",
    "protein",
    "fat",
    "carbs",
    "servings",
    "total_time_minutes",
    "ingredients",
    "rating",
    "review_count",
    "allergen_categories",
)


def _none_if_missing(value: object) -> object:
    """Преобразовать NaN/NA-подобные значения pandas в None для payload."""
    if value is None:
        return None

    try:
        is_missing = pd.isna(value)
    except (TypeError, ValueError):
        return value

    return None if is_missing else value


def build_payload(recipe: pd.Series) -> dict:
    """Построить payload-словарь Qdrant-точки из строки curated-DataFrame.

    Строка должна уже содержать ``meal_types`` (из ``add_meal_types``) и
    ``allergen_categories`` (из ``add_allergen_categories``). Функция не
    ходит в Qdrant — чистое преобразование данных, пригодное для юнит-тестов.
    ``total_time_minutes``/``rating``/``review_count`` сохраняются как
    ``None``, если отсутствуют — Qdrant это поддерживает. ``ingredients`` и
    ``instructions`` сохраняются целиком — полный рецепт, а не только
    структурированные constraints, чтобы его можно было показать пользователю.
    """
    return {
        "recipe_id": int(recipe["recipe_id"]),
        "name": recipe["name"],
        "source_category": _none_if_missing(recipe["source_category"]),
        "meal_types": list(recipe["meal_types"]),
        "calories": _none_if_missing(recipe["calories"]),
        "protein": _none_if_missing(recipe["protein"]),
        "fat": _none_if_missing(recipe["fat"]),
        "carbs": _none_if_missing(recipe["carbs"]),
        "servings": _none_if_missing(recipe["servings"]),
        "total_time_minutes": _none_if_missing(recipe["total_time_minutes"]),
        "ingredients": list(recipe["ingredients"]),
        "instructions": list(recipe["instructions"]),
        "rating": _none_if_missing(recipe["rating"]),
        "review_count": _none_if_missing(recipe["review_count"]),
        "allergen_categories": list(recipe["allergen_categories"]),
    }


def ensure_collection(client: QdrantClient) -> None:
    """Идемпотентно создать коллекцию ``COLLECTION_NAME`` с payload-индексами.

    Если коллекция уже существует, функция ничего не делает — повторный
    вызов безопасен.
    """
    if client.collection_exists(COLLECTION_NAME):
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=EMBEDDING_DIMENSION, distance=models.Distance.COSINE
        ),
    )

    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="meal_types",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="allergen_categories",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="calories",
        field_schema=models.PayloadSchemaType.FLOAT,
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="protein",
        field_schema=models.PayloadSchemaType.FLOAT,
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="total_time_minutes",
        field_schema=models.PayloadSchemaType.INTEGER,
    )


def _iter_batches(recipes: pd.DataFrame, batch_size: int) -> Iterator[pd.DataFrame]:
    """Разбить DataFrame на последовательные срезы по ``batch_size`` строк."""
    for start in range(0, len(recipes), batch_size):
        yield recipes.iloc[start : start + batch_size]


def index_recipes(
    client: QdrantClient,
    curated_recipes: pd.DataFrame,
    embed_fn: Callable[[list[str]], list[list[float]]],
    batch_size: int = 256,
) -> int:
    """Проиндексировать curated-рецепты в Qdrant батчами.

    Для каждой строки строится document-side текст через
    ``build_embedding_text``, эмбеддинги считаются переданным ``embed_fn``
    (не хардкодится конкретная модель — инверсия зависимости для
    тестируемости), точка формируется с ``id=recipe_id`` и payload из
    ``build_payload``. ``curated_recipes`` должен уже содержать колонки
    ``meal_types`` и ``allergen_categories`` (см. докстроку модуля).

    Возвращает общее количество проиндексированных точек.

    Исключения:
        ValueError: отсутствует колонка, необходимая для индексации.
    """
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in curated_recipes.columns
    ]
    if missing_columns:
        columns = ", ".join(missing_columns)
        raise ValueError(f"Missing columns required to index recipes: {columns}")

    ensure_collection(client)

    total_indexed = 0
    for batch in _iter_batches(curated_recipes, batch_size):
        rows = [row for _, row in batch.iterrows()]
        texts = [
            build_embedding_text(
                row["name"],
                row["source_category"],
                row["keywords"],
                row["ingredients"],
            )
            for row in rows
        ]
        vectors = embed_fn(texts)

        points = [
            models.PointStruct(
                id=int(row["recipe_id"]),
                vector=vector,
                payload=build_payload(row),
            )
            for row, vector in zip(rows, vectors)
        ]

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        total_indexed += len(points)

    return total_indexed
