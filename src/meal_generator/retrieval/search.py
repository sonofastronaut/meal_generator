"""Поиск рецептов: structured constraints + семантический поиск по query_text.

``query_text`` эмбеддится напрямую, без обработки через
``build_embedding_text`` — та функция собирает document-side текст из
name/source_category/keywords/ingredients curated-рецепта для индексации, а
здесь на входе уже пользовательский запрос в свободной форме, который нужно
эмбеддить как есть, без document-side сборки.

``embed_fn`` инъектируется, а не хардкодится конкретная модель — по той же
причине тестируемости без реальной модели, что и в ``qdrant_index.py``:
логику построения фильтра и запроса к Qdrant можно проверить с фейковым
эмбеддером и ``QdrantClient(":memory:")``.

Для MiniLM (текущая модель, см. ``COLLECTION_NAME`` в ``qdrant_index.py``)
query-префикс не нужен — в отличие от bge-base, где query-side текст
принято оборачивать в отдельный промпт-префикс (``"query: ..."``), отличный
от document-side. При будущем апгрейде модели на bge-base эту функцию
придётся ревизировать: потребуется добавить префикс к ``query_text`` перед
эмбеддингом.

``search_recipes_by_rating`` — второй, non-semantic путь retrieval, поверх
той же фильтрации. Он появился из эмпирически обнаруженной проблемы:
пустая строка на входе эмбеддинг-модели даёт вырожденный, бессмысленный
вектор, и косинусное сходство с ним — это не "нейтральный" ранкинг, а
фактически случайный шум, который тем не менее выглядит как осмысленный
score. Когда осмысленного query_text нет (или семантический поиск заведомо
неуместен, как для завтрака — см. ``_fetch_candidate_pools`` в
``planning/optimizer.py``), честная фильтрация по structured constraints с
сортировкой по популярности (``rating``/``review_count``) даёт куда более
предсказуемый и объяснимый результат, чем векторный поиск по мусорному
вектору.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from qdrant_client import QdrantClient, models

from ..indexing.qdrant_index import COLLECTION_NAME

OVERSAMPLE_FACTOR = 10
MAX_SCROLL_LIMIT = 300


@dataclass
class SearchConstraints:
    query_text: str
    meal_type: str | None = None
    calories_min: float | None = None
    calories_max: float | None = None
    protein_min: float | None = None
    max_total_time_minutes: int | None = None
    excluded_categories: list[str] = field(default_factory=list)
    limit: int = 20


def build_search_filter(constraints: SearchConstraints) -> models.Filter:
    """Построить Qdrant-фильтр из structured constraints.

    Чистая функция — не ходит в Qdrant. Условие по каждому полю
    добавляется в ``must`` только если хотя бы одна связанная граница
    задана; ``excluded_categories`` превращается в отдельные
    ``must_not``-условия, по одному на категорию.
    """
    must: list[models.FieldCondition] = []

    if constraints.meal_type is not None:
        must.append(
            models.FieldCondition(
                key="meal_types",
                match=models.MatchValue(value=constraints.meal_type),
            )
        )

    if constraints.calories_min is not None or constraints.calories_max is not None:
        must.append(
            models.FieldCondition(
                key="calories",
                range=models.Range(
                    gte=constraints.calories_min, lte=constraints.calories_max
                ),
            )
        )

    if constraints.protein_min is not None:
        must.append(
            models.FieldCondition(
                key="protein",
                range=models.Range(gte=constraints.protein_min),
            )
        )

    if constraints.max_total_time_minutes is not None:
        must.append(
            models.FieldCondition(
                key="total_time_minutes",
                range=models.Range(lte=constraints.max_total_time_minutes),
            )
        )

    must_not = [
        models.FieldCondition(
            key="allergen_categories", match=models.MatchValue(value=category)
        )
        for category in constraints.excluded_categories
    ]

    if not must and not must_not:
        return models.Filter()

    return models.Filter(must=must, must_not=must_not)


def search_recipes(
    client: QdrantClient,
    embed_fn: Callable[[list[str]], list[list[float]]],
    constraints: SearchConstraints,
) -> list[dict]:
    """Найти рецепты, комбинируя семантический поиск с structured-фильтром.

    Вектор запроса считается ``embed_fn`` из ``constraints.query_text``,
    фильтр строится через ``build_search_filter``. Возвращает список
    payload-словарей найденных точек с добавленным ключом ``score``.
    """
    query_vector = embed_fn([constraints.query_text])[0]
    query_filter = build_search_filter(constraints)

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=query_filter,
        limit=constraints.limit,
    )

    return [{"score": hit.score, **hit.payload} for hit in response.points]


def search_recipes_by_rating(
    client: QdrantClient, constraints: SearchConstraints
) -> list[dict]:
    """Найти рецепты по structured-фильтру, без семантического поиска.

    Фильтр строится через ``build_search_filter`` — та же логика, что и в
    ``search_recipes``, не дублируется. Вместо векторного поиска
    (``query_points``) используется ``client.scroll``: он не умеет
    сортировать на стороне Qdrant, поэтому зачёрпывается пул с запасом
    (``limit * OVERSAMPLE_FACTOR``, но не больше ``MAX_SCROLL_LIMIT``),
    сортировка по ``rating``/``review_count`` (не-``None`` и по убыванию,
    в этом порядке приоритета) выполняется на своей стороне уже после, и
    только затем результат обрезается до ``constraints.limit``.

    Ключ ``"score"`` есть в каждом результате (всегда ``None``) для
    консистентности со ``search_recipes`` — downstream-коду не нужно
    условно проверять его наличие в зависимости от того, какой путь
    retrieval был использован.

    Известное ограничение: для breakfast-слотов этот путь используется
    всегда, даже когда query_text осмысленно описывает вайб всего дня (не
    только non-breakfast части) — breakfast полностью не учитывает
    cuisine/vibe предпочтения пользователя, отбор идёт только по рейтингу.
    Это осознанный компромисс ради устранения query bleed (см. выше), не
    баг: более тонкое решение — гибридный ранкинг (rating как основной
    сигнал + лёгкий semantic re-rank поверх топ-N) — возможный будущий шаг,
    не реализовано в MVP.
    """
    query_filter = build_search_filter(constraints)
    scroll_limit = min(constraints.limit * OVERSAMPLE_FACTOR, MAX_SCROLL_LIMIT)

    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=query_filter,
        limit=scroll_limit,
    )

    candidates = [point.payload for point in points]
    candidates.sort(
        key=lambda c: (
            c["rating"] is None,
            -(c["rating"] or 0),
            c["review_count"] is None,
            -(c["review_count"] or 0),
        )
    )

    return [{"score": None, **candidate} for candidate in candidates[: constraints.limit]]
