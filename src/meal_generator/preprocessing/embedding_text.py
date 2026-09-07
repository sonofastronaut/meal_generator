"""Построение текста для эмбеддинга рецепта перед индексацией в Qdrant.

Дедупликация нужна потому, что в Food.com ``source_category`` часто
дублируется внутри ``keywords`` (например ``source_category="Chicken"`` и
``keywords=["Chicken", "Easy"]``): без дедупликации такое слово попадало бы
в текст дважды и получало бы непреднамеренный лишний вес в эмбеддинге.

Текст собирается без какой-либо структурной разметки (без меток вроде
"Категория:" или "Ингредиенты:") — в отличие от промптинга LLM,
sentence-transformer модели не выигрывают от служебных меток, им нужен
просто плотный набор токенов для векторизации.
"""

from __future__ import annotations


def _normalize_part(value: object) -> str | None:
    """Привести часть текста к непустой строке или вернуть None."""
    if not isinstance(value, str):
        return None

    normalized = value.strip()
    return normalized or None


def build_embedding_text(
    name: str,
    source_category: object,
    keywords: list[str],
    ingredients: list[str],
) -> str:
    """Построить document-side текст для эмбеддинга curated-рецепта.

    Порядок частей: ``name``, затем ``source_category`` (если задана),
    затем ``keywords`` по порядку, затем ``ingredients`` по порядку.
    Пустые, ``None`` и пустые после ``strip()`` части пропускаются.

    Дедупликация выполняется без учёта регистра по всему объединённому
    списку сразу (name против category против keywords против ingredients),
    с сохранением первого по порядку вхождения каждого значения.

    Префикс для retrieval-запросов (например, у BGE-моделей) сюда не
    добавляется — это отдельный query-side шаг, не относящийся к индексации
    корпуса.
    """
    raw_parts = [name, source_category, *keywords, *ingredients]

    seen: set[str] = set()
    deduplicated_parts: list[str] = []
    for raw_part in raw_parts:
        normalized_part = _normalize_part(raw_part)
        if normalized_part is None:
            continue

        dedup_key = normalized_part.casefold()
        if dedup_key in seen:
            continue

        seen.add(dedup_key)
        deduplicated_parts.append(normalized_part)

    return " ".join(deduplicated_parts)
