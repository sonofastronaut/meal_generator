"""Разовый диагностический scratch-скрипт: query_text "перетягивает" breakfast-пул?

Сравнивает breakfast-пул для одного и того же excluded_categories-фильтра с
vibe-запросом ("spicy asian dinner") и с пустым query_text, чтобы понять,
действительно ли семантический поиск по несвойственному завтраку query_text
подмешивает в пул нехарактерные для завтрака блюда, или наблюдаемые
странности объясняются качеством/разметкой самих curated-данных (например,
слабой breakfast-классификацией source_category/keywords), а не retrieval-
слоем. Не тест, не часть постоянной структуры проекта — падает с обычным
трейсбеком, если Qdrant не поднят или модель не найдена.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from meal_generator.retrieval.search import (  # noqa: E402
    SearchConstraints,
    search_recipes,
)

if __name__ == "__main__":
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = QdrantClient(host="localhost", port=6333)

    def embed_fn(texts: list[str]) -> list[list[float]]:
        return model.encode(texts).tolist()

    for query_text in ["spicy asian dinner", ""]:
        print(f"=== breakfast pool, query_text={query_text!r} ===")
        constraints = SearchConstraints(
            query_text=query_text,
            meal_type="breakfast",
            excluded_categories=["dairy"],
            limit=10,
        )
        results = search_recipes(client, embed_fn, constraints)
        for r in results:
            print(f"  {r['score']:.3f}  {r['calories']:>6.1f} kcal  {r['name']}")
        print()
