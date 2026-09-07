"""Разовый ручной sanity-check retrieval на живом Qdrant.

Печатает top-N результатов поиска для одного захардкоженного запроса, чтобы
визуально оценить качество семантического поиска и структурных фильтров на
уже проиндексированных 336 509 рецептах (docker-compose up -d и
scripts/build_index.py должны быть выполнены заранее). Не тест — падает с
обычным трейсбеком, если Qdrant не поднят или модель не найдена.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from meal_generator.retrieval.search import SearchConstraints, search_recipes  # noqa: E402

if __name__ == "__main__":
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = QdrantClient(host="localhost", port=6333)

    def embed_fn(texts: list[str]) -> list[list[float]]:
        return model.encode(texts).tolist()

    constraints = SearchConstraints(
        query_text="spicy asian chicken stir fry",
        meal_type="main_meal",
        calories_min=300,
        calories_max=700,
        excluded_categories=["dairy"],
        limit=10,
    )

    results = search_recipes(client, embed_fn, constraints)
    for r in results:
        print(
            f"{r['score']:.3f}  {r['name']}  "
            f"({r['calories']} kcal, {r['allergen_categories']})"
        )
