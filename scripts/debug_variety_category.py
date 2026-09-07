"""Разовый диагностический scratch-скрипт: насколько часто source_category

повторяется внутри пула кандидатов на слот — до того, как закладывать
category-based сигнал в _select_variety. Если source_category внутри пула
почти не повторяется, дополнительный сигнал по категории не даст выигрыша
поверх уже существующего ingredients-based штрафа. Не тест, не часть
постоянной структуры проекта — падает с обычным трейсбеком, если Qdrant не
поднят, модель не найдена, или пул меньше 2 кандидатов (деление на
total_pairs).
"""

import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from meal_generator.planning.optimizer import (  # noqa: E402
    MealPlanConstraints,
    _fetch_candidate_pools,
)

if __name__ == "__main__":
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = QdrantClient(host="localhost", port=6333)

    def embed_fn(texts: list[str]) -> list[list[float]]:
        return model.encode(texts).tolist()

    constraints = MealPlanConstraints(
        meal_slots=["breakfast", "main_meal", "main_meal"],
        target_calories=2100.0,
        excluded_categories=["fish"],
        query_text="",
    )

    pools = _fetch_candidate_pools(client, embed_fn, constraints)

    for meal_type, pool in pools.items():
        print(f"=== pool for {meal_type} ({len(pool)} candidates) ===")
        for c in pool:
            print(f"  id={c['recipe_id']:>6}  calories={c['calories']:>6.1f}  source_category={c['source_category']!r}")
        print()

        category_counts = Counter(c["source_category"] for c in pool)
        print(f"  category distribution: {category_counts.most_common()}")
        total_pairs = len(pool) * (len(pool) - 1) // 2
        same_category_pairs = sum(n * (n - 1) // 2 for n in category_counts.values())
        print(f"  same-category pairs: {same_category_pairs}/{total_pairs} ({same_category_pairs/total_pairs:.1%})")
        print()
