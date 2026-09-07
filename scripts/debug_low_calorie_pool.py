"""Разовый диагностический scratch-скрипт: откуда в плане steak на 102 ккал.

Последний прогон бота выдал странные результаты (steak на 102 ккал, курица
на 10 ккал, коктейль как десерт, хлебные роллы на 3.3 ккал) для запроса на
3000 ккал/день с protein_min=200. Печатает пулы кандидатов из
_fetch_candidate_pools, отсортированные по calories по возрастанию, чтобы
проверить гипотезу: пул изначально забит низкокалорийным мусором (плохая
нормализация на порцию, см. build_curated_recipes) — или дело в самом
отборе (_select_*). Не тест, не часть постоянной структуры проекта — падает
с обычным трейсбеком, если Qdrant не поднят или модель не найдена.
"""

import sys
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
        meal_slots=["breakfast", "main_meal", "lunch_snack", "main_meal"],
        target_calories=3000.0,
        protein_min=200.0,
        query_text="",
    )

    pools = _fetch_candidate_pools(client, embed_fn, constraints)

    for meal_type, pool in pools.items():
        print(f"=== pool for {meal_type} ({len(pool)} candidates) ===")
        for c in sorted(pool, key=lambda x: x["calories"]):
            print(
                f"  {c['calories']:>8.2f} kcal  servings={c.get('servings')}  "
                f"cat={c['source_category']!r}  {c['name']}"
            )
        print()
