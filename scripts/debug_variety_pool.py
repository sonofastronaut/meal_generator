"""Разовый диагностический scratch-скрипт: почему variety == balanced на реальном прогоне.

Печатает пулы кандидатов из _fetch_candidate_pools и попарные пересечения
ingredients внутри пула main_meal, чтобы проверить гипотезу: если пул
слабо пересекается по ingredients (или вообще не пересекается), штраф
_select_variety для всех кандидатов равен 0, и выбор схлопывается к тому же
tie-break по calories, что и в _select_balanced — отсюда идентичный
результат. Не тест, не часть постоянной структуры проекта — падает с
обычным трейсбеком, если Qdrant не поднят или модель не найдена.
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
        meal_slots=["breakfast", "main_meal", "main_meal"],
        target_calories=2100.0,
        excluded_categories=["fish"],
        query_text="",
    )

    pools = _fetch_candidate_pools(client, embed_fn, constraints)

    for meal_type, pool in pools.items():
        print(f"=== pool for {meal_type} ({len(pool)} candidates) ===")
        for c in pool:
            ingredients_preview = ", ".join(c["ingredients"][:5])
            print(f"  {c['calories']:>6.1f} kcal  id={c['recipe_id']}  [{ingredients_preview}]")
        print()

    main_meal_pool = pools["main_meal"]
    print("=== pairwise ingredient overlap within main_meal pool ===")
    for i in range(len(main_meal_pool)):
        for j in range(i + 1, len(main_meal_pool)):
            a = {x.casefold() for x in main_meal_pool[i]["ingredients"]}
            b = {x.casefold() for x in main_meal_pool[j]["ingredients"]}
            overlap = len(a & b)
            if overlap > 0:
                print(
                    f"  id={main_meal_pool[i]['recipe_id']} vs "
                    f"id={main_meal_pool[j]['recipe_id']}: overlap={overlap} "
                    f"({a & b})"
                )
