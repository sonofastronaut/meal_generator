"""Разовый ручной sanity-check optimizer'а на живом Qdrant.

Печатает три варианта дневного рациона (balanced/quick_cooking/variety) для
одного захардкоженного набора constraints, чтобы визуально оценить качество
жадных эвристик на уже проиндексированных 336 509 рецептах (docker-compose
up -d и scripts/build_index.py должны быть выполнены заранее). Сценарий —
breakfast + 2x main_meal с непустым vibe-запросом ("spicy asian dinner") —
проверяет фикс breakfast-bleed: с введением non-semantic пути в
_fetch_candidate_pools (см. planning/optimizer.py) breakfast-слот всегда
идёт через search_recipes_by_rating, а не через семантический поиск по
query_text, поэтому вывод включает rating — по нему теперь и должен быть
выбран breakfast, а не по близости к "spicy asian dinner". Не тест — падает
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
    build_meal_plan_variants,
)

if __name__ == "__main__":
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = QdrantClient(host="localhost", port=6333)

    def embed_fn(texts: list[str]) -> list[list[float]]:
        return model.encode(texts).tolist()

    constraints = MealPlanConstraints(
        meal_slots=["breakfast", "main_meal", "main_meal"],
        target_calories=2000.0,
        excluded_categories=["dairy"],
        query_text="spicy asian dinner",
    )

    variants = build_meal_plan_variants(client, embed_fn, constraints)

    for variant in variants:
        print(
            f"=== {variant.label} ===  total={variant.total_calories:.1f}  "
            f"within_tolerance={variant.within_tolerance}"
        )
        for slot in variant.slots:
            print(f"  {slot['calories']:>6.1f} kcal  {slot['total_time_minutes']}min  cat={slot['source_category']!r}  rating={slot['rating']}  {slot['name']}")
        print()
