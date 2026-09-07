"""Разовый ручной end-to-end sanity-check всего пайплайна на живых сервисах.

От natural language запроса пользователя (через OpenAI structured outputs)
до трёх вариантов дневного рациона на живом Qdrant с уже проиндексированными
336 509 рецептами (docker-compose up -d и scripts/build_index.py должны
быть выполнены заранее; .env с реальным OPENAI_API_KEY должен существовать
— см. .env.example). Не тест — падает с обычным трейсбеком, если Qdrant не
поднят, модель не найдена, или OPENAI_API_KEY не задан.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from meal_generator.config import get_openai_api_key  # noqa: E402
from meal_generator.nlp.parse_request import (  # noqa: E402
    parse_meal_request,
    to_meal_plan_constraints,
)
from meal_generator.planning.optimizer import build_meal_plan_variants  # noqa: E402

if __name__ == "__main__":
    openai_client = OpenAI(api_key=get_openai_api_key())
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    qdrant_client = QdrantClient(host="localhost", port=6333)

    def embed_fn(texts: list[str]) -> list[list[float]]:
        return embedding_model.encode(texts).tolist()

    user_message = (
        "I want a spicy asian-inspired day of meals, 3 meals today, around "
        "2000 calories, no dairy, as much protein as possible"
    )

    parsed = parse_meal_request(openai_client, user_message)
    print("=== parsed request ===")
    print(parsed)
    print()

    constraints = to_meal_plan_constraints(parsed)
    print("=== meal plan constraints ===")
    print(constraints)
    print()

    variants = build_meal_plan_variants(qdrant_client, embed_fn, constraints)

    for variant in variants:
        print(
            f"=== {variant.label} ===  total={variant.total_calories:.1f}  "
            f"within_tolerance={variant.within_tolerance}"
        )
        for slot in variant.slots:
            print(f"  {slot['calories']:>6.1f} kcal  {slot['total_time_minutes']}min  cat={slot['source_category']!r}  {slot['name']}")
        print()
