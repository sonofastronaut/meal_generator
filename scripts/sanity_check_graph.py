"""Разовый ручной sanity-check LangGraph-обвязки целиком на живых сервисах.

В отличие от sanity_check_full_pipeline.py (который вызывает parse_meal_request
и build_meal_plan_variants напрямую), этот скрипт собирает граф через
build_meal_plan_graph и гоняет .invoke() дважды подряд, как в реальном
диалоге: первый вызов с пустой историей проходит classify_intent (без LLM,
первое сообщение) -> parse_request -> build_plan -> (relax | explain), а
второй — с историей из первого обмена плюс новое уточняющее сообщение —
идёт через classify_intent (уже с LLM) -> parse_refinement -> replace_slot
-> explain, точечно заменяя один слот вместо полного replan. Нужны
docker-compose up -d + scripts/build_index.py (живой Qdrant с
проиндексированными рецептами) и .env с реальным OPENAI_API_KEY (см.
.env.example). Не тест — падает с обычным трейсбеком, если Qdrant не
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
from meal_generator.graph.graph import build_meal_plan_graph  # noqa: E402

if __name__ == "__main__":
    openai_client = OpenAI(api_key=get_openai_api_key())
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    qdrant_client = QdrantClient(host="localhost", port=6333)

    def embed_fn(texts: list[str]) -> list[list[float]]:
        return embedding_model.encode(texts).tolist()

    graph = build_meal_plan_graph(openai_client, qdrant_client, embed_fn)

    initial_state = {
        "messages": [
            {
                "role": "user",
                "content": (
                    "I want a spicy asian-inspired day of meals, 3 meals "
                    "today, around 2000 calories, no dairy, as much protein "
                    "as possible"
                ),
            }
        ],
        "parsed_request": None,
        "constraints": None,
        "pool_size": 30,
        "variants": None,
        "working_variant": None,
        "error_message": None,
        "explanation": None,
        "intent": None,
        "refine_slot_index": None,
        "refine_avoid_terms": None,
    }

    turn1_result = graph.invoke(initial_state)

    print("=== turn 1: intent ===")
    print(turn1_result["intent"])
    print()

    print("=== parsed request ===")
    print(turn1_result["parsed_request"])
    print()

    print("=== constraints ===")
    print(turn1_result["constraints"])
    print()

    if turn1_result["variants"] is not None:
        for variant in turn1_result["variants"]:
            print(
                f"=== {variant.label} ===  total={variant.total_calories:.1f}  "
                f"within_tolerance={variant.within_tolerance}"
            )
            for slot in variant.slots:
                print(f"  {slot['calories']:>6.1f} kcal  {slot['total_time_minutes']}min  cat={slot['source_category']!r}  {slot['name']}")
            print()
    else:
        print("=== no variants (pool exhausted) ===")
        print(turn1_result["error_message"])
        print()

    print("=== turn 1: explanation ===")
    print(turn1_result["explanation"])
    print()

    if turn1_result["working_variant"] is None:
        sys.exit(0)

    turn2_state = dict(turn1_result)
    turn2_state["messages"] = turn1_result["messages"] + [
        {"role": "assistant", "content": turn1_result["explanation"]},
        {
            "role": "user",
            "content": "actually, swap out the first meal for something else",
        },
    ]

    turn2_result = graph.invoke(turn2_state)

    print("=== turn 2: intent ===")
    print(turn2_result["intent"])
    print()

    print("=== turn 2: working plan ===")
    if turn2_result["working_variant"] is not None:
        for slot in turn2_result["working_variant"].slots:
            print(f"  {slot['calories']:>6.1f} kcal  {slot['total_time_minutes']}min  cat={slot['source_category']!r}  {slot['name']}")
    print()

    print("=== turn 2: explanation ===")
    print(turn2_result["explanation"])
