"""Полный rebuild коллекции curated-рецептов в Qdrant.

Читает data/recipes.parquet, строит curated-DataFrame (calories/protein/fat/
carbs уже на одну порцию — см. докстроку ``build_curated_recipes``),
отсекает записи с неправдоподобной калорийностью на порцию
(``filter_plausible_calories``), добавляет meal_types и allergen_categories,
поднимает all-MiniLM-L6-v2 и индексирует весь корпус в локальный Qdrant
(docker-compose up -d перед запуском).

Каждый запуск скрипта — полный rebuild, а не инкрементальное обновление:
если коллекция уже существует, она удаляется перед индексацией и строится
заново с нуля. ``index_recipes``/``ensure_collection`` сами по себе
идемпотентны и используют upsert — они никогда не удаляют точки, которых
больше нет в новом датасете (например, отсеянные ``filter_plausible_calories``
выбросы из предыдущего прогона). Без явного удаления коллекции здесь
повторный запуск оставлял бы старые точки перемешанными с новыми. Поэтому
повторный запуск этого скрипта безопасен и всегда даёт результат,
консистентный с текущим состоянием ``data/recipes.parquet`` и текущей
логикой фильтрации.
"""

import sys
import time
from pathlib import Path

import pandas as pd
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from meal_generator.indexing.qdrant_index import (  # noqa: E402
    COLLECTION_NAME,
    index_recipes,
)
from meal_generator.preprocessing.recipes import (  # noqa: E402
    add_allergen_categories,
    add_meal_types,
    build_curated_recipes,
    filter_plausible_calories,
    filter_technically_valid_recipes,
)

DATA_PATH = PROJECT_ROOT / "data" / "recipes.parquet"
BATCH_SIZE = 64


def main() -> None:
    print(f"Loading raw recipes from {DATA_PATH}...")
    raw = pd.read_parquet(DATA_PATH)
    print(f"Raw rows: {len(raw)}")

    valid = filter_technically_valid_recipes(raw)
    curated = build_curated_recipes(valid)
    curated = filter_plausible_calories(curated)
    print(f"Rows after filtering implausible calories/serving: {len(curated)}")
    curated = add_meal_types(curated)
    curated = add_allergen_categories(curated)
    print(f"Curated rows to index: {len(curated)}")

    print("Loading model all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Connecting to Qdrant at localhost:6333...")
    client = QdrantClient(host="localhost", port=6333)

    if client.collection_exists(COLLECTION_NAME):
        print(f"Deleting existing collection {COLLECTION_NAME} for a clean rebuild...")
        client.delete_collection(COLLECTION_NAME)

    batches_total = (len(curated) + BATCH_SIZE - 1) // BATCH_SIZE
    batches_done = 0

    def embed_fn(texts: list[str]) -> list[list[float]]:
        nonlocal batches_done
        vectors = model.encode(texts, batch_size=BATCH_SIZE).tolist()
        batches_done += 1
        print(f"Encoded batch {batches_done}/{batches_total}")
        return vectors

    start = time.perf_counter()
    total_indexed = index_recipes(
        client, curated, embed_fn=embed_fn, batch_size=BATCH_SIZE
    )
    elapsed = time.perf_counter() - start

    print(f"Indexed {total_indexed} points in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
