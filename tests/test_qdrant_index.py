import sys
import unittest
from pathlib import Path

import pandas as pd
from qdrant_client import QdrantClient, models


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from meal_generator.indexing.qdrant_index import (  # noqa: E402
    COLLECTION_NAME,
    EMBEDDING_DIMENSION,
    build_payload,
    ensure_collection,
    index_recipes,
)


def fake_embed_fn(texts: list[str]) -> list[list[float]]:
    """Детерминированный фейковый эмбеддер: нулевой вектор с одним
    отличающимся компонентом на основе индекса текста в батче."""
    vectors = []
    for index, _ in enumerate(texts):
        vector = [0.0] * EMBEDDING_DIMENSION
        vector[index % EMBEDDING_DIMENSION] = 1.0
        vectors.append(vector)
    return vectors


class BuildPayloadTest(unittest.TestCase):
    def test_builds_expected_payload(self) -> None:
        recipe = pd.Series(
            {
                "recipe_id": 42,
                "name": "Grilled Chicken",
                "source_category": "Poultry",
                "keywords": ["Easy", "Dinner"],
                "meal_types": ["main_meal"],
                "calories": 450.0,
                "protein": 35.0,
                "fat": 12.0,
                "carbs": 48.0,
                "servings": 2,
                "total_time_minutes": 80,
                "ingredients": ["chicken breast", "olive oil"],
                "instructions": [
                    "Preheat oven to 350F.",
                    "Mix ingredients.",
                    "Bake for 20 minutes.",
                ],
                "rating": 4.5,
                "review_count": 10,
                "allergen_categories": [],
            }
        )

        result = build_payload(recipe)

        self.assertEqual(
            result,
            {
                "recipe_id": 42,
                "name": "Grilled Chicken",
                "source_category": "Poultry",
                "meal_types": ["main_meal"],
                "calories": 450.0,
                "protein": 35.0,
                "fat": 12.0,
                "carbs": 48.0,
                "servings": 2,
                "total_time_minutes": 80,
                "ingredients": ["chicken breast", "olive oil"],
                "instructions": [
                    "Preheat oven to 350F.",
                    "Mix ingredients.",
                    "Bake for 20 minutes.",
                ],
                "rating": 4.5,
                "review_count": 10,
                "allergen_categories": [],
            },
        )

    def test_none_total_time_minutes_is_preserved_as_none(self) -> None:
        recipe = pd.Series(
            {
                "recipe_id": 1,
                "name": "Water",
                "source_category": None,
                "keywords": [],
                "meal_types": [],
                "calories": 0.0,
                "protein": 0.0,
                "fat": 0.0,
                "carbs": 0.0,
                "servings": 1,
                "total_time_minutes": None,
                "ingredients": ["water"],
                "instructions": ["Pour water into a glass."],
                "rating": None,
                "review_count": None,
                "allergen_categories": [],
            }
        )

        result = build_payload(recipe)

        self.assertIsNone(result["total_time_minutes"])
        self.assertIsNone(result["rating"])
        self.assertIsNone(result["review_count"])
        self.assertIsNone(result["source_category"])


class EnsureCollectionTest(unittest.TestCase):
    def test_creates_collection_with_expected_config(self) -> None:
        client = QdrantClient(":memory:")

        ensure_collection(client)

        self.assertTrue(client.collection_exists(COLLECTION_NAME))
        info = client.get_collection(COLLECTION_NAME)
        self.assertEqual(info.config.params.vectors.size, EMBEDDING_DIMENSION)
        self.assertEqual(info.config.params.vectors.distance, models.Distance.COSINE)

    def test_is_idempotent(self) -> None:
        client = QdrantClient(":memory:")

        ensure_collection(client)
        ensure_collection(client)

        self.assertTrue(client.collection_exists(COLLECTION_NAME))


class IndexRecipesTest(unittest.TestCase):
    @staticmethod
    def curated_recipes() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "recipe_id": 1,
                    "name": "Grilled Salmon",
                    "source_category": "Fish",
                    "keywords": ["Easy"],
                    "meal_types": ["main_meal"],
                    "calories": 400.0,
                    "protein": 30.0,
                    "fat": 20.0,
                    "carbs": 5.0,
                    "servings": 2,
                    "total_time_minutes": 30,
                    "ingredients": ["salmon fillet", "lemon"],
                    "instructions": ["Season the salmon.", "Grill for 10 minutes."],
                    "rating": 4.5,
                    "review_count": 12,
                    "allergen_categories": ["fish"],
                },
                {
                    "recipe_id": 2,
                    "name": "Pancakes",
                    "source_category": "Breakfast",
                    "keywords": ["Sweet"],
                    "meal_types": ["breakfast"],
                    "calories": 350.0,
                    "protein": 8.0,
                    "fat": 10.0,
                    "carbs": 55.0,
                    "servings": 4,
                    "total_time_minutes": 20,
                    "ingredients": ["flour", "egg", "milk"],
                    "instructions": ["Mix the batter.", "Fry on a hot griddle."],
                    "rating": 4.0,
                    "review_count": 5,
                    "allergen_categories": ["dairy", "egg", "gluten_wheat"],
                },
                {
                    "recipe_id": 3,
                    "name": "Chicken Stew",
                    "source_category": "Chicken",
                    "keywords": ["Comfort Food"],
                    "meal_types": ["main_meal"],
                    "calories": 500.0,
                    "protein": 40.0,
                    "fat": 18.0,
                    "carbs": 25.0,
                    "servings": 4,
                    "total_time_minutes": None,
                    "ingredients": ["chicken breast", "carrots", "potatoes"],
                    "instructions": ["Brown the chicken.", "Simmer with vegetables."],
                    "rating": None,
                    "review_count": None,
                    "allergen_categories": [],
                },
            ],
            index=[10, 20, 30],
        )

    def test_returns_count_matching_number_of_rows(self) -> None:
        client = QdrantClient(":memory:")
        recipes = self.curated_recipes()

        result = index_recipes(client, recipes, embed_fn=fake_embed_fn)

        self.assertEqual(result, 3)

    def test_client_count_matches_indexed_points(self) -> None:
        client = QdrantClient(":memory:")
        recipes = self.curated_recipes()

        index_recipes(client, recipes, embed_fn=fake_embed_fn)

        count_result = client.count(COLLECTION_NAME)
        self.assertEqual(count_result.count, 3)

    def test_scroll_with_meal_types_filter_returns_expected_subset(self) -> None:
        client = QdrantClient(":memory:")
        recipes = self.curated_recipes()

        index_recipes(client, recipes, embed_fn=fake_embed_fn)

        scroll_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="meal_types", match=models.MatchValue(value="main_meal")
                )
            ]
        )
        points, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=scroll_filter,
            limit=10,
        )

        returned_ids = sorted(point.id for point in points)
        self.assertEqual(returned_ids, [1, 3])

    def test_respects_batch_size(self) -> None:
        client = QdrantClient(":memory:")
        recipes = self.curated_recipes()

        result = index_recipes(client, recipes, embed_fn=fake_embed_fn, batch_size=1)

        self.assertEqual(result, 3)
        self.assertEqual(client.count(COLLECTION_NAME).count, 3)

    def test_raises_for_missing_required_columns(self) -> None:
        client = QdrantClient(":memory:")
        recipes = self.curated_recipes().drop(columns="allergen_categories")

        with self.assertRaisesRegex(
            ValueError,
            "Missing columns required to index recipes: allergen_categories",
        ):
            index_recipes(client, recipes, embed_fn=fake_embed_fn)


if __name__ == "__main__":
    unittest.main()
