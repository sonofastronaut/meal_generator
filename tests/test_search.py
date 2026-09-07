import sys
import unittest
from pathlib import Path

from qdrant_client import QdrantClient, models


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from meal_generator.indexing.qdrant_index import index_recipes  # noqa: E402
from meal_generator.retrieval.search import (  # noqa: E402
    SearchConstraints,
    build_search_filter,
    search_recipes,
    search_recipes_by_rating,
)

from test_qdrant_index import IndexRecipesTest, fake_embed_fn  # noqa: E402


def _indexed_client() -> QdrantClient:
    client = QdrantClient(":memory:")
    index_recipes(client, IndexRecipesTest.curated_recipes(), embed_fn=fake_embed_fn)
    return client


class BuildSearchFilterTest(unittest.TestCase):
    def test_empty_constraints_produce_empty_filter(self) -> None:
        constraints = SearchConstraints(query_text="anything")

        result = build_search_filter(constraints)

        self.assertEqual(result, models.Filter())

    def test_meal_type_produces_must_condition(self) -> None:
        constraints = SearchConstraints(query_text="anything", meal_type="breakfast")

        result = build_search_filter(constraints)

        self.assertEqual(
            result.must,
            [
                models.FieldCondition(
                    key="meal_types", match=models.MatchValue(value="breakfast")
                )
            ],
        )
        self.assertEqual(result.must_not, [])

    def test_both_calorie_bounds_produce_range(self) -> None:
        constraints = SearchConstraints(
            query_text="anything", calories_min=200.0, calories_max=600.0
        )

        result = build_search_filter(constraints)

        self.assertEqual(
            result.must,
            [
                models.FieldCondition(
                    key="calories",
                    range=models.Range(gte=200.0, lte=600.0),
                )
            ],
        )

    def test_only_calories_min_produces_range_with_none_upper_bound(self) -> None:
        constraints = SearchConstraints(query_text="anything", calories_min=200.0)

        result = build_search_filter(constraints)

        self.assertEqual(
            result.must,
            [
                models.FieldCondition(
                    key="calories",
                    range=models.Range(gte=200.0, lte=None),
                )
            ],
        )

    def test_excluded_categories_produce_must_not_conditions(self) -> None:
        constraints = SearchConstraints(
            query_text="anything", excluded_categories=["dairy", "fish"]
        )

        result = build_search_filter(constraints)

        self.assertEqual(result.must, [])
        self.assertEqual(
            result.must_not,
            [
                models.FieldCondition(
                    key="allergen_categories", match=models.MatchValue(value="dairy")
                ),
                models.FieldCondition(
                    key="allergen_categories", match=models.MatchValue(value="fish")
                ),
            ],
        )

    def test_combination_of_constraints_includes_all_conditions(self) -> None:
        constraints = SearchConstraints(
            query_text="anything",
            meal_type="main_meal",
            calories_min=100.0,
            protein_min=20.0,
            max_total_time_minutes=45,
            excluded_categories=["dairy"],
        )

        result = build_search_filter(constraints)

        self.assertEqual(
            result.must,
            [
                models.FieldCondition(
                    key="meal_types", match=models.MatchValue(value="main_meal")
                ),
                models.FieldCondition(
                    key="calories", range=models.Range(gte=100.0, lte=None)
                ),
                models.FieldCondition(
                    key="protein", range=models.Range(gte=20.0)
                ),
                models.FieldCondition(
                    key="total_time_minutes", range=models.Range(lte=45)
                ),
            ],
        )
        self.assertEqual(
            result.must_not,
            [
                models.FieldCondition(
                    key="allergen_categories", match=models.MatchValue(value="dairy")
                )
            ],
        )


class SearchRecipesTest(unittest.TestCase):
    def test_search_without_constraints_respects_limit(self) -> None:
        client = _indexed_client()
        constraints = SearchConstraints(query_text="salmon dinner", limit=2)

        results = search_recipes(client, fake_embed_fn, constraints)

        self.assertLessEqual(len(results), 2)
        self.assertGreater(len(results), 0)
        for result in results:
            self.assertIn("score", result)
            self.assertIsInstance(result["score"], float)

    def test_search_with_meal_type_filters_to_breakfast(self) -> None:
        client = _indexed_client()
        constraints = SearchConstraints(query_text="sweet breakfast", meal_type="breakfast")

        results = search_recipes(client, fake_embed_fn, constraints)

        self.assertTrue(results)
        for result in results:
            self.assertIn("breakfast", result["meal_types"])

    def test_search_with_excluded_categories_removes_dairy_recipe(self) -> None:
        client = _indexed_client()
        constraints = SearchConstraints(
            query_text="anything", excluded_categories=["dairy"]
        )

        results = search_recipes(client, fake_embed_fn, constraints)

        returned_ids = {result["recipe_id"] for result in results}
        self.assertNotIn(2, returned_ids)

    def test_combination_of_calorie_range_and_excluded_categories(self) -> None:
        client = _indexed_client()
        constraints = SearchConstraints(
            query_text="anything",
            calories_min=380.0,
            calories_max=550.0,
            excluded_categories=["fish"],
        )

        results = search_recipes(client, fake_embed_fn, constraints)

        returned_ids = {result["recipe_id"] for result in results}
        self.assertEqual(returned_ids, {3})

    def test_score_present_and_is_float_for_every_result(self) -> None:
        client = _indexed_client()
        constraints = SearchConstraints(query_text="anything")

        results = search_recipes(client, fake_embed_fn, constraints)

        self.assertTrue(results)
        for result in results:
            self.assertIsInstance(result["score"], float)


class SearchRecipesByRatingTest(unittest.TestCase):
    def test_sorts_by_rating_descending_with_none_last(self) -> None:
        client = _indexed_client()
        constraints = SearchConstraints(query_text="")

        results = search_recipes_by_rating(client, constraints)

        returned_ids = [result["recipe_id"] for result in results]
        # recipe 1 rating=4.5, recipe 2 rating=4.0, recipe 3 rating=None.
        self.assertEqual(returned_ids, [1, 2, 3])

    def test_filters_by_meal_type_same_as_search_recipes(self) -> None:
        client = _indexed_client()
        constraints = SearchConstraints(query_text="", meal_type="breakfast")

        results = search_recipes_by_rating(client, constraints)

        self.assertTrue(results)
        for result in results:
            self.assertIn("breakfast", result["meal_types"])

    def test_limit_is_applied_after_sorting(self) -> None:
        client = _indexed_client()
        constraints = SearchConstraints(query_text="", limit=1)

        results = search_recipes_by_rating(client, constraints)

        self.assertLessEqual(len(results), 1)
        self.assertEqual(results[0]["recipe_id"], 1)

    def test_score_key_present_and_none(self) -> None:
        client = _indexed_client()
        constraints = SearchConstraints(query_text="")

        results = search_recipes_by_rating(client, constraints)

        self.assertTrue(results)
        for result in results:
            self.assertIn("score", result)
            self.assertIsNone(result["score"])


if __name__ == "__main__":
    unittest.main()
