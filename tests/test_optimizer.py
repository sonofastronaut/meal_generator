import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from qdrant_client import QdrantClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from meal_generator.indexing.qdrant_index import index_recipes  # noqa: E402
from meal_generator.planning.optimizer import (  # noqa: E402
    MealPlanConstraints,
    MealPlanVariant,
    _fetch_candidate_pools,
    _is_within_tolerance,
    _select_balanced,
    _select_quick_cooking,
    _select_variety,
    build_meal_plan_variants,
)

from test_qdrant_index import IndexRecipesTest, fake_embed_fn  # noqa: E402


class IsWithinToleranceTest(unittest.TestCase):
    def test_within_tolerance(self) -> None:
        self.assertTrue(_is_within_tolerance(1050.0, 1000.0, 0.07))

    def test_exactly_on_boundary(self) -> None:
        self.assertTrue(_is_within_tolerance(1070.0, 1000.0, 0.07))

    def test_outside_tolerance(self) -> None:
        self.assertFalse(_is_within_tolerance(1080.0, 1000.0, 0.07))


class SelectBalancedTest(unittest.TestCase):
    def test_picks_distinct_recipes_closest_to_residual_targets(self) -> None:
        pools = {
            "main_meal": [
                {"recipe_id": 1, "calories": 490.0, "total_time_minutes": 20, "ingredients": ["a"]},
                {"recipe_id": 2, "calories": 650.0, "total_time_minutes": 20, "ingredients": ["b"]},
                {"recipe_id": 3, "calories": 520.0, "total_time_minutes": 20, "ingredients": ["c"]},
            ]
        }
        meal_slots = ["main_meal", "main_meal"]

        result = _select_balanced(meal_slots, pools, target_calories=1000.0)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["recipe_id"], 1)
        self.assertEqual(result[1]["recipe_id"], 3)
        self.assertNotEqual(result[0]["recipe_id"], result[1]["recipe_id"])

    def test_raises_when_pool_exhausted(self) -> None:
        pools = {
            "main_meal": [
                {"recipe_id": 1, "calories": 500.0, "total_time_minutes": 20, "ingredients": ["a"]},
            ]
        }
        meal_slots = ["main_meal", "main_meal"]

        with self.assertRaisesRegex(
            ValueError,
            r"Not enough distinct candidates for meal_type 'main_meal': "
            r"pool size 1, needed at least 2 for 2 slot\(s\)",
        ):
            _select_balanced(meal_slots, pools, target_calories=1000.0)


class SelectQuickCookingTest(unittest.TestCase):
    def test_prefers_faster_candidate_within_local_tolerance(self) -> None:
        pools = {
            "main_meal": [
                {"recipe_id": 1, "calories": 600.0, "total_time_minutes": 60, "ingredients": ["a"]},
                {"recipe_id": 2, "calories": 650.0, "total_time_minutes": 20, "ingredients": ["b"]},
            ]
        }
        meal_slots = ["main_meal"]

        result = _select_quick_cooking(meal_slots, pools, target_calories=600.0)

        self.assertEqual(result[0]["recipe_id"], 2)

    def test_falls_back_to_closest_calories_when_nothing_in_range(self) -> None:
        pools = {
            "main_meal": [
                {"recipe_id": 1, "calories": 800.0, "total_time_minutes": 10, "ingredients": ["a"]},
                {"recipe_id": 2, "calories": 1000.0, "total_time_minutes": 5, "ingredients": ["b"]},
            ]
        }
        meal_slots = ["main_meal"]

        result = _select_quick_cooking(meal_slots, pools, target_calories=600.0)

        self.assertEqual(result[0]["recipe_id"], 1)

    def test_raises_when_pool_exhausted(self) -> None:
        pools = {
            "main_meal": [
                {"recipe_id": 1, "calories": 500.0, "total_time_minutes": 20, "ingredients": ["a"]},
            ]
        }
        meal_slots = ["main_meal", "main_meal"]

        with self.assertRaises(ValueError):
            _select_quick_cooking(meal_slots, pools, target_calories=1000.0)


class SelectVarietyTest(unittest.TestCase):
    def test_prefers_less_overlapping_ingredients_when_categories_tie(self) -> None:
        pools = {
            "main_meal": [
                {
                    "recipe_id": 1,
                    "source_category": "Chicken",
                    "calories": 500.0,
                    "total_time_minutes": 30,
                    "ingredients": ["chicken", "rice"],
                },
                {
                    "recipe_id": 2,
                    "source_category": "Poultry",
                    "calories": 500.0,
                    "total_time_minutes": 25,
                    "ingredients": ["chicken", "broccoli"],
                },
                {
                    "recipe_id": 3,
                    "source_category": "Beef",
                    "calories": 520.0,
                    "total_time_minutes": 40,
                    "ingredients": ["beef", "potato"],
                },
            ]
        }
        meal_slots = ["main_meal", "main_meal"]

        result = _select_variety(meal_slots, pools, target_calories=1000.0)

        # Recipe 1 and 2 have distinct source_category, so the second slot's
        # choice is decided by ingredient jaccard, not the category signal:
        # recipe 3 shares no ingredients with recipe 1, recipe 2 shares
        # "chicken" — recipe 3 wins despite being calorically further.
        self.assertEqual(result[0]["recipe_id"], 1)
        self.assertEqual(result[1]["recipe_id"], 3)

    def test_avoids_repeating_source_category_over_closer_calorie_match(self) -> None:
        pools = {
            "main_meal": [
                {
                    "recipe_id": 1,
                    "source_category": "Meat",
                    "calories": 500.0,
                    "total_time_minutes": 20,
                    "ingredients": ["beef", "onion"],
                },
                {
                    "recipe_id": 2,
                    "source_category": "Meat",
                    "calories": 510.0,
                    "total_time_minutes": 20,
                    "ingredients": ["chicken", "garlic"],
                },
                {
                    "recipe_id": 3,
                    "source_category": "Meat",
                    "calories": 900.0,
                    "total_time_minutes": 20,
                    "ingredients": ["pork", "pepper"],
                },
                {
                    "recipe_id": 4,
                    "source_category": "Spaghetti",
                    "calories": 700.0,
                    "total_time_minutes": 20,
                    "ingredients": ["pasta", "tomato"],
                },
            ]
        }
        meal_slots = ["main_meal", "main_meal", "main_meal"]

        result = _select_variety(meal_slots, pools, target_calories=1500.0)

        result_ids = [slot["recipe_id"] for slot in result]
        # Slot 1 picks recipe 1 (closest to 500 residual target). For slot 2
        # (residual 500), recipe 2 is calorically far closer (diff 10) than
        # recipe 4 (diff 200) — a raw calorie-only or raw-ingredient-count
        # heuristic would repeat the "Meat" category. The category-repeat
        # penalty makes recipe 4 (a different category) win instead.
        self.assertEqual(result_ids, [1, 4, 2])

    def test_none_source_category_is_not_treated_as_a_repeat(self) -> None:
        pools = {
            "main_meal": [
                {
                    "recipe_id": 1,
                    "source_category": None,
                    "calories": 500.0,
                    "total_time_minutes": 20,
                    "ingredients": ["x"],
                },
                {
                    "recipe_id": 2,
                    "source_category": None,
                    "calories": 500.0,
                    "total_time_minutes": 20,
                    "ingredients": ["y"],
                },
                {
                    "recipe_id": 3,
                    "source_category": "Meat",
                    "calories": 800.0,
                    "total_time_minutes": 20,
                    "ingredients": ["z"],
                },
            ]
        }
        meal_slots = ["main_meal", "main_meal"]

        result = _select_variety(meal_slots, pools, target_calories=1000.0)

        result_ids = [slot["recipe_id"] for slot in result]
        # Both recipe 1 and recipe 2 have source_category=None. If None were
        # treated as a matching category, recipe 2 would be penalized on the
        # second slot and recipe 3 (calorically much further) would win
        # instead. It shouldn't be penalized, so recipe 2 (closer calories)
        # is picked.
        self.assertEqual(result_ids, [1, 2])

    def test_jaccard_normalizes_by_union_not_raw_intersection_count(self) -> None:
        pools = {
            "main_meal": [
                {
                    "recipe_id": 1,
                    "source_category": "Meat",
                    "calories": 500.0,
                    "total_time_minutes": 20,
                    "ingredients": ["shared_ing", "onion"],
                },
                {
                    "recipe_id": 2,
                    "source_category": "Meat",
                    "calories": 500.0,
                    "total_time_minutes": 20,
                    "ingredients": ["shared_ing", "carrot"],
                },
                {
                    "recipe_id": 3,
                    "source_category": "Meat",
                    "calories": 500.0,
                    "total_time_minutes": 20,
                    "ingredients": [
                        "shared_ing",
                        "i2",
                        "i3",
                        "i4",
                        "i5",
                        "i6",
                        "i7",
                        "i8",
                        "i9",
                        "i10",
                    ],
                },
            ]
        }
        meal_slots = ["main_meal", "main_meal"]

        result = _select_variety(meal_slots, pools, target_calories=1000.0)

        # Slot 1 ties on everything and picks recipe 1 (first in pool order).
        # For slot 2, recipe 2 and recipe 3 both share exactly one ingredient
        # ("shared_ing") with recipe 1 — same raw intersection count — but
        # recipe 3's ingredient list is much longer, so its jaccard
        # (1/11) is far smaller than recipe 2's (1/3). Recipe 3 should win.
        self.assertEqual(result[0]["recipe_id"], 1)
        self.assertEqual(result[1]["recipe_id"], 3)

    def test_raises_when_pool_exhausted(self) -> None:
        pools = {
            "main_meal": [
                {
                    "recipe_id": 1,
                    "source_category": "Meat",
                    "calories": 500.0,
                    "total_time_minutes": 20,
                    "ingredients": ["a"],
                },
            ]
        }
        meal_slots = ["main_meal", "main_meal"]

        with self.assertRaises(ValueError):
            _select_variety(meal_slots, pools, target_calories=1000.0)


class BuildMealPlanVariantsTest(unittest.TestCase):
    def test_returns_three_variants_with_expected_shape(self) -> None:
        client = QdrantClient(":memory:")
        index_recipes(client, IndexRecipesTest.curated_recipes(), embed_fn=fake_embed_fn)

        constraints = MealPlanConstraints(
            meal_slots=["breakfast", "main_meal"],
            target_calories=750.0,
            query_text="anything",
        )

        variants = build_meal_plan_variants(client, fake_embed_fn, constraints)

        self.assertEqual(len(variants), 3)
        labels = [variant.label for variant in variants]
        self.assertEqual(labels, ["balanced", "quick_cooking", "variety"])

        for variant in variants:
            self.assertIsInstance(variant, MealPlanVariant)
            self.assertEqual(len(variant.slots), len(constraints.meal_slots))
            expected_total = sum(slot["calories"] for slot in variant.slots)
            self.assertEqual(variant.total_calories, expected_total)
            self.assertEqual(
                variant.within_tolerance,
                _is_within_tolerance(
                    expected_total, constraints.target_calories, constraints.calorie_tolerance
                ),
            )


class FetchCandidatePoolsTest(unittest.TestCase):
    def test_uses_semantic_search_for_non_breakfast_with_query_text(self) -> None:
        constraints = MealPlanConstraints(
            meal_slots=["main_meal"],
            target_calories=1000.0,
            query_text="spicy asian",
        )

        with (
            patch("meal_generator.planning.optimizer.search_recipes") as mock_semantic,
            patch(
                "meal_generator.planning.optimizer.search_recipes_by_rating"
            ) as mock_rating,
        ):
            mock_semantic.return_value = [{"recipe_id": 1}]

            _fetch_candidate_pools(
                client=object(), embed_fn=lambda texts: [[0.0]], constraints=constraints
            )

        mock_semantic.assert_called_once()
        mock_rating.assert_not_called()

        called_constraints = mock_semantic.call_args[0][2]
        self.assertEqual(called_constraints.meal_type, "main_meal")
        self.assertEqual(called_constraints.query_text, "spicy asian")

    def test_breakfast_always_uses_non_semantic_even_with_query_text(self) -> None:
        constraints = MealPlanConstraints(
            meal_slots=["breakfast", "main_meal"],
            target_calories=1500.0,
            query_text="spicy asian",
        )

        with (
            patch("meal_generator.planning.optimizer.search_recipes") as mock_semantic,
            patch(
                "meal_generator.planning.optimizer.search_recipes_by_rating"
            ) as mock_rating,
        ):
            mock_semantic.return_value = [{"recipe_id": 1}]
            mock_rating.return_value = [{"recipe_id": 2}]

            _fetch_candidate_pools(
                client=object(), embed_fn=lambda texts: [[0.0]], constraints=constraints
            )

        mock_rating.assert_called_once()
        rating_constraints = mock_rating.call_args[0][1]
        self.assertEqual(rating_constraints.meal_type, "breakfast")

        mock_semantic.assert_called_once()
        semantic_constraints = mock_semantic.call_args[0][2]
        self.assertEqual(semantic_constraints.meal_type, "main_meal")

    def test_uses_non_semantic_search_when_query_text_empty(self) -> None:
        constraints = MealPlanConstraints(
            meal_slots=["main_meal"],
            target_calories=1000.0,
            query_text="",
        )

        with (
            patch("meal_generator.planning.optimizer.search_recipes") as mock_semantic,
            patch(
                "meal_generator.planning.optimizer.search_recipes_by_rating"
            ) as mock_rating,
        ):
            mock_rating.return_value = [{"recipe_id": 1}]

            _fetch_candidate_pools(
                client=object(), embed_fn=lambda texts: [[0.0]], constraints=constraints
            )

        mock_rating.assert_called_once()
        mock_semantic.assert_not_called()

        rating_constraints = mock_rating.call_args[0][1]
        self.assertEqual(rating_constraints.meal_type, "main_meal")

    def test_protein_min_is_not_passed_through_to_search_constraints(self) -> None:
        constraints = MealPlanConstraints(
            meal_slots=["main_meal"],
            target_calories=1000.0,
            protein_min=200.0,
            query_text="spicy asian",
        )

        with (
            patch("meal_generator.planning.optimizer.search_recipes") as mock_semantic,
            patch(
                "meal_generator.planning.optimizer.search_recipes_by_rating"
            ) as mock_rating,
        ):
            mock_semantic.return_value = [{"recipe_id": 1}]

            _fetch_candidate_pools(
                client=object(), embed_fn=lambda texts: [[0.0]], constraints=constraints
            )

        mock_rating.assert_not_called()
        called_constraints = mock_semantic.call_args[0][2]
        self.assertIsNone(called_constraints.protein_min)


if __name__ == "__main__":
    unittest.main()
