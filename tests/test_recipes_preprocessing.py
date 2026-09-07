import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from meal_generator.preprocessing.recipes import (  # noqa: E402
    CURATED_RECIPE_COLUMNS,
    _parse_total_time_minutes,
    add_allergen_categories,
    add_meal_types,
    build_curated_recipes,
    filter_plausible_calories,
    filter_technically_valid_recipes,
)


class ParseTotalTimeMinutesTest(unittest.TestCase):
    def test_parses_valid_values_and_rejects_invalid_values(self) -> None:
        cases = [
            ("PT35M", 35),
            ("PT1H20M", 80),
            ("PT4H", 240),
            ("PT24H45M", 1485),
            (" PT1H5M ", 65),
            ("PT0M", 0),
            ("PT", None),
            ("", None),
            ("abc", None),
            ("PT30S", None),
            (None, None),
            (np.nan, None),
        ]

        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(_parse_total_time_minutes(value), expected)


class FilterTechnicallyValidRecipesTest(unittest.TestCase):
    @staticmethod
    def valid_recipe(**overrides: object) -> dict[str, object]:
        recipe: dict[str, object] = {
            "RecipeId": 1,
            "Name": "Chicken and rice",
            "RecipeIngredientParts": ["chicken breast", "rice"],
            "RecipeIngredientQuantities": ["500 g", "200 g"],
            "RecipeInstructions": ["Cook the chicken and rice."],
            "RecipeServings": 2,
            "Calories": 450.0,
            "ProteinContent": 35.0,
            "FatContent": 12.0,
            "CarbohydrateContent": 48.0,
        }
        recipe.update(overrides)
        return recipe

    def assert_recipe_is_removed(self, **overrides: object) -> None:
        recipes = pd.DataFrame([self.valid_recipe(**overrides)], index=[42])
        original = recipes.copy(deep=True)

        result = filter_technically_valid_recipes(recipes)

        self.assertEqual(len(result), 0)
        assert_frame_equal(recipes, original)

    def test_keeps_valid_recipe_without_modifying_input(self) -> None:
        recipes = pd.DataFrame([self.valid_recipe()], index=[42])
        original = recipes.copy(deep=True)

        result = filter_technically_valid_recipes(recipes)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.index.tolist(), [42])
        assert_frame_equal(recipes, original)

    def test_removes_recipe_with_zero_recipe_id(self) -> None:
        self.assert_recipe_is_removed(RecipeId=0)

    def test_removes_recipe_with_inf_recipe_id(self) -> None:
        self.assert_recipe_is_removed(RecipeId=np.inf)

    def test_removes_recipe_with_non_numeric_recipe_id(self) -> None:
        self.assert_recipe_is_removed(RecipeId="abc")

    def test_removes_recipe_with_empty_name(self) -> None:
        self.assert_recipe_is_removed(Name="   ")

    def test_removes_recipe_with_empty_ingredients(self) -> None:
        self.assert_recipe_is_removed(RecipeIngredientParts=[])

    def test_removes_recipe_with_empty_instructions(self) -> None:
        self.assert_recipe_is_removed(RecipeInstructions=[])

    def test_removes_recipe_with_zero_servings(self) -> None:
        self.assert_recipe_is_removed(RecipeServings=0)

    def test_removes_recipe_with_non_finite_servings(self) -> None:
        self.assert_recipe_is_removed(RecipeServings=np.inf)

    def test_removes_recipe_with_zero_calories(self) -> None:
        self.assert_recipe_is_removed(Calories=0)

    def test_removes_recipe_with_non_finite_calories(self) -> None:
        self.assert_recipe_is_removed(Calories=np.inf)

    def test_removes_recipe_with_negative_macro(self) -> None:
        self.assert_recipe_is_removed(ProteinContent=-1)

    def test_removes_recipe_with_non_finite_macro(self) -> None:
        self.assert_recipe_is_removed(FatContent=np.inf)

    def test_allows_zero_macros(self) -> None:
        recipes = pd.DataFrame([self.valid_recipe(ProteinContent=0)], index=[42])

        result = filter_technically_valid_recipes(recipes)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.index.tolist(), [42])

    def test_does_not_require_ingredient_quantities(self) -> None:
        recipe = self.valid_recipe()
        recipe.pop("RecipeIngredientQuantities")
        recipes = pd.DataFrame([recipe], index=[42])

        result = filter_technically_valid_recipes(recipes)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.index.tolist(), [42])

    def test_raises_for_missing_required_column(self) -> None:
        recipes = pd.DataFrame([self.valid_recipe()]).drop(columns="Calories")

        with self.assertRaisesRegex(
            ValueError, "Missing required recipe columns: Calories"
        ):
            filter_technically_valid_recipes(recipes)

    def test_reports_all_missing_required_columns(self) -> None:
        missing_columns = ["RecipeId", "Name", "Calories"]
        recipes = pd.DataFrame([self.valid_recipe()]).drop(columns=missing_columns)

        with self.assertRaises(ValueError) as context:
            filter_technically_valid_recipes(recipes)

        error_message = str(context.exception)
        for column in missing_columns:
            self.assertIn(column, error_message)

    def test_filters_invalid_rows_and_keeps_valid_rows(self) -> None:
        recipes = pd.DataFrame(
            [
                self.valid_recipe(RecipeId=1, Name="Valid one"),
                self.valid_recipe(RecipeId=0, Name="Invalid ID"),
                self.valid_recipe(
                    RecipeId=2, Name="Valid zero macro", ProteinContent=0
                ),
                self.valid_recipe(
                    RecipeId=3,
                    Name="No instructions",
                    RecipeInstructions=[],
                ),
            ],
            index=[10, 20, 30, 40],
        )
        original = recipes.copy(deep=True)

        result = filter_technically_valid_recipes(recipes)

        self.assertEqual(result.index.tolist(), [10, 30])
        self.assertEqual(result["Name"].tolist(), ["Valid one", "Valid zero macro"])
        assert_frame_equal(recipes, original)


class BuildCuratedRecipesTest(unittest.TestCase):
    @staticmethod
    def raw_recipe(**overrides: object) -> dict[str, object]:
        recipe: dict[str, object] = {
            "RecipeId": 38.0,
            "Name": "Chicken and rice",
            "RecipeCategory": "One Dish Meal",
            "Keywords": ["Chicken", "Easy"],
            "Calories": 450.0,
            "ProteinContent": 35.0,
            "FatContent": 12.0,
            "CarbohydrateContent": 48.0,
            "RecipeServings": 2,
            "TotalTime": "PT1H20M",
            "RecipeIngredientParts": ["chicken breast", "rice"],
            "RecipeIngredientQuantities": ["500 g", "200 g"],
            "RecipeInstructions": ["Cook the chicken and rice."],
            "AggregatedRating": 4.5,
            "ReviewCount": 10,
        }
        recipe.update(overrides)
        return recipe

    def build_one(self, **overrides: object) -> pd.DataFrame:
        recipes = pd.DataFrame([self.raw_recipe(**overrides)], index=[42])
        return build_curated_recipes(recipes)

    def test_returns_exact_curated_recipe_columns(self) -> None:
        result = self.build_one()

        self.assertEqual(tuple(result.columns), CURATED_RECIPE_COLUMNS)

    def test_renames_raw_columns(self) -> None:
        result = self.build_one()

        expected_values = {
            "recipe_id": 38,
            "name": "Chicken and rice",
            "source_category": "One Dish Meal",
            "keywords": ["Chicken", "Easy"],
            "calories": 225.0,
            "protein": 17.5,
            "fat": 6.0,
            "carbs": 24.0,
            "servings": 2,
            "ingredients": ["chicken breast", "rice"],
            "instructions": ["Cook the chicken and rice."],
            "rating": 4.5,
            "review_count": 10,
        }
        for column, expected in expected_values.items():
            with self.subTest(column=column):
                self.assertEqual(result.at[42, column], expected)

    def test_parses_total_time(self) -> None:
        result = self.build_one(TotalTime="PT1H20M")

        self.assertEqual(result.at[42, "total_time_minutes"], 80)

    def test_sets_none_for_invalid_total_time(self) -> None:
        result = self.build_one(TotalTime="bad value")

        self.assertIsNone(result.at[42, "total_time_minutes"])

    def test_sets_none_when_total_time_column_is_missing(self) -> None:
        recipe = self.raw_recipe()
        recipe.pop("TotalTime")

        result = build_curated_recipes(pd.DataFrame([recipe], index=[42]))

        self.assertIsNone(result.at[42, "total_time_minutes"])

    def test_adds_null_rating_and_review_count_when_columns_are_missing(self) -> None:
        recipe = self.raw_recipe()
        recipe.pop("AggregatedRating")
        recipe.pop("ReviewCount")

        result = build_curated_recipes(pd.DataFrame([recipe], index=[42]))

        self.assertIn("rating", result.columns)
        self.assertIn("review_count", result.columns)
        self.assertTrue(pd.isna(result.at[42, "rating"]))
        self.assertTrue(pd.isna(result.at[42, "review_count"]))

    def test_converts_numeric_strings_to_numbers(self) -> None:
        result = self.build_one(
            Calories="450",
            ProteinContent="35",
            FatContent="12",
            CarbohydrateContent="48",
            RecipeServings="2",
            AggregatedRating="4.5",
            ReviewCount="10",
        )

        for column in (
            "calories",
            "protein",
            "fat",
            "carbs",
            "servings",
            "rating",
            "review_count",
        ):
            with self.subTest(column=column):
                self.assertTrue(pd.api.types.is_numeric_dtype(result[column]))

    def test_converts_integral_float_recipe_id_to_integer(self) -> None:
        result = self.build_one(RecipeId=38.0)

        self.assertEqual(result.at[42, "recipe_id"], 38)
        self.assertTrue(pd.api.types.is_integer_dtype(result["recipe_id"]))

    def test_converts_collections_to_python_lists(self) -> None:
        result = self.build_one(
            RecipeIngredientParts=np.array(["chicken", "rice"]),
            Keywords=("Chicken", "Easy"),
            RecipeInstructions=np.array(["Cook", "Serve"]),
        )

        for column in ("ingredients", "keywords", "instructions"):
            with self.subTest(column=column):
                self.assertIsInstance(result.at[42, column], list)
        self.assertEqual(result.at[42, "ingredients"], ["chicken", "rice"])
        self.assertEqual(result.at[42, "keywords"], ["Chicken", "Easy"])
        self.assertEqual(result.at[42, "instructions"], ["Cook", "Serve"])

    def test_strips_whitespace_from_strings(self) -> None:
        result = self.build_one(
            Name="  Chicken and rice  ",
            RecipeCategory="  One Dish Meal  ",
            RecipeIngredientParts=[" chicken ", " rice "],
            Keywords=[" Chicken ", " Easy "],
            RecipeInstructions=[" Cook. "],
        )

        self.assertEqual(result.at[42, "name"], "Chicken and rice")
        self.assertEqual(result.at[42, "source_category"], "One Dish Meal")
        self.assertEqual(result.at[42, "ingredients"], ["chicken", "rice"])
        self.assertEqual(result.at[42, "keywords"], ["Chicken", "Easy"])
        self.assertEqual(result.at[42, "instructions"], ["Cook."])

    def test_sets_food_com_source(self) -> None:
        result = self.build_one()

        self.assertEqual(result.at[42, "source"], "food.com")

    def test_excludes_extra_raw_column(self) -> None:
        result = self.build_one(UnexpectedColumn="not curated")

        self.assertNotIn("UnexpectedColumn", result.columns)

    def test_preserves_original_index(self) -> None:
        result = self.build_one()

        self.assertEqual(result.index.tolist(), [42])

    def test_does_not_modify_input_dataframe(self) -> None:
        recipes = pd.DataFrame([self.raw_recipe()], index=[42])
        original = recipes.copy(deep=True)

        build_curated_recipes(recipes)

        assert_frame_equal(recipes, original)

    def test_raises_for_missing_required_source_column(self) -> None:
        recipe = self.raw_recipe()
        recipe.pop("Calories")

        with self.assertRaisesRegex(
            ValueError,
            "Missing columns required to build curated recipes: Calories",
        ):
            build_curated_recipes(pd.DataFrame([recipe]))

    def test_filters_and_builds_expected_curated_dataframe(self) -> None:
        raw = pd.DataFrame(
            [
                self.raw_recipe(
                    RecipeId="38",
                    Name="  Valid recipe  ",
                    Calories="450",
                    ProteinContent="35",
                    FatContent="12",
                    CarbohydrateContent="48",
                    RecipeServings="2",
                    TotalTime="PT1H20M",
                    RecipeIngredientParts=np.array([" chicken ", " rice "]),
                    RecipeInstructions=(" Cook. ", " Serve. "),
                ),
                self.raw_recipe(RecipeId=0, Name="Invalid recipe"),
            ],
            index=[10, 20],
        )

        valid = filter_technically_valid_recipes(raw)
        result = build_curated_recipes(valid)

        expected = pd.DataFrame(
            [
                {
                    "recipe_id": 38,
                    "name": "Valid recipe",
                    "source_category": "One Dish Meal",
                    "keywords": ["Chicken", "Easy"],
                    "calories": 225,
                    "protein": 17.5,
                    "fat": 6,
                    "carbs": 24,
                    "servings": 2,
                    "total_time_minutes": 80,
                    "ingredients": ["chicken", "rice"],
                    "instructions": ["Cook.", "Serve."],
                    "rating": 4.5,
                    "review_count": 10,
                    "source": "food.com",
                }
            ],
            index=[10],
            columns=CURATED_RECIPE_COLUMNS,
        )
        assert_frame_equal(result, expected, check_dtype=False)


class FilterPlausibleCaloriesTest(unittest.TestCase):
    @staticmethod
    def curated_recipe(**overrides: object) -> dict[str, object]:
        recipe: dict[str, object] = {"calories": 500.0}
        recipe.update(overrides)
        return recipe

    def test_keeps_recipe_at_or_below_max_calories(self) -> None:
        recipes = pd.DataFrame([self.curated_recipe(calories=1500.0)], index=[42])

        result = filter_plausible_calories(recipes)

        self.assertEqual(result.index.tolist(), [42])

    def test_removes_recipe_above_max_calories(self) -> None:
        recipes = pd.DataFrame([self.curated_recipe(calories=2500.0)], index=[42])

        result = filter_plausible_calories(recipes)

        self.assertEqual(len(result), 0)

    def test_boundary_value_is_kept_not_strict_inequality(self) -> None:
        recipes = pd.DataFrame([self.curated_recipe(calories=2000.0)], index=[42])

        result = filter_plausible_calories(recipes)

        self.assertEqual(result.index.tolist(), [42])

    def test_custom_max_calories_per_serving_affects_result(self) -> None:
        recipes = pd.DataFrame([self.curated_recipe(calories=800.0)], index=[42])

        result = filter_plausible_calories(recipes, max_calories_per_serving=500.0)

        self.assertEqual(len(result), 0)

    def test_raises_for_missing_calories_column(self) -> None:
        recipes = pd.DataFrame([{"name": "no calories column"}])

        with self.assertRaisesRegex(
            ValueError,
            "Missing columns required to filter plausible calories: calories",
        ):
            filter_plausible_calories(recipes)

    def test_does_not_modify_input_dataframe(self) -> None:
        recipes = pd.DataFrame(
            [self.curated_recipe(calories=2500.0)], index=[42]
        )
        original = recipes.copy(deep=True)

        filter_plausible_calories(recipes)

        assert_frame_equal(recipes, original)

    def test_preserves_original_index_of_remaining_rows(self) -> None:
        recipes = pd.DataFrame(
            [
                self.curated_recipe(calories=500.0),
                self.curated_recipe(calories=2500.0),
                self.curated_recipe(calories=1000.0),
            ],
            index=[10, 20, 30],
        )

        result = filter_plausible_calories(recipes)

        self.assertEqual(result.index.tolist(), [10, 30])


class AddMealTypesTest(unittest.TestCase):
    @staticmethod
    def curated_recipe(**overrides: object) -> dict[str, object]:
        recipe: dict[str, object] = {
            "source_category": "Salad Dressings",
            "keywords": ["Sweet", "Baking"],
        }
        recipe.update(overrides)
        return recipe

    def test_empty_or_missing_keywords_and_category_result_in_empty_meal_types(
        self,
    ) -> None:
        recipes = pd.DataFrame(
            [self.curated_recipe(source_category="Salad Dressings", keywords=[])],
            index=[42],
        )

        result = add_meal_types(recipes)

        self.assertEqual(result.at[42, "meal_types"], [])

    def test_classifies_row_with_breakfast_signal(self) -> None:
        recipes = pd.DataFrame(
            [self.curated_recipe(source_category="Breakfast")], index=[42]
        )

        result = add_meal_types(recipes)

        self.assertEqual(result.at[42, "meal_types"], ["breakfast"])

    def test_classifies_row_with_main_meal_signal(self) -> None:
        recipes = pd.DataFrame(
            [self.curated_recipe(source_category="Chicken")], index=[42]
        )

        result = add_meal_types(recipes)

        self.assertEqual(result.at[42, "meal_types"], ["main_meal"])

    def test_classifies_multiple_rows_independently_in_one_call(self) -> None:
        recipes = pd.DataFrame(
            [
                self.curated_recipe(source_category="Breakfast", keywords=[]),
                self.curated_recipe(source_category="Chicken", keywords=[]),
                self.curated_recipe(source_category="Salad Dressings", keywords=[]),
            ],
            index=[10, 20, 30],
        )

        result = add_meal_types(recipes)

        self.assertEqual(
            result["meal_types"].tolist(),
            [["breakfast"], ["main_meal"], []],
        )

    def test_classifies_row_with_multiple_matched_types(self) -> None:
        recipes = pd.DataFrame(
            [
                self.curated_recipe(
                    source_category="Breakfast", keywords=["Lunch/Snacks"]
                )
            ],
            index=[42],
        )

        result = add_meal_types(recipes)

        self.assertEqual(result.at[42, "meal_types"], ["breakfast", "lunch_snack"])

    def test_raises_for_missing_required_columns(self) -> None:
        recipes = pd.DataFrame([self.curated_recipe()]).drop(columns="keywords")

        with self.assertRaisesRegex(
            ValueError, "Missing columns required to add meal types: keywords"
        ):
            add_meal_types(recipes)

    def test_does_not_modify_input_dataframe(self) -> None:
        recipes = pd.DataFrame([self.curated_recipe()], index=[42])
        original = recipes.copy(deep=True)

        add_meal_types(recipes)

        assert_frame_equal(recipes, original)


class AddAllergenCategoriesTest(unittest.TestCase):
    @staticmethod
    def curated_recipe(**overrides: object) -> dict[str, object]:
        recipe: dict[str, object] = {
            "ingredients": ["chopped carrots", "olive oil"],
        }
        recipe.update(overrides)
        return recipe

    def test_empty_ingredients_result_in_empty_allergen_categories(self) -> None:
        recipes = pd.DataFrame([self.curated_recipe(ingredients=[])], index=[42])

        result = add_allergen_categories(recipes)

        self.assertEqual(result.at[42, "allergen_categories"], [])

    def test_no_matching_ingredients_result_in_empty_allergen_categories(self) -> None:
        recipes = pd.DataFrame(
            [self.curated_recipe(ingredients=["chopped carrots", "olive oil"])],
            index=[42],
        )

        result = add_allergen_categories(recipes)

        self.assertEqual(result.at[42, "allergen_categories"], [])

    def test_classifies_single_allergen_category(self) -> None:
        recipes = pd.DataFrame(
            [self.curated_recipe(ingredients=["salmon fillet"])], index=[42]
        )

        result = add_allergen_categories(recipes)

        self.assertEqual(result.at[42, "allergen_categories"], ["fish"])

    def test_classifies_multiple_allergen_categories_in_fixed_order(self) -> None:
        recipes = pd.DataFrame(
            [self.curated_recipe(ingredients=["shrimp", "peanuts", "carrots"])],
            index=[42],
        )

        result = add_allergen_categories(recipes)

        self.assertEqual(
            result.at[42, "allergen_categories"], ["shellfish", "peanut"]
        )

    def test_classifies_multiple_rows_independently_in_one_call(self) -> None:
        recipes = pd.DataFrame(
            [
                self.curated_recipe(ingredients=["salmon fillet"]),
                self.curated_recipe(ingredients=["peanuts"]),
                self.curated_recipe(ingredients=["chopped carrots"]),
            ],
            index=[10, 20, 30],
        )

        result = add_allergen_categories(recipes)

        self.assertEqual(
            result["allergen_categories"].tolist(),
            [["fish"], ["peanut"], []],
        )

    def test_raises_for_missing_required_columns(self) -> None:
        recipes = pd.DataFrame([self.curated_recipe()]).drop(columns="ingredients")

        with self.assertRaisesRegex(
            ValueError,
            "Missing columns required to add allergen categories: ingredients",
        ):
            add_allergen_categories(recipes)

    def test_does_not_modify_input_dataframe(self) -> None:
        recipes = pd.DataFrame([self.curated_recipe()], index=[42])
        original = recipes.copy(deep=True)

        add_allergen_categories(recipes)

        assert_frame_equal(recipes, original)


if __name__ == "__main__":
    unittest.main()
