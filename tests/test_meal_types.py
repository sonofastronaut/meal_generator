import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from meal_generator.preprocessing.meal_types import (  # noqa: E402
    classify_meal_types,
)


class ClassifyMealTypesTest(unittest.TestCase):
    def test_classifies_breakfast_from_source_category(self) -> None:
        result = classify_meal_types("Breakfast", [])

        self.assertEqual(result, ["breakfast"])

    def test_classifies_breakfast_from_brunch_keyword(self) -> None:
        result = classify_meal_types(None, ["Brunch"])

        self.assertEqual(result, ["breakfast"])

    def test_classifies_brunch_from_source_category(self) -> None:
        result = classify_meal_types("Brunch", [])

        self.assertEqual(result, ["breakfast"])

    def test_ignores_case_and_surrounding_whitespace(self) -> None:
        cases = [" BREAKFAST ", "breakfast", "BrEaKfAsT"]

        for source_category in cases:
            with self.subTest(source_category=source_category):
                result = classify_meal_types(source_category, [])
                self.assertEqual(result, ["breakfast"])

    def test_classifies_main_meal_from_source_category(self) -> None:
        result = classify_meal_types("Chicken", [])

        self.assertEqual(result, ["main_meal"])

    def test_classifies_main_meal_from_stew_keyword(self) -> None:
        result = classify_meal_types(None, ["Stew"])

        self.assertEqual(result, ["main_meal"])

    def test_returns_single_main_meal_for_multiple_main_meal_signals(self) -> None:
        result = classify_meal_types("Chicken", ["Meat", "Pork", "Stew"])

        self.assertEqual(result, ["main_meal"])

    def test_classifies_main_meal_from_poultry_source_category(self) -> None:
        result = classify_meal_types("Poultry", [])

        self.assertEqual(result, ["main_meal"])

    def test_classifies_main_meal_from_steak_source_category(self) -> None:
        result = classify_meal_types("Steak", [])

        self.assertEqual(result, ["main_meal"])

    def test_classifies_dessert_from_source_category(self) -> None:
        result = classify_meal_types("Dessert", [])

        self.assertEqual(result, ["dessert"])

    def test_classifies_dessert_from_frozen_desserts_keyword(self) -> None:
        result = classify_meal_types(None, ["Frozen Desserts"])

        self.assertEqual(result, ["dessert"])

    def test_classifies_lunch_snack_from_source_category(self) -> None:
        result = classify_meal_types("Lunch/Snacks", [])

        self.assertEqual(result, ["lunch_snack"])

    def test_returns_all_matched_types_in_fixed_order(self) -> None:
        result = classify_meal_types(
            "Breakfast", ["Chicken", "Dessert", "Lunch/Snacks"]
        )

        self.assertEqual(result, ["breakfast", "main_meal", "dessert", "lunch_snack"])

    def test_does_not_use_substring_matching(self) -> None:
        result = classify_meal_types("Chicken Soup", [])

        self.assertEqual(result, [])

    def test_returns_empty_for_unknown_values(self) -> None:
        result = classify_meal_types("Salad Dressings", ["Sweet", "Baking"])

        self.assertEqual(result, [])

    def test_handles_empty_or_missing_keywords(self) -> None:
        for keywords in ([], None):
            with self.subTest(keywords=keywords):
                result = classify_meal_types("Salad Dressings", keywords)
                self.assertEqual(result, [])

    def test_ignores_non_string_keyword_items(self) -> None:
        result = classify_meal_types(None, [None, 42, True, "unknown"])

        self.assertEqual(result, [])

    def test_accepts_single_string_as_keywords(self) -> None:
        result = classify_meal_types(None, "Stew")

        self.assertEqual(result, ["main_meal"])


if __name__ == "__main__":
    unittest.main()
