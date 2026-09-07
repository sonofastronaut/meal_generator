import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from meal_generator.preprocessing.exclusions import (  # noqa: E402
    classify_allergen_categories,
    ingredient_matches_exclusion,
    ingredient_matches_terms,
    recipe_is_excluded,
)


class IngredientMatchesExclusionTest(unittest.TestCase):
    def test_matches_exact_signal_word(self) -> None:
        result = ingredient_matches_exclusion("salmon fillet", ["fish"])

        self.assertTrue(result)

    def test_respects_token_boundaries_scallion_vs_scallop(self) -> None:
        result = ingredient_matches_exclusion("chopped scallion", ["shellfish"])

        self.assertFalse(result)

    def test_matches_scallop_for_shellfish(self) -> None:
        result = ingredient_matches_exclusion("bay scallops", ["shellfish"])

        self.assertTrue(result)

    def test_unknown_category_is_ignored_fail_open(self) -> None:
        result = ingredient_matches_exclusion("salmon fillet", ["not_a_real_category"])

        self.assertFalse(result)

    def test_known_and_unknown_categories_mixed(self) -> None:
        result = ingredient_matches_exclusion(
            "salmon fillet", ["not_a_real_category", "fish"]
        )

        self.assertTrue(result)

    def test_case_insensitive_and_whitespace_tolerant(self) -> None:
        cases = ["  SALMON fillet  ", "Salmon Fillet", "salmon fillet"]

        for ingredient in cases:
            with self.subTest(ingredient=ingredient):
                result = ingredient_matches_exclusion(ingredient, ["fish"])
                self.assertTrue(result)

    def test_no_match_returns_false(self) -> None:
        result = ingredient_matches_exclusion("chopped carrots", ["fish", "dairy"])

        self.assertFalse(result)

    def test_flour_matches_gluten_wheat_even_for_almond_flour(self) -> None:
        result = ingredient_matches_exclusion("almond flour", ["gluten_wheat"])

        self.assertTrue(result)


class IngredientMatchesTermsTest(unittest.TestCase):
    def test_exact_token_match(self) -> None:
        result = ingredient_matches_terms("bay scallops", ["scallops"])

        self.assertTrue(result)

    def test_respects_token_boundaries_scallion_vs_scallop(self) -> None:
        result = ingredient_matches_terms("chopped scallion", ["scallop"])

        self.assertFalse(result)

    def test_case_insensitive(self) -> None:
        result = ingredient_matches_terms("Bay Scallops", ["SCALLOPS"])

        self.assertTrue(result)

    def test_empty_avoid_terms_returns_false(self) -> None:
        result = ingredient_matches_terms("salmon fillet", [])

        self.assertFalse(result)

    def test_matches_any_of_multiple_terms(self) -> None:
        result = ingredient_matches_terms("rolled oats", ["scallops", "oats"])

        self.assertTrue(result)


class RecipeIsExcludedTest(unittest.TestCase):
    def test_excluded_when_any_ingredient_matches_any_category(self) -> None:
        ingredients = ["chopped carrots", "peanut butter", "olive oil"]

        result = recipe_is_excluded(ingredients, ["fish", "peanut"])

        self.assertTrue(result)

    def test_not_excluded_when_no_ingredient_matches(self) -> None:
        ingredients = ["chopped carrots", "olive oil", "chopped scallion"]

        result = recipe_is_excluded(ingredients, ["fish", "shellfish"])

        self.assertFalse(result)

    def test_uses_any_not_all_across_ingredients(self) -> None:
        ingredients = ["chopped carrots", "shrimp", "olive oil"]

        result = recipe_is_excluded(ingredients, ["shellfish"])

        self.assertTrue(result)

    def test_unknown_category_does_not_raise_and_does_not_exclude(self) -> None:
        ingredients = ["chopped carrots", "olive oil"]

        result = recipe_is_excluded(ingredients, ["made_up_category"])

        self.assertFalse(result)


class ClassifyAllergenCategoriesTest(unittest.TestCase):
    def test_multiple_matching_categories_in_fixed_order(self) -> None:
        ingredients = ["shrimp", "peanuts", "chopped carrots"]

        result = classify_allergen_categories(ingredients)

        # Порядок фиксирован через EXCLUSION_CATEGORY_SIGNALS
        # (fish, shellfish, peanut, ...), а не через порядок в ingredients,
        # где peanuts упомянуты раньше shrimp.
        self.assertEqual(result, ["shellfish", "peanut"])

    def test_no_matches_returns_empty_list(self) -> None:
        result = classify_allergen_categories(["chopped carrots", "olive oil"])

        self.assertEqual(result, [])

    def test_empty_ingredients_returns_empty_list(self) -> None:
        result = classify_allergen_categories([])

        self.assertEqual(result, [])

    def test_single_matching_category(self) -> None:
        result = classify_allergen_categories(["salmon fillet"])

        self.assertEqual(result, ["fish"])

    def test_consistent_with_recipe_is_excluded(self) -> None:
        ingredients = ["shrimp", "peanuts", "chopped carrots"]

        matched_categories = classify_allergen_categories(ingredients)

        self.assertTrue(recipe_is_excluded(ingredients, matched_categories))
        self.assertFalse(recipe_is_excluded(ingredients, ["dairy"]))


if __name__ == "__main__":
    unittest.main()
