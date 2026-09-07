import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from meal_generator.preprocessing.embedding_text import (  # noqa: E402
    build_embedding_text,
)


class BuildEmbeddingTextTest(unittest.TestCase):
    def test_basic_case_orders_all_parts(self) -> None:
        result = build_embedding_text(
            "Grilled Chicken",
            "Poultry",
            ["Easy", "Dinner"],
            ["chicken breast", "olive oil"],
        )

        self.assertEqual(
            result, "Grilled Chicken Poultry Easy Dinner chicken breast olive oil"
        )

    def test_source_category_duplicate_of_keyword_is_deduplicated(self) -> None:
        result = build_embedding_text(
            "Roast Chicken", "Chicken", ["chicken", "Easy"], []
        )

        self.assertEqual(result, "Roast Chicken Chicken Easy")
        self.assertEqual(result.casefold().count("chicken"), 2)

    def test_keyword_duplicate_of_ingredient_is_deduplicated(self) -> None:
        result = build_embedding_text(
            "Soup", "Lunch/Snacks", ["Tomato"], ["tomato", "onion"]
        )

        self.assertEqual(result, "Soup Lunch/Snacks Tomato onion")

    def test_none_source_category_is_skipped(self) -> None:
        result = build_embedding_text("Salad", None, ["Fresh"], ["lettuce"])

        self.assertEqual(result, "Salad Fresh lettuce")
        self.assertNotIn("None", result)

    def test_empty_keywords_and_ingredients_do_not_raise(self) -> None:
        result = build_embedding_text("Plain Toast", "Breakfast", [], [])

        self.assertEqual(result, "Plain Toast Breakfast")

    def test_all_empty_optional_fields(self) -> None:
        result = build_embedding_text("Water", None, [], [])

        self.assertEqual(result, "Water")

    def test_blank_and_whitespace_only_parts_are_skipped(self) -> None:
        result = build_embedding_text(
            "Stew", "  ", ["", "   ", "Hearty"], ["", "beef"]
        )

        self.assertEqual(result, "Stew Hearty beef")

    def test_name_kept_as_first_occurrence_when_duplicated_later(self) -> None:
        result = build_embedding_text("Chicken", "Chicken", ["chicken"], ["chicken"])

        self.assertEqual(result, "Chicken")


if __name__ == "__main__":
    unittest.main()
