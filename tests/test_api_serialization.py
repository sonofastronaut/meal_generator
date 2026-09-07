import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from meal_generator.api.serialization import model_to_state, state_to_model  # noqa: E402
from meal_generator.nlp.parse_request import ParsedMealRequest  # noqa: E402
from meal_generator.planning.optimizer import (  # noqa: E402
    MealPlanConstraints,
    MealPlanVariant,
)


def _populated_state() -> dict:
    return {
        "messages": [
            {"role": "user", "content": "spicy asian dinner, 2000 kcal"},
            {"role": "assistant", "content": "Here's your plan..."},
        ],
        "parsed_request": ParsedMealRequest(
            meal_slots=["breakfast", "main_meal"],
            target_calories=2000.0,
            protein_min=100.0,
            excluded_categories=["dairy"],
            query_text="spicy asian",
            max_total_time_minutes=30,
        ),
        "constraints": MealPlanConstraints(
            meal_slots=["breakfast", "main_meal"],
            target_calories=2000.0,
            protein_min=100.0,
            excluded_categories=["dairy"],
            max_total_time_minutes=30,
            query_text="spicy asian",
        ),
        "pool_size": 60,
        "variants": [
            MealPlanVariant(
                label="balanced",
                slots=[{"recipe_id": 1, "name": "Pancakes", "calories": 350.0}],
                total_calories=350.0,
                within_tolerance=True,
            ),
            MealPlanVariant(
                label="quick_cooking",
                slots=[{"recipe_id": 2, "name": "Oatmeal", "calories": 300.0}],
                total_calories=300.0,
                within_tolerance=False,
            ),
        ],
        "working_variant": MealPlanVariant(
            label="balanced",
            slots=[{"recipe_id": 1, "name": "Pancakes", "calories": 350.0}],
            total_calories=350.0,
            within_tolerance=True,
        ),
        "error_message": None,
        "explanation": "The balanced plan is your best bet.",
        "intent": "new_request",
        "refine_slot_index": None,
        "refine_avoid_terms": None,
    }


def _empty_state() -> dict:
    return {
        "messages": [],
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


class SerializationRoundTripTest(unittest.TestCase):
    def test_round_trip_preserves_populated_state(self) -> None:
        state = _populated_state()

        model = state_to_model(state)
        result = model_to_state(model)

        self.assertEqual(result, state)

    def test_round_trip_preserves_none_optional_fields(self) -> None:
        state = _empty_state()

        model = state_to_model(state)
        result = model_to_state(model)

        self.assertEqual(result, state)
        self.assertIsNone(result["parsed_request"])
        self.assertIsNone(result["constraints"])
        self.assertIsNone(result["variants"])
        self.assertIsNone(result["working_variant"])


if __name__ == "__main__":
    unittest.main()
