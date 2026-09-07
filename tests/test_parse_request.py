import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from meal_generator.nlp.parse_request import (  # noqa: E402
    ParsedMealRequest,
    _ensure_breakfast_first,
    build_json_schema,
    parse_meal_request,
    to_meal_plan_constraints,
)
from meal_generator.planning.optimizer import MealPlanConstraints  # noqa: E402
from meal_generator.preprocessing.exclusions import (  # noqa: E402
    EXCLUSION_CATEGORY_SIGNALS,
)
from meal_generator.preprocessing.meal_types import _MEAL_TYPE_SIGNALS  # noqa: E402


class BuildJsonSchemaTest(unittest.TestCase):
    def test_meal_slots_enum_matches_meal_types_source_of_truth(self) -> None:
        schema = build_json_schema()

        expected = [meal_type for meal_type, _ in _MEAL_TYPE_SIGNALS]
        self.assertEqual(
            schema["properties"]["meal_slots"]["items"]["enum"], expected
        )

    def test_excluded_categories_enum_matches_exclusions_source_of_truth(self) -> None:
        schema = build_json_schema()

        expected = list(EXCLUSION_CATEGORY_SIGNALS.keys())
        self.assertEqual(
            schema["properties"]["excluded_categories"]["items"]["enum"], expected
        )

    def test_nullable_fields_and_required_shape(self) -> None:
        schema = build_json_schema()

        self.assertEqual(schema["properties"]["protein_min"]["type"], ["number", "null"])
        self.assertEqual(
            schema["properties"]["max_total_time_minutes"]["type"], ["integer", "null"]
        )
        self.assertEqual(schema["properties"]["target_calories"]["type"], "number")
        self.assertEqual(schema["properties"]["query_text"]["type"], "string")
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(
            set(schema["required"]),
            {
                "meal_slots",
                "target_calories",
                "protein_min",
                "excluded_categories",
                "query_text",
                "max_total_time_minutes",
            },
        )


class EnsureBreakfastFirstTest(unittest.TestCase):
    def test_already_first_is_unchanged(self) -> None:
        result = _ensure_breakfast_first(["breakfast", "main_meal", "main_meal"])

        self.assertEqual(result, ["breakfast", "main_meal", "main_meal"])

    def test_moves_breakfast_to_front_preserving_rest_order(self) -> None:
        result = _ensure_breakfast_first(["main_meal", "breakfast", "main_meal"])

        self.assertEqual(result, ["breakfast", "main_meal", "main_meal"])

    def test_replaces_first_element_when_breakfast_missing(self) -> None:
        result = _ensure_breakfast_first(["main_meal", "main_meal", "main_meal"])

        self.assertEqual(result, ["breakfast", "main_meal", "main_meal"])

    def test_replaces_first_element_not_inserts_when_breakfast_missing(self) -> None:
        result = _ensure_breakfast_first(["main_meal", "dessert", "lunch_snack"])

        self.assertEqual(result, ["breakfast", "dessert", "lunch_snack"])

    def test_empty_list_returns_single_breakfast(self) -> None:
        result = _ensure_breakfast_first([])

        self.assertEqual(result, ["breakfast"])

    def test_does_not_mutate_input_when_moving_breakfast(self) -> None:
        original = ["main_meal", "breakfast", "main_meal"]

        _ensure_breakfast_first(original)

        self.assertEqual(original, ["main_meal", "breakfast", "main_meal"])

    def test_does_not_mutate_input_when_replacing_first_element(self) -> None:
        original = ["main_meal", "main_meal", "main_meal"]

        _ensure_breakfast_first(original)

        self.assertEqual(original, ["main_meal", "main_meal", "main_meal"])


def _fake_client(payload: dict):
    content = json.dumps(payload)
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    response = SimpleNamespace(choices=[choice])

    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return response

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    return client, calls


class ParseMealRequestTest(unittest.TestCase):
    def test_parses_response_into_parsed_meal_request(self) -> None:
        payload = {
            "meal_slots": ["breakfast", "main_meal"],
            "target_calories": 2000.0,
            "protein_min": 100.0,
            "excluded_categories": ["dairy"],
            "query_text": "spicy asian food",
            "max_total_time_minutes": 30,
        }
        client, calls = _fake_client(payload)

        result = parse_meal_request(client, "spicy asian dinner, no dairy")

        self.assertEqual(
            result,
            ParsedMealRequest(
                meal_slots=["breakfast", "main_meal"],
                target_calories=2000.0,
                protein_min=100.0,
                excluded_categories=["dairy"],
                query_text="spicy asian food",
                max_total_time_minutes=30,
            ),
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["model"], "gpt-4o-mini")

    def test_handles_null_optional_fields(self) -> None:
        payload = {
            "meal_slots": ["breakfast"],
            "target_calories": 500.0,
            "protein_min": None,
            "excluded_categories": [],
            "query_text": "anything",
            "max_total_time_minutes": None,
        }
        client, _ = _fake_client(payload)

        result = parse_meal_request(client, "just breakfast")

        self.assertIsNone(result.protein_min)
        self.assertIsNone(result.max_total_time_minutes)
        self.assertEqual(result.excluded_categories, [])

    def test_corrects_meal_slots_missing_breakfast_from_the_model(self) -> None:
        payload = {
            "meal_slots": ["main_meal", "main_meal"],
            "target_calories": 1800.0,
            "protein_min": None,
            "excluded_categories": [],
            "query_text": "anything",
            "max_total_time_minutes": None,
        }
        client, _ = _fake_client(payload)

        result = parse_meal_request(client, "two meals today")

        self.assertEqual(result.meal_slots, ["breakfast", "main_meal"])


class ToMealPlanConstraintsTest(unittest.TestCase):
    def test_maps_fields_with_values(self) -> None:
        parsed = ParsedMealRequest(
            meal_slots=["breakfast", "main_meal"],
            target_calories=1800.0,
            protein_min=80.0,
            excluded_categories=["fish", "soy"],
            query_text="light mediterranean",
            max_total_time_minutes=45,
        )

        result = to_meal_plan_constraints(parsed)

        self.assertEqual(
            result,
            MealPlanConstraints(
                meal_slots=["breakfast", "main_meal"],
                target_calories=1800.0,
                protein_min=80.0,
                excluded_categories=["fish", "soy"],
                max_total_time_minutes=45,
                query_text="light mediterranean",
            ),
        )

    def test_maps_none_optional_fields(self) -> None:
        parsed = ParsedMealRequest(
            meal_slots=["breakfast"],
            target_calories=600.0,
            protein_min=None,
            excluded_categories=[],
            query_text="anything",
            max_total_time_minutes=None,
        )

        result = to_meal_plan_constraints(parsed)

        self.assertIsNone(result.protein_min)
        self.assertIsNone(result.max_total_time_minutes)
        self.assertEqual(result.calorie_tolerance, MealPlanConstraints.calorie_tolerance)


if __name__ == "__main__":
    unittest.main()
