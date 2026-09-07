import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from meal_generator.nlp.parse_refinement import (  # noqa: E402
    ParsedRefinement,
    build_refinement_json_schema,
    parse_refinement,
)
from meal_generator.planning.optimizer import MealPlanVariant  # noqa: E402


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


class BuildRefinementJsonSchemaTest(unittest.TestCase):
    def test_schema_shape(self) -> None:
        schema = build_refinement_json_schema()

        self.assertEqual(schema["type"], "object")
        self.assertEqual(schema["properties"]["slot_index"]["type"], "integer")
        self.assertEqual(schema["properties"]["avoid_terms"]["type"], "array")
        self.assertEqual(
            schema["properties"]["avoid_terms"]["items"]["type"], "string"
        )
        self.assertEqual(set(schema["required"]), {"slot_index", "avoid_terms"})
        self.assertFalse(schema["additionalProperties"])


class ParseRefinementTest(unittest.TestCase):
    def test_parses_response_into_parsed_refinement(self) -> None:
        payload = {"slot_index": 1, "avoid_terms": ["mushrooms", "onions"]}
        client, calls = _fake_client(payload)
        working_variant = MealPlanVariant(
            label="balanced",
            slots=[
                {"name": "Oatmeal", "calories": 350.0, "meal_types": ["breakfast"]},
                {"name": "Beef Stew", "calories": 600.0, "meal_types": ["main_meal"]},
            ],
            total_calories=950.0,
            within_tolerance=True,
        )

        result = parse_refinement(client, working_variant, "no mushrooms or onions in the stew")

        self.assertEqual(
            result, ParsedRefinement(slot_index=1, avoid_terms=["mushrooms", "onions"])
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["model"], "gpt-4o-mini")
        self.assertIn("response_format", calls[0])

    def test_prompt_includes_serialized_slots(self) -> None:
        payload = {"slot_index": 0, "avoid_terms": ["oatmeal"]}
        client, calls = _fake_client(payload)
        working_variant = MealPlanVariant(
            label="balanced",
            slots=[
                {"name": "Oatmeal", "calories": 350.0, "meal_types": ["breakfast"]},
            ],
            total_calories=350.0,
            within_tolerance=True,
        )

        parse_refinement(client, working_variant, "not a fan of oatmeal")

        system_message = calls[0]["messages"][0]
        self.assertEqual(system_message["role"], "system")
        self.assertIn("Oatmeal", system_message["content"])
        self.assertIn("0:", system_message["content"])


if __name__ == "__main__":
    unittest.main()
