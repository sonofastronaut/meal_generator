import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from qdrant_client import QdrantClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from meal_generator.graph.graph import build_meal_plan_graph  # noqa: E402
from meal_generator.graph.nodes import (  # noqa: E402
    make_build_plan_node,
    make_classify_intent_node,
    make_explain_node,
    make_parse_refinement_node,
    make_parse_request_node,
    make_replace_slot_node,
    relax_pool_size_node,
    route_after_build_plan,
)
from meal_generator.indexing.qdrant_index import build_payload, index_recipes  # noqa: E402
from meal_generator.nlp.parse_request import ParsedMealRequest  # noqa: E402
from meal_generator.planning.optimizer import (  # noqa: E402
    MealPlanConstraints,
    MealPlanVariant,
)

from test_parse_request import _fake_client  # noqa: E402
from test_qdrant_index import IndexRecipesTest, fake_embed_fn  # noqa: E402


def _indexed_client() -> QdrantClient:
    client = QdrantClient(":memory:")
    index_recipes(client, IndexRecipesTest.curated_recipes(), embed_fn=fake_embed_fn)
    return client


def _payload_by_recipe_id() -> dict[int, dict]:
    """Real payload dicts (as build_payload would produce them) for the shared
    test fixtures, keyed by recipe_id — avoids hand-typing slot dicts that
    drift from the actual payload shape."""
    recipes = IndexRecipesTest.curated_recipes()
    return {
        int(row["recipe_id"]): build_payload(row) for _, row in recipes.iterrows()
    }


def _indexed_client_with_diverging_calorie_candidates() -> QdrantClient:
    """An in-memory Qdrant client whose main_meal pool is built so that
    "first by retrieval rank" (rating-sorted, since query_text is empty)
    and "closest by calories to the replaced slot" pick different recipes
    — needed to prove replace_slot_node picks by calorie closeness, not by
    whatever order the retrieval function happens to return."""
    recipes = pd.DataFrame(
        [
            {
                "recipe_id": 101,
                "name": "Order-First Dish",
                "source_category": "Test",
                "keywords": [],
                "meal_types": ["main_meal"],
                "calories": 300.0,
                "protein": 20.0,
                "fat": 10.0,
                "carbs": 20.0,
                "servings": 2,
                "total_time_minutes": 20,
                "ingredients": ["dish a"],
                "instructions": ["Cook dish a."],
                "rating": 5.0,
                "review_count": 50,
                "allergen_categories": [],
            },
            {
                "recipe_id": 102,
                "name": "Middle Dish",
                "source_category": "Test",
                "keywords": [],
                "meal_types": ["main_meal"],
                "calories": 548.0,
                "protein": 30.0,
                "fat": 15.0,
                "carbs": 30.0,
                "servings": 2,
                "total_time_minutes": 25,
                "ingredients": ["dish b"],
                "instructions": ["Cook dish b."],
                "rating": 3.0,
                "review_count": 20,
                "allergen_categories": [],
            },
            {
                "recipe_id": 103,
                "name": "Calorie-Closest Dish",
                "source_category": "Test",
                "keywords": [],
                "meal_types": ["main_meal"],
                "calories": 560.0,
                "protein": 35.0,
                "fat": 18.0,
                "carbs": 25.0,
                "servings": 2,
                "total_time_minutes": 30,
                "ingredients": ["dish c"],
                "instructions": ["Cook dish c."],
                "rating": 1.0,
                "review_count": 1,
                "allergen_categories": [],
            },
        ]
    )
    client = QdrantClient(":memory:")
    index_recipes(client, recipes, embed_fn=fake_embed_fn)
    return client


def _fake_text_client(text: str):
    """Fake OpenAI client returning a plain-text response (no JSON schema)."""
    message = SimpleNamespace(content=text)
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


def _fake_multi_client(schema_responses: dict, text_response: str):
    """Fake OpenAI client serving multiple structured-output call sites plus a
    plain-text one, dispatched by ``response_format["json_schema"]["name"]``
    (structured calls) or absence of ``response_format`` (plain-text calls,
    e.g. the explain node)."""
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        response_format = kwargs.get("response_format")
        if response_format is None:
            content = text_response
        else:
            schema_name = response_format["json_schema"]["name"]
            content = json.dumps(schema_responses[schema_name])
        message = SimpleNamespace(content=content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    return client, calls


class RelaxPoolSizeNodeTest(unittest.TestCase):
    def test_doubles_pool_size(self) -> None:
        result = relax_pool_size_node({"pool_size": 30})

        self.assertEqual(result, {"pool_size": 60})

    def test_caps_at_max_pool_size(self) -> None:
        result = relax_pool_size_node({"pool_size": 80})

        self.assertEqual(result, {"pool_size": 100})


class RouteAfterBuildPlanTest(unittest.TestCase):
    def test_routes_to_explain_when_variants_present(self) -> None:
        state = {"variants": [object()], "error_message": None, "pool_size": 30}

        self.assertEqual(route_after_build_plan(state), "explain")

    def test_routes_to_relax_when_error_and_pool_not_maxed(self) -> None:
        state = {"variants": None, "error_message": "boom", "pool_size": 30}

        self.assertEqual(route_after_build_plan(state), "relax")

    def test_routes_to_explain_when_error_and_pool_maxed(self) -> None:
        state = {"variants": None, "error_message": "boom", "pool_size": 100}

        self.assertEqual(route_after_build_plan(state), "explain")


class MakeClassifyIntentNodeTest(unittest.TestCase):
    def test_empty_messages_is_new_request_without_llm_call(self) -> None:
        client, calls = _fake_client({"intent": "refine_slot"})
        node = make_classify_intent_node(client)

        result = node({"messages": []})

        self.assertEqual(result, {"intent": "new_request"})
        self.assertEqual(len(calls), 0)

    def test_first_message_is_new_request_without_llm_call(self) -> None:
        client, calls = _fake_client({"intent": "refine_slot"})
        node = make_classify_intent_node(client)

        result = node({"messages": [{"role": "user", "content": "hi"}]})

        self.assertEqual(result, {"intent": "new_request"})
        self.assertEqual(len(calls), 0)

    def test_non_empty_history_classifies_via_llm_new_request(self) -> None:
        client, calls = _fake_client({"intent": "new_request"})
        node = make_classify_intent_node(client)
        messages = [
            {"role": "user", "content": "spicy asian dinner today"},
            {"role": "assistant", "content": "Here is your plan..."},
            {"role": "user", "content": "actually give me a totally different day"},
        ]

        result = node({"messages": messages})

        self.assertEqual(result, {"intent": "new_request"})
        self.assertEqual(len(calls), 1)

    def test_non_empty_history_classifies_via_llm_refine_slot(self) -> None:
        client, calls = _fake_client({"intent": "refine_slot"})
        node = make_classify_intent_node(client)
        messages = [
            {"role": "user", "content": "spicy asian dinner today"},
            {"role": "assistant", "content": "Here is your plan..."},
            {"role": "user", "content": "no mushrooms in the stew please"},
        ]

        result = node({"messages": messages})

        self.assertEqual(result, {"intent": "refine_slot"})
        self.assertEqual(len(calls), 1)


class MakeParseRequestNodeTest(unittest.TestCase):
    def test_returns_parsed_request_constraints_and_starting_pool_size(self) -> None:
        payload = {
            "meal_slots": ["breakfast", "main_meal"],
            "target_calories": 2000.0,
            "protein_min": 100.0,
            "excluded_categories": ["dairy"],
            "query_text": "spicy asian food",
            "max_total_time_minutes": 30,
        }
        client, _ = _fake_client(payload)
        node = make_parse_request_node(client)

        result = node(
            {"messages": [{"role": "user", "content": "spicy asian dinner, no dairy"}]}
        )

        self.assertEqual(result["pool_size"], 30)
        self.assertEqual(
            result["parsed_request"],
            ParsedMealRequest(
                meal_slots=["breakfast", "main_meal"],
                target_calories=2000.0,
                protein_min=100.0,
                excluded_categories=["dairy"],
                query_text="spicy asian food",
                max_total_time_minutes=30,
            ),
        )
        self.assertEqual(
            result["constraints"],
            MealPlanConstraints(
                meal_slots=["breakfast", "main_meal"],
                target_calories=2000.0,
                protein_min=100.0,
                excluded_categories=["dairy"],
                max_total_time_minutes=30,
                query_text="spicy asian food",
            ),
        )


class MakeBuildPlanNodeTest(unittest.TestCase):
    def test_success_returns_variants_working_variant_and_no_error(self) -> None:
        client = _indexed_client()
        node = make_build_plan_node(client, fake_embed_fn)
        constraints = MealPlanConstraints(
            meal_slots=["breakfast", "main_meal"],
            target_calories=750.0,
            query_text="anything",
        )

        result = node({"constraints": constraints, "pool_size": 30})

        self.assertIsNone(result["error_message"])
        self.assertIsNotNone(result["variants"])
        self.assertEqual(len(result["variants"]), 3)
        self.assertIs(result["working_variant"], result["variants"][0])

    def test_pool_exhaustion_returns_none_variants_and_error_message(self) -> None:
        client = _indexed_client()
        node = make_build_plan_node(client, fake_embed_fn)
        # Test data has only 2 distinct main_meal recipes; 3 slots exhaust the pool.
        constraints = MealPlanConstraints(
            meal_slots=["main_meal", "main_meal", "main_meal"],
            target_calories=1500.0,
            query_text="",
        )

        result = node({"constraints": constraints, "pool_size": 30})

        self.assertIsNone(result["variants"])
        self.assertTrue(result["error_message"])


class MakeParseRefinementNodeTest(unittest.TestCase):
    def test_returns_slot_index_and_avoid_terms(self) -> None:
        payload = {"slot_index": 1, "avoid_terms": ["mushrooms"]}
        client, calls = _fake_client(payload)
        node = make_parse_refinement_node(client)
        working_variant = MealPlanVariant(
            label="balanced",
            slots=[
                {"name": "Pancakes", "calories": 350.0, "meal_types": ["breakfast"]},
                {"name": "Chicken Stew", "calories": 500.0, "meal_types": ["main_meal"]},
            ],
            total_calories=850.0,
            within_tolerance=True,
        )

        result = node(
            {
                "working_variant": working_variant,
                "messages": [
                    {"role": "user", "content": "no mushrooms in the stew"}
                ],
            }
        )

        self.assertEqual(
            result, {"refine_slot_index": 1, "refine_avoid_terms": ["mushrooms"]}
        )
        self.assertEqual(len(calls), 1)


class MakeReplaceSlotNodeTest(unittest.TestCase):
    def test_success_replaces_slot_and_recalculates_totals(self) -> None:
        client = _indexed_client()
        node = make_replace_slot_node(client, fake_embed_fn)
        payloads = _payload_by_recipe_id()
        # breakfast=id2 (Pancakes), main_meal=id3 (Chicken Stew); only other
        # main_meal candidate in the test pool is id1 (Grilled Salmon).
        working_variant = MealPlanVariant(
            label="balanced",
            slots=[payloads[2], payloads[3]],
            total_calories=850.0,
            within_tolerance=False,
        )
        constraints = MealPlanConstraints(
            meal_slots=["breakfast", "main_meal"],
            target_calories=750.0,
            query_text="",
        )

        result = node(
            {
                "working_variant": working_variant,
                "refine_slot_index": 1,
                "refine_avoid_terms": [],
                "constraints": constraints,
                "pool_size": 30,
            }
        )

        self.assertIsNone(result["error_message"])
        updated = result["working_variant"]
        self.assertEqual(updated.slots[0]["recipe_id"], 2)
        self.assertEqual(updated.slots[1]["recipe_id"], 1)
        self.assertEqual(updated.total_calories, 350.0 + 400.0)

    def test_picks_candidate_closest_by_calories_not_first_by_retrieval_order(
        self,
    ) -> None:
        client = _indexed_client_with_diverging_calorie_candidates()
        node = make_replace_slot_node(client, fake_embed_fn)
        # Replaced slot itself isn't in the candidate pool (id 999), so all
        # three fixture recipes (101/102/103) remain after filtering.
        # Rating-sorted (non-semantic, empty query_text) retrieval order is
        # [101 (rating 5.0), 102 (rating 3.0), 103 (rating 1.0)] — "first by
        # order" would be 101 (300 kcal). Closest by calories to the
        # replaced slot's 555 kcal is 103 (560 kcal, diff 5), not 101
        # (diff 255) or 102 (548 kcal, diff 7).
        target_slot = {
            "recipe_id": 999,
            "name": "Old Dish",
            "calories": 555.0,
            "meal_types": ["main_meal"],
        }
        working_variant = MealPlanVariant(
            label="balanced",
            slots=[target_slot],
            total_calories=555.0,
            within_tolerance=True,
        )
        constraints = MealPlanConstraints(
            meal_slots=["main_meal"],
            target_calories=555.0,
            query_text="",
        )

        result = node(
            {
                "working_variant": working_variant,
                "refine_slot_index": 0,
                "refine_avoid_terms": [],
                "constraints": constraints,
                "pool_size": 30,
            }
        )

        self.assertIsNone(result["error_message"])
        self.assertEqual(result["working_variant"].slots[0]["recipe_id"], 103)

    def test_no_candidates_left_leaves_working_variant_untouched(self) -> None:
        client = _indexed_client()
        node = make_replace_slot_node(client, fake_embed_fn)
        payloads = _payload_by_recipe_id()
        # breakfast slot's own recipe (id2) is the only breakfast candidate
        # in the test pool, and it's already "existing" (it's the slot being
        # replaced) — so the pool is exhausted regardless of avoid_terms.
        working_variant = MealPlanVariant(
            label="balanced",
            slots=[payloads[2], payloads[1]],
            total_calories=750.0,
            within_tolerance=True,
        )
        constraints = MealPlanConstraints(
            meal_slots=["breakfast", "main_meal"],
            target_calories=750.0,
            query_text="",
        )

        result = node(
            {
                "working_variant": working_variant,
                "refine_slot_index": 0,
                "refine_avoid_terms": [],
                "constraints": constraints,
                "pool_size": 30,
            }
        )

        self.assertNotIn("working_variant", result)
        self.assertTrue(result["error_message"])
        self.assertIn("slot 0", result["error_message"])

    def test_out_of_range_slot_index_returns_error_instead_of_raising(self) -> None:
        client = _indexed_client()
        node = make_replace_slot_node(client, fake_embed_fn)
        payloads = _payload_by_recipe_id()
        working_variant = MealPlanVariant(
            label="balanced",
            slots=[payloads[2], payloads[1]],
            total_calories=750.0,
            within_tolerance=True,
        )
        constraints = MealPlanConstraints(
            meal_slots=["breakfast", "main_meal"],
            target_calories=750.0,
            query_text="",
        )

        # working_variant only has 2 slots (indices 0-1); the LLM's
        # slot_index is only type-validated by the JSON schema, not
        # range-checked, so an out-of-range value must not raise IndexError.
        result = node(
            {
                "working_variant": working_variant,
                "refine_slot_index": 5,
                "refine_avoid_terms": [],
                "constraints": constraints,
                "pool_size": 30,
            }
        )

        self.assertNotIn("working_variant", result)
        self.assertTrue(result["error_message"])
        self.assertIn("5", result["error_message"])

    def test_avoid_terms_filters_out_matching_candidates(self) -> None:
        client = _indexed_client()
        node = make_replace_slot_node(client, fake_embed_fn)
        payloads = _payload_by_recipe_id()
        working_variant = MealPlanVariant(
            label="balanced",
            slots=[payloads[2], payloads[1]],
            total_calories=750.0,
            within_tolerance=True,
        )
        constraints = MealPlanConstraints(
            meal_slots=["breakfast", "main_meal"],
            target_calories=750.0,
            query_text="",
        )

        # Only remaining main_meal candidate (id3, Chicken Stew) contains
        # "carrots" — avoiding it exhausts the pool entirely.
        result = node(
            {
                "working_variant": working_variant,
                "refine_slot_index": 1,
                "refine_avoid_terms": ["carrots"],
                "constraints": constraints,
                "pool_size": 30,
            }
        )

        self.assertNotIn("working_variant", result)
        self.assertTrue(result["error_message"])


class MakeExplainNodeTest(unittest.TestCase):
    def test_explains_variants_via_openai_without_response_format(self) -> None:
        client, calls = _fake_text_client("Balanced is your best bet.")
        node = make_explain_node(client)
        variants = [
            MealPlanVariant(
                label="balanced",
                slots=[
                    {"name": "Oatmeal", "calories": 400.0, "total_time_minutes": 10}
                ],
                total_calories=400.0,
                within_tolerance=True,
            )
        ]

        result = node(
            {
                "variants": variants,
                "error_message": None,
                "intent": "new_request",
                "messages": [{"role": "user", "content": "give me something balanced"}],
            }
        )

        self.assertEqual(result["explanation"], "Balanced is your best bet.")
        self.assertEqual(len(calls), 1)
        self.assertNotIn("response_format", calls[0])

    def test_error_case_skips_openai_call(self) -> None:
        client, calls = _fake_text_client("should not be used")
        node = make_explain_node(client)

        result = node(
            {
                "variants": None,
                "error_message": "Not enough distinct candidates for meal_type 'main_meal'",
                "intent": "new_request",
            }
        )

        self.assertIn(
            "Not enough distinct candidates for meal_type 'main_meal'",
            result["explanation"],
        )
        self.assertEqual(len(calls), 0)

    def test_refine_slot_error_skips_openai_call(self) -> None:
        client, calls = _fake_text_client("should not be used")
        node = make_explain_node(client)

        result = node(
            {
                "intent": "refine_slot",
                "error_message": "Couldn't find a replacement for slot 1 avoiding ['salmon']",
                "variants": None,
            }
        )

        self.assertEqual(
            result["explanation"],
            "Couldn't find a replacement for slot 1 avoiding ['salmon']",
        )
        self.assertEqual(len(calls), 0)

    def test_refine_slot_success_explains_the_change_via_openai(self) -> None:
        client, calls = _fake_text_client("Swapped the stew for something else.")
        node = make_explain_node(client)
        working_variant = MealPlanVariant(
            label="balanced",
            slots=[
                {"name": "Pancakes", "calories": 350.0, "total_time_minutes": 20},
                {"name": "Grilled Salmon", "calories": 400.0, "total_time_minutes": 30},
            ],
            total_calories=750.0,
            within_tolerance=True,
        )

        result = node(
            {
                "intent": "refine_slot",
                "error_message": None,
                "working_variant": working_variant,
                "refine_slot_index": 1,
                "refine_avoid_terms": ["mushrooms"],
                "messages": [{"role": "user", "content": "no mushrooms in the stew"}],
            }
        )

        self.assertEqual(result["explanation"], "Swapped the stew for something else.")
        self.assertEqual(len(calls), 1)
        self.assertNotIn("response_format", calls[0])


class BuildMealPlanGraphTest(unittest.TestCase):
    def test_end_to_end_success_produces_variants_and_explanation(self) -> None:
        qdrant_client = _indexed_client()
        parse_payload = {
            "meal_slots": ["breakfast", "main_meal"],
            "target_calories": 750.0,
            "protein_min": None,
            "excluded_categories": [],
            "query_text": "",
            "max_total_time_minutes": None,
        }
        openai_client, calls = _fake_multi_client(
            {"meal_request": parse_payload}, "Here's how the plans compare."
        )

        graph = build_meal_plan_graph(openai_client, qdrant_client, fake_embed_fn)

        initial_state = {
            "messages": [
                {"role": "user", "content": "breakfast and a main meal today"}
            ],
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

        result = graph.invoke(initial_state)

        self.assertEqual(result["intent"], "new_request")
        self.assertEqual(result["explanation"], "Here's how the plans compare.")
        self.assertIsNotNone(result["variants"])
        self.assertEqual(len(result["variants"]), 3)
        self.assertIs(result["working_variant"], result["variants"][0])
        self.assertEqual(len(calls), 2)

    def test_end_to_end_refine_path_replaces_one_slot(self) -> None:
        qdrant_client = _indexed_client()
        parse_payload = {
            "meal_slots": ["breakfast", "main_meal"],
            "target_calories": 750.0,
            "protein_min": None,
            "excluded_categories": [],
            "query_text": "",
            "max_total_time_minutes": None,
        }
        schema_responses = {
            "meal_request": parse_payload,
            "intent_classification": {"intent": "refine_slot"},
            "meal_refinement": {"slot_index": 1, "avoid_terms": ["salmon"]},
        }
        openai_client, calls = _fake_multi_client(
            schema_responses, "OK, here's what changed."
        )

        graph = build_meal_plan_graph(openai_client, qdrant_client, fake_embed_fn)

        turn1_state = {
            "messages": [
                {"role": "user", "content": "breakfast and a main meal today"}
            ],
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

        turn1_result = graph.invoke(turn1_state)

        # balanced picks breakfast=id2 (Pancakes, 350 kcal) and, closest to
        # the remaining 400 kcal residual, main_meal=id1 (Grilled Salmon).
        self.assertEqual(turn1_result["working_variant"].slots[0]["recipe_id"], 2)
        self.assertEqual(turn1_result["working_variant"].slots[1]["recipe_id"], 1)

        turn2_state = dict(turn1_result)
        turn2_state["messages"] = turn1_result["messages"] + [
            {"role": "assistant", "content": turn1_result["explanation"]},
            {"role": "user", "content": "no salmon, something else for the main meal"},
        ]

        turn2_result = graph.invoke(turn2_state)

        self.assertEqual(turn2_result["intent"], "refine_slot")
        self.assertIsNone(turn2_result["error_message"])
        # id1 (salmon) is avoided; the only remaining main_meal candidate is id3.
        self.assertEqual(turn2_result["working_variant"].slots[0]["recipe_id"], 2)
        self.assertEqual(turn2_result["working_variant"].slots[1]["recipe_id"], 3)
        self.assertTrue(turn2_result["explanation"])


if __name__ == "__main__":
    unittest.main()
