import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from qdrant_client import QdrantClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from meal_generator.api.server import app, get_graph  # noqa: E402
from meal_generator.graph.graph import build_meal_plan_graph  # noqa: E402
from meal_generator.indexing.qdrant_index import index_recipes  # noqa: E402

from test_graph import _fake_multi_client  # noqa: E402
from test_qdrant_index import IndexRecipesTest, fake_embed_fn  # noqa: E402


class HealthEndpointTest(unittest.TestCase):
    def test_health_returns_ok_without_touching_any_resource(self) -> None:
        client = TestClient(app)

        response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})


class ChatEndpointTest(unittest.TestCase):
    def test_chat_returns_explanation_and_variants(self) -> None:
        qdrant_client = QdrantClient(":memory:")
        index_recipes(
            qdrant_client, IndexRecipesTest.curated_recipes(), embed_fn=fake_embed_fn
        )

        parse_payload = {
            "meal_slots": ["breakfast", "main_meal"],
            "target_calories": 750.0,
            "protein_min": None,
            "excluded_categories": [],
            "query_text": "",
            "max_total_time_minutes": None,
        }
        openai_client, _ = _fake_multi_client(
            {"meal_request": parse_payload}, "Here's how the plans compare."
        )

        fake_graph = build_meal_plan_graph(openai_client, qdrant_client, fake_embed_fn)
        app.dependency_overrides[get_graph] = lambda: fake_graph

        try:
            client = TestClient(app)
            response = client.post(
                "/chat",
                json={
                    "messages": [
                        {
                            "role": "user",
                            "content": "breakfast and a main meal today",
                        }
                    ]
                },
            )
        finally:
            app.dependency_overrides.clear()

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["explanation"])
        self.assertIsNotNone(body["variants"])
        self.assertEqual(len(body["variants"]), 3)
        self.assertEqual(body["intent"], "new_request")


if __name__ == "__main__":
    unittest.main()
