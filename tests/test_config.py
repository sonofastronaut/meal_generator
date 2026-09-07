import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from meal_generator.config import get_openai_api_key  # noqa: E402


class GetOpenaiApiKeyTest(unittest.TestCase):
    def test_returns_value_when_set(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-123"}, clear=True):
            with patch("meal_generator.config.load_dotenv"):
                result = get_openai_api_key()

        self.assertEqual(result, "sk-test-123")

    def test_raises_runtime_error_when_missing(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with patch("meal_generator.config.load_dotenv"):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"OPENAI_API_KEY is not set\. Copy \.env\.example to \.env "
                    r"and fill in your key\.",
                ):
                    get_openai_api_key()


if __name__ == "__main__":
    unittest.main()
