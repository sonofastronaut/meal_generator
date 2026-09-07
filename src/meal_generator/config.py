"""Тонкая обёртка над переменными окружения — не полноценная система конфигурации."""

from __future__ import annotations

import os

from dotenv import load_dotenv


def get_openai_api_key() -> str:
    """Прочитать ``OPENAI_API_KEY`` из окружения, подхватив ``.env`` из корня проекта.

    Исключения:
        RuntimeError: переменная ``OPENAI_API_KEY`` не установлена.
    """
    load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill in your key."
        )

    return api_key


def get_api_url() -> str:
    """Прочитать ``API_URL`` из окружения, с дефолтом для локальной разработки.

    В отличие от ``get_openai_api_key``, не обязательный — отсутствие
    переменной не ошибка, а просто "бэкенд поднят локально на дефолтном
    порту uvicorn".
    """
    load_dotenv()

    return os.environ.get("API_URL", "http://127.0.0.1:8000")
