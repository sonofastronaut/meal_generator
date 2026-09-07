"""Разовый диагностический scratch-скрипт: почему пользователь получил 5 блюд вместо 4.

Пользователь попросил "4 блюда, 3000 ккал, снек между обедом и ужином, 200г
белка" и получил план из 5 блюд. Печатает сырой ``ParsedMealRequest`` от
``parse_meal_request`` для этого конкретного русскоязычного запроса, чтобы
проверить гипотезу: LLM неверно интерпретирует "снек между обедом и ужином"
как дополнительный слот сверх заявленных 4, а не один из них (см.
build_system_prompt в parse_request.py — там нет явного слота "обед"/"ужин",
только meal_slots/lunch_snack/main_meal). Не тест, не часть постоянной
структуры проекта — падает с обычным трейсбеком, если OPENAI_API_KEY не
настроен.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from openai import OpenAI

from meal_generator.config import get_openai_api_key  # noqa: E402
from meal_generator.nlp.parse_request import parse_meal_request  # noqa: E402

if __name__ == "__main__":
    client = OpenAI(api_key=get_openai_api_key())

    user_message = "Составь рацион из 4 блюд, 3000 ккал, снек между обедом и ужином, 200г белка"

    parsed = parse_meal_request(client, user_message)

    print("meal_slots:", parsed.meal_slots)
    print("count:", len(parsed.meal_slots))
    print()
    print("full parsed request:", parsed)
