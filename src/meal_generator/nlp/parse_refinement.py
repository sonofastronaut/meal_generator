"""Парсинг пользовательского уточнения одного слота существующего плана.

Использует OpenAI structured outputs — тот же подход и то же обоснование,
что и в ``parse_request.py``: strict JSON schema гарантирует валидную форму
ответа на стороне OpenAI API, поэтому ручная валидация после парсинга не
нужна.

``client`` инъецируется в ``parse_refinement``, а не создаётся внутри неё —
та же причина тестируемости без реального похода в сеть/API-ключа, что и у
``embed_fn``/``client`` во всём остальном проекте: логику парсинга ответа и
сборки ``ParsedRefinement`` можно проверить с фейковым клиентом.

В отличие от ``parse_request.py``, здесь модели нужен контекст — текущий
рабочий план (``working_variant``), чтобы она могла указать конкретный
``slot_index`` (0-based, по порядку в ``working_variant.slots``), поэтому
system-промпт строится не статически, а как функция от ``working_variant``
(``build_refinement_system_prompt``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ..planning.optimizer import MealPlanVariant


@dataclass
class ParsedRefinement:
    slot_index: int
    avoid_terms: list[str]


def build_refinement_json_schema() -> dict:
    """Построить JSON-схему ``ParsedRefinement`` для OpenAI structured outputs.

    Чистая функция — не ходит в OpenAI.
    """
    return {
        "type": "object",
        "properties": {
            "slot_index": {"type": "integer"},
            "avoid_terms": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["slot_index", "avoid_terms"],
        "additionalProperties": False,
    }


def _format_working_variant_for_prompt(working_variant: MealPlanVariant) -> str:
    """Сериализовать слоты ``working_variant`` в компактный пронумерованный список."""
    lines = []
    for index, slot in enumerate(working_variant.slots):
        meal_types = slot.get("meal_types")
        meal_type = meal_types[0] if meal_types else None
        lines.append(
            f"{index}: {slot['name']} (meal_type={meal_type!r}, "
            f"{slot['calories']:.0f} kcal)"
        )
    return "\n".join(lines)


def build_refinement_system_prompt(working_variant: MealPlanVariant) -> str:
    """Построить system-промпт для парсинга уточнения одного слота ``working_variant``.

    Включает сериализованный (через ``_format_working_variant_for_prompt``)
    пронумерованный список текущих слотов плана, чтобы модель могла явно
    указать ``slot_index`` из последнего сообщения пользователя, а не
    угадывать его вслепую.
    """
    slots_text = _format_working_variant_for_prompt(working_variant)
    return f"""The user already has a meal plan for today with these meals:

{slots_text}

The user's latest message asks to change exactly one specific meal in this \
plan (e.g. swap it out, remove a disliked ingredient or dish). From it,
identify:

- slot_index: the 0-based index (matching the numbered list above) of the \
single meal the user wants changed.
- avoid_terms: the specific word(s)/ingredient(s)/dish(es) the user wants \
excluded from the replacement, taken from their message (e.g. "oatmeal", \
"mushrooms").

Respond with a JSON object matching the schema — no extra commentary."""


def parse_refinement(
    client,
    working_variant: MealPlanVariant,
    user_message: str,
    model: str = "gpt-4o-mini",
) -> ParsedRefinement:
    """Распарсить уточняющее сообщение пользователя в ``ParsedRefinement`` через OpenAI.

    ``client`` — экземпляр ``openai.OpenAI``, создаётся и передаётся
    вызывающим кодом (см. докстроку модуля).
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": build_refinement_system_prompt(working_variant),
            },
            {"role": "user", "content": user_message},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "meal_refinement",
                "strict": True,
                "schema": build_refinement_json_schema(),
            },
        },
    )

    data = json.loads(response.choices[0].message.content)

    return ParsedRefinement(
        slot_index=data["slot_index"],
        avoid_terms=data["avoid_terms"],
    )
