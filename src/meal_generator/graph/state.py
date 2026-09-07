"""LangGraph state для пайплайна построения и уточнения дневного рациона.

``messages`` — полная история диалога (``{"role": "user"|"assistant", "content": str}``
на элемент), включая только что отправленное сообщение пользователя как
последний элемент. ``user_message`` как отдельное поле не хранится:
"текущее" сообщение пользователя — это последняя запись с ``role == "user"``
в ``messages`` (см. ``_latest_user_message`` в ``nodes.py``).

``working_variant`` — текущий рабочий план (изначально ``variants[0]``,
``"balanced"``), который точечно правится узлом ``replace_slot`` при
уточнении, вместо полного replan. ``constraints``/``pool_size`` из
исходного запроса сохраняются в state и после построения плана — они нужны
``replace_slot`` для повторного retrieval с теми же structured constraints.
"""

from __future__ import annotations

from typing import TypedDict

from ..nlp.parse_request import ParsedMealRequest
from ..planning.optimizer import MealPlanConstraints, MealPlanVariant


class MealPlanState(TypedDict):
    messages: list[dict]
    parsed_request: ParsedMealRequest | None
    constraints: MealPlanConstraints | None
    pool_size: int
    variants: list[MealPlanVariant] | None
    working_variant: MealPlanVariant | None
    error_message: str | None
    explanation: str | None
    intent: str | None
    refine_slot_index: int | None
    refine_avoid_terms: list[str] | None
