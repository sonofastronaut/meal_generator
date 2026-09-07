"""Чистые функции конвертации между ``MealPlanState`` и ``MealPlanStateModel``.

Без побочных эффектов и без похода в сеть/клиентов — только структурное
преобразование dataclass-полей ``MealPlanState`` (``parsed_request``,
``constraints``, ``variants``, ``working_variant``) в/из plain dict, которые
умеет сериализовать Pydantic/JSON на границе HTTP API (см. ``schemas.py``).
"""

from __future__ import annotations

import dataclasses

from ..graph.state import MealPlanState
from ..nlp.parse_request import ParsedMealRequest
from ..planning.optimizer import MealPlanConstraints, MealPlanVariant
from .schemas import MealPlanStateModel


def state_to_model(state: MealPlanState) -> MealPlanStateModel:
    """Сериализовать ``MealPlanState`` (с живыми dataclass-инстансами) в ``MealPlanStateModel``."""
    parsed_request = state["parsed_request"]
    constraints = state["constraints"]
    variants = state["variants"]
    working_variant = state["working_variant"]

    return MealPlanStateModel(
        messages=state["messages"],
        parsed_request=(
            dataclasses.asdict(parsed_request) if parsed_request is not None else None
        ),
        constraints=(
            dataclasses.asdict(constraints) if constraints is not None else None
        ),
        pool_size=state["pool_size"],
        variants=(
            [dataclasses.asdict(variant) for variant in variants]
            if variants is not None
            else None
        ),
        working_variant=(
            dataclasses.asdict(working_variant) if working_variant is not None else None
        ),
        error_message=state["error_message"],
        explanation=state["explanation"],
        intent=state["intent"],
        refine_slot_index=state["refine_slot_index"],
        refine_avoid_terms=state["refine_avoid_terms"],
    )


def model_to_state(model: MealPlanStateModel) -> MealPlanState:
    """Восстановить ``MealPlanState`` с живыми dataclass-инстансами из ``MealPlanStateModel``."""
    return MealPlanState(
        messages=model.messages,
        parsed_request=(
            ParsedMealRequest(**model.parsed_request)
            if model.parsed_request is not None
            else None
        ),
        constraints=(
            MealPlanConstraints(**model.constraints)
            if model.constraints is not None
            else None
        ),
        pool_size=model.pool_size,
        variants=(
            [MealPlanVariant(**variant) for variant in model.variants]
            if model.variants is not None
            else None
        ),
        working_variant=(
            MealPlanVariant(**model.working_variant)
            if model.working_variant is not None
            else None
        ),
        error_message=model.error_message,
        explanation=model.explanation,
        intent=model.intent,
        refine_slot_index=model.refine_slot_index,
        refine_avoid_terms=model.refine_avoid_terms,
    )
