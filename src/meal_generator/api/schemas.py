"""Pydantic-модели границы HTTP API между Streamlit-клиентом и FastAPI-бэкендом.

``MealPlanStateModel`` используется и как request, и как response body у
``POST /chat`` — тот же контракт "текущий state -> обновлённый state", что
раньше был прямым вызовом ``graph.invoke()`` в ``app.py``, теперь просто
проходит через HTTP. Поля дублируют ``MealPlanState`` (см.
``graph/state.py``), но с dataclass-полями (``parsed_request``,
``constraints``, ``variants``, ``working_variant``), развёрнутыми в
plain dict — Pydantic/JSON не знают про наши dataclass-ы напрямую, поэтому
границу пересекает уже сериализованная форма (см. ``serialization.py``).
"""

from __future__ import annotations

from pydantic import BaseModel


class MealPlanStateModel(BaseModel):
    messages: list[dict]
    parsed_request: dict | None = None
    constraints: dict | None = None
    pool_size: int = 30
    variants: list[dict] | None = None
    working_variant: dict | None = None
    error_message: str | None = None
    explanation: str | None = None
    intent: str | None = None
    refine_slot_index: int | None = None
    refine_avoid_terms: list[str] | None = None
