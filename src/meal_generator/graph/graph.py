"""Сборка и компиляция LangGraph-графа построения и уточнения дневного рациона."""

from __future__ import annotations

from collections.abc import Callable

from langgraph.graph import END, START, StateGraph

from .nodes import (
    make_build_plan_node,
    make_classify_intent_node,
    make_explain_node,
    make_parse_refinement_node,
    make_parse_request_node,
    make_replace_slot_node,
    relax_pool_size_node,
    route_after_build_plan,
    route_after_classify_intent,
)
from .state import MealPlanState


def build_meal_plan_graph(
    openai_client,
    qdrant_client,
    embed_fn: Callable[[list[str]], list[list[float]]],
):
    """Собрать и скомпилировать граф с двумя путями: полный replan и точечное уточнение.

    ``classify_intent`` роутит каждое сообщение пользователя:

    - ``"new_request"`` (в т.ч. всегда для самого первого сообщения в
      диалоге — см. ``make_classify_intent_node``) идёт по прежнему пути
      ``parse_request -> build_plan -> (relax | explain)``: ``build_plan``
      при нехватке кандидатов (``ValueError``) роутится на ``relax``
      (удвоение ``pool_size``) и возвращается обратно на ``build_plan``,
      пока пул не дойдёт до потолка (``MAX_POOL_SIZE`` в ``nodes.py``) —
      тогда роутер отправляет граф на ``explain`` уже с ошибкой вместо
      вариантов.
    - ``"refine_slot"`` идёт по новому пути ``parse_refinement ->
      replace_slot -> explain``: вместо полного replan точечно заменяется
      один слот текущего ``working_variant``.

    Оба пути сходятся в ``explain``, который сам различает, что объяснять
    (сравнение трёх вариантов или что изменилось при точечном уточнении) по
    ``state["intent"]``.
    """
    graph = StateGraph(MealPlanState)

    graph.add_node("classify_intent", make_classify_intent_node(openai_client))
    graph.add_node("parse_request", make_parse_request_node(openai_client))
    graph.add_node("build_plan", make_build_plan_node(qdrant_client, embed_fn))
    graph.add_node("relax", relax_pool_size_node)
    graph.add_node("parse_refinement", make_parse_refinement_node(openai_client))
    graph.add_node("replace_slot", make_replace_slot_node(qdrant_client, embed_fn))
    graph.add_node("explain", make_explain_node(openai_client))

    graph.add_edge(START, "classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_after_classify_intent,
        {"new_request": "parse_request", "refine_slot": "parse_refinement"},
    )
    graph.add_edge("parse_request", "build_plan")
    graph.add_conditional_edges(
        "build_plan",
        route_after_build_plan,
        {"explain": "explain", "relax": "relax"},
    )
    graph.add_edge("relax", "build_plan")
    graph.add_edge("parse_refinement", "replace_slot")
    graph.add_edge("replace_slot", "explain")
    graph.add_edge("explain", END)

    return graph.compile()
