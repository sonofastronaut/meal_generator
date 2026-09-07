"""Тонкий Streamlit-чат поверх FastAPI-бэкенда (``meal_generator.api.server``).

Никакой тяжёлой логики здесь: ни LLM-клиента, ни Qdrant, ни embedding-модели,
ни самого LangGraph-графа — всё это живёт в FastAPI-процессе. Этот файл —
только HTTP-провод между ``st.session_state`` и эндпоинтом ``POST /chat``.
Нужен запущенный uvicorn (``python -m uvicorn meal_generator.api.server:app``)
— без него запросы падают на ``httpx.ConnectError``, обработанный ниже как
понятное сообщение пользователю, а не трейсбек.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import httpx
import streamlit as st

from meal_generator.config import get_api_url  # noqa: E402

st.set_page_config(page_title="Meal Planner")

EMPTY_MEAL_PLAN_STATE = {
    "messages": [],
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

if "meal_plan_state" not in st.session_state:
    st.session_state["meal_plan_state"] = dict(EMPTY_MEAL_PLAN_STATE)

for message in st.session_state["meal_plan_state"]["messages"]:
    st.chat_message(message["role"]).write(message["content"])


def _render_slots_table(slots: list[dict]) -> None:
    st.table(
        [
            {
                "meal": slot["name"],
                "calories": slot["calories"],
                "protein (g)": slot["protein"],
                "fat (g)": slot["fat"],
                "carbs (g)": slot["carbs"],
                "time (min)": slot["total_time_minutes"],
            }
            for slot in slots
        ]
    )


def _render_recipes(slots: list[dict]) -> None:
    for slot in slots:
        with st.expander(f"Recipe: {slot['name']}"):
            for step_number, step in enumerate(slot["instructions"], start=1):
                st.write(f"{step_number}. {step}")


def _render_result(result: dict) -> None:
    """Показать структурированный итог по ``result["intent"]``.

    ``"refine_slot"`` — один рабочий план, одна таблица и один набор
    expander'ов с рецептами. Иначе (полный replan, включая случай ``intent``
    ещё не проставлен) — три отдельных варианта, каждый под своим
    заголовком, своей таблицей и своим независимым набором expander'ов. Если
    ни ``variants``, ни ``working_variant`` не пришли (ошибка построения
    плана), ничего из этого не показываем — explanation уже содержит
    сообщение об ошибке.
    """
    if result["intent"] == "refine_slot":
        if result["working_variant"] is not None:
            slots = result["working_variant"]["slots"]
            _render_slots_table(slots)
            _render_recipes(slots)
        return

    if result["variants"] is not None:
        for variant in result["variants"]:
            st.subheader(variant["label"])
            _render_slots_table(variant["slots"])
            _render_recipes(variant["slots"])


if user_input := st.chat_input("What would you like to eat today?"):
    state = st.session_state["meal_plan_state"]
    state["messages"].append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    api_url = get_api_url()
    try:
        response = httpx.post(f"{api_url}/chat", json=state, timeout=60.0)
        response.raise_for_status()
    except httpx.ConnectError:
        st.error(f"Backend API is unavailable. Expected API at {api_url}")
        st.stop()
    except httpx.TimeoutException:
        st.error("Backend request timed out.")
        st.stop()
    except httpx.HTTPStatusError as exception:
        st.error(
            f"Backend request failed with status {exception.response.status_code}."
        )
        st.stop()

    result = response.json()
    st.session_state["meal_plan_state"] = result

    st.chat_message("assistant").write(result["explanation"])
    _render_result(result)
