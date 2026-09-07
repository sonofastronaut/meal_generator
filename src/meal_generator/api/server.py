"""FastAPI-приложение, держащее тяжёлые ресурсы (LLM-клиент, embedding-модель,
Qdrant-клиент, LangGraph-граф) — единственный процесс, которому они нужны.

Каждый ресурс собирается через ``@lru_cache``-геттер: конструируется один раз
за жизнь процесса при первом обращении, а не на каждый запрос. ``get_graph``
собирает граф прямым вызовом трёх остальных геттеров (не через ``Depends``) —
граф не нужен нигде, кроме себя самого, поэтому не нужно тащить его через
FastAPI DI ради единственного места использования.
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, FastAPI
from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from ..config import get_openai_api_key
from ..graph.graph import build_meal_plan_graph
from .schemas import MealPlanStateModel
from .serialization import model_to_state, state_to_model

app = FastAPI()


@lru_cache
def get_openai_client() -> OpenAI:
    return OpenAI(api_key=get_openai_api_key())


@lru_cache
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


@lru_cache
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(host="localhost", port=6333)


@lru_cache
def get_graph():
    openai_client = get_openai_client()
    qdrant_client = get_qdrant_client()
    embedding_model = get_embedding_model()

    def embed_fn(texts: list[str]) -> list[list[float]]:
        return embedding_model.encode(texts).tolist()

    return build_meal_plan_graph(openai_client, qdrant_client, embed_fn)


@app.get("/health")
def health() -> dict:
    """Живость процесса — без похода ни в какие ресурсы."""
    return {"status": "ok"}


@app.post("/chat", response_model=MealPlanStateModel)
def chat(request: MealPlanStateModel, graph=Depends(get_graph)) -> MealPlanStateModel:
    state = model_to_state(request)
    result = graph.invoke(state)
    result["messages"] = result["messages"] + [
        {"role": "assistant", "content": result["explanation"]}
    ]
    return state_to_model(result)
