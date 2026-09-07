## Qdrant

Start the vector database locally:

```
docker-compose up -d
```

Dashboard UI: http://localhost:6333/dashboard

## Running the app

The app runs as two processes: a FastAPI backend that holds the heavy
resources (LLM client, embedding model, Qdrant client, the LangGraph graph),
and a thin Streamlit client that only talks HTTP to it.

With Qdrant up, recipes indexed (`scripts/build_index.py`), and `.env`
filled in from `.env.example`, start the backend in one terminal:

```
python -m uvicorn meal_generator.api.server:app --reload
```

and the Streamlit client in another:

```
streamlit run app.py
```

By default the client talks to `http://127.0.0.1:8000`; set `API_URL` in
`.env` to point it elsewhere.
