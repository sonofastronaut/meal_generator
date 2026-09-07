## Qdrant

Запуск векторной базы данных локально:

```
docker-compose up -d
```

UI дашборда: http://localhost:6333/dashboard

## Запуск приложения

Приложение состоит из двух процессов: FastAPI-бэкенд держит тяжёлые ресурсы
(LLM-клиент, embedding-модель, Qdrant-клиент, сам LangGraph-граф), а тонкий
Streamlit-клиент только ходит к нему по HTTP.

После того как Qdrant поднят, рецепты проиндексированы
(`scripts/build_index.py`), а `.env` заполнен на основе `.env.example`,
запусти бэкенд в одном терминале:

```
python -m uvicorn meal_generator.api.server:app --reload
```

и Streamlit-клиент в другом:

```
streamlit run app.py
```

По умолчанию клиент обращается к `http://127.0.0.1:8000`; чтобы указать
другой адрес, задай `API_URL` в `.env`.
