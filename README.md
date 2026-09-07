# Meal Generator

An AI meal planning assistant. You describe what you want in plain language —
"3 meals, ~2000 kcal, high protein, no dairy, something Asian-inspired" — and
it turns that into structured constraints, retrieves matching recipes from a
curated corpus of over 300k real recipes, and greedily assembles three
different daily meal plans, each explained in plain language. You can then
refine the plan conversationally, one meal at a time ("swap the lunch for
something without mushrooms").

## Architecture

```
raw Food.com data (522k recipes)
  → preprocessing (technical validation, per-serving nutrition normalization,
     multi-label meal-type classification, allergen-category extraction,
     plausibility filtering)
  → sentence-transformer embeddings → Qdrant (dense vectors + payload filters)
  → LangGraph agent (OpenAI structured outputs: classify intent → parse
     request → retrieve candidates → build 3 plan variants → explain;
     conversational refinement of individual meals)
  → FastAPI backend (holds the LLM client, embedding model, Qdrant client,
     and compiled graph)
  → Streamlit chat client (thin HTTP layer, no ML dependencies of its own)
```

The pipeline lives in `src/meal_generator/`:

- `preprocessing/` — turns raw Food.com rows into a curated schema
  (`recipes.py`), classifies meal type (`meal_types.py`) and allergen
  categories (`exclusions.py`), and builds the text that gets embedded
  (`embedding_text.py`).
- `indexing/` — builds Qdrant payloads and upserts the curated corpus
  (`qdrant_index.py`).
- `retrieval/` — combines semantic search with structured payload filters
  (`search.py`).
- `planning/` — assembles a full day's plan from retrieved candidates with
  greedy heuristics (`optimizer.py`).
- `nlp/` — parses free-form user text into structured requests and
  refinements via OpenAI structured outputs (`parse_request.py`,
  `parse_refinement.py`).
- `graph/` — wires all of the above into a LangGraph state machine
  (`graph.py`, `nodes.py`, `state.py`).
- `api/` — the FastAPI app that owns every heavy resource for the process
  (`server.py`, plus request/response schemas and serialization).

## Key engineering decisions

- **Multi-label meal-type classification, not single-label with priority.**
  A recipe can legitimately be both a `main_meal` and a `lunch_snack`;
  forcing a single label with a priority order would silently drop valid
  matches. `classify_meal_types` returns every type with a matching signal
  instead of picking one winner.

- **Token-based matching for allergen exclusion, exact-value matching for
  meal-type classification — different problems, different matching.**
  Meal-type signals come from Food.com's own structured `source_category`
  and `keywords` fields, so an exact normalized-string comparison is
  correct. Ingredient text is free-form ("almond flour", "2 cups milk"), so
  allergen exclusion tokenizes each ingredient and matches on individual
  words. The trade-off is explicit in the code: "flour" as a token matches
  `gluten_wheat` even in "almond flour" — a known, accepted v1 simplification,
  not a bug.

- **Per-serving nutrition normalization.** Food.com's `Calories`/`ProteinContent`/etc.
  are totals for the whole recipe (all `RecipeServings` portions), not per
  serving — a fact that silently broke every downstream calorie filter,
  the optimizer's targets, and the numbers shown to the user until it was
  found and fixed. `build_curated_recipes` now divides by `servings` once,
  at the schema-conversion step, so everything downstream can treat
  `calories`/`protein`/`fat`/`carbs` as per-serving values.

- **Greedy heuristics instead of an ILP solver for daily plan assembly.**
  Building three variants (balanced / quick-cooking / high-variety) is done
  with a straightforward greedy pass over slots — pick the locally best
  candidate for a running residual-calorie target. This is a deliberate MVP
  trade-off: far simpler to implement, debug, and explain to a user than
  formulating the problem as global ILP optimization (OR-Tools/PuLP) across
  calories, time, and variety simultaneously. Upgrading to a solver is a
  future step if the greedy approach turns out to produce noticeably
  suboptimal plans in practice.

- **sentence-transformers/MiniLM instead of OpenAI embeddings.** Recipe
  embeddings are computed locally with `all-MiniLM-L6-v2` rather than an
  OpenAI embedding endpoint — the corpus is self-contained and indexing
  doesn't create a runtime dependency on an external API or add per-call
  latency/cost to a step that only needs to run once per corpus rebuild.

- **Separate semantic and rating-based retrieval paths (the query-bleed
  problem).** Embedding an empty `query_text` produces a degenerate vector
  whose cosine similarity is effectively noise dressed up as a score. Worse,
  a vibe-heavy query like "spicy asian dinner" would semantically pull
  breakfast recipes toward dinner-ish results. `search_recipes_by_rating`
  is a second, non-semantic retrieval path — structured filters plus a sort
  by rating/review count — used whenever `query_text` is empty, and always
  for breakfast slots regardless of query content, as a deliberate point-fix
  for that bleed.

- **FastAPI + Streamlit as two separate processes.** The backend owns every
  heavy, slow-to-load resource (LLM client, embedding model, Qdrant client,
  compiled LangGraph graph) behind `@lru_cache` getters, constructed once per
  process. The Streamlit client only speaks HTTP to it. This avoids paying
  ML-library cold-start cost on every Streamlit rerun and follows the same
  process-split pattern used elsewhere for apps with heavy ML dependencies.

## Stack

Python · pandas/numpy · sentence-transformers · Qdrant · LangGraph ·
OpenAI API (structured outputs) · FastAPI · Streamlit · Docker · unittest

## Data

Recipes come from the [Food.com Recipes and Reviews dataset on
Kaggle](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews)
(~522k recipes). `data/recipes.parquet` and `data/reviews.parquet` are
**not included in this repository** — they exceed GitHub's file size limit.
Download the dataset from Kaggle and place the two parquet files in `data/`
before running the indexing step below.

## Setup and running

1. Clone the repository and install dependencies:

   ```
   pip install -e .
   ```

2. Copy `.env.example` to `.env` and fill in your OpenAI API key:

   ```
   cp .env.example .env
   ```

3. Download the Food.com dataset from Kaggle (see [Data](#data)) and place
   `recipes.parquet`/`reviews.parquet` in `data/`.

4. Start Qdrant locally:

   ```
   docker-compose up -d
   ```

   Dashboard UI: http://localhost:6333/dashboard

5. Build the index — a one-time step that loads the embedding model,
   filters and classifies the full dataset, and indexes it into Qdrant.
   Takes roughly 20-40 minutes for the full corpus:

   ```
   python scripts/build_index.py
   ```

6. Run the app as two processes. In one terminal, the FastAPI backend:

   ```
   python -m uvicorn meal_generator.api.server:app --reload
   ```

   In another, the Streamlit client:

   ```
   streamlit run app.py
   ```

   By default the client talks to `http://127.0.0.1:8000`; set `API_URL` in
   `.env` to point it elsewhere.

## Tests

```
python -m unittest discover -s tests -v
```

214 tests across the pipeline. Tests use an in-memory Qdrant client
(`QdrantClient(":memory:")`) and fake LLM clients — no real Qdrant
container, embedding model, or OpenAI API key is needed to run the suite.
