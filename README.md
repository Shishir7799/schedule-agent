# Agentic RAG Schedule Assistant

An agentic RAG (Retrieval-Augmented Generation) assistant that manages a user's
schedule for the next 30 days. The agent decides, per message, whether to
**retrieve** schedule info or **mutate** the schedule, by calling one of two
tools.

## Architecture

```
data/generate_schedule.py   -> generates 30 days of sample events (meetings,
                                workshops, tasks, appointments) into data/schedule.json
app/schedule_store.py       -> JSON source of truth, thread-safe CRUD
app/vectorstore.py          -> ChromaDB wrapper (TF-IDF embeddings, persisted
                                vectorizer so it survives restarts) — the RAG index
app/tools.py                -> get_schedule (RAG retrieval) + update_schedule (CRUD),
                                keeps ChromaDB in sync with the JSON store
app/agent.py                -> agentic decision layer:
                                  - GeminiAgent: Gemini function calling decides
                                    which tool to call, then composes the reply
                                  - RuleBasedAgent: deterministic fallback used
                                    automatically when GEMINI_API_KEY isn't set,
                                    so the app is fully runnable out of the box
main.py                     -> FastAPI app: /chat, /schedule, /reset, /health,
                                and the playground UI mounted at /agent/playground
static/index.html           -> chat UI for demoing the agent
notebook/schedule_agent_colab.ipynb -> Colab notebook for building/testing
```

### Why TF-IDF embeddings instead of downloading a model?
ChromaDB's default embedding function downloads a model from Hugging Face at
runtime, which is slow to cold-start on a free Render instance. Instead, a
`TfidfVectorizer` is fit over the schedule corpus and the resulting vectors are
passed directly into Chroma via `embeddings=`. This is fully offline, fast, and
good enough for retrieval over short structured event text. The vectorizer is
pickled to disk (`chroma_db/vectorizer.pkl`) so it survives server restarts —
swap in Gemini's `text-embedding-004` or OpenAI embeddings in `app/vectorstore.py`
if you want stronger semantics later.

### Agent decision logic
`app/agent.py` holds the system instruction and two function declarations
(`get_schedule`, `update_schedule`). Gemini decides which tool fits the user's
message (read vs. write intent), calls it, and the tool's JSON result is fed
back to the model to produce a natural-language reply. If `GEMINI_API_KEY` is
not set, `RuleBasedAgent` handles the same example queries with keyword/date/time
parsing — useful for demos without burning API quota, and as a transparent
reference implementation of the same tool contracts.

## Local run

```bash
pip install -r requirements.txt
python data/generate_schedule.py     # (re)generate 30 days of sample data
export GEMINI_API_KEY=your_key_here  # optional — omit to use the rule-based fallback
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/agent/playground` to chat with the assistant.

## Testing in Google Colab

Use `notebook/schedule_agent_colab.ipynb`:
1. Clone your GitHub repo of this project.
2. `pip install -r requirements.txt`.
3. Optionally set `GEMINI_API_KEY`.
4. Run the agent directly in-notebook, or launch the FastAPI server and tunnel
   it with `pyngrok` for a quick shareable link while iterating.

## Deploying to Render (final deployment)

1. Push this project to a GitHub repo (public or private).
2. On [render.com](https://render.com) → **New +** → **Web Service** → connect
   the repo.
3. Render should auto-detect `render.yaml`. If configuring manually instead, use:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Under **Environment**, add `GEMINI_API_KEY` with your key (optional — the
   app works without it via the rule-based fallback, but Gemini function
   calling gives noticeably better natural-language handling).
5. Deploy. Render gives you a URL like `https://<your-service>.onrender.com`.
6. Your playground is at:
   `https://<your-service>.onrender.com/agent/playground`

Free-tier Render services spin down after inactivity — the first request after
idling can take ~30-50s to respond while the instance wakes up.

## Example queries to try in the playground

- "What do I have scheduled tomorrow?"
- "Am I free Friday afternoon?"
- "Add a meeting on August 15 at 3 PM."
- "Move my meeting from 3 PM to 4 PM on August 15."
- "Cancel my meeting on August 15 at 4 PM"

## API reference

| Method | Path                 | Description                                  |
|--------|----------------------|-----------------------------------------------|
| POST   | `/chat`               | `{"message": "..."}` → agent reply + tool trace |
| GET    | `/schedule?date=YYYY-MM-DD` | List all events, or events for one date |
| POST   | `/reset`              | Regenerate the 30-day sample schedule + reindex |
| GET    | `/health`             | Health check |
| GET    | `/agent/playground`   | Chat UI |
