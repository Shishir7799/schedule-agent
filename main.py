"""
main.py
FastAPI entrypoint for the Agentic RAG Schedule Assistant.

Endpoints:
  GET  /                    -> redirects to /agent/playground
  GET  /agent/playground    -> simple chat UI for demoing the agent
  POST /chat                -> {"message": "..."} -> agent reply + tool trace
  GET  /schedule            -> list all schedule events (debug/inspection)
  POST /reset               -> regenerate the 30-day sample schedule
  GET  /health              -> health check for Render
"""
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.agent import get_agent
from app import schedule_store as store
from app.vectorstore import get_store
from data.generate_schedule import generate_schedule, save_schedule

app = FastAPI(title="Agentic RAG Schedule Assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
app.mount("/agent/playground", StaticFiles(directory=str(BASE_DIR / "static"), html=True), name="playground")


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def root():
    return RedirectResponse(url="/agent/playground")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    agent = get_agent()
    result = agent.chat(req.message)
    return result


@app.get("/schedule")
def list_schedule(date: str = None):
    if date:
        return {"events": store.get_by_date(date)}
    events = store.list_all()
    events.sort(key=lambda e: (e["date"], e["start_time"]))
    return {"events": events, "count": len(events)}


@app.post("/reset")
def reset_schedule():
    events = generate_schedule(30)
    path = save_schedule(events)
    vstore = get_store()
    n = vstore.rebuild_from_json(path)
    return {"message": f"Regenerated {n} sample events and rebuilt the vector index."}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
