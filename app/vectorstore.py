"""
vectorstore.py
Wraps ChromaDB as the vector database for the schedule RAG pipeline.

Embeddings: we fit a TF-IDF vectorizer over the schedule corpus and pass
precomputed vectors into Chroma via `embeddings=`. This avoids downloading
any external embedding model at runtime (fast cold start, works offline,
no extra API key needed) while still giving genuine semantic-ish retrieval
over event titles/types/dates/descriptions. Swap in Gemini/OpenAI embeddings
later by changing `Embedder` if you want stronger semantics.
"""
import json
import pickle
import threading
from pathlib import Path

import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = Path(__file__).parent.parent / "data"
SCHEDULE_JSON = DATA_DIR / "schedule.json"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
VECTORIZER_PATH = CHROMA_DIR / "vectorizer.pkl"

_lock = threading.Lock()


def _doc_text(event: dict) -> str:
    """Text representation of an event used for embedding + retrieval."""
    return (
        f"{event['title']}. Type: {event['type']}. Date: {event['date']}. "
        f"Time: {event['start_time']} to {event['end_time']}. "
        f"Location: {event.get('location', '')}. {event.get('description', '')}"
    )


class Embedder:
    """TF-IDF based embedder, refit whenever the corpus changes."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=2048)
        self._fitted = False

    def fit(self, texts: list[str]):
        if not texts:
            return
        self.vectorizer.fit(texts)
        self._fitted = True

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self._fitted:
            self.fit(texts)
        return self.vectorizer.transform(texts).toarray().tolist()

    def save(self, path: Path = VECTORIZER_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.vectorizer, f)

    def load(self, path: Path = VECTORIZER_PATH) -> bool:
        if not path.exists():
            return False
        with open(path, "rb") as f:
            self.vectorizer = pickle.load(f)
        self._fitted = True
        return True


class ScheduleVectorStore:
    def __init__(self, persist_dir: str = None):
        persist_dir = persist_dir or str(CHROMA_DIR)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="schedule", metadata={"hnsw:space": "cosine"}
        )
        self.embedder = Embedder()
        self.embedder.load()  # reuse a previously-fitted vocabulary if present

    # ---------- bulk load / rebuild ----------
    def rebuild_from_json(self, path: str = None):
        path = path or str(SCHEDULE_JSON)
        with open(path) as f:
            events = json.load(f)
        with _lock:
            # wipe collection
            existing = self.collection.get()
            if existing and existing.get("ids"):
                self.collection.delete(ids=existing["ids"])

            if not events:
                return 0

            texts = [_doc_text(e) for e in events]
            self.embedder.fit(texts)
            self.embedder.save()
            vectors = self.embedder.embed(texts)

            self.collection.add(
                ids=[e["id"] for e in events],
                documents=texts,
                embeddings=vectors,
                metadatas=[
                    {
                        "title": e["title"],
                        "type": e["type"],
                        "date": e["date"],
                        "start_time": e["start_time"],
                        "end_time": e["end_time"],
                        "location": e.get("location", ""),
                    }
                    for e in events
                ],
            )
        return len(events)

    # ---------- CRUD used by update_schedule tool ----------
    def upsert_event(self, event: dict):
        text = _doc_text(event)
        # Reuse the existing fitted vocabulary (loaded from disk in __init__ or
        # built by rebuild_from_json). If somehow still unfitted, rebuild the
        # whole index from the JSON source of truth so dimensions stay
        # consistent with what's already stored in the collection.
        if not self.embedder._fitted:
            self.rebuild_from_json()
        vector = self.embedder.embed([text])[0]
        with _lock:
            self.collection.upsert(
                ids=[event["id"]],
                documents=[text],
                embeddings=[vector],
                metadatas=[
                    {
                        "title": event["title"],
                        "type": event["type"],
                        "date": event["date"],
                        "start_time": event["start_time"],
                        "end_time": event["end_time"],
                        "location": event.get("location", ""),
                    }
                ],
            )

    def delete_event(self, event_id: str):
        with _lock:
            self.collection.delete(ids=[event_id])

    # ---------- retrieval ----------
    def query(self, query_text: str, n_results: int = 5, where: dict = None):
        if not self.embedder._fitted:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]]}
        vector = self.embedder.embed([query_text])[0]
        return self.collection.query(
            query_embeddings=[vector],
            n_results=n_results,
            where=where,
        )

    def get_by_date(self, date_str: str):
        return self.collection.get(where={"date": date_str})

    def get_all(self):
        return self.collection.get()


_store_singleton = None


def get_store() -> ScheduleVectorStore:
    global _store_singleton
    if _store_singleton is None:
        _store_singleton = ScheduleVectorStore()
        # build index if empty
        existing = _store_singleton.collection.get()
        if not existing or not existing.get("ids"):
            if SCHEDULE_JSON.exists():
                _store_singleton.rebuild_from_json()
    return _store_singleton
