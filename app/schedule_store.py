"""
schedule_store.py
JSON file acts as the source of truth for schedule events (id, title, type,
date, start_time, end_time, location, description). ChromaDB is kept in sync
with this file on every write, so retrieval always reflects the latest state.
"""
import json
import threading
import uuid
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
SCHEDULE_JSON = DATA_DIR / "schedule.json"

_lock = threading.Lock()


def _load() -> list[dict]:
    if not SCHEDULE_JSON.exists():
        return []
    with open(SCHEDULE_JSON) as f:
        return json.load(f)


def _save(events: list[dict]):
    with open(SCHEDULE_JSON, "w") as f:
        json.dump(events, f, indent=2)


def _make_description(e: dict) -> str:
    from datetime import datetime
    try:
        d = datetime.strptime(e["date"], "%Y-%m-%d")
        day_str = d.strftime("%A, %B %d")
    except Exception:
        day_str = e["date"]
    try:
        start_disp = datetime.strptime(e["start_time"], "%H:%M").strftime("%I:%M %p")
        end_disp = datetime.strptime(e["end_time"], "%H:%M").strftime("%I:%M %p")
    except Exception:
        start_disp, end_disp = e["start_time"], e["end_time"]
    return (f"{e['title']} ({e['type']}) scheduled on {day_str} "
            f"from {start_disp} to {end_disp} at {e.get('location', '')}.")


def list_all() -> list[dict]:
    with _lock:
        return _load()


def get_by_date(date_str: str) -> list[dict]:
    with _lock:
        return [e for e in _load() if e["date"] == date_str]


def get_by_id(event_id: str) -> dict | None:
    with _lock:
        for e in _load():
            if e["id"] == event_id:
                return e
    return None


def add_event(title: str, type_: str, date_str: str, start_time: str,
              end_time: str, location: str = "") -> dict:
    with _lock:
        events = _load()
        event = {
            "id": str(uuid.uuid4())[:8],
            "title": title,
            "type": type_,
            "date": date_str,
            "start_time": start_time,
            "end_time": end_time,
            "location": location,
        }
        event["description"] = _make_description(event)
        events.append(event)
        _save(events)
        return event


def update_event(event_id: str, **fields) -> dict | None:
    with _lock:
        events = _load()
        for e in events:
            if e["id"] == event_id:
                e.update({k: v for k, v in fields.items() if v is not None})
                e["description"] = _make_description(e)
                _save(events)
                return e
    return None


def delete_event(event_id: str) -> bool:
    with _lock:
        events = _load()
        new_events = [e for e in events if e["id"] != event_id]
        changed = len(new_events) != len(events)
        if changed:
            _save(new_events)
        return changed


def find_events(title_contains: str = None, date_str: str = None,
                 start_time: str = None) -> list[dict]:
    """Loose match helper used to resolve 'move my meeting from 2PM to 4PM'
    style requests where the agent doesn't have an exact event id."""
    with _lock:
        events = _load()
    results = events
    if date_str:
        results = [e for e in results if e["date"] == date_str]
    if start_time:
        results = [e for e in results if e["start_time"] == start_time]
    if title_contains:
        tl = title_contains.lower()
        results = [e for e in results if tl in e["title"].lower()]
    return results
