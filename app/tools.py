"""
tools.py
The two tools exposed to the agent:

1. get_schedule(query, date=None, date_range_start=None, date_range_end=None)
   -> RAG retrieval over ChromaDB, optionally filtered by exact date or a
      date range, then re-ranked/returned as structured events.

2. update_schedule(action, ...) -> add / update / remove entries.
   Writes to the JSON source of truth AND keeps ChromaDB in sync.

Both are plain Python functions with JSON-serializable inputs/outputs so they
can be wired directly into Gemini function calling (see agent.py) or called
by the rule-based fallback router.
"""
from datetime import datetime, timedelta

from . import schedule_store as store
from .vectorstore import get_store


def _today():
    return datetime.today().date()


def get_schedule(query: str = "", date: str = None, date_range_start: str = None,
                  date_range_end: str = None, top_k: int = 8) -> dict:
    """
    Retrieve schedule information relevant to a query and/or date range.

    Args:
        query: free-text description of what the user wants (used for semantic
            retrieval, e.g. "meetings", "am I free", "workshop").
        date: exact ISO date (YYYY-MM-DD) to filter to a single day.
        date_range_start / date_range_end: ISO dates for a range query
            (e.g. "this week").
        top_k: max number of results to return from semantic search when no
            date filter is given.

    Returns:
        dict with "events": list of matching events (each with id, title,
        type, date, start_time, end_time, location, description), and "count".
    """
    vstore = get_store()

    # Exact date filter -> pull directly from the source of truth (precise,
    # not dependent on embedding similarity).
    if date:
        events = store.get_by_date(date)
        return {"events": events, "count": len(events), "mode": "exact_date"}

    if date_range_start and date_range_end:
        all_events = store.list_all()
        events = [e for e in all_events if date_range_start <= e["date"] <= date_range_end]
        events.sort(key=lambda e: (e["date"], e["start_time"]))
        return {"events": events, "count": len(events), "mode": "date_range"}

    # Otherwise fall back to semantic RAG search over the vector DB.
    if not query:
        events = store.list_all()
        events.sort(key=lambda e: (e["date"], e["start_time"]))
        return {"events": events[:top_k], "count": len(events), "mode": "all"}

    result = vstore.query(query, n_results=top_k)
    ids = result.get("ids", [[]])[0]
    events = [store.get_by_id(eid) for eid in ids]
    events = [e for e in events if e]
    events.sort(key=lambda e: (e["date"], e["start_time"]))
    return {"events": events, "count": len(events), "mode": "semantic"}


def find_candidates(date: str = None, start_time: str = None, title: str = None) -> list:
    """Thin wrapper over schedule_store.find_events for the agent layer."""
    return store.find_events(title_contains=title, date_str=date, start_time=start_time)


def update_schedule(action: str, event_id: str = None, title: str = None,
                     type: str = None, date: str = None, start_time: str = None,
                     end_time: str = None, location: str = "",
                     new_date: str = None, new_start_time: str = None,
                     new_end_time: str = None) -> dict:
    """
    Add, update, or remove a schedule entry. Keeps ChromaDB and the JSON
    source of truth in sync.

    Args:
        action: one of "add", "update", "remove".
        event_id: required for "update"/"remove" if known.
        title, type, date, start_time, end_time, location: fields for "add",
            or the fields used to *locate* an event for "update"/"remove"
            when event_id isn't known (e.g. "meeting" at "14:00" on a date).
        new_date, new_start_time, new_end_time: new values to apply on "update".

    Returns:
        dict with "success", "message", and "event" (the resulting event, if any).
    """
    vstore = get_store()

    if action == "add":
        if not (title and date and start_time and end_time):
            return {"success": False, "message": "title, date, start_time and end_time are required to add an event."}
        event = store.add_event(title, type or "meeting", date, start_time, end_time, location or "")
        vstore.upsert_event(event)
        return {"success": True, "message": f"Added '{event['title']}' on {event['date']} at {event['start_time']}.", "event": event}

    if action in ("update", "remove"):
        target = None
        if event_id:
            target = store.get_by_id(event_id)
        if not target:
            # try to resolve via title/date/start_time
            candidates = store.find_events(title_contains=title, date_str=date, start_time=start_time)
            if len(candidates) == 1:
                target = candidates[0]
            elif len(candidates) > 1:
                return {"success": False, "message": "Multiple matching events found, please specify which one.",
                        "candidates": candidates}
        if not target:
            return {"success": False, "message": "Could not find a matching event to modify."}

        if action == "remove":
            store.delete_event(target["id"])
            vstore.delete_event(target["id"])
            return {"success": True, "message": f"Removed '{target['title']}' on {target['date']} at {target['start_time']}."}

        # update
        fields = {}
        if title:
            fields["title"] = title
        if type:
            fields["type"] = type
        if new_date:
            fields["date"] = new_date
        if new_start_time:
            fields["start_time"] = new_start_time
        if new_end_time:
            fields["end_time"] = new_end_time
        if location:
            fields["location"] = location
        updated = store.update_event(target["id"], **fields)
        vstore.upsert_event(updated)
        return {"success": True, "message": f"Updated '{updated['title']}' -> {updated['date']} {updated['start_time']}-{updated['end_time']}.", "event": updated}

    return {"success": False, "message": f"Unknown action '{action}'. Use add, update, or remove."}
