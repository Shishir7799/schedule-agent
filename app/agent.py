"""
agent.py
The agentic decision layer. Two tools are exposed to the model:
  - get_schedule    (RAG retrieval)
  - update_schedule (CRUD)

Primary path: Google Gemini function calling (google-generativeai) decides
which tool to call (if any) based on the user's message, calls it, then
composes a natural-language answer from the tool result.

Fallback path: if GEMINI_API_KEY is not set (e.g. quick local/Colab demo
without a key), a lightweight rule-based intent router handles the same
example queries deterministically, so the app is fully runnable out of the box.
"""
import os
import re
import json
from datetime import datetime, timedelta

from . import tools

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()  # best-effort default, not relied on below

SYSTEM_INSTRUCTION = """You are a Schedule Assistant agent that manages a user's \
calendar for the next 30 days. You have two tools:

1. get_schedule(query, date, date_range_start, date_range_end, top_k) - use this to \
ANSWER questions about what's on the schedule, check free/busy time, or look things up. \
Use `date` for a specific day (resolve relative dates like "tomorrow" or "Friday" to an \
actual YYYY-MM-DD date yourself using today's date). Use date_range_start/end for ranges \
like "this week". Use `query` for open-ended/semantic lookups ("do I have any workshops?").

2. update_schedule(action, ...) - use this whenever the user wants to ADD, MOVE/UPDATE, or \
REMOVE/CANCEL something on their schedule. action is "add", "update", or "remove". For \
"update"/"remove" you can pass event_id if known, otherwise pass title/date/start_time to \
help find the right event, plus new_date/new_start_time/new_end_time for what should change.

Decide which tool (if any) is needed based on the user's request; call get_schedule for \
read/lookup requests and update_schedule for write/mutating requests. After receiving a \
tool result, answer the user in a short, friendly, natural sentence or two - don't dump raw \
JSON. If update_schedule reports multiple candidates or a failure, ask a brief clarifying \
question instead of guessing.
"""

FUNCTION_DECLARATIONS = [
    {
        "name": "get_schedule",
        "description": "Retrieve schedule information based on date, date range, or a free-text query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Free-text query for semantic retrieval."},
                "date": {"type": "string", "description": "Exact ISO date YYYY-MM-DD."},
                "date_range_start": {"type": "string", "description": "ISO start date for a range query."},
                "date_range_end": {"type": "string", "description": "ISO end date for a range query."},
            },
        },
    },
    {
        "name": "update_schedule",
        "description": "Add, update, or remove a schedule entry.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["add", "update", "remove"]},
                "event_id": {"type": "string"},
                "title": {"type": "string"},
                "type": {"type": "string", "enum": ["meeting", "workshop", "task", "appointment"]},
                "date": {"type": "string"},
                "start_time": {"type": "string", "description": "HH:MM 24h"},
                "end_time": {"type": "string", "description": "HH:MM 24h"},
                "location": {"type": "string"},
                "new_date": {"type": "string"},
                "new_start_time": {"type": "string"},
                "new_end_time": {"type": "string"},
            },
            "required": ["action"],
        },
    },
]

TOOL_IMPL = {"get_schedule": tools.get_schedule, "update_schedule": tools.update_schedule}


# ----------------------------------------------------------------------
# Gemini-backed agent
# ----------------------------------------------------------------------
class GeminiAgent:
    # Try Google's stable "latest" alias first (it always points at whatever
    # flash model Google currently recommends for new API keys), then fall
    # back through pinned versions in case the alias isn't available on your
    # account. This avoids the app breaking again the next time Google
    # retires a specific model version.
    CANDIDATE_MODELS = [
        "gemini-flash-latest",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
    ]

    def __init__(self, api_key: str):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.genai = genai
        self.model = None
        last_err = None
        for name in self.CANDIDATE_MODELS:
            try:
                candidate = genai.GenerativeModel(
                    model_name=name,
                    system_instruction=SYSTEM_INSTRUCTION,
                    tools=[{"function_declarations": FUNCTION_DECLARATIONS}],
                )
                # cheap call to confirm this model name actually works with this key
                candidate.generate_content("ping", request_options={"timeout": 10})
                self.model = candidate
                self.model_name = name
                break
            except Exception as e:
                last_err = e
                continue
        if self.model is None:
            raise RuntimeError(f"No usable Gemini model found for this API key: {last_err}")

    REQUEST_TIMEOUT_SECONDS = 20

    def chat(self, message: str, history: list = None) -> dict:
        chat = self.model.start_chat(history=history or [])
        today_ctx = f"Today's date is {datetime.today().date().isoformat()}. User: {message}"
        opts = {"timeout": self.REQUEST_TIMEOUT_SECONDS}
        response = chat.send_message(today_ctx, request_options=opts)
        tool_calls_log = []

        # handle (possibly chained) function calls
        for _ in range(4):
            parts = response.candidates[0].content.parts
            fn_call = next((p.function_call for p in parts if getattr(p, "function_call", None)), None)
            if not fn_call:
                break
            fn_name = fn_call.name
            fn_args = {k: v for k, v in fn_call.args.items()}
            result = TOOL_IMPL[fn_name](**fn_args)
            tool_calls_log.append({"tool": fn_name, "args": fn_args, "result": result})
            response = chat.send_message(
                self.genai.protos.Content(
                    parts=[self.genai.protos.Part(
                        function_response=self.genai.protos.FunctionResponse(
                            name=fn_name, response={"result": json.dumps(result, default=str)}
                        )
                    )]
                ),
                request_options=opts,
            )

        return {"reply": response.text, "tool_calls": tool_calls_log, "backend": "gemini"}


# ----------------------------------------------------------------------
# Rule-based fallback agent (no API key required)
# ----------------------------------------------------------------------
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def _resolve_date(text: str) -> str | None:
    text = text.lower()
    today = datetime.today().date()
    if "today" in text:
        return today.isoformat()
    if "tomorrow" in text:
        return (today + timedelta(days=1)).isoformat()
    for i, wd in enumerate(WEEKDAYS):
        if wd in text:
            days_ahead = (i - today.weekday()) % 7
            days_ahead = days_ahead or 7  # "friday" means the upcoming Friday
            return (today + timedelta(days=days_ahead)).isoformat()
    m = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{1,2})", text)
    if m:
        month_map = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
                     "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
        month = month_map[m.group(1)[:3]]
        day = int(m.group(2))
        year = today.year
        try:
            d = datetime(year, month, day).date()
        except ValueError:
            return None
        if d < today:
            d = datetime(year + 1, month, day).date()
        return d.isoformat()
    return None


def _resolve_time(text: str) -> str | None:
    m = re.search(r"(\d{1,2})(:(\d{2}))?\s*(am|pm)", text.lower())
    if not m:
        return None
    hour = int(m.group(1))
    minute = int(m.group(3) or 0)
    ampm = m.group(4)
    if ampm == "pm" and hour != 12:
        hour += 12
    if ampm == "am" and hour == 12:
        hour = 0
    return f"{hour:02d}:{minute:02d}"


def _resolve_all_times(text: str) -> list[str]:
    return [
        _resolve_time(f"{h}:{mi or '00'} {ap}")
        for h, _, mi, ap in re.findall(r"(\d{1,2})(:(\d{2}))?\s*(am|pm)", text.lower())
    ]


GREETING_WORDS = {"hi", "hello", "hey", "yo", "sup", "hiya", "greetings", "good morning",
                   "good afternoon", "good evening", "thanks", "thank you", "ok", "okay"}


class RuleBasedAgent:
    """Deterministic fallback so the app works without a Gemini API key."""

    def chat(self, message: str, history: list = None) -> dict:
        text = message.lower().strip(" !.?")

        if text in GREETING_WORDS or len(text) <= 3:
            reply = ("Hi! I'm your Schedule Assistant. Ask me things like "
                     "\"what do I have tomorrow?\", \"am I free Friday afternoon?\", "
                     "or \"add a meeting on August 15 at 3 PM.\"")
            return {"reply": reply, "tool_calls": [], "backend": "rule_based"}

        date = _resolve_date(text)

        is_write = any(w in text for w in ["add", "schedule a", "book", "create",
                                            "move", "reschedule", "change", "cancel",
                                            "remove", "delete", "update"])

        if is_write:
            return self._handle_update(text, date)
        return self._handle_get(text, date)

    def _handle_get(self, text: str, date: str | None) -> dict:
        if date:
            result = tools.get_schedule("", date=date)
        elif "week" in text:
            today = datetime.today().date()
            end = today + timedelta(days=7)
            result = tools.get_schedule("", date_range_start=today.isoformat(), date_range_end=end.isoformat())
        else:
            result = tools.get_schedule(text)

        events = result["events"]
        if not events:
            reply = f"You're free{' on ' + date if date else ''} - nothing on the schedule."
        else:
            lines = [f"- {e['title']} ({e['type']}) {e['start_time']}-{e['end_time']} on {e['date']}" for e in events]
            reply = f"Here's what I found ({len(events)}):\n" + "\n".join(lines)
        return {"reply": reply, "tool_calls": [{"tool": "get_schedule", "args": {"date": date}, "result": result}], "backend": "rule_based"}

    def _handle_update(self, text: str, date: str | None) -> dict:
        times = _resolve_all_times(text)

        if any(w in text for w in ["cancel", "remove", "delete"]):
            args = {"action": "remove", "date": date, "start_time": times[0] if times else None}
            result = tools.update_schedule(**args)
            return {"reply": result["message"], "tool_calls": [{"tool": "update_schedule", "args": args, "result": result}], "backend": "rule_based"}

        if any(w in text for w in ["move", "reschedule", "change"]) and len(times) >= 2:
            args = {"action": "update", "date": date, "start_time": times[0], "new_start_time": times[1]}
            # try to preserve original duration when shifting the start time
            candidates = tools.find_candidates(date, times[0])
            if len(candidates) == 1:
                orig = candidates[0]
                dur = (datetime.strptime(orig["end_time"], "%H:%M") - datetime.strptime(orig["start_time"], "%H:%M")).seconds // 60
                new_end_dt = datetime.strptime(times[1], "%H:%M") + timedelta(minutes=dur)
                args["new_end_time"] = new_end_dt.strftime("%H:%M")
            result = tools.update_schedule(**args)
            return {"reply": result["message"], "tool_calls": [{"tool": "update_schedule", "args": args, "result": result}], "backend": "rule_based"}

        # default: add
        title_match = re.search(r"(add|schedule|book|create)\s+(a|an)?\s*(.*?)\s+(on|for|at)\b", text)
        title = title_match.group(3).strip().title() if title_match else "New Event"
        start = times[0] if times else "09:00"
        end_hour = (int(start.split(":")[0]) + 1) % 24
        end = f"{end_hour:02d}:{start.split(':')[1]}"
        etype = "meeting" if "meeting" in text else (
            "workshop" if "workshop" in text else ("appointment" if "appointment" in text else "task"))
        args = {"action": "add", "title": title, "type": etype, "date": date or datetime.today().date().isoformat(),
                "start_time": start, "end_time": end}
        result = tools.update_schedule(**args)
        return {"reply": result["message"], "tool_calls": [{"tool": "update_schedule", "args": args, "result": result}], "backend": "rule_based"}


class SafeAgent:
    """Wraps GeminiAgent with a per-message fallback to RuleBasedAgent so a
    transient Gemini error (rate limit, network blip, deprecated model name,
    etc.) degrades gracefully instead of returning a 500 to the client."""

    def __init__(self, primary, fallback):
        self.primary = primary
        self.fallback = fallback

    def chat(self, message: str, history: list = None) -> dict:
        try:
            return self.primary.chat(message, history)
        except Exception as e:
            result = self.fallback.chat(message, history)
            result["backend"] = "rule_based_fallback"
            result["gemini_error"] = str(e)
            return result


_agent_singleton = None


def get_agent():
    global _agent_singleton
    if _agent_singleton is None:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        print(f"[agent] Fresh env check -> key_present={bool(api_key)} key_length={len(api_key)}")
        if api_key:
            try:
                _agent_singleton = SafeAgent(GeminiAgent(api_key), RuleBasedAgent())
                print(f"[agent] Using Gemini model: {_agent_singleton.primary.model_name}")
            except Exception as e:
                print(f"[agent] Gemini init FAILED, falling back to rule-based. Reason: {e}")
                _agent_singleton = RuleBasedAgent()
        else:
            print("[agent] No GEMINI_API_KEY set in this terminal session -> using rule-based agent.")
            _agent_singleton = RuleBasedAgent()
    return _agent_singleton
