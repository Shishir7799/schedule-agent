"""
generate_schedule.py
Generates 30 days of sample schedule data (meetings, workshops, tasks, appointments)
starting from today. Writes to data/schedule.json which acts as the source of truth.
The vector store (ChromaDB) is (re)built from this file.
"""
import json
import random
import uuid
from datetime import datetime, timedelta, date
from pathlib import Path

random.seed(42)

EVENT_TYPES = ["meeting", "workshop", "task", "appointment"]

MEETING_TITLES = [
    "Team Standup", "Sprint Planning", "1:1 with Manager", "Client Sync Call",
    "Product Review", "Design Review", "Budget Discussion", "Marketing Sync",
    "Investor Update Call", "Cross-team Alignment", "Quarterly Planning",
    "Stakeholder Review",
]
WORKSHOP_TITLES = [
    "AI/ML Bootcamp", "Cloud Architecture Workshop", "UX Design Workshop",
    "Hackathon Kickoff", "Public Speaking Workshop", "Leadership Training",
    "Data Engineering Workshop", "Robotics Lab Session", "Career Development Workshop",
]
TASK_TITLES = [
    "Finish project report", "Review pull requests", "Prepare slide deck",
    "Update documentation", "Submit assignment", "Grocery shopping",
    "Pay electricity bill", "Backup project files", "Write blog post",
    "Research competitor products", "Clean up email inbox", "Renew gym membership",
]
APPOINTMENT_TITLES = [
    "Dentist Appointment", "Doctor Checkup", "Haircut", "Car Service",
    "Bank Visit", "Eye Checkup", "Physiotherapy Session", "Consultation Call",
]

TITLE_MAP = {
    "meeting": MEETING_TITLES,
    "workshop": WORKSHOP_TITLES,
    "task": TASK_TITLES,
    "appointment": APPOINTMENT_TITLES,
}

LOCATIONS = ["Zoom", "Google Meet", "Office - Room 204", "Home", "Client Office",
             "Community Hall", "Clinic", "Online", "Conference Room A"]


def random_time(event_type: str):
    """Return a plausible (start_hour, start_min, duration_minutes) for the event type."""
    if event_type == "meeting":
        hour = random.choice([9, 10, 11, 14, 15, 16, 17])
        minute = random.choice([0, 15, 30, 45])
        duration = random.choice([30, 45, 60])
    elif event_type == "workshop":
        hour = random.choice([10, 11, 14])
        minute = 0
        duration = random.choice([120, 180])
    elif event_type == "task":
        hour = random.choice([9, 13, 18, 20])
        minute = random.choice([0, 30])
        duration = random.choice([30, 60, 90])
    else:  # appointment
        hour = random.choice([9, 10, 11, 16, 17])
        minute = random.choice([0, 30])
        duration = 30
    return hour, minute, duration


def make_event(day: date) -> dict:
    etype = random.choices(EVENT_TYPES, weights=[0.4, 0.15, 0.3, 0.15])[0]
    title = random.choice(TITLE_MAP[etype])
    hour, minute, duration = random_time(etype)
    start_dt = datetime(day.year, day.month, day.day, hour, minute)
    end_dt = start_dt + timedelta(minutes=duration)
    location = random.choice(LOCATIONS)
    return {
        "id": str(uuid.uuid4())[:8],
        "title": title,
        "type": etype,
        "date": day.isoformat(),               # YYYY-MM-DD
        "start_time": start_dt.strftime("%H:%M"),
        "end_time": end_dt.strftime("%H:%M"),
        "location": location,
        "description": f"{title} ({etype}) scheduled on {day.strftime('%A, %B %d')} "
                        f"from {start_dt.strftime('%I:%M %p')} to {end_dt.strftime('%I:%M %p')} "
                        f"at {location}.",
    }


def generate_schedule(days: int = 30, start: date | None = None) -> list[dict]:
    start = start or date.today()
    events = []
    for i in range(days):
        day = start + timedelta(days=i)
        # 0-3 events per day, weighted so most days have 1-2
        n_events = random.choices([0, 1, 2, 3], weights=[0.1, 0.35, 0.35, 0.2])[0]
        day_events = [make_event(day) for _ in range(n_events)]
        day_events.sort(key=lambda e: e["start_time"])
        events.extend(day_events)
    return events


def save_schedule(events: list[dict], path: str = None):
    path = path or str(Path(__file__).parent / "schedule.json")
    with open(path, "w") as f:
        json.dump(events, f, indent=2)
    return path


if __name__ == "__main__":
    events = generate_schedule(30)
    out = save_schedule(events)
    print(f"Generated {len(events)} events across 30 days -> {out}")
