import json
from pathlib import Path
from kg_rag.config import settings
from kg_rag.models import JiraTicket


def read_json(file_path: str | Path) -> dict:
    path = Path(file_path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_tickets(file_path: str | Path | None = None) -> list[JiraTicket]:
    """Load and parse tickets from JSON into JiraTicket dataclasses.
    Falls back to settings.json_file if no path is provided.
    """
    raw = read_json(file_path or settings.json_file)

    # Support both a bare list [...] and a wrapped {"tickets": [...]}
    if isinstance(raw, dict):
        raw = raw.get("tickets", list(raw.values()))

    tickets, errors = [], 0
    for item in raw:
        try:
            tickets.append(JiraTicket.from_dict(item))
        except Exception as exc:
            print(f"Skipping {item.get('ticket_id', '?')}: {exc}")
            errors += 1

    print(f"Loaded {len(tickets)} tickets ({errors} skipped).")
    return tickets


# Guard: only run at script entry, not on import
if __name__ == "__main__":
    data = read_json(settings.json_file)