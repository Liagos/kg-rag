from dataclasses import dataclass
from typing import Any


@dataclass
class RetrievedDocument:
    id: str
    content: str
    score: float
    metadata: dict[str, Any]
