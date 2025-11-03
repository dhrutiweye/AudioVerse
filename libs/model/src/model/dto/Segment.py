from dataclasses import dataclass
from typing import Optional


@dataclass
class Segment:
    text: str
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    speaker: Optional[str] = None
    lang: Optional[str] = None
