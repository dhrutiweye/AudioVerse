from dataclasses import dataclass
from typing import Optional, List

from .Segment import Segment


@dataclass
class IndexRequest:
    call_id: str
    date_time: Optional[str]  # raw string as provided
    agent_name: Optional[str]
    agent_code: Optional[str]
    call_duration: Optional[int]
    operator_phone: Optional[str]
    lang: Optional[str]
    segments: List[Segment]