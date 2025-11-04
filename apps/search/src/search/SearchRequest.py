from dataclasses import dataclass
from typing import Optional


@dataclass()
class SearchRequest:
    query: str
    page: int = 0
    size: int = 10
    start_date: Optional[str] = None  # YYYY-MM-DD
    end_date: Optional[str] = None
    agent_name: Optional[str] = None
    operator_phone: Optional[str] = None
    lang: Optional[str] = None
    min_score: float = 0.2
    rerank_gate_prob: float = 0.001
    group_hits: int = 3
    with_context_neighbors: int = 1  # 0,1,2...
