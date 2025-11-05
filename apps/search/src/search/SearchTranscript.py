import asyncio
import os
import time
from typing import Any, Dict, List, Optional

import torch
from SearchRequest import SearchRequest
from model import Embedder, ReRanking
from vector_db import ensure_collection, build_filter, search_groups as q_search_groups

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "ds-test-recall-0511")
_bi = Embedder(os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"))
_rm = ReRanking(os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"))
EMBED_DIM = _bi.dim
ensure_collection(EMBED_DIM, COLLECTION_NAME)

# ---- Query helpers ----
def _embed_query_multi(text: str, enhance_short: bool = True) -> List[float]:
    text = (text or "").strip()
    variants = [text]
    if enhance_short and len(text.split()) <= 2:
        v = text.lower()
        variants.extend({v, v.rstrip("s"), v + " details", "information about " + v})
        variants = list({x for x in variants if x})
    vecs = _bi.embed_text_encode(variants)
    return vecs.mean(axis=0).tolist()

def minmax_normalize(values):
    vmin, vmax = min(values), max(values)
    return [(v - vmin) / (vmax - vmin + 1e-8) for v in values]


def softmax_normalize(values):
    return torch.softmax(torch.tensor(values), dim=0).tolist()


# ---- PUBLIC: async search API (one result per call_id) ----
async def search_transcripts(
        query: str,
        page: int = 0,
        size: int = 10,
        start_date: Optional[str] = None,  # "YYYY-MM-DD" or ISO datetime
        end_date: Optional[str] = None,
        agent_name: Optional[str] = None,
        operator_phone: Optional[str] = None,
        min_score: float = 0.7,
        rerank_gate_prob: float = 0.01,
        group_hits: int = 1,  # top chunks per audio to consider
        oversample_factor: int = 8,  # when falling back to client-side grouping
        enhance_short_query: bool = True,
) -> Dict[str, Any]:
    """
    Returns one row per audio (call_id).
    """

    # ---- parse date strings to epoch seconds for numeric range filter ----
    def _to_ts(s: Optional[str]) -> Optional[int]:
        from datetime import datetime
        if not s: return None
        for fmt in (
        "%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
            try:
                return int(datetime.strptime(s[:len(fmt)], fmt).timestamp())
            except Exception:
                continue
        try:
            return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())
        except Exception:
            return None

    start_ts, end_ts = _to_ts(start_date), _to_ts(end_date)
    flt, order_by = build_filter(start_ts=start_ts, end_ts=end_ts, agent_name=agent_name, operator_phone=operator_phone,
                       lang=None, sort_field="date_ts", sort_order="desc")

    # ---- embed query (optionally enhanced for very short) ----
    qvec = _embed_query_multi(query, enhance_short=enhance_short_query)

    # ---- try server-side grouping first ----
    results: List[Dict[str, Any]] = []
    try:
        _size = int(size * 2.5)
        _t1=time.time_ns()
        groups = q_search_groups(
            query_vector=qvec,
            group_by="call_id",
            group_size=max(1, group_hits),
            limit=_size + page * _size,  # fetch enough then slice by page
            score_threshold=min_score,
            query_filter=flt,
            collection=COLLECTION_NAME,
            with_payload=True,
            with_vectors=False,
            order_by=order_by
        )
        _t2=time.time_ns()
        # paginate groups
        start_idx = page * _size
        groups = groups.groups[start_idx:start_idx + _size] if groups else []
        cands = []
        for g in groups or []:
            hits = getattr(g, "hits", None) or getattr(g, "scored_points", None) or []
            if not hits:
                continue

            for h in hits:
                pl = h.payload or {}
                cands.append({
                    "text": pl.get("text", ""),
                    "score": float(h.score),
                    "start_ms": pl.get("start_ms"),
                    "end_ms": pl.get("end_ms"),
                    "payload": pl,
                })
        _t3=time.time_ns()
        bests, _s = _rm._rerank_and_choose_top(query, cands,
                         rerank_gate_prob=rerank_gate_prob,
                         dense_gate=min_score,
                         top_k=size, _search_v2=True)
        _t4 = time.time_ns()
        # pl0 = hits[0].payload or {}
        # call_id = pl0.get("call_id")
        results = [{
            "call_id": best['payload']['call_id'],
            "best_text": best["text"],
            "diarized_transcript_str": best.get("diarized_transcript_str", ""),
            "raw_transcript_str": best.get('raw_transcript_str', ""),
            "start_ms": best["start_ms"],
            "end_ms": best["end_ms"],
            "dense_score": best.get("score", 0),
            "final_score": best.get("final_score", 0),
            "rerank_score": best.get("rerank_prob", 0.0),
            "time_m": str(f'em_take {_t2-_t1} gader {_t3-_t2} reranker {_t4-_t3}'),
            "reranking_m": str(_s),
            "meta": {
                "speaker":  best['payload'].get("speaker", ""),
                "prev_id": best['payload'].get("prev_id", None),
                "next_id": best['payload'].get("next_id", None),
                "agent_name": best['payload'].get("agent_name", None),
                "agent_code": best['payload'].get("agent_code", None),
                "operator_phone": best['payload'].get("operator_phone", None),
                "lang": best['payload'].get("lang", None),
                "date": best['payload'].get("date", None),
            }
        } for best in bests]
    except Exception as e:
        print(e)
        print(f"not get group data for queri {query}")

    return {"query": query, "page": page, "size": size, "results": results}


def search(req: SearchRequest):
    return asyncio.run(search_transcripts(
        query=req.query,
        page=req.page,
        size=req.size,
        start_date=req.start_date,  # "YYYY-MM-DD" or ISO datetime
        end_date=req.end_date,
        agent_name=req.agent_name,
        operator_phone=req.operator_phone,
        min_score=req.min_score,
        rerank_gate_prob=req.rerank_gate_prob,
        group_hits=req.group_hits))
