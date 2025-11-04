import time
from typing import Tuple

from util import get_device
from sentence_transformers import CrossEncoder
from util.ReRankerHelper import _minmax_normalize, _detect_and_normalize_rerank



class ReRanking:
    def __init__(self, model_name: str | None):
        self.device = get_device()
        model_name = model_name or "BAAI/bge-reranker-v2-m3"
        self.model = CrossEncoder(model_name).to(get_device())
        print(f"Loaded embedding model: {model_name} on {self.device}")

    def predict_list(self, pairs: list[Tuple[str, str]]):
        model = self.model
        return model.predict(pairs)

    def predict(self, text: str, queri: str):
        return self.predict_list([(text, queri)])
    def _rerank_and_choose_top(self,
    query: str,
    candidates: list[dict],
    top_k: int = 10,
    alpha: float = 0.65,  # 70% rerank + 30% dense
    beta: float = 0.20,
    gamma: float = 0.25,
    dense_gate: float = 0.38,   # cosine similarity (multilingual)
    rerank_gate_prob: float = 0.01,  # gate on rerank *probability* (after autodetect)
    return_when_empty: bool = False,  # if False → return []; if True → return best dense as fallback
    _search_v2: bool = False) -> (list[dict], str):
        """
        Expects each candidate like:
          {"text": str, "score": dense_cosine_float, ...}
        Adds:
          c["rerank_score_raw"], c["rerank_prob"], c["final_score"]
        Applies gates: dense, rerank_prob, lexical.
        """
        if not candidates:
            return [], ""

        _ce = self.model

        # If no reranker wired or only one item, fallback to dense-only
        if _ce is None or len(candidates) == 1:
            ranked = sorted(candidates, key=lambda c: c.get("score", 0.0), reverse=True)
            return ranked[:top_k], str(f'')

        # 1) Get raw rerank scores
        pairs = [(query, c.get("text", "")) for c in candidates]
        try:
            _t1 = time.time_ns()
            ce_scores = _ce.predict(pairs, batch_size=top_k)  # list/np.ndarray of floats (logits or probs)
            _t2 = time.time_ns()
        except Exception:
            ranked = sorted(candidates, key=lambda c: c.get("score", 0.0), reverse=True)
            return ranked[:top_k], str(f're_scor {_t2-_t1}')

        # 2) Attach raw + prob (auto-detected)
        ce_scores = list(map(float, ce_scores))
        rerank_probs, scale_tag = _detect_and_normalize_rerank(ce_scores)
        for c, raw, pr in zip(candidates, ce_scores, rerank_probs):
            c["rerank_score_raw"] = raw
            c["rerank_prob"] = pr

        _t3 = time.time_ns()
        # 3) Normalize within-batch for fusion (to [0,1])
        #    - rerank: already a probability in [0,1]
        #    - dense: min-max to [0,1] (cosine across this candidate set)
        dense_vals = [c.get("score", 0.0) for c in candidates]
        dense_norm = _minmax_normalize(dense_vals)
        rerank_norm = rerank_probs  # already [0,1]

        _t4 = time.time_ns()
        # 4) Weighted fusion
        for c, r, d in zip(candidates, rerank_norm, dense_norm):
            _score = alpha * r + (1.0 - alpha) * d
            if not _search_v2:
                _score += gamma * c.get("kw_score_raw", 0.0)
                _score -= beta * c.get("greeting_ratio", 0.0)
            c["final_score"] = _score

        _t5 = time.time_ns()
        # 5) Relevance gates
        filtered: list[dict] = []
        for c in candidates:
            if c.get("score", 0.0) < dense_gate:
                c["reject_reason"] = f"dense_gate-{c.get('scor', 0.0)}"
                continue
            if c.get("rerank_prob", 0.0) < rerank_gate_prob:
                c["reject_reason"] = f"rerank_gate-{c.get('rerank_prob', 0.0)}"
                continue
            filtered.append(c)

        if not filtered:
            # nothing passed strict gates → safest is to return empty (caller should refuse)
            if return_when_empty:
                # optional: return best dense (still risky)
                fallback = sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)[:min(top_k, len(candidates))]
                for f in fallback: f.setdefault("note", "fallback_dense_no_relevant_hit")
                return fallback
            return [], str(f're_scor {_t2-_t1} auto-detected {_t3-_t2} '
                                   f'Weighted fusion {_t4- _t3} Relevance gates {_t5 - _t4} final {time.time_ns() - _t5}')

        # 6) Rank by fused score and cap
        ranked = sorted(filtered, key=lambda c: c["final_score"], reverse=True)
        return ranked[:top_k], str(f're_scor {_t2-_t1} auto-detected {_t3-_t2} '
                                   f'Weighted fusion {_t4- _t3} Relevance gates {_t5 - _t4} final {time.time_ns() - _t5}')

