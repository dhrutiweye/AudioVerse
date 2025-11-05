import math
import numpy as np
def _minmax_normalize(vals: list[float]) -> list[float]:
    arr = np.asarray(vals, dtype=float)
    vmin, vmax = float(np.min(arr)), float(np.max(arr))
    if math.isclose(vmin, vmax, rel_tol=1e-9, abs_tol=1e-12):
        return [0.5] * len(arr)  # all equal → neutral midpoint
    return ((arr - vmin) / (vmax - vmin)).tolist()

def _sigmoid(x: np.ndarray) -> np.ndarray:
    # numerically stable sigmoid
    return 1.0 / (1.0 + np.exp(-x))

def _detect_and_normalize_rerank(scores: list[float]) -> tuple[list[float], str]:
    """
    Returns (probabilities_in_[0,1], scale_tag).
    - If scores already in [0,1] → treat as probabilities (no change).
    - Else → treat as logits and apply sigmoid to get probabilities.
    """
    arr = np.asarray(scores, dtype=float)
    if np.all(arr >= 0.0) and np.all(arr <= 1.0):
        return arr.tolist(), "prob"     # already calibrated probs
    # assume logits or uncalibrated → map to probs
    return _sigmoid(arr).tolist(), "logit"

def _lexical_overlap(query: str, text: str) -> float:
    # very cheap Jaccard overlap; replace with BM25 if you have it
    qt = set(re.findall(r"\w+", (query or "").lower()))
    tt = set(re.findall(r"\w+", (text or "").lower()))
    if not qt or not tt:
        return 0.0
    return len(qt & tt) / len(qt | tt)