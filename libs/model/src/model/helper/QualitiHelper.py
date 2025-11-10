from __future__ import annotations
import regex as re
from typing import Dict, Any, List, Optional

# keep small, high-signal lists; extend over time
GREETINGS = {
    "hi", "hello", "hey", "good", "morning", "afternoon", "evening",
    "नमस्ते", "नमस्कार", "हेलो", "हैलो", "जी", "बोलिए", "सुनिए", "हाँ", "हां", "हांजी", "जी सर", "सर"
}
STOPWORDS = GREETINGS | {
    "the", "a", "an", "to", "is", "are", "am", "ok", "okay", "haan", "haanji", "acha", "achha", "hmm", "hmmm"
}

KEY_BOOSTS = {
    "gps": 0.95, "जीपीएस": 0.90,
    "dcm": 0.95, "डीसीएम": 0.95,
}

_tok = re.compile(r"\p{L}+", re.UNICODE)


def tokens(text: str) -> List[str]:
    # return _tok.findall((text or "").lower())
    return [a.lower() for a in text.split(" ")]

def keyword_hits(text: str):
    toks = [t.lower() for t in _tok.findall(text or "")]
    present = {k: (k.lower() in toks) for k in KEY_BOOSTS}
    score = sum(KEY_BOOSTS[k] for k, ok in present.items() if ok)
    return {
        "kw_present": present,             # {"gps": True, "dms": False, ...}
        "kw_score_raw": float(score),      # e.g., 1.0 if "gps" present
        "has_kw": any(present.values())
    }


def content_density(text: str) -> float:
    toks = tokens(text)
    if not toks: return 0.0
    non_stop = [t for t in toks if t not in STOPWORDS]
    return len(non_stop) / len(toks)


def greeting_ratio(text: str) -> float:
    toks = tokens(text)
    if not toks: return 0.0
    g = sum(1 for t in toks if t in GREETINGS)
    return g / len(toks)


def repetition_ratio(text: str) -> float:
    toks = tokens(text)
    if not toks: return 0.0
    from collections import Counter
    cnt = Counter(toks)
    most = cnt.most_common(1)[0][1]
    return most / max(1, len(toks))


def tag_section(start_ms: Optional[int] , speech_ms: Optional[int] = None ) -> str:
    """
    Heuristic: first 20s as 'opening' (greetings/introductions), else 'body'.
    If you store turn types later, replace with diarization-aware tagging.
    """
    s = start_ms or 0
    return "opening" if s < 20_000 else "body"


def compute_quality_signals(text: str, start_ms: Optional[int], key_booster: bool = False) -> Dict[str, Any]:
    cd = content_density(text)
    gr = greeting_ratio(text)
    rr = repetition_ratio(text)
    sec = tag_section(start_ms)
    kb = keyword_hits(text) if key_booster else {}
    return {
        "content_density": cd,
        "greeting_ratio": gr,
        "repetition_ratio": rr,
        "section": sec,
        "char_len": len(text or ""),
        "kw_score_raw": kb.get("kw_score_raw", 0),
    }
