from typing import List, Dict, Any, Optional, Tuple

def chunk_by_tokens(units: List[Dict[str, Any]],
                    count_tokens_fn,
                    target_tokens: int,
                    overlap_tokens: int,
                    hard_max: Optional[int] = 256) -> List[Dict[str, Any]]:
    budget = min(target_tokens, max(32, hard_max - 16))
    overlap = min(overlap_tokens, max(0, budget // 3))

    chunks, window, win_tok = [], [], 0

    def bounds(ws):
        starts = [w.get("start_ms") for w in ws if w.get("start_ms")]
        ends = [w.get("end_ms") for w in ws if w.get("end_ms")]
        return (min(starts) if starts else None, max(ends) if ends else None)

    def flush(keep_overlap: bool):
        nonlocal window, win_tok
        if not window: return
        txt = " ".join([w["text"] for w in window]).strip()
        s_ms, e_ms = bounds(window)
        spks = {w.get("speaker") for w in window}
        speaker = window[0].get("speaker") if len(spks) == 1 else None
        chunks.append({"text": txt, "start_ms": s_ms, "end_ms": e_ms, "speaker": speaker, "tokens": win_tok})
        if keep_overlap:
            tail, tok_c = [], 0
            for w in reversed(window):
                if tok_c >= overlap: break
                tail.insert(0, w)
                tok_c += count_tokens_fn(w["text"])
            window, win_tok = tail, sum(count_tokens_fn(w["text"]) for w in window)
        else:
            window, win_tok = [], 0

    for u in units:
        t = count_tokens_fn(u["text"])
        if window and win_tok + t > budget:
            flush(keep_overlap=True)
        window.append(u)
        win_tok += t
    flush(keep_overlap=False)
    return [c for c in chunks if c["text"]]
