import torch
from sentence_transformers import SentenceTransformer
from datetime import datetime
import re
from typing import List, Dict, Any, Optional
from util import get_device

from .dto import Segment

_SENT_SPLIT_RE = re.compile(r"(?<=[\.!\?।！？])\s+")


def parse_date_to_ts(date_str: Optional[str]) -> Optional[int]:
    """
    Convert your metadata['date_time'] to a numeric epoch seconds for range filters.
    Tries a few common formats; returns None if parsing fails.
    """
    if not date_str:
        return None
    trials = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
    ]
    for fmt in trials:
        try:
            return int(datetime.strptime(date_str[:len(fmt)], fmt).timestamp())
        except Exception:
            continue
    # last resort: try fromisoformat
    try:
        return int(datetime.fromisoformat(date_str.replace("Z", "+00:00")).timestamp())
    except Exception:
        return None


def split_to_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = _SENT_SPLIT_RE.split(text)
    if len(parts) == 1 and len(parts[0].split()) > 120:
        toks = parts[0].split()
        parts = [" ".join(toks[i:i + 25]) for i in range(0, len(toks), 25)]
    return parts


def build_sentence_units(segments: List[Segment]) -> List[Dict[str, Any]]:
    units: List[Dict[str, Any]] = []
    for seg in segments:
        sents = split_to_sentences(seg.text)
        if not sents:
            continue
        s_ms, e_ms, spk = seg.start_ms, seg.end_ms, seg.speaker
        if s_ms is not None and e_ms is not None and len(sents) > 1:
            dur = max(1, e_ms - s_ms)
            tot = sum(max(1, len(s)) for s in sents)
            t = s_ms
            for i, s in enumerate(sents):
                frac = max(1, len(s)) / tot
                sdur = int(round(dur * frac))
                start_i, end_i = t, t + sdur
                t = end_i
                if i == len(sents) - 1:
                    end_i = e_ms
                units.append({"text": s, "start_ms": start_i, "end_ms": end_i, "speaker": spk})
        else:
            for s in sents:
                units.append({"text": s, "start_ms": s_ms, "end_ms": e_ms, "speaker": spk})
    return units




class Embedder:
    def __init__(self, model_name: str):
        self.device = get_device()
        self.model = SentenceTransformer(model_name).to(self.device)
        print(f"Loaded embedding model: {model_name} on {self.device}")

    @property
    def dim(self):
        return self.model.get_sentence_embedding_dimension()

    def count_tokens(self, text: str) -> int:
        return len(self.model.tokenizer.encode(text, add_special_tokens=True))

    def embed_texts(self, texts, batch_size: int = 64):
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
            device=self.device
        ).tolist()

    def embed_text_encode(self, texts):
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
            device=self.device
        ).tolist()

    def _embed_query_multi(self, text: str, enhance_short: bool = True) -> List[float]:
        text = (text or "").strip()
        variants = [text]
        if enhance_short and len(text.split()) <= 2:
            v = text.lower()
            variants.extend({v, v.rstrip("s"), v + " details", "information about " + v})
            variants = list({x for x in variants if x})
        vecs = self.model(variants, convert_to_numpy=True,
                          normalize_embeddings=True,
                          show_progress_bar=False,
                          device=self.device())
        return vecs.mean(axis=0).tolist()

