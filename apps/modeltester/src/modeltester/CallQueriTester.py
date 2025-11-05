import os
import uuid
from typing import Dict, Any, List
import numpy as np
from database.mongo_client import _get_default_mongo_url
from dotenv import load_dotenv

from model import Embedder, ReRanking, compute_quality_signals
from model.EmbedderModel import build_sentence_units, parse_date_to_ts
from model.Chunker import chunk_by_tokens
from model.dto import IndexRequest, Segment
from database import get_document_by_id

load_dotenv()
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "ds-test-recall-3010")
embedder = Embedder(os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"))
_rm = ReRanking(os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"))
CHUNK_SIZE = os.getenv("CHUNK_SIZE", 128)
CHUNK_OVERLAP = os.getenv("CHUNK_OVERLAP", 30)
CHAR_LENGTH = os.getenv("CHAR_LENGTH", 15)
CONTENT_DENSITY = os.getenv("CONTENT_DENSITY", 0.18)
EMBED_DIM = embedder.dim



def index_transcript(req: IndexRequest) -> Dict[str, Any]:
    units = build_sentence_units(req.segments)
    chunks = chunk_by_tokens(units,
                             count_tokens_fn=embedder.count_tokens,
                             target_tokens=CHUNK_SIZE,
                             overlap_tokens=CHUNK_OVERLAP)

    enriched = []
    for c in chunks:
        sig = compute_quality_signals(c["text"], c.get("start_ms"))
        if sig["char_len"] > CHAR_LENGTH and sig["content_density"] > CONTENT_DENSITY:
            c["_signals"] = sig
            enriched.append(c)

    vecs = embedder.embed_texts([c["text"] for c in enriched])
    point_ids = [f"{req.call_id}:{i}" for i in range(len(enriched))]
    date_ts = parse_date_to_ts(req.date_time)

    points = []
    for pid, v, c in zip(point_ids, vecs, enriched):
        payload = {
            "call_id": req.call_id,
            "text": c["text"],
            "agent_name": req.agent_name,
            "agent_code": req.agent_code,
            "operator_phone": req.operator_phone,
            "date_ts": date_ts,
            "lang": req.lang,
            "tokens": c["tokens"],
            **c["_signals"]
        }
        points.append({"id": str(uuid.uuid4()), "vector": v, "payload": payload})
    return points

def index_from_mongo_record(data: Dict[str, Any], default_lang: str = "hi") -> Dict[str, Any]:
    """
    Expects your 'data' dict from Mongo like you described.
    """
    x = data.get("metadata", {}).get("diarized_transcript", {}).get("entries", []) or []
    segs: List[Segment] = []

    for i in x:
        # prefer end_time_seconds if present; else approximate 2s window
        start_s = float(i.get("start_time_seconds", 0) or 0.0)
        end_s = float(i.get("end_time_seconds", start_s) or start_s)
        if end_s <= start_s: end_s = start_s + 2.0
        segs.append(Segment(
            text=i.get("transcript", "") or "",
            start_ms=int(start_s * 1000),
            end_ms=int(end_s * 1000),
            speaker=str(i.get("speaker_id")) if i.get("speaker_id") is not None else None,
            lang=default_lang
        ))

    ts = IndexRequest(
        call_id=str(data.get("_id")),
        date_time=str(data.get("metadata", {}).get("date_time", "")) if data.get("metadata") else None,
        agent_name=(data.get("metadata") or {}).get("agent_name"),
        agent_code=(data.get("metadata") or {}).get("agent_code"),
        lang=default_lang,
        segments=segs,
        operator_phone=(data.get("metadata") or {}).get("operator_phone"),
        call_duration=(data.get('duration', 0))
    )
    return index_transcript(ts)

def _embed_query_multi(text: str, enhance_short: bool = True) -> List[float]:
    text = (text or "").strip()
    variants = [text]
    if enhance_short and len(text.split()) <= 2:
        v = text.lower()
        variants.extend({v, v.rstrip("s"), v + " details", "information about " + v})
        variants = list({x for x in variants if x})
    vecs = embedder.embed_text_encode(variants)
    return vecs.mean(axis=0).tolist()

def get_data(p_vector, q_vector, threshold = 0.5):

    # Assuming points_vector is a list of vectors (each vector is a list or numpy array)
    # and q_vector is a single vector
    points_vector = np.array(p_vector)
    q_vector = np.array(q_vector)

    # Normalize the vectors to unit length
    points_norm = points_vector / np.linalg.norm(points_vector, axis=1, keepdims=True)
    q_norm = q_vector / np.linalg.norm(q_vector)

    # Compute cosine similarities (dot product of normalized vectors)
    cosine_similarities = np.dot(points_norm, q_norm)

    # Apply score threshold
    score_threshold = threshold  # for example
    filtered_indices = np.where(cosine_similarities >= score_threshold)[0]

    # Get scores and vectors above threshold
    filtered_scores = cosine_similarities[filtered_indices]
    filtered_vectors = points_vector[filtered_indices]

    # Print or use as needed
    for idx, score in zip(filtered_indices, filtered_scores):
        print(f"Index: {idx}, Cosine Similarity: {score:.4f}")
    return filtered_indices


if __name__=="__main__":
    call_id =45591
    _q="Conversation reached till price stage in sales conversation for diesel sensor"
    print(_get_default_mongo_url())
    doc = get_document_by_id(
        db_name="call_iq",
        collection_name="audios",
        doc_id=int(call_id)
    )
    print(doc)
    points = index_from_mongo_record(doc)
    # print(points)

    p_vector = [i.get('vector') for i in points]
    q_vector = _embed_query_multi(_q)
    _ex = get_data(p_vector, q_vector, 0.1)

    [print(points[i].get('payload',{}).get('text')) for i in _ex]






