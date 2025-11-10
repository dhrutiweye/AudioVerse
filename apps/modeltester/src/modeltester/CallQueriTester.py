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
model1='sentence-transformers/LaBSE'
model2='google/embeddinggemma-300m'
model0='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
embedder = Embedder(os.getenv("EMBED_MODEL", model2))
print(embedder.dim)
_rm = ReRanking(os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"))
_rm2 = ReRanking("jinaai/jina-reranker-v2-base-multilingual", 2)
CHUNK_SIZE = os.getenv("CHUNK_SIZE", 100)
CHUNK_OVERLAP = os.getenv("CHUNK_OVERLAP", 20)
CHAR_LENGTH = os.getenv("CHAR_LENGTH", 5)
CONTENT_DENSITY = os.getenv("CONTENT_DENSITY", 0.18)



def index_transcript(req: IndexRequest) -> Dict[str, Any]:
    units = build_sentence_units(req.segments)
    [a.update({"_signals": compute_quality_signals(a["text"], a.get("start_ms"))}) for a in units ]
    units = [a for a in units if a['_signals']['greeting_ratio'] < 0.60]
    units = [a for a in units if a['_signals']['repetition_ratio'] < 0.80]
    units = [a for a in units if a['_signals']['content_density'] > CONTENT_DENSITY]

    chunks_small = chunk_by_tokens(units,
                             count_tokens_fn=embedder.count_tokens,
                             target_tokens=90,
                             overlap_tokens=10)
    chunks_medium = chunk_by_tokens(units,
                                   count_tokens_fn=embedder.count_tokens,
                                   target_tokens=200,
                                   overlap_tokens=30)
    chunks_large = chunk_by_tokens(units,
                                   count_tokens_fn=embedder.count_tokens,
                                   target_tokens=400,
                                   overlap_tokens=50)

    enriched = []
    for c in chunks_medium:
        sig = compute_quality_signals(c["text"], c.get("start_ms"))
        if sig["char_len"] > CHAR_LENGTH and sig["content_density"] > CONTENT_DENSITY:
            c["_signals"] = sig
            c['chunk_type']='medium'
            enriched.append(c)
    for c in chunks_small:
        sig = compute_quality_signals(c["text"], c.get("start_ms"))
        if sig["char_len"] > CHAR_LENGTH and sig["content_density"] > CONTENT_DENSITY:
            c["_signals"] = sig
            c['chunk_type'] = 'small'
            enriched.append(c)

    for c in chunks_large:
        sig = compute_quality_signals(c["text"], c.get("start_ms"))
        if sig["char_len"] > CHAR_LENGTH and sig["content_density"] > CONTENT_DENSITY:
            c["_signals"] = sig
            c['chunk_type'] = 'large'
            enriched.append(c)

    print(f"chunks {chunks_small}\n{chunks_medium}\n{chunks_large}")
    print(f"enriched {enriched}")


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
    return points, enriched

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
    return index_transcript(ts), [x.text for x in segs]

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
    return (filtered_indices, filtered_scores)


samp= "काम नहीं कर रहा सब कुछ तो रुक जा रही है"

if __name__=="__main__":
    call_id =45371 # 35799 36102
    _q="Certain feature in product is not working properly"
    print(_get_default_mongo_url())
    doc = get_document_by_id(
        db_name="call_iq",
        collection_name="audios",
        doc_id=int(call_id)
    )
    print(doc)
    (points, enriched) , texts = index_from_mongo_record(doc)

    print(texts)
    print(enriched)

    p_vector = [i.get('vector') for i in points]
    q_vector = _embed_query_multi(_q, False)
    t_vector = _embed_query_multi(samp, False)
    _ex, _sx = get_data(p_vector, q_vector, 0.2)

    cand = [(points[i].get('payload',{}).get('text'), s) for i,s in zip(_ex, _sx)]
    [print(a) for a in cand]
    # _ca = [
    #     {'text':"Certain feature in product is not working properly"},
    #     {'text':'उत्पाद में कुछ विशेषताएँ ठीक से काम नहीं कर रही हैं'},
    #     {'text': 'Certain feature in Desial sensor is not working properly'},
    #     {'text': "Everything is not working, it's stopping"},
    #     {'text': "Hello, hello Manoj ji, Namaskar Sir, I told you on whom we have done this, this is number 70, it is not working for all, it is not working, everything is getting stuck. Sir, its payment etc. is pending, right? Yes, everything is not on time. I mean, did you make any payment? No, no, I made ₹200, I made ₹250. Yes, so let's see, since when it is not working, how many days have passed?"},
    #     {'text': "काम नहीं कर रहा सब कुछ तो रुक जा रही है"},
    #     {'text': "हेलो हेलो जी मनोज जी नमस्कार सर कहा ना हमने किस पे करा है ये 70 नंबर है ये सब का काम नहीं करता है काम नहीं कर रहा सब कुछ तो रुक जा रही है। ये सर इसका पेमेंट वगैरह तो कुछ पेंडिंग ही है ना? हां सब कुछ टाइम नहीं है। मतलब पेमेंट कर दिया था आपने कोई पेमेंट का? नहीं नहीं ₹200 किया था हमने ₹250 का। हां तो चलिए जब से काम नहीं कर रहे हैं कितने दिन में"},
    # ]
    # [print(a) for a in cand]
    d= _rm2.filter_relevant_chunks(_q, candidates=[{'text':a} for a,b in cand], rerank_gate_prob=0.000001)
    # d1 = _rm2.filter_relevant_chunks(_q, _ca, rerank_gate_prob=0.0)
    # print(d1)
    # d2 = _rm.filter_relevant_chunks(_q, _ca, rerank_gate_prob=0.0)
    print(d)
    # print(d1)
    # print(d2)






