import os

from database import get_documents_by_filter
from dotenv import load_dotenv

from CallQueriTester import _embed_query_multi
from model import Embedder, ReRanking
from vector_db import search_groups as q_search_groups

load_dotenv()
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "ds-test-recall-3010")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "call_test_data")
embedder = Embedder(os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"))
_rm = ReRanking(os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"))
CHUNK_SIZE = os.getenv("CHUNK_SIZE", 128)
CHUNK_OVERLAP = os.getenv("CHUNK_OVERLAP", 30)
CHAR_LENGTH = os.getenv("CHAR_LENGTH", 15)
CONTENT_DENSITY = os.getenv("CONTENT_DENSITY", 0.18)
EMBED_DIM = embedder.dim



def getCallByQueri(_q) -> list[int]:
    docs = get_documents_by_filter(
        db_name='call_iq',
        collection_name=MONGO_COLLECTION,
        data={
            'query': _q,
            'llm_is_related': True
        }
    )
    return [i.get('call_id') for i in docs]

def getEmbarding(_q, _size=10, scor=0.1, rerank_gate_prob=0.001, group_size=1):
    groups = q_search_groups(
        query_vector=_embed_query_multi(_q),
        group_by="call_id",
        group_size=group_size,
        limit=int(_size*1.5),
        score_threshold=scor,
        query_filter=None,
        collection=COLLECTION_NAME,
        with_payload=True,
        with_vectors=False,
        order_by=None
    )
    groups = groups.groups[0:0 + int(_size*1.5)] if groups else []
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

    bests, _s = _rm._rerank_and_choose_top(_q, cands,
                                           rerank_gate_prob=rerank_gate_prob,
                                           dense_gate=scor,
                                           top_k=_size, _search_v2=True)

    return (cands, bests)


if __name__ == "__main__":
    queri = "Certain feature in product is not working properly"
    t_data = getCallByQueri(queri)
    c, b = getEmbarding(queri, 100, 0.2, 0.0004, 5)
    c_map = {
        i.get('payload', {}).get('call_id'): i.get('score', 0)
        for i in c if i.get('payload', {}).get('call_id')  # ensures key exis
    }
    c_data = list(set([i.get('payload', {}).get('call_id') for i in c]))
    b_data = list(set([i.get('payload', {}).get('call_id') for i in b]))
    c_com = [x for x in c_data if int(x) in t_data]
    b_com = [x for x in b_data if int(x) in t_data]
    c_p = len(c_com)/len(c_data)
    b_p = len(b_com)/len(b_data)
    print(c_p, b_p)

    print(t_data)
    print(f"c presition {c_p}")
    print(f"c recall {len(c_com)/ len(t_data)}")
    print(c_data)

    print(f"b presition {b_p}")
    print(f"b recall {len(b_com)/ len(t_data)}")
    print(set(b_data))




