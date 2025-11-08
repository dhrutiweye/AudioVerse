import csv
import os

from database import get_documents_by_filter, get_unique_values_by_filter
from dotenv import load_dotenv

from CallQueriTester import _embed_query_multi
from model import Embedder, ReRanking
from model.FlagModel import FlagModel
from vector_db import search_groups as q_search_groups

load_dotenv()
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "ds-test-recall-3010")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "call_test_data")
embedder = Embedder(os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"))
_rm = ReRanking(os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"))
_fm = FlagModel(os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"))
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
        limit=int(_size),
        score_threshold=scor,
        query_filter=None,
        collection=COLLECTION_NAME,
        with_payload=True,
        with_vectors=False,
        order_by=None
    )
    groups = groups.groups[0:0 + int(_size)] if groups else []
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
                                           top_k=_size, _search_v2=False)
    # bests = _fm.filter_relevant_chunks_compute_score(query=_q, candidates=cands, rerank_gate_prob=rerank_gate_prob)

    return (cands, bests)

def logQueriProference(_q, data = []):
    queri = _q
    t_data = data if len(data) > 0 else getCallByQueri(queri)
    c, b = getEmbarding(queri, 100, 0.2, 0.0000001, 5)
    # c, b = getEmbarding(queri, 100, 0.2, 0.0, 5)

    c_map = {
        i.get('payload', {}).get('call_id'): (i.get('score', 0), i.get('final_score', 0), i.get('text', ""))
        for i in c if i.get('payload', {}).get('call_id')  # ensures key exis
    }

    b_map = {
        i.get('payload', {}).get('call_id'): (i.get('score', 0), i.get('r_score', 0), i.get('text', ""))
        for i in b if i.get('payload', {}).get('call_id')  # ensures key exis
    }

    c_data = list(set([i.get('payload', {}).get('call_id') for i in c]))
    b_data = list(set([i.get('payload', {}).get('call_id') for i in b]))
    c_com = [x for x in c_data if int(x) in t_data]
    b_com = [x for x in b_data if int(x) in t_data]
    c_p = len(c_com) / len(c_data)
    b_p = len(b_com) / len(b_data)
    print(t_data)
    print(c_p, b_p)
    print(c_map)
    print(b_map)
    # embarding
    print(c_com)
    print(c_data)
    TP = len(c_com)
    FP = len(c_data) - len(c_com)
    FN = len(t_data) - len(c_com)
    _p = TP / (TP + FP) if (TP + FP) > 0 else 0
    _r = TP / (TP + FN) if (TP + FN) > 0 else 0
    _f1s = 2 * (_p*_r/(_p+_r)) if (_p + _r) > 0 else 0
    print(f"c presition {_p}")
    print(f"c recall {_r}")
    print(f" cF1 score {_f1s}")

    print(b_com)
    print(b_data)
    TP = len(b_com)
    FP = len(b_data) - len(b_com)
    FN = len(t_data) - len(b_com)
    _pr = TP / (TP + FP) if (TP + FP) > 0 else 0
    _rr = TP / (TP + FN) if (TP + FN) > 0 else 0
    _f1sr = 2 * (_pr * _rr / (_pr + _rr)) if (_pr + _rr) > 0 else 0
    print(f"r presition {_pr}")
    print(f"r recall {_rr}")
    print(f"r F1 score {_f1sr}")

    print([a for a in c_data if a not in b_data])
    return [_q, _p, _r, _f1s, _pr, _rr, _f1sr]


if __name__ == "__main__":
    queries = all_id = get_unique_values_by_filter(
        db_name='call_iq',
        collection_name='call_test_data',
        column_name='query',
        filter_data=None,
        offset=0,
        limit=1000
    )
    data=[]
    for i in queries:
        print(f"data for {i}")
        data.append(logQueriProference(i))

    with open("call_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query", "m_p", "m_r", "m_f1", "r_p", "r_r", "r_f1"])  # header
        writer.writerows(data)

