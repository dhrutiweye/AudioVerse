import uuid
from typing import Dict, Any, List
from .dto import RagConfig

# Core components
from .EmbedderModel import Embedder, build_sentence_units, parse_date_to_ts
from .Chunker import chunk_by_tokens
from vector_db import ensure_collection, upsert_points
from .helper import compute_quality_signals
from .dto import IndexRequest


# Import version config
# config = RagConfig(
#     EMBED_MODEL="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
#     CHUNK_SIZE=128,
#     CHUNK_OVERLAP=30,
#     COLLECTION_NAME="ds-test-recall-3010",
#     CONTENT_DENSITY=0.18,
#     CHAR_LENGTH=15
# )
#
#
# embedder = Embedder(config.EMBED_MODEL)
# ensure_collection(embedder.dim, config.COLLECTION_NAME)

def index_transcript(req: IndexRequest) -> Dict[str, Any]:
    # units = build_sentence_units(req.segments)
    # chunks = chunk_by_tokens(units,
    #                          count_tokens_fn=embedder.count_tokens,
    #                          target_tokens=config.CHUNK_SIZE,
    #                          overlap_tokens=config.CHUNK_OVERLAP)
    #
    # enriched = []
    # for c in chunks:
    #     sig = compute_quality_signals(c["text"], c.get("start_ms"))
    #     if sig["char_len"] > config.CHAR_LENGTH and sig["content_density"] > config.CONTENT_DENSITY:
    #         c["_signals"] = sig
    #         enriched.append(c)
    #
    # vecs = embedder.embed_texts([c["text"] for c in enriched])
    # point_ids = [f"{req.call_id}:{i}" for i in range(len(enriched))]
    # date_ts = parse_date_to_ts(req.date_time)
    #
    # points = []
    # for pid, v, c in zip(point_ids, vecs, enriched):
    #     payload = {
    #         "call_id": req.call_id,
    #         "text": c["text"],
    #         "agent_name": req.agent_name,
    #         "agent_code": req.agent_code,
    #         "operator_phone": req.operator_phone,
    #         "date_ts": date_ts,
    #         "lang": req.lang,
    #         "tokens": c["tokens"],
    #         **c["_signals"]
    #     }
    #     points.append({"id": str(uuid.uuid4()), "vector": v, "payload": payload})
    #
    # upsert_points(points, config.COLLECTION_NAME)
    # return {"call_id": req.call_id, "chunks_indexed": len(points), "collection": config.COLLECTION_NAME}
    pass