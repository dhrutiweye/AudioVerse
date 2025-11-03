import os
from typing import Any, Dict, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Filter, FieldCondition,
    MatchValue, Range, VectorParams,
    OrderBy
)

# ---- Qdrant connection & defaults ----
QDRANT_URL = f"http://{os.getenv('QDRANT_HOST', '10.20.4.235')}:{os.getenv('QDRANT_PORT', '6333')}"
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
QDRANT_TIMEOUT = int(os.getenv("QDRANT_TIMEOUT", "30"))
DEFAULT_VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "384"))  # fallback

def get_client() -> QdrantClient:
    """Initialize a Qdrant client using env vars."""
    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=QDRANT_TIMEOUT,
        prefer_grpc=False
    )


# ---- Collection management ----
def ensure_collection(embed_dim: Optional[int] = None,
                      collection: Optional[str] = None,
                      distance: str = "Cosine") -> None:
    """
    Ensure that the Qdrant collection exists with given embedding dimension.
    If not found, create it.
    """
    if not collection:
        raise ValueError("Collection name must be provided.")
    dim = embed_dim or DEFAULT_VECTOR_SIZE

    client = get_client()
    try:
        client.get_collection(collection)
        print(f"✅ Collection '{collection}' already exists.")
    except Exception:
        print(f"⚙️  Creating new collection '{collection}' (dim={dim}, distance={distance}) ...")
        client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=distance)
        )
        print(f"✅ Collection '{collection}' created.")


# ---- Data operations ----
def upsert_points(points: List[Dict[str, Any]], collection: str) -> None:
    """Insert or update points in the given collection."""
    if not points:
        print("⚠️ No points to upsert.")
        return
    client = get_client()
    client.upsert(collection_name=collection, points=points)
    print(f"✅ Upserted {len(points)} points into collection '{collection}'.")


def update_payload(point_ids: List[str], payload: Dict[str, Any], collection: str) -> None:
    """Update metadata for existing points."""
    client = get_client()
    client.set_payload(collection_name=collection, payload=payload, points=point_ids)
    print(f"✅ Updated payload for {len(point_ids)} points in '{collection}'.")


## read queri ##
def get_point(point_id: str, collection: str) -> Optional[Dict[str, Any]]:
    """Retrieve a single point by ID."""
    client = get_client()
    res = client.retrieve(collection_name=collection,
                          ids=[point_id],
                          with_payload=True,
                          with_vectors=False)
    return res[0].dict() if res else None

# ---- Search helpers ----
def build_filter(
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        agent_name: Optional[str] = None,
        operator_phone: Optional[str] = None,
        lang: Optional[str] = None,
        sort_field: Optional[str] = None,
        sort_order: str = "desc",   # "asc" or "desc"
) -> Optional[Filter]:
    must = []
    if agent_name:
        must.append(FieldCondition(key="agent_name", match=MatchValue(value=agent_name)))
    if operator_phone:
        must.append(FieldCondition(key="agent_phone", match=MatchValue(value=operator_phone)))
    if lang:
        must.append(FieldCondition(key="lang", match=MatchValue(value=lang)))
    if start_ts is not None or end_ts is not None:
        rng = {}
        if start_ts is not None:
            rng["gte"] = start_ts
        if end_ts is not None:
            rng["lte"] = end_ts
        must.append(FieldCondition(key="date_ts", range=Range(**rng)))  # numeric timestamp (sec)
    qdrant_filter = Filter(must=must) if must else None

    order_by = None
    if sort_field:
        order_by = OrderBy(
            key=sort_field,
            direction="desc" if sort_order.lower() == "desc" else "asc"
        )

    return qdrant_filter, order_by


def search(
        query_vector: List[float],
        collection: str,
        limit: int = 10,
        score_threshold: Optional[float] = None,
        query_filter: Optional[Filter] = None,
        offset: int = 0,
        with_payload: bool = True,
        with_vectors: bool = False,
):
    client = get_client()
    return client.query_points(
        collection_name=collection,
        query=query_vector,
        with_payload=with_payload,
        with_vectors=with_vectors,
        score_threshold=score_threshold,
        query_filter=query_filter,
        limit=limit,
        offset=offset,
    )


def search_groups(
        query_vector: List[float],
        group_by: str,
        group_size: int,
        limit: int,
        collection: str,
        score_threshold: Optional[float] = None,
        query_filter: Optional[Filter] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
        order_by: Optional[OrderBy] = None
):
    client = get_client()
    # available on Qdrant >= 1.7
    return client.query_points_groups(
        collection_name=collection,
        query=query_vector,
        group_by=group_by,
        group_size=group_size,
        limit=limit,
        with_payload=with_payload,
        with_vectors=with_vectors,
        score_threshold=score_threshold,
        query_filter=query_filter,
    )

