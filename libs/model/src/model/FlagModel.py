from FlagEmbedding import FlagReranker
from .helper import get_device

class FlagModel:
    def __init__(self, model_name: str | None):
        self.device = get_device()
        model_name = model_name or "BAAI/bge-reranker-v2-m3"
        print(f"use FlagReranker {model_name} ")
        self.model = FlagReranker(model_name,
                use_fp16=(self.device == 'cuda'),
                device=self.device
            )
        print(f"Loaded embedding model: {model_name} on {self.device}")
        # self.model = CrossEncoder(
        #     "jinaai/jina-reranker-v2-base-multilingual",
        #     automodel_args={"dtype": "auto"},
        #     trust_remote_code=True,
        # )

    def filter_relevant_chunks_compute_score(
            self,
            query: str,
            candidates: list[dict],
            rerank_gate_prob: float = 0.01,
    ):
        """
        Filters candidate chunks by query relevance using a cross-encoder.

        Args:
            model: Loaded CrossEncoder model.
            query (str): The user query.
            candidates (list[dict]): List of dicts, each with at least {'text': str}.
            rerank_gate_prob (float): Threshold probability for filtering.

        Returns:
            list[dict]: List of relevant candidates with 'score' added.
        """
        if not candidates:
            return []

        # Create pairs (query, text)
        pairs = [(query, c.get("text", "")) for c in candidates]

        # Get relevance probabilities
        scores = self.model.compute_score(pairs)

        # Add scores back into candidate dicts
        for cand, score in zip(candidates, scores):
            cand["r_score"] = float(score)

        # Filter by threshold
        relevant = [c for c in candidates if c["r_score"] >= rerank_gate_prob]
        return relevant
