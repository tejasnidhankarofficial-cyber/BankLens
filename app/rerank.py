"""Cross-encoder reranker (optional dependency: sentence-transformers)."""

from __future__ import annotations

from app.retrieve import Hit


class Reranker:
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "Reranking needs sentence-transformers: pip install -r requirements-local.txt"
            ) from e
        self.model = CrossEncoder(model_name, max_length=512)

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        if not hits:
            return []
        scores = self.model.predict([(query, h.chunk.embed_text) for h in hits])
        order = sorted(range(len(hits)), key=lambda i: -scores[i])[:top_k]
        return [Hit(hits[i].chunk, float(scores[i]), r) for r, i in enumerate(order, start=1)]
