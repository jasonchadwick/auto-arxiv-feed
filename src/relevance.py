"""Cosine-similarity-based relevance filtering."""

import logging
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Return the cosine similarity between vectors *a* and *b* (range −1 to 1)."""
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


def find_relevant_papers(
    new_embedding: List[float],
    library_embeddings: List[Tuple[str, List[float]]],
    threshold: float = 0.8,
    top_k: int = 5,
) -> Tuple[bool, float, List[Tuple[str, float]]]:
    """Decide whether a new paper is relevant given library embeddings.

    Args:
        new_embedding: Embedding of the new arXiv paper.
        library_embeddings: ``[(paper_id, embedding), …]`` for the Zotero library.
        threshold: Minimum cosine similarity for the paper to be considered relevant.
        top_k: How many of the most-similar library papers to return.

    Returns:
        ``(is_relevant, max_similarity, top_matches)`` where *top_matches* is a
        list of ``(paper_id, similarity)`` pairs sorted from highest to lowest.
    """
    if not library_embeddings:
        return False, 0.0, []

    scores: List[Tuple[str, float]] = [
        (pid, cosine_similarity(new_embedding, emb))
        for pid, emb in library_embeddings
    ]
    scores.sort(key=lambda x: x[1], reverse=True)

    max_sim = scores[0][1]
    top_matches = scores[:top_k]
    is_relevant = max_sim >= threshold

    return is_relevant, max_sim, top_matches
