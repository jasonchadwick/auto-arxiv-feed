"""Relevance filtering using continuous LOF density scoring."""

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


def _lof_density_score(
    new_embedding: List[float],
    library_vectors: List[List[float]],
    n_neighbors: int = 20,
) -> float:
    """Return a normalized LOF inlier-density score in [0, 1].

    A higher value means the new embedding looks more like the existing library
    distribution. This uses ``LocalOutlierFactor`` with ``novelty=True`` and a
    cosine metric, then maps ``decision_function`` output to [0, 1] with a
    sigmoid transform.
    """
    if not library_vectors:
        return 0.0

    # LOF is unstable for very small sample sets; use a smooth fallback.
    if len(library_vectors) < 3:
        mean_sim = float(
            np.mean([cosine_similarity(new_embedding, emb) for emb in library_vectors])
        )
        return (mean_sim + 1.0) / 2.0

    try:
        from sklearn.neighbors import LocalOutlierFactor
    except ImportError as exc:  # pragma: no cover - dependency issue
        raise RuntimeError(
            "scikit-learn is required for LOF relevance scoring. "
            "Install it with: pip install scikit-learn"
        ) from exc

    vectors = np.asarray(library_vectors, dtype=np.float64)
    query = np.asarray([new_embedding], dtype=np.float64)

    # For novelty mode, n_neighbors must be <= n_samples - 1.
    k = max(2, min(n_neighbors, len(library_vectors) - 1))

    lof = LocalOutlierFactor(n_neighbors=k, metric="cosine", novelty=True)
    lof.fit(vectors)

    decision = float(lof.decision_function(query)[0])
    decision = float(np.clip(decision, -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(-decision)))


def find_relevant_papers(
    new_embedding: List[float],
    library_embeddings: List[Tuple[str, List[float]]],
    threshold: float = 0.5,
    top_k: int = 5,
    lof_neighbors: int = 20,
) -> Tuple[bool, float, List[Tuple[str, float]]]:
    """Decide whether a new paper is relevant given library embeddings.

    Args:
        new_embedding: Embedding of the new arXiv paper.
        library_embeddings: ``[(paper_id, embedding), …]`` for the Zotero library.
        threshold: Minimum LOF density score (0 to 1).
        top_k: How many of the most-similar library papers to return.
        lof_neighbors: Number of neighbors LOF uses for local density.

    Returns:
        ``(is_relevant, density_score, top_matches)`` where *top_matches*
        is a list of ``(paper_id, similarity)`` pairs sorted from highest to
        lowest. ``top_matches`` are cosine-ranked and are provided for
        interpretability, while relevance gating uses LOF density.
    """
    if not library_embeddings:
        return False, 0.0, []

    scores: List[Tuple[str, float]] = [
        (pid, cosine_similarity(new_embedding, emb))
        for pid, emb in library_embeddings
    ]
    scores.sort(key=lambda x: x[1], reverse=True)

    density_score = _lof_density_score(
        new_embedding,
        [emb for _, emb in library_embeddings],
        n_neighbors=lof_neighbors,
    )

    top_matches = scores[:top_k]
    is_relevant = density_score >= threshold

    return is_relevant, density_score, top_matches


def find_closest_collection(
    embedding: List[float],
    collection_centroids: List[Tuple[str, List[float]]],
) -> Tuple[str, float]:
    """Return ``(collection_path, similarity)`` for the nearest collection centroid.

    Returns ``("", 0.0)`` when *collection_centroids* is empty.
    """
    if not collection_centroids:
        return "", 0.0

    best_path = ""
    best_score = -2.0
    for path, centroid in collection_centroids:
        score = cosine_similarity(embedding, centroid)
        if score > best_score:
            best_score = score
            best_path = path

    return best_path, float(best_score)
