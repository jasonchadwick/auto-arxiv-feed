"""Tests for src/relevance.py"""

import math

import pytest

from src.relevance import cosine_similarity, find_relevant_papers


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------


def test_identical_vectors():
    v = [1.0, 2.0, 3.0]
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_orthogonal_vectors():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_opposite_vectors():
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_zero_vector():
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_known_similarity():
    a = [1.0, 1.0, 0.0]
    b = [1.0, 0.0, 0.0]
    # cos(45°) = 1/sqrt(2)
    expected = 1.0 / math.sqrt(2)
    assert cosine_similarity(a, b) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# find_relevant_papers
# ---------------------------------------------------------------------------


def test_relevant_above_threshold():
    new_emb = [1.0, 0.0]
    library = [("lib1", [1.0, 0.0])]  # tiny-library fallback path
    is_rel, max_sim, top = find_relevant_papers(new_emb, library, threshold=0.9)
    assert is_rel is True
    assert max_sim == pytest.approx(1.0)
    assert top[0] == ("lib1", pytest.approx(1.0))


def test_not_relevant_below_threshold():
    new_emb = [1.0, 0.0]
    library = [("lib1", [0.0, 1.0])]  # tiny-library fallback path
    is_rel, max_sim, _ = find_relevant_papers(new_emb, library, threshold=0.75)
    assert is_rel is False
    assert max_sim == pytest.approx(0.5)


def test_empty_library():
    is_rel, max_sim, top = find_relevant_papers([1.0, 2.0], [], threshold=0.5)
    assert is_rel is False
    assert max_sim == 0.0
    assert top == []


def test_top_k_limit():
    new_emb = [1.0, 0.0, 0.0]
    library = [
        ("a", [1.0, 0.0, 0.0]),  # similarity 1.0
        ("b", [0.9, 0.1, 0.0]),
        ("c", [0.8, 0.2, 0.0]),
        ("d", [0.7, 0.3, 0.0]),
        ("e", [0.6, 0.4, 0.0]),
        ("f", [0.5, 0.5, 0.0]),
    ]
    _, _, top = find_relevant_papers(new_emb, library, threshold=0.0, top_k=3)
    assert len(top) == 3
    # Should be the 3 most similar
    assert top[0][0] == "a"


def test_exact_threshold_boundary():
    new_emb = [1.0, 0.0]
    library = [
        ("a", [1.0, 0.0]),
        ("b", [0.95, 0.05]),
        ("c", [0.9, 0.1]),
        ("d", [0.92, 0.08]),
    ]
    _, score, _ = find_relevant_papers(new_emb, library, threshold=0.0)
    is_rel, _, _ = find_relevant_papers(new_emb, library, threshold=score)
    assert is_rel is True


def test_sorted_descending():
    new_emb = [1.0, 0.0]
    library = [
        ("low", [0.0, 1.0]),   # similarity 0.0
        ("high", [1.0, 0.0]),  # similarity 1.0
        ("mid", [0.7071, 0.7071]),  # similarity ~0.707
    ]
    _, _, top = find_relevant_papers(new_emb, library, threshold=0.0, top_k=3)
    sims = [sim for _, sim in top]
    assert sims == sorted(sims, reverse=True)


def test_lof_density_prefers_in_distribution_point():
    library = [
        ("a", [1.0, 0.0]),
        ("b", [0.98, 0.02]),
        ("c", [0.95, 0.05]),
        ("d", [0.92, 0.08]),
        ("e", [0.9, 0.1]),
    ]

    inlier = [0.96, 0.04]
    outlier = [-1.0, 0.0]

    _, inlier_score, _ = find_relevant_papers(inlier, library, threshold=0.0)
    _, outlier_score, _ = find_relevant_papers(outlier, library, threshold=0.0)

    assert inlier_score > outlier_score

    mid_threshold = (inlier_score + outlier_score) / 2.0
    inlier_rel, _, _ = find_relevant_papers(inlier, library, threshold=mid_threshold)
    outlier_rel, _, _ = find_relevant_papers(outlier, library, threshold=mid_threshold)
    assert inlier_rel is True
    assert outlier_rel is False
