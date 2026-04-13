"""Tests for scripts/daily_digest.py helpers."""

from src.arxiv_feed import ArxivPaper
from scripts.daily_digest import _normalize_terms, _paper_matching_terms


def test_normalize_terms_filters_and_lowercases():
    raw = ["  Quantum ", "", " Error Correction", 123, "  "]
    assert _normalize_terms(raw) == ["quantum", "error correction", "123"]


def test_normalize_terms_non_list_returns_empty():
    assert _normalize_terms("quantum") == []


def test_paper_matching_terms_title_and_abstract_case_insensitive():
    paper = ArxivPaper(
        paper_id="2401.00001",
        title="A New Quantum Error Correction Scheme",
        authors=["A. Example"],
        abstract="We benchmark superconducting qubit architectures.",
        url="https://arxiv.org/abs/2401.00001",
    )
    terms = ["error correction", "superconducting qubit", "does-not-match"]

    assert _paper_matching_terms(paper, terms) == [
        "error correction",
        "superconducting qubit",
    ]


def test_paper_matching_terms_empty_terms_returns_empty():
    paper = ArxivPaper(
        paper_id="2401.00002",
        title="Anything",
        authors=[],
        abstract="Anything",
        url="https://arxiv.org/abs/2401.00002",
    )
    assert _paper_matching_terms(paper, []) == []
