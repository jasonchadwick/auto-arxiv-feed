"""Tests for scripts/daily_digest.py helpers."""

from src.email_digest import _build_html_body, _build_text_body
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


def test_digest_body_shows_new_listings_in_full_and_updates_as_titles_only():
    papers = [
        {
            "paper_id": "2401.00001",
            "title": "New Listing via Relevance",
            "authors": ["Alice Smith"],
            "abstract": "Detailed abstract for the new listing.",
            "url": "https://arxiv.org/abs/2401.00001",
            "max_similarity": 0.91,
            "closest_collection": "hardware/superconducting",
            "closest_collection_score": 0.88,
            "is_update": False,
        },
        {
            "paper_id": "2401.00002",
            "title": "Keyword Fallback New Listing",
            "authors": ["Bob Jones"],
            "abstract": "Detailed abstract for the keyword fallback listing.",
            "url": "https://arxiv.org/abs/2401.00002",
            "max_similarity": 0.12,
            "override_by_terms": True,
            "override_terms": ["quantum error correction"],
            "is_update": False,
        },
        {
            "paper_id": "2401.00003v2",
            "title": "Updated Existing Paper",
            "authors": ["Carol Lee"],
            "abstract": "This should not appear in full in the updates section.",
            "url": "https://arxiv.org/abs/2401.00003v2",
            "max_similarity": 0.55,
            "is_update": True,
        },
    ]

    text_body = _build_text_body(papers, "2026-04-14")

    assert "New listings" in text_body
    assert "1. New Listing via Relevance" in text_body
    assert "2. Keyword Fallback New Listing" in text_body
    assert "Override:   keyword(s): quantum error correction" in text_body
    assert "Updates" in text_body
    assert "1. Updated Existing Paper" in text_body
    assert "This should not appear in full in the updates section." not in text_body


def test_html_digest_renders_updates_as_title_list_only():
    papers = [
        {
            "paper_id": "2401.00001",
            "title": "New Listing",
            "authors": ["Alice Smith"],
            "abstract": "Detailed abstract.",
            "url": "https://arxiv.org/abs/2401.00001",
            "max_similarity": 0.91,
            "is_update": False,
        },
        {
            "paper_id": "2401.00002v2",
            "title": "Updated Listing",
            "authors": ["Bob Jones"],
            "abstract": "Update abstract should stay out of the updates section.",
            "url": "https://arxiv.org/abs/2401.00002v2",
            "max_similarity": 0.52,
            "is_update": True,
        },
    ]

    html_body = _build_html_body(papers, "2026-04-14")

    assert "<h2>New listings</h2>" in html_body
    assert "Detailed abstract." in html_body
    assert "<h2>Updates</h2>" in html_body
    assert "Updated Listing" in html_body
    assert "Update abstract should stay out of the updates section." not in html_body
