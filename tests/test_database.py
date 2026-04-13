"""Tests for src/database.py"""

import json
import os
import tempfile

import pytest

from src.database import PaperDatabase


@pytest.fixture
def db(tmp_path):
    return PaperDatabase(str(tmp_path / "test.db"))


# ---------------------------------------------------------------------------
# Paper CRUD
# ---------------------------------------------------------------------------


def test_upsert_and_get_paper(db):
    db.upsert_paper(
        paper_id="zotero:ABC123",
        source="zotero",
        title="Quantum Supremacy",
        authors=["Alice", "Bob"],
        abstract="We demonstrate quantum supremacy.",
        url="https://example.com",
        date_added="2024-01-01",
        date_published="2024-01-01",
    )
    paper = db.get_paper("zotero:ABC123")
    assert paper is not None
    assert paper["title"] == "Quantum Supremacy"
    assert paper["authors"] == ["Alice", "Bob"]
    assert paper["source"] == "zotero"


def test_get_nonexistent_paper_returns_none(db):
    assert db.get_paper("does:not:exist") is None


def test_upsert_overwrites_existing(db):
    db.upsert_paper("p1", "zotero", title="Old title")
    db.upsert_paper("p1", "zotero", title="New title")
    assert db.get_paper("p1")["title"] == "New title"


def test_get_papers_by_source(db):
    db.upsert_paper("p1", "zotero", title="Zotero Paper")
    db.upsert_paper("p2", "arxiv", title="arXiv Paper")
    zotero_papers = db.get_papers_by_source("zotero")
    arxiv_papers = db.get_papers_by_source("arxiv")
    assert len(zotero_papers) == 1
    assert len(arxiv_papers) == 1
    assert zotero_papers[0]["title"] == "Zotero Paper"


# ---------------------------------------------------------------------------
# Embedding CRUD
# ---------------------------------------------------------------------------


def test_store_and_get_embedding(db):
    db.upsert_paper("p1", "zotero")
    emb = [0.1, 0.2, 0.3]
    db.store_embedding("p1", "openai/text-embedding-3-small", emb)
    retrieved = db.get_embedding("p1", "openai/text-embedding-3-small")
    assert retrieved is not None
    assert len(retrieved) == 3
    assert abs(retrieved[0] - 0.1) < 1e-9


def test_has_embedding(db):
    db.upsert_paper("p1", "zotero")
    assert not db.has_embedding("p1", "openai/text-embedding-3-small")
    db.store_embedding("p1", "openai/text-embedding-3-small", [0.1])
    assert db.has_embedding("p1", "openai/text-embedding-3-small")


def test_store_embedding_overwrites(db):
    db.upsert_paper("p1", "zotero")
    db.store_embedding("p1", "model", [1.0, 2.0])
    db.store_embedding("p1", "model", [9.0, 8.0])
    retrieved = db.get_embedding("p1", "model")
    assert retrieved[0] == pytest.approx(9.0)


def test_get_papers_without_embedding(db):
    db.upsert_paper("p1", "zotero", title="Paper 1")
    db.upsert_paper("p2", "zotero", title="Paper 2")
    db.store_embedding("p1", "mymodel", [0.5, 0.5])

    unembedded = db.get_papers_without_embedding("zotero", "mymodel")
    ids = [p["id"] for p in unembedded]
    assert "p1" not in ids
    assert "p2" in ids


def test_get_all_embeddings_for_source(db):
    db.upsert_paper("p1", "zotero")
    db.upsert_paper("p2", "zotero")
    db.upsert_paper("p3", "arxiv")
    db.store_embedding("p1", "m", [1.0])
    db.store_embedding("p2", "m", [2.0])
    db.store_embedding("p3", "m", [3.0])

    result = db.get_all_embeddings_for_source("zotero", "m")
    paper_ids = [r[0] for r in result]
    assert set(paper_ids) == {"p1", "p2"}


def test_get_all_embeddings_with_titles_for_source(db):
    db.upsert_paper("p1", "zotero", title="First paper")
    db.upsert_paper("p2", "zotero", title="")
    db.upsert_paper("p3", "arxiv", title="Ignored")
    db.store_embedding("p1", "m", [1.0])
    db.store_embedding("p2", "m", [2.0])
    db.store_embedding("p3", "m", [3.0])

    result = db.get_all_embeddings_with_titles_for_source("zotero", "m")
    rows = {paper_id: (title, emb) for paper_id, title, emb in result}

    assert set(rows) == {"p1", "p2"}
    assert rows["p1"][0] == "First paper"
    assert rows["p2"][0] == ""
    assert rows["p1"][1] == [1.0]


def test_null_authors_handled(db):
    db.upsert_paper("p1", "zotero", authors=None)
    paper = db.get_papers_by_source("zotero")[0]
    assert paper["authors"] == []
