"""Tests for src/embeddings.py"""

import pytest

from src.embeddings import BaseEmbedder, LocalEmbedder, AnthropicEmbedder, get_embedder


# ---------------------------------------------------------------------------
# BaseEmbedder.paper_to_text
# ---------------------------------------------------------------------------


def test_paper_to_text_full():
    text = BaseEmbedder.paper_to_text(
        title="Quantum Supremacy",
        authors=["Alice Smith", "Bob Jones"],
        abstract="We show quantum advantage.",
    )
    assert "Title: Quantum Supremacy" in text
    assert "Authors: Alice Smith, Bob Jones" in text
    assert "Abstract: We show quantum advantage." in text


def test_paper_to_text_missing_authors():
    text = BaseEmbedder.paper_to_text(
        title="My Paper", authors=[], abstract="Some abstract."
    )
    assert "Authors:" not in text
    assert "Title: My Paper" in text


def test_paper_to_text_all_empty():
    text = BaseEmbedder.paper_to_text(title="", authors=[], abstract="")
    assert text == ""


def test_paper_to_text_no_abstract():
    text = BaseEmbedder.paper_to_text(title="T", authors=["A"], abstract="")
    assert "Abstract:" not in text


# ---------------------------------------------------------------------------
# LocalEmbedder raises NotImplementedError on init
# ---------------------------------------------------------------------------


def test_local_embedder_raises_on_init():
    with pytest.raises(NotImplementedError, match="LocalEmbedder"):
        LocalEmbedder(model="some-model")


# ---------------------------------------------------------------------------
# AnthropicEmbedder raises NotImplementedError on init
# ---------------------------------------------------------------------------


def test_anthropic_embedder_raises_on_init():
    with pytest.raises(NotImplementedError, match="Anthropic"):
        AnthropicEmbedder(model="claude-3", api_key="fake")


# ---------------------------------------------------------------------------
# get_embedder factory
# ---------------------------------------------------------------------------


def test_get_embedder_unknown_provider():
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        get_embedder("nonexistent", "model-x")


def test_get_embedder_local_raises():
    with pytest.raises(NotImplementedError):
        get_embedder("local", "some-model")


def test_get_embedder_anthropic_raises():
    with pytest.raises(NotImplementedError):
        get_embedder("anthropic", "claude-3")


def test_get_embedder_claude_alias_raises():
    with pytest.raises(NotImplementedError):
        get_embedder("claude", "claude-3")


def test_get_embedder_openai_missing_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        get_embedder("openai", "text-embedding-3-small")


def test_get_embedder_gemini_missing_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        get_embedder("gemini", "models/text-embedding-004")
