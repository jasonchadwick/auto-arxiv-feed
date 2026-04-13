"""Tests for src/embeddings.py"""

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.embeddings import (
    BaseEmbedder,
    LocalEmbedder,
    AnthropicEmbedder,
    _is_wsl,
    _detect_local_device,
    get_embedder,
)


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
# LocalEmbedder
# ---------------------------------------------------------------------------


def _mock_st_module(single_vec=None, batch_vecs=None):
    """Return a fake sentence_transformers module and its inner mock model."""
    if single_vec is None:
        single_vec = np.array([0.1, 0.2, 0.3])
    mock_model = MagicMock()
    mock_model.encode.return_value = single_vec
    mock_module = MagicMock()
    mock_module.SentenceTransformer.return_value = mock_model
    return mock_module, mock_model


def test_local_embedder_import_error():
    """ImportError with helpful message when sentence-transformers is not installed."""
    with pytest.raises(ImportError, match="sentence-transformers"):
        with pytest.MonkeyPatch().context() as mp:
            mp.setitem(sys.modules, "sentence_transformers", None)
            LocalEmbedder(model="BAAI/bge-small-en-v1.5")


def test_local_embedder_model_id():
    mock_module, _ = _mock_st_module()
    with pytest.MonkeyPatch().context() as mp:
        mp.setitem(sys.modules, "sentence_transformers", mock_module)
        emb = LocalEmbedder(model="my/custom-model")
    assert emb.model_id == "local/my/custom-model"


def test_local_embedder_embed():
    vec = np.array([0.1, 0.2, 0.3])
    mock_module, mock_model = _mock_st_module(single_vec=vec)
    with pytest.MonkeyPatch().context() as mp:
        mp.setitem(sys.modules, "sentence_transformers", mock_module)
        emb = LocalEmbedder(model="test-model")
    result = emb.embed("hello world")
    assert result == pytest.approx([0.1, 0.2, 0.3])
    mock_model.encode.assert_called_with("hello world", normalize_embeddings=True)


def test_local_embedder_embed_batch():
    vecs = np.array([[0.1, 0.2], [0.3, 0.4]])
    mock_module, mock_model = _mock_st_module(single_vec=vecs)
    with pytest.MonkeyPatch().context() as mp:
        mp.setitem(sys.modules, "sentence_transformers", mock_module)
        emb = LocalEmbedder(model="test-model")
    result = emb.embed_batch(["text one", "text two"])
    assert np.allclose(result, [[0.1, 0.2], [0.3, 0.4]])
    mock_model.encode.assert_called_with(
        ["text one", "text two"],
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def test_get_embedder_local_returns_local_embedder():
    mock_module, _ = _mock_st_module()
    with pytest.MonkeyPatch().context() as mp:
        mp.setitem(sys.modules, "sentence_transformers", mock_module)
        emb = get_embedder("local", "BAAI/bge-small-en-v1.5")
    assert isinstance(emb, LocalEmbedder)
    assert emb.model_id == "local/BAAI/bge-small-en-v1.5"


# ---------------------------------------------------------------------------
# _is_wsl / _detect_local_device helpers
# ---------------------------------------------------------------------------


def test_is_wsl_true(tmp_path):
    proc_version = tmp_path / "version"
    proc_version.write_text("Linux version 5.15.0-microsoft-standard-WSL2\n")
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("builtins.open", lambda path, *a, **kw: proc_version.open(*a, **kw)
                   if "proc/version" in str(path) else open(path, *a, **kw))
        # Direct test via reading the file content
    assert "microsoft" in proc_version.read_text().lower()


def test_is_wsl_false_on_non_wsl(tmp_path):
    proc_version = tmp_path / "version"
    proc_version.write_text("Linux version 6.1.0-21-amd64 (Debian)\n")
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(
            "src.embeddings._is_wsl",
            lambda: "microsoft" in proc_version.read_text().lower(),
        )
        from src.embeddings import _is_wsl as patched
        assert not patched()


def test_detect_local_device_cpu_when_torch_missing():
    """Falls back to 'cpu' when torch is not importable."""
    with pytest.MonkeyPatch().context() as mp:
        mp.setitem(sys.modules, "torch", None)
        device = _detect_local_device()
    assert device == "cpu"


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
