"""Embedding providers: OpenAI, Gemini, Anthropic stub, and local (sentence-transformers)."""

import logging
import os
import sys
from abc import ABC, abstractmethod
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_embedder(
    provider: str, model: str, api_key: Optional[str] = None
) -> "BaseEmbedder":
    """Return an embedder for the given *provider* and *model*.

    *api_key* is optional; if not provided the implementation will fall back to
    the relevant environment variable (``OPENAI_API_KEY``, ``GEMINI_API_KEY``).
    """
    provider = provider.lower()
    if provider == "openai":
        return OpenAIEmbedder(model=model, api_key=api_key)
    if provider in ("gemini", "google"):
        return GeminiEmbedder(model=model, api_key=api_key)
    if provider in ("anthropic", "claude"):
        return AnthropicEmbedder(model=model, api_key=api_key)
    if provider == "local":
        return LocalEmbedder(model=model)
    raise ValueError(
        f"Unknown embedding provider '{provider}'. "
        "Choose from: openai, gemini, anthropic, local."
    )


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseEmbedder(ABC):
    """Abstract base class for all embedding providers."""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Embed *text* and return a float vector."""

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts.  Override for providers that support batching."""
        return [self.embed(t) for t in texts]

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Unique identifier string for this model (used as DB key)."""

    # ------------------------------------------------------------------
    # Helper shared by all providers
    # ------------------------------------------------------------------

    @staticmethod
    def paper_to_text(
        title: str, authors: List[str], abstract: str
    ) -> str:
        """Concatenate paper metadata into a single string suitable for embedding."""
        parts: List[str] = []
        if title:
            parts.append(f"Title: {title}")
        if authors:
            parts.append(f"Authors: {', '.join(authors)}")
        if abstract:
            parts.append(f"Abstract: {abstract}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI text-embedding models (e.g. ``text-embedding-3-small``)."""

    DEFAULT_MODEL = "text-embedding-3-small"

    def __init__(self, model: str = DEFAULT_MODEL, api_key: Optional[str] = None):
        import openai  # lazy import so the package is only required when used

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OpenAI API key not found. "
                "Set the OPENAI_API_KEY environment variable or pass api_key."
            )
        self.client = openai.OpenAI(api_key=key)
        self._model = model

    @property
    def model_id(self) -> str:
        return f"openai/{self._model}"

    def embed(self, text: str) -> List[float]:
        response = self.client.embeddings.create(input=text, model=self._model)
        return response.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(input=texts, model=self._model)
        # API guarantees ordering by `index`
        items = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in items]


# ---------------------------------------------------------------------------
# Gemini / Google
# ---------------------------------------------------------------------------


class GeminiEmbedder(BaseEmbedder):
    """Google Gemini embedding models (e.g. ``models/text-embedding-004``)."""

    DEFAULT_MODEL = "models/text-embedding-004"

    def __init__(self, model: str = DEFAULT_MODEL, api_key: Optional[str] = None):
        import google.genai as genai  # lazy import

        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError(
                "Gemini API key not found. "
                "Set the GEMINI_API_KEY environment variable or pass api_key."
            )
        self._client = genai.Client(api_key=key)
        self._model = model

    @property
    def model_id(self) -> str:
        return f"gemini/{self._model}"

    def embed(self, text: str) -> List[float]:
        response = self._client.models.embed_content(model=self._model, contents=text)
        return list(response.embeddings[0].values)


# ---------------------------------------------------------------------------
# Anthropic / Claude (stub)
# ---------------------------------------------------------------------------


class AnthropicEmbedder(BaseEmbedder):
    """Placeholder for Anthropic embedding support.

    .. note::
        As of 2024/2025 Anthropic does **not** offer a public text-embedding
        API.  This class exists so the provider can be specified in config and
        will raise a clear error at initialisation time.  A future PR can
        replace this stub once Anthropic releases an embedding endpoint.
    """

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        raise NotImplementedError(
            "Anthropic/Claude does not currently provide a text-embedding API. "
            "Use 'openai', 'gemini', or 'local' as the embedding provider instead."
        )

    def embed(self, text: str) -> List[float]:  # pragma: no cover
        raise NotImplementedError

    @property
    def model_id(self) -> str:  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------


def _is_wsl() -> bool:
    """Return *True* when running inside WSL (any version)."""
    try:
        with open("/proc/version") as fh:
            return "microsoft" in fh.read().lower()
    except OSError:
        return False


def _detect_local_device() -> str:
    """Return the best available PyTorch device string.

    Detection order: CUDA → MPS (Apple Silicon) → CPU.
    On WSL2 with an NVIDIA driver, CUDA is accessible via the Windows driver
    passthrough (``/dev/dxg``); no separate Linux CUDA toolkit is needed.
    """
    wsl = _is_wsl()
    try:
        import torch  # type: ignore[import]

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            suffix = " via WSL2 passthrough" if wsl else ""
            logger.info("LocalEmbedder: CUDA GPU detected%s — '%s'", suffix, gpu_name)
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("LocalEmbedder: Apple MPS detected")
            return "mps"
    except ImportError:
        pass

    if wsl:
        logger.info(
            "LocalEmbedder: no CUDA available (WSL2 detected). "
            "For GPU support, install PyTorch with CUDA: "
            "pip install torch --index-url https://download.pytorch.org/whl/cu121"
        )
    else:
        logger.info("LocalEmbedder: no GPU detected, using CPU")
    return "cpu"


# ---------------------------------------------------------------------------
# Local
# ---------------------------------------------------------------------------


class LocalEmbedder(BaseEmbedder):
    """Local embedding model powered by ``sentence-transformers``.

    The model is downloaded from HuggingFace on first use and cached in
    ``~/.cache/huggingface/``.  Subsequent runs load from cache.

    **Recommended models**:

    +-----------------------------------------+------+--------+-------------------------------------+
    | Model name                              | Dims | Size   | Notes                               |
    +=========================================+======+========+=====================================+
    | ``BAAI/bge-small-en-v1.5`` (default)   |  384 | ~133 MB| Best speed/quality trade-off        |
    | ``BAAI/bge-large-en-v1.5``             | 1024 | ~1.3 GB| Highest quality, general purpose    |
    | ``allenai-specter``                     |  768 | ~400 MB| Fine-tuned on scientific papers     |
    | ``all-MiniLM-L6-v2``                   |  384 |  ~22 MB| Smallest/fastest, lower quality     |
    +-----------------------------------------+------+--------+-------------------------------------+

    **Installation** (CPU only)::

        pip install sentence-transformers

    **Installation** (CUDA GPU — Linux or WSL2 with NVIDIA driver)::

        pip install torch --index-url https://download.pytorch.org/whl/cu121
        pip install sentence-transformers

    On WSL2 the Windows NVIDIA driver exposes CUDA via ``/dev/dxg``; no
    separate Linux CUDA toolkit installation is required.

    **Config** (``config.yaml``)::

        embedding:
          provider: local
          model: BAAI/bge-small-en-v1.5
    """

    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

    def __init__(self, model: str = DEFAULT_MODEL):
        self._model_name = model
        self._st_model = None
        self._device = _detect_local_device()
        self._load_model()

    def _load_model(self) -> None:
        """Import sentence-transformers and load the model onto ``self._device``."""
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for provider='local'.\n"
                "Install options:\n"
                "  CPU only:         pip install sentence-transformers\n"
                "  CUDA (WSL2/Linux): pip install torch --index-url "
                "https://download.pytorch.org/whl/cu121 "
                "&& pip install sentence-transformers"
            ) from exc

        logger.info(
            "LocalEmbedder: loading '%s' on device '%s'", self._model_name, self._device
        )
        self._st_model = SentenceTransformer(self._model_name, device=self._device)
        logger.info("LocalEmbedder: model ready")

    @property
    def model_id(self) -> str:
        return f"local/{self._model_name}"

    def embed(self, text: str) -> List[float]:
        return self._st_model.encode(text, normalize_embeddings=True).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._st_model.encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
        )
        return [e.tolist() for e in embeddings]
