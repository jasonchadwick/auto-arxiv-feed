#!/usr/bin/env python3
"""Daily script – fetch new arXiv papers, find relevant ones, send email digest.

Designed to be run once per day (e.g. shortly after arXiv's daily announcement).
Requires the Zotero library to be up-to-date in the local database
(run ``scripts/update_zotero.py`` first).

Usage::

    python scripts/daily_digest.py --config config.yaml
    python scripts/daily_digest.py --config config.yaml --dry-run
    python scripts/daily_digest.py --config config.yaml --dry-run --threshold 0.75
"""

import argparse
import logging
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from src.arxiv_feed import get_new_papers
from src.database import PaperDatabase
from src.email_digest import send_digest
from src.embeddings import BaseEmbedder, get_embedder
from src.relevance import find_relevant_papers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def _api_key_for(emb_cfg: dict) -> Optional[str]:
    provider = emb_cfg.get("provider", "").lower()
    return emb_cfg.get(f"{provider}_api_key") or None


def _print_digest(papers: list) -> None:
    """Fallback: print the digest to stdout when no email config is present."""
    print(f"\nFound {len(papers)} relevant paper(s):\n")
    for i, p in enumerate(papers, start=1):
        print(f"{i}. {p.get('title') or '(no title)'}")
        authors = p.get("authors") or []
        if authors:
            print(f"   Authors:    {', '.join(authors)}")
        print(f"   URL:        {p.get('url') or ''}")
        print(f"   Similarity: {p.get('max_similarity', 0.0):.4f}")
        print()


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def run(
    config: dict,
    dry_run: bool = False,
    threshold_override: Optional[float] = None,
) -> None:
    """Fetch today's arXiv papers, compare to library, optionally email results."""

    db = PaperDatabase(config["database"]["path"])

    # --- Embedder ---
    emb_cfg = config["embedding"]
    embedder = get_embedder(
        provider=emb_cfg["provider"],
        model=emb_cfg["model"],
        api_key=_api_key_for(emb_cfg),
    )

    rel_cfg = config.get("relevance", {})
    threshold = (
        threshold_override
        if threshold_override is not None
        else rel_cfg.get("threshold", 0.8)
    )
    top_k: int = rel_cfg.get("top_k", 5)

    # ------------------------------------------------------------------
    # 1. Fetch new arXiv papers for the configured categories
    # ------------------------------------------------------------------
    categories: list = config["arxiv"]["categories"]
    logger.info("Fetching new arXiv papers for: %s", categories)
    new_papers = get_new_papers(categories)
    logger.info("Retrieved %d new paper(s) from arXiv.", len(new_papers))

    if not new_papers:
        logger.info("No new papers – nothing to do.")
        return

    # ------------------------------------------------------------------
    # 2. Load Zotero library embeddings
    # ------------------------------------------------------------------
    logger.info("Loading Zotero library embeddings (model: %s)…", embedder.model_id)
    library_embeddings = db.get_all_embeddings_for_source("zotero", embedder.model_id)
    logger.info("Loaded %d Zotero embedding(s).", len(library_embeddings))

    if not library_embeddings:
        logger.warning(
            "No Zotero embeddings found in the database. "
            "Run scripts/update_zotero.py first."
        )
        return

    # ------------------------------------------------------------------
    # 3. Embed each new arXiv paper and check relevance
    # ------------------------------------------------------------------
    logger.info("Checking relevance at threshold=%.4f…", threshold)
    relevant: list = []

    for paper in new_papers:
        text = BaseEmbedder.paper_to_text(paper.title, paper.authors, paper.abstract)
        if not text.strip():
            continue
        try:
            embedding = embedder.embed(text)
            is_relevant, max_sim, top_matches = find_relevant_papers(
                embedding, library_embeddings, threshold=threshold, top_k=top_k
            )
            if is_relevant:
                relevant.append(
                    {
                        "paper_id": paper.paper_id,
                        "title": paper.title,
                        "authors": paper.authors,
                        "abstract": paper.abstract,
                        "url": paper.url,
                        "date_published": paper.date_published,
                        "max_similarity": max_sim,
                        "top_matches": top_matches,
                    }
                )
                logger.info(
                    "RELEVANT (%.4f): %s", max_sim, (paper.title or "")[:70]
                )
            else:
                logger.debug(
                    "not relevant (%.4f): %s", max_sim, (paper.title or "")[:70]
                )
        except Exception as exc:  # noqa: BLE001
            logger.error("Error processing paper %s: %s", paper.paper_id, exc)

    logger.info(
        "%d/%d paper(s) are relevant at threshold=%.4f.",
        len(relevant),
        len(new_papers),
        threshold,
    )

    # ------------------------------------------------------------------
    # 4. Send / print digest
    # ------------------------------------------------------------------
    email_cfg = config.get("email") or {}

    if not email_cfg.get("smtp_host"):
        # No email configured – just print to stdout
        if relevant:
            _print_digest(relevant)
        else:
            logger.info("No relevant papers found.")
        return

    send_digest(
        relevant_papers=relevant,
        smtp_host=email_cfg.get("smtp_host", ""),
        smtp_port=int(email_cfg.get("smtp_port", 587)),
        smtp_user=email_cfg.get("smtp_user", ""),
        smtp_password=email_cfg.get("smtp_password")
        or os.environ.get("EMAIL_PASSWORD", ""),
        from_address=email_cfg.get("from_address", ""),
        to_address=email_cfg.get("to_address", ""),
        subject_prefix=email_cfg.get("subject_prefix", "[arXiv Feed]"),
        dry_run=dry_run,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Daily arXiv feed: find relevant new papers and email a digest."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration YAML file (default: config.yaml).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the email digest to stdout instead of sending it. "
            "Embeddings are still computed so you can experiment with --threshold."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "Cosine similarity threshold for relevance (0–1). "
            "Overrides the value in config.yaml. "
            "Lower values cast a wider net; higher values are more selective."
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run(config, dry_run=args.dry_run, threshold_override=args.threshold)


if __name__ == "__main__":
    main()
