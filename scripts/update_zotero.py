#!/usr/bin/env python3
"""Hourly script – sync Zotero library and embed any papers not yet embedded.

Run this on a schedule (e.g. via cron every hour) so the local database stays
up-to-date with your Zotero library.

Usage::

    python scripts/update_zotero.py --config config.yaml
    python scripts/update_zotero.py --config config.yaml --dry-run
"""

import argparse
import logging
import os
import sys
from typing import Optional

# Allow running the script directly from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from src.database import PaperDatabase
from src.embeddings import BaseEmbedder, get_embedder
from src.zotero_client import ZoteroClient

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
    """Pull the provider-specific API key from the config block."""
    provider = emb_cfg.get("provider", "").lower()
    key_field = f"{provider}_api_key"
    return emb_cfg.get(key_field) or None


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def run(config: dict, dry_run: bool = False) -> None:
    """Sync Zotero → DB → embed unembedded papers."""

    db = PaperDatabase(config["database"]["path"])

    # --- Zotero client ---
    zot_cfg = config["zotero"]
    client = ZoteroClient(
        user_id=str(zot_cfg["user_id"]),
        api_key=zot_cfg["api_key"],
        library_type=zot_cfg.get("library_type", "user"),
    )

    # --- Embedder ---
    emb_cfg = config["embedding"]
    embedder = get_embedder(
        provider=emb_cfg["provider"],
        model=emb_cfg["model"],
        api_key=_api_key_for(emb_cfg),
    )

    # ------------------------------------------------------------------
    # 1. Fetch papers from Zotero and upsert into the database
    # ------------------------------------------------------------------
    logger.info("Fetching papers from Zotero…")
    papers = client.get_all_papers()
    logger.info("Zotero library contains %d paper(s).", len(papers))

    new_count = 0
    for paper in papers:
        paper_id = f"zotero:{paper.item_key}"
        if db.get_paper(paper_id) is None:
            new_count += 1
        db.upsert_paper(
            paper_id=paper_id,
            source="zotero",
            title=paper.title,
            authors=paper.authors,
            abstract=paper.abstract,
            url=paper.url,
            date_added=paper.date_added,
            date_published=paper.date_published,
        )

    logger.info("%d new paper(s) added to the database.", new_count)

    # ------------------------------------------------------------------
    # 2. Find papers that still lack embeddings
    # ------------------------------------------------------------------
    unembedded = db.get_papers_without_embedding("zotero", embedder.model_id)
    logger.info(
        "%d paper(s) need embedding with model '%s'.",
        len(unembedded),
        embedder.model_id,
    )

    if not unembedded:
        logger.info("All Zotero papers are already embedded.")
        return

    if dry_run:
        logger.info("DRY RUN – would embed %d paper(s):", len(unembedded))
        for p in unembedded[:10]:
            logger.info("  • %s", p.get("title") or p["id"])
        return

    # ------------------------------------------------------------------
    # 3. Embed each unembedded paper
    # ------------------------------------------------------------------
    errors = 0
    for idx, paper in enumerate(unembedded, start=1):
        text = BaseEmbedder.paper_to_text(
            paper.get("title") or "",
            paper.get("authors") or [],
            paper.get("abstract") or "",
        )
        if not text.strip():
            logger.warning("Skipping %s – no text to embed.", paper["id"])
            continue
        try:
            embedding = embedder.embed(text)
            db.store_embedding(paper["id"], embedder.model_id, embedding)
            if idx % 10 == 0 or idx == len(unembedded):
                logger.info("  Embedded %d/%d paper(s)…", idx, len(unembedded))
        except Exception as exc:  # noqa: BLE001
            logger.error("Error embedding %s: %s", paper["id"], exc)
            errors += 1

    logger.info(
        "Done. %d paper(s) embedded, %d error(s).",
        len(unembedded) - errors,
        errors,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync Zotero library and embed new papers into the local database."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration YAML file (default: config.yaml).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without calling any embedding API.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
