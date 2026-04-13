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
import traceback
from typing import Optional

# Allow running the script directly from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from src.database import PaperDatabase
from src.email_digest import send_error_notification
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
    # 1b. Sync collections and paper→collection memberships
    # ------------------------------------------------------------------
    logger.info("Syncing Zotero collections…")
    collections = client.get_all_collections()
    db.replace_all_collections(collections)
    for paper in papers:
        paper_id = f"zotero:{paper.item_key}"
        db.set_paper_collections(paper_id, paper.collections)
    logger.info("Collection memberships synced for %d paper(s).", len(papers))

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

    # ------------------------------------------------------------------
    # 4. Recompute collection centroids
    # ------------------------------------------------------------------
    _recompute_collection_centroids(db, embedder.model_id)


def _recompute_collection_centroids(db: PaperDatabase, model_id: str) -> None:
    """Compute and store a centroid embedding for every Zotero collection.

    Only collections that have at least one embedded paper contribute a
    centroid.  Stale centroids are cleared before re-computation so removed
    collections don't linger.
    """
    import numpy as np

    collections = db.get_all_collections()
    if not collections:
        logger.info("No Zotero collections found – skipping centroid update.")
        return

    paths = PaperDatabase.build_collection_paths(collections)

    # Preload all Zotero embeddings once (avoids N per-collection queries).
    all_embs: dict = {
        pid: emb
        for pid, emb in db.get_all_embeddings_for_source("zotero", model_id)
    }

    db.delete_all_collection_centroids_for_model(model_id)

    updated = 0
    for col in collections:
        key = col["key"]
        path = paths.get(key) or key
        paper_ids = db.get_paper_ids_in_collection(key)
        vectors = [all_embs[pid] for pid in paper_ids if pid in all_embs]
        if not vectors:
            continue
        centroid = np.mean(np.asarray(vectors, dtype=np.float64), axis=0).tolist()
        db.upsert_collection_centroid(path, model_id, centroid)
        updated += 1

    logger.info("Recomputed %d collection centroid(s).", updated)


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

    config: Optional[dict] = None
    try:
        config = load_config(args.config)
        run(config, dry_run=args.dry_run)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled error in update_zotero.")

        if config is None:
            try:
                config = load_config(args.config)
            except Exception:  # noqa: BLE001
                config = None

        email_cfg = (config or {}).get("email") or {}
        if email_cfg.get("smtp_host"):
            send_error_notification(
                script_name="scripts/update_zotero.py",
                error_message=str(exc),
                traceback_text=traceback.format_exc(),
                smtp_host=email_cfg.get("smtp_host", ""),
                smtp_port=int(email_cfg.get("smtp_port", 587)),
                smtp_user=email_cfg.get("smtp_user", ""),
                smtp_password=email_cfg.get("smtp_password")
                or os.environ.get("EMAIL_PASSWORD", ""),
                from_address=email_cfg.get("from_address", ""),
                to_address=email_cfg.get("to_address", ""),
                subject_prefix=email_cfg.get("subject_prefix", "[arXiv Feed]"),
                dry_run=args.dry_run,
            )

        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
