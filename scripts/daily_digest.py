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
import time
import traceback
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from src.arxiv_feed import get_new_papers
from src.database import PaperDatabase
from src.email_digest import send_digest, send_error_notification
from src.embeddings import BaseEmbedder, get_embedder
from src.relevance import find_relevant_papers, find_closest_collection

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


def _resilience_config(config: Optional[dict]) -> dict:
    return (config or {}).get("resilience") or {}


def _should_retry(attempt: int, retry_forever: bool, max_attempts: int) -> bool:
    if retry_forever:
        return True
    return attempt < max_attempts


def _print_digest(papers: list) -> None:
    """Fallback: print the digest to stdout when no email config is present."""
    new_listings = [paper for paper in papers if not paper.get("is_update")]
    updates = [paper for paper in papers if paper.get("is_update")]

    print(
        f"\nFound {len(papers)} relevant item(s): "
        f"{len(new_listings)} new listing(s), {len(updates)} update(s).\n"
    )

    if new_listings:
        print("New listings")
        print("-" * 50)
    for i, p in enumerate(new_listings, start=1):
        print(f"{i}. {p.get('title') or '(no title)'}")
        authors = p.get("authors") or []
        if authors:
            print(f"   Authors:    {', '.join(authors)}")
        print(f"   URL:        {p.get('url') or ''}")
        print(f"   Similarity: {p.get('max_similarity', 0.0):.4f}")
        override_terms = p.get("override_terms") or []
        if p.get("override_by_terms") and override_terms:
            print(f"   Override:   keyword(s): {', '.join(override_terms)}")
        print()

    if updates:
        print("Updates")
        print("-" * 50)
        for i, p in enumerate(updates, start=1):
            print(f"{i}. {p.get('title') or '(no title)'}")
        print()


def _normalize_terms(raw_terms: object) -> list[str]:
    """Normalize config-provided always-include terms to lowercase strings."""
    if not isinstance(raw_terms, list):
        return []
    return [str(term).strip().lower() for term in raw_terms if str(term).strip()]


def _paper_matching_terms(paper: object, terms: list[str]) -> list[str]:
    """Return which *terms* appear in the paper title/abstract (case-insensitive)."""
    if not terms:
        return []

    title = getattr(paper, "title", "") or ""
    abstract = getattr(paper, "abstract", "") or ""
    haystack = f"{title}\n{abstract}".lower()
    return [term for term in terms if term in haystack]


def _build_excluded_collection_keys(all_collections: list, excluded_names: list) -> set:
    """Return the set of collection keys that are excluded.

    A collection is excluded if its own name (case-insensitive) matches any
    entry in *excluded_names*, or if any of its ancestors' names match
    (recursive).  A paper is only removed from clustering if *all* of its
    collections are excluded (see ``_paper_fully_excluded``).
    """
    if not excluded_names:
        return set()

    normalized = {n.lower().strip() for n in excluded_names}
    by_key = {c["key"]: c for c in all_collections}
    cache: dict = {}

    def is_excluded(key: str, visiting: frozenset = frozenset()) -> bool:
        if key in cache:
            return cache[key]
        if key not in by_key or key in visiting:
            return False
        col = by_key[key]
        if (col["name"] or "").lower().strip() in normalized:
            cache[key] = True
            return True
        parent = col.get("parent_key")
        result = bool(parent and is_excluded(parent, visiting | {key}))
        cache[key] = result
        return result

    return {c["key"] for c in all_collections if is_excluded(c["key"])}


def _paper_fully_excluded(
    paper_id: str, paper_col_map: dict, excluded_keys: set
) -> bool:
    """Return True only if the paper has collections AND every one is excluded."""
    col_keys = paper_col_map.get(paper_id, [])
    if not col_keys:
        return False  # no collection info → keep it
    return all(k in excluded_keys for k in col_keys)


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
        else rel_cfg.get("threshold", 0.5)
    )
    top_k: int = rel_cfg.get("top_k", 5)
    lof_neighbors: int = int(rel_cfg.get("lof_neighbors", 20))
    always_include_terms = _normalize_terms(rel_cfg.get("always_include_terms", []))
    excluded_collection_names = _normalize_terms(rel_cfg.get("excluded_collections", []))

    # ------------------------------------------------------------------
    # 1. Fetch new arXiv papers for the configured categories
    # ------------------------------------------------------------------
    categories: list = config["arxiv"]["categories"]
    arxiv_request_delay = float(config.get("arxiv", {}).get("request_delay", 3.0))
    arxiv_timeout = float(config.get("arxiv", {}).get("timeout_seconds", 60.0))
    logger.info("Fetching new arXiv papers for: %s", categories)
    new_papers = get_new_papers(
        categories,
        request_delay=arxiv_request_delay,
        api_timeout=arxiv_timeout,
    )
    logger.info("Retrieved %d new paper(s) from arXiv.", len(new_papers))

    if not new_papers:
        logger.info("No new papers – nothing to do.")
        return

    # ------------------------------------------------------------------
    # 2. Load Zotero library embeddings
    # ------------------------------------------------------------------
    logger.info("Loading Zotero library embeddings (model: %s)…", embedder.model_id)
    library_rows = db.get_all_embeddings_with_titles_for_source(
        "zotero", embedder.model_id
    )
    logger.info("Loaded %d Zotero embedding(s).", len(library_rows))


    all_collections: list = []
    excluded_keys: set = set()

    # Filter out papers whose every collection is in the excluded set.
    if excluded_collection_names:
        all_collections = db.get_all_collections()
        excluded_keys = _build_excluded_collection_keys(all_collections, excluded_collection_names)
        if excluded_keys:
            paper_col_map = db.get_all_paper_collection_keys()
            before = len(library_rows)
            library_rows = [
                row for row in library_rows
                if not _paper_fully_excluded(row[0], paper_col_map, excluded_keys)
            ]
            logger.info(
                "Excluded %d/%d Zotero paper(s) in excluded-only collections from clustering.",
                before - len(library_rows),
                before,
            )

    library_embeddings = [(paper_id, embedding) for paper_id, _, embedding in library_rows]
    paper_id_to_title = {
        paper_id: (title.strip() if isinstance(title, str) else "")
        for paper_id, title, _ in library_rows
    }

    if not library_embeddings:
        logger.warning(
            "No Zotero embeddings found in the database. "
            "Run scripts/update_zotero.py first."
        )
        return

    # Load collection centroids for nearest-collection annotation.
    collection_centroids = db.get_collection_centroids(embedder.model_id)
    if excluded_keys and all_collections:
        path_by_key = PaperDatabase.build_collection_paths(all_collections)
        excluded_paths = {
            path_by_key[key]
            for key in excluded_keys
            if key in path_by_key
        }
        before = len(collection_centroids)
        collection_centroids = [
            (path, emb)
            for path, emb in collection_centroids
            if path not in excluded_paths
        ]
        logger.info(
            "Excluded %d/%d collection centroid(s) from nearest-collection labels.",
            before - len(collection_centroids),
            before,
        )
    logger.info("Loaded %d collection centroid(s).", len(collection_centroids))

    # ------------------------------------------------------------------
    # 3. Embed each new arXiv paper and check relevance
    # ------------------------------------------------------------------
    logger.info("Checking relevance at threshold=%.4f…", threshold)
    logger.info("LOF neighbors: %d", lof_neighbors)
    if always_include_terms:
        logger.info(
            "Always-include sanity terms active (%d): %s",
            len(always_include_terms),
            always_include_terms,
        )
    relevant: list = []

    for paper in new_papers:
        text = BaseEmbedder.paper_to_text(paper.title, paper.authors, paper.abstract)
        if not text.strip():
            continue
        matched_terms = _paper_matching_terms(paper, always_include_terms)
        forced_by_terms = bool(matched_terms)

        max_sim = 0.0
        top_matches: list[tuple[str, float]] = []
        is_relevant = False
        try:
            embedding = embedder.embed(text)
            is_relevant, max_sim, top_matches = find_relevant_papers(
                embedding,
                library_embeddings,
                threshold=threshold,
                top_k=top_k,
                lof_neighbors=lof_neighbors,
            )
            closest_col, col_score = find_closest_collection(embedding, collection_centroids)
            if is_relevant or forced_by_terms:
                relevant.append(
                    {
                        "paper_id": paper.paper_id,
                        "title": paper.title,
                        "authors": paper.authors,
                        "abstract": paper.abstract,
                        "url": paper.url,
                        "date_published": paper.date_published,
                        "date_updated": paper.date_updated,
                        "is_update": paper.is_update,
                        "max_similarity": max_sim,
                        "top_matches": [
                            (
                                paper_id_to_title.get(match_id) or match_id,
                                sim,
                            )
                            for match_id, sim in top_matches
                        ],
                        "closest_collection": closest_col,
                        "closest_collection_score": col_score,
                        "override_by_terms": bool(forced_by_terms and not is_relevant),
                        "override_terms": matched_terms if (forced_by_terms and not is_relevant) else [],
                    }
                )
                if forced_by_terms and not is_relevant:
                    logger.info(
                        "FORCED INCLUDE (terms=%s, score=%.4f, collection=%s): %s",
                        matched_terms,
                        max_sim,
                        closest_col or "—",
                        (paper.title or "")[:70],
                    )
                else:
                    logger.info(
                        "RELEVANT (%.4f, collection=%s): %s",
                        max_sim,
                        closest_col or "—",
                        (paper.title or "")[:70],
                    )
            else:
                logger.debug(
                    "not relevant (%.4f): %s", max_sim, (paper.title or "")[:70]
                )
        except Exception as exc:  # noqa: BLE001
            logger.error("Error processing paper %s: %s", paper.paper_id, exc)
            if forced_by_terms:
                # Embedding/relevance failed, but sanity terms require inclusion.
                relevant.append(
                    {
                        "paper_id": paper.paper_id,
                        "title": paper.title,
                        "authors": paper.authors,
                        "abstract": paper.abstract,
                        "url": paper.url,
                        "date_published": paper.date_published,
                        "date_updated": paper.date_updated,
                        "is_update": paper.is_update,
                        "max_similarity": 0.0,
                        "top_matches": [],
                        "override_by_terms": True,
                        "override_terms": matched_terms,
                    }
                )
                logger.warning(
                    "FORCED INCLUDE after processing error (terms=%s): %s",
                    matched_terms,
                    (paper.title or "")[:70],
                )

    if always_include_terms:
        expected_ids = {
            paper.paper_id
            for paper in new_papers
            if _paper_matching_terms(paper, always_include_terms)
        }
        included_ids = {paper["paper_id"] for paper in relevant}
        missing_ids = sorted(expected_ids - included_ids)

        if missing_ids:
            logger.error(
                "Sanity check failed: %d term-matching paper(s) missing from digest; adding them now.",
                len(missing_ids),
            )
            paper_by_id = {paper.paper_id: paper for paper in new_papers}
            for missing_id in missing_ids:
                paper = paper_by_id.get(missing_id)
                if not paper:
                    continue
                relevant.append(
                    {
                        "paper_id": paper.paper_id,
                        "title": paper.title,
                        "authors": paper.authors,
                        "abstract": paper.abstract,
                        "url": paper.url,
                        "date_published": paper.date_published,
                        "date_updated": paper.date_updated,
                        "is_update": paper.is_update,
                        "max_similarity": 0.0,
                        "top_matches": [],
                        "override_by_terms": True,
                        "override_terms": _paper_matching_terms(paper, always_include_terms),
                    }
                )
                logger.warning(
                    "Backfilled forced include due to sanity check: %s",
                    (paper.title or "")[:70],
                )
        else:
            logger.info(
                "Sanity check passed: all %d term-matching paper(s) are included.",
                len(expected_ids),
            )

    relevant.sort(
        key=lambda p: float(p.get("max_similarity", 0.0) or 0.0),
        reverse=True,
    )

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
            "LOF density-score threshold for relevance (0–1). "
            "Overrides the value in config.yaml. "
            "Lower values cast a wider net; higher values are more selective."
        ),
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load config: %s", args.config)
        raise SystemExit(1) from exc

    resilience_cfg = _resilience_config(config)
    retry_forever = bool(resilience_cfg.get("retry_until_success", False))
    max_attempts = max(1, int(resilience_cfg.get("max_attempts", 3)))
    retry_delay_seconds = max(1.0, float(resilience_cfg.get("retry_delay_seconds", 30)))
    retry_backoff = max(1.0, float(resilience_cfg.get("retry_backoff", 2.0)))
    max_retry_delay_seconds = max(
        retry_delay_seconds,
        float(resilience_cfg.get("max_retry_delay_seconds", 600)),
    )

    attempt = 0
    delay = retry_delay_seconds
    while True:
        attempt += 1
        try:
            run(config, dry_run=args.dry_run, threshold_override=args.threshold)
            if attempt > 1:
                logger.info("daily_digest recovered after %d attempt(s).", attempt)
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unhandled error in daily_digest (attempt %d).", attempt)

            email_cfg = (config or {}).get("email") or {}
            if email_cfg.get("smtp_host"):
                send_error_notification(
                    script_name="scripts/daily_digest.py",
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
                    max_attempts=int(resilience_cfg.get("error_email_max_attempts", 6)),
                    retry_delay_seconds=float(
                        resilience_cfg.get("error_email_retry_delay_seconds", 30)
                    ),
                    retry_backoff=float(resilience_cfg.get("error_email_retry_backoff", 2.0)),
                    max_retry_delay_seconds=float(
                        resilience_cfg.get("error_email_max_retry_delay_seconds", 300)
                    ),
                    fallback_log_path=str(
                        resilience_cfg.get(
                            "error_email_fallback_log_path",
                            "log/unsent_error_notifications.log",
                        )
                    ),
                )

            if not _should_retry(attempt, retry_forever, max_attempts):
                raise SystemExit(1) from exc

            max_attempts_label = "inf" if retry_forever else str(max_attempts)
            logger.warning(
                "Retrying daily_digest in %.1f seconds (attempt %d/%s)...",
                delay,
                attempt + 1,
                max_attempts_label,
            )
            time.sleep(delay)
            delay = min(max_retry_delay_seconds, delay * retry_backoff)


if __name__ == "__main__":
    main()
