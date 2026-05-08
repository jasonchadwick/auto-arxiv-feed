#!/usr/bin/env python3
"""Send a one-off digest for arXiv links listed in a text file.

The input file may contain raw arXiv IDs, arXiv URLs, or arbitrary non-arXiv
links. Non-arXiv lines are skipped and reported. Duplicate arXiv IDs are
deduplicated while preserving their first-seen order.

Usage::

    python scripts/send_links_digest.py --config config.yaml --links-file links.txt
    python scripts/send_links_digest.py --config config.yaml --links-file links.txt --dry-run
"""

import argparse
import logging
import os
import re
import sys
import traceback
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from src.arxiv_feed import _fetch_paper_details, extract_arxiv_id
from src.database import PaperDatabase
from src.email_digest import send_digest, send_error_notification
from src.embeddings import BaseEmbedder, get_embedder
from src.relevance import find_closest_collection, find_relevant_papers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ParsedLinks:
    arxiv_ids: list[str]
    skipped_lines: list[tuple[int, str]]


def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def _api_key_for(emb_cfg: dict) -> Optional[str]:
    provider = emb_cfg.get("provider", "").lower()
    return emb_cfg.get(f"{provider}_api_key") or None


def _normalize_terms(raw_terms: object) -> list[str]:
    if not isinstance(raw_terms, list):
        return []
    return [str(term).strip().lower() for term in raw_terms if str(term).strip()]


def _paper_matching_terms(paper: object, terms: list[str]) -> list[str]:
    if not terms:
        return []

    title = getattr(paper, "title", "") or ""
    abstract = getattr(paper, "abstract", "") or ""
    haystack = f"{title}\n{abstract}".lower()
    return [term for term in terms if term in haystack]


def _build_excluded_collection_keys(all_collections: list, excluded_names: list) -> set:
    if not excluded_names:
        return set()

    normalized = {name.lower().strip() for name in excluded_names}
    by_key = {collection["key"]: collection for collection in all_collections}
    cache: dict = {}

    def is_excluded(key: str, visiting: frozenset = frozenset()) -> bool:
        if key in cache:
            return cache[key]
        if key not in by_key or key in visiting:
            return False
        collection = by_key[key]
        if (collection["name"] or "").lower().strip() in normalized:
            cache[key] = True
            return True
        parent = collection.get("parent_key")
        result = bool(parent and is_excluded(parent, visiting | {key}))
        cache[key] = result
        return result

    return {collection["key"] for collection in all_collections if is_excluded(collection["key"])}


def _paper_fully_excluded(paper_id: str, paper_col_map: dict, excluded_keys: set) -> bool:
    col_keys = paper_col_map.get(paper_id, [])
    if not col_keys:
        return False
    return all(key in excluded_keys for key in col_keys)


def canonicalize_arxiv_id(arxiv_id: str) -> str:
    if not arxiv_id:
        return ""
    return re.sub(r"v\d+$", "", arxiv_id)


def extract_supported_arxiv_id(value: str) -> Optional[str]:
    normalized = value.strip()
    if not normalized:
        return None

    parsed = urlparse(normalized)
    if parsed.scheme or parsed.netloc:
        hostname = (parsed.netloc or parsed.path).lower()
        if hostname.startswith("www."):
            hostname = hostname[4:]
        if not hostname.endswith("arxiv.org"):
            return None

        extracted = extract_arxiv_id(normalized)
        return canonicalize_arxiv_id(extracted) if extracted else None

    extracted = extract_arxiv_id(normalized)
    return canonicalize_arxiv_id(extracted) if extracted else None


def parse_links_file(path: str) -> ParsedLinks:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Links file not found: {path}")

    arxiv_ids: list[str] = []
    skipped_lines: list[tuple[int, str]] = []
    seen_ids: set[str] = set()

    with open(path) as fh:
        for line_number, raw_line in enumerate(fh, start=1):
            value = raw_line.strip()

            if not value or value.startswith("#"):
                continue

            arxiv_id = extract_supported_arxiv_id(value)
            if not arxiv_id:
                skipped_lines.append((line_number, value))
                continue

            if arxiv_id in seen_ids:
                continue

            seen_ids.add(arxiv_id)
            arxiv_ids.append(arxiv_id)

    return ParsedLinks(arxiv_ids=arxiv_ids, skipped_lines=skipped_lines)


def fetch_papers_for_ids(arxiv_ids: list[str]) -> list[object]:
    if not arxiv_ids:
        return []

    papers: list[object] = []
    batch_size = 100

    for start in range(0, len(arxiv_ids), batch_size):
        batch = arxiv_ids[start : start + batch_size]
        logger.info(
            "Fetching arXiv metadata for batch %d-%d of %d.",
            start + 1,
            min(start + len(batch), len(arxiv_ids)),
            len(arxiv_ids),
        )
        batch_papers = _fetch_paper_details(batch)
        by_id = {
            canonicalize_arxiv_id(paper.paper_id): paper for paper in batch_papers
        }

        for arxiv_id in batch:
            paper = by_id.get(arxiv_id)
            if not paper:
                logger.warning("arXiv metadata not found for %s", arxiv_id)
                continue

            papers.append(paper)

    return papers


def filter_relevant_papers(config: dict, papers: list[object]) -> list[dict]:
    if not papers:
        return []

    db = PaperDatabase(config["database"]["path"])

    emb_cfg = config["embedding"]
    embedder = get_embedder(
        provider=emb_cfg["provider"],
        model=emb_cfg["model"],
        api_key=_api_key_for(emb_cfg),
    )

    rel_cfg = config.get("relevance", {})
    threshold = rel_cfg.get("threshold", 0.5)
    top_k: int = rel_cfg.get("top_k", 5)
    lof_neighbors: int = int(rel_cfg.get("lof_neighbors", 20))
    always_include_terms = _normalize_terms(rel_cfg.get("always_include_terms", []))
    excluded_collection_names = _normalize_terms(rel_cfg.get("excluded_collections", []))

    logger.info("Loading Zotero library embeddings (model: %s)...", embedder.model_id)
    library_rows = db.get_all_embeddings_with_titles_for_source("zotero", embedder.model_id)
    logger.info("Loaded %d Zotero embedding(s).", len(library_rows))

    all_collections: list = []
    excluded_keys: set = set()
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
            "No Zotero embeddings found in the database. Run scripts/update_zotero.py first."
        )
        return []

    collection_centroids = db.get_collection_centroids(embedder.model_id)
    if excluded_keys and all_collections:
        path_by_key = PaperDatabase.build_collection_paths(all_collections)
        excluded_paths = {path_by_key[key] for key in excluded_keys if key in path_by_key}
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

    logger.info("Checking relevance at threshold=%.4f...", threshold)
    logger.info("LOF neighbors: %d", lof_neighbors)
    if always_include_terms:
        logger.info(
            "Always-include sanity terms active (%d): %s",
            len(always_include_terms),
            always_include_terms,
        )

    relevant: list[dict] = []
    for paper in papers:
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
        except Exception as exc:  # noqa: BLE001
            logger.error("Error processing paper %s: %s", paper.paper_id, exc)
            if forced_by_terms:
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

    if always_include_terms:
        expected_ids = {
            paper.paper_id
            for paper in papers
            if _paper_matching_terms(paper, always_include_terms)
        }
        included_ids = {paper["paper_id"] for paper in relevant}
        missing_ids = sorted(expected_ids - included_ids)
        if missing_ids:
            logger.error(
                "Sanity check failed: %d term-matching paper(s) missing from digest; adding them now.",
                len(missing_ids),
            )
            paper_by_id = {paper.paper_id: paper for paper in papers}
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

    relevant.sort(
        key=lambda paper: float(paper.get("max_similarity", 0.0) or 0.0),
        reverse=True,
    )

    filtered_out = len(papers) - len(relevant)
    logger.info("%d/%d paper(s) are relevant at threshold=%.4f.", len(relevant), len(papers), threshold)
    logger.info("Filtered out %d/%d fetched arXiv paper(s) by relevance screening.", filtered_out, len(papers))
    return relevant


def print_skip_summary(skipped_lines: list[tuple[int, str]]) -> None:
    if not skipped_lines:
        return

    logger.warning("Skipped %d non-arXiv line(s) from links file.", len(skipped_lines))
    preview_count = min(10, len(skipped_lines))
    for line_number, value in skipped_lines[:preview_count]:
        logger.warning("Skipped line %d: %s", line_number, value)
    if len(skipped_lines) > preview_count:
        logger.warning("... plus %d more skipped line(s).", len(skipped_lines) - preview_count)


def run(config: dict, links_file: str, dry_run: bool = False) -> None:
    parsed = parse_links_file(links_file)
    print_skip_summary(parsed.skipped_lines)

    if not parsed.arxiv_ids:
        logger.info("No arXiv IDs found in %s. Nothing to send.", links_file)
        return

    logger.info("Found %d unique arXiv link(s).", len(parsed.arxiv_ids))
    papers = fetch_papers_for_ids(parsed.arxiv_ids)

    if not papers:
        logger.warning("No arXiv metadata could be fetched. Nothing to send.")
        return

    relevant_papers = filter_relevant_papers(config, papers)
    if not relevant_papers:
        logger.info("No relevant papers found after applying the normal digest filter.")
        return

    email_cfg = config.get("email") or {}
    smtp_host = email_cfg.get("smtp_host", "")

    if not smtp_host:
        logger.warning("No email configuration found; printing digest instead.")
        send_digest(
            relevant_papers=relevant_papers,
            smtp_host="",
            smtp_port=int(email_cfg.get("smtp_port", 587)),
            smtp_user=email_cfg.get("smtp_user", ""),
            smtp_password="",
            from_address=email_cfg.get("from_address", ""),
            to_address=email_cfg.get("to_address", ""),
            subject_prefix=email_cfg.get("subject_prefix", "[arXiv Feed]"),
            dry_run=True,
        )
        return

    send_digest(
        relevant_papers=relevant_papers,
        smtp_host=smtp_host,
        smtp_port=int(email_cfg.get("smtp_port", 587)),
        smtp_user=email_cfg.get("smtp_user", ""),
        smtp_password=email_cfg.get("smtp_password") or os.environ.get("EMAIL_PASSWORD", ""),
        from_address=email_cfg.get("from_address", ""),
        to_address=email_cfg.get("to_address", ""),
        subject_prefix=email_cfg.get("subject_prefix", "[arXiv Feed]"),
        dry_run=dry_run,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send a one-off email digest from a list of arXiv links."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration YAML file (default: config.yaml).",
    )
    parser.add_argument(
        "--links-file",
        default="links.txt",
        help="Path to a text file containing URLs or arXiv IDs (default: links.txt).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the email that would be sent instead of sending it.",
    )
    args = parser.parse_args()

    config: Optional[dict] = None
    try:
        config = load_config(args.config)
        run(config=config, links_file=args.links_file, dry_run=args.dry_run)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled error in send_links_digest.")

        if config is None:
            try:
                config = load_config(args.config)
            except Exception:  # noqa: BLE001
                config = None

        email_cfg = (config or {}).get("email") or {}
        if email_cfg.get("smtp_host"):
            send_error_notification(
                script_name="scripts/send_links_digest.py",
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