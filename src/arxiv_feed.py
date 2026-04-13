"""arXiv RSS feed parsing and full paper detail retrieval via the arXiv API."""

import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Optional

import feedparser
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARXIV_RSS_BASE = "https://rss.arxiv.org/rss/{category}"
ARXIV_API_BASE = "http://export.arxiv.org/api/query"

# Atom namespaces used by the arXiv API
_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

# arXiv ID pattern: YYMM.NNNNN (old or new format) optionally followed by vN
_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)")


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class ArxivPaper:
    paper_id: str
    title: str
    authors: List[str]
    abstract: str
    url: str
    date_published: Optional[str] = None
    categories: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_new_papers(
    categories: List[str],
    request_delay: float = 1.0,
) -> List[ArxivPaper]:
    """Fetch today's new papers from arXiv for the given *categories*.

    Steps:
    1. Parse each category's RSS feed to collect paper IDs.
    2. Query the arXiv Atom API in batches of 100 to retrieve full metadata
       (title, authors, abstract, …).  The RSS feed does not include authors,
       so this step is necessary.

    Args:
        categories: List of arXiv category identifiers, e.g. ``["quant-ph", "cs.ET"]``.
        request_delay: Seconds to sleep between HTTP requests (be a good citizen).

    Returns:
        List of :class:`ArxivPaper` objects with full metadata.
    """
    paper_ids: List[str] = []
    seen: set = set()

    for category in categories:
        logger.info("Fetching RSS feed for category: %s", category)
        ids = _fetch_rss_paper_ids(category)
        for pid in ids:
            if pid not in seen:
                seen.add(pid)
                paper_ids.append(pid)
        time.sleep(request_delay)

    logger.info(
        "Found %d unique paper IDs across %d category feed(s)",
        len(paper_ids),
        len(categories),
    )

    papers: List[ArxivPaper] = []
    batch_size = 100
    for i in range(0, len(paper_ids), batch_size):
        batch = paper_ids[i : i + batch_size]
        papers.extend(_fetch_paper_details(batch))
        if i + batch_size < len(paper_ids):
            time.sleep(request_delay)

    return papers


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fetch_rss_paper_ids(category: str) -> List[str]:
    """Return paper IDs found in the RSS feed for *category*."""
    url = ARXIV_RSS_BASE.format(category=category)
    feed = feedparser.parse(url)
    if feed.bozo:
        logger.warning(
            "RSS parser reported problems for %s: %s", category, feed.bozo_exception
        )

    ids: List[str] = []
    for entry in feed.entries:
        raw = entry.get("id") or entry.get("link") or ""
        pid = extract_arxiv_id(raw)
        if pid:
            ids.append(pid)

    logger.info("Found %d papers in %s RSS feed", len(ids), category)
    return ids


def extract_arxiv_id(url_or_id: str) -> Optional[str]:
    """Extract a normalised arXiv paper ID from a URL or raw ID string.

    Handles:
    * ``https://arxiv.org/abs/2401.12345``
    * ``http://arxiv.org/abs/2401.12345v2``
    * ``oai:arXiv.org:2401.12345``
    * ``2401.12345`` / ``2401.12345v2``
    """
    if not url_or_id:
        return None
    m = _ARXIV_ID_RE.search(url_or_id)
    return m.group(1) if m else None


def _fetch_paper_details(paper_ids: List[str]) -> List[ArxivPaper]:
    """Retrieve full metadata for *paper_ids* via the arXiv Atom API."""
    id_list = ",".join(paper_ids)
    params = {"id_list": id_list, "max_results": len(paper_ids)}

    try:
        resp = requests.get(ARXIV_API_BASE, params=params, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.error("Error fetching paper details from arXiv API: %s", exc)
        return []

    return _parse_api_response(resp.text)


def _parse_api_response(xml_text: str) -> List[ArxivPaper]:
    """Parse the Atom XML returned by the arXiv API."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.error("Failed to parse arXiv API XML: %s", exc)
        return []

    papers: List[ArxivPaper] = []
    for entry in root.findall("atom:entry", _NS):
        try:
            paper = _parse_entry(entry)
            if paper is not None:
                papers.append(paper)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error parsing arXiv API entry: %s", exc)

    return papers


def _parse_entry(entry: ET.Element) -> Optional[ArxivPaper]:
    """Parse a single Atom ``<entry>`` element into an :class:`ArxivPaper`."""
    id_elem = entry.find("atom:id", _NS)
    if id_elem is None or not id_elem.text:
        return None

    paper_id = extract_arxiv_id(id_elem.text.strip())
    if not paper_id:
        return None

    title_elem = entry.find("atom:title", _NS)
    title = (
        " ".join((title_elem.text or "").split())  # normalise whitespace
        if title_elem is not None
        else ""
    )

    authors: List[str] = []
    for author_elem in entry.findall("atom:author", _NS):
        name_elem = author_elem.find("atom:name", _NS)
        if name_elem is not None and name_elem.text:
            authors.append(name_elem.text.strip())

    summary_elem = entry.find("atom:summary", _NS)
    abstract = (
        " ".join((summary_elem.text or "").split())
        if summary_elem is not None
        else ""
    )

    published_elem = entry.find("atom:published", _NS)
    date_published = (
        published_elem.text.strip() if published_elem is not None else None
    )

    categories: List[str] = [
        cat.get("term", "") for cat in entry.findall("atom:category", _NS)
        if cat.get("term")
    ]

    return ArxivPaper(
        paper_id=paper_id,
        title=title,
        authors=authors,
        abstract=abstract,
        url=f"https://arxiv.org/abs/{paper_id}",
        date_published=date_published,
        categories=categories,
    )
