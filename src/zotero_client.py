"""Zotero API client – fetch papers and their metadata."""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class ZoteroPaper:
    item_key: str
    title: str
    authors: List[str]
    abstract: str
    url: str
    doi: Optional[str] = None
    date_published: Optional[str] = None
    date_added: Optional[str] = None


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class ZoteroClient:
    """Thin wrapper around the ``pyzotero`` library."""

    #: Zotero item types treated as "papers" by default.
    DEFAULT_ITEM_TYPES = ("journalArticle", "conferencePaper", "preprint")

    def __init__(self, user_id: str, api_key: str, library_type: str = "user"):
        from pyzotero import zotero  # lazy import

        self.library = zotero.Zotero(user_id, library_type, api_key)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def get_all_papers(
        self, item_types: Optional[List[str]] = None
    ) -> List[ZoteroPaper]:
        """Fetch all papers from the Zotero library.

        Args:
            item_types: Zotero item types to include.  Defaults to
                ``['journalArticle', 'conferencePaper', 'preprint']``.

        Returns:
            List of :class:`ZoteroPaper` objects.
        """
        if item_types is None:
            item_types = list(self.DEFAULT_ITEM_TYPES)

        all_items: list = []
        for itype in item_types:
            logger.info("Fetching %s items from Zotero…", itype)
            items = self.library.everything(self.library.items(itemType=itype))
            all_items.extend(items)
            logger.info("  → %d items fetched", len(items))

        papers: List[ZoteroPaper] = []
        for item in all_items:
            paper = self._parse_item(item)
            if paper is not None:
                papers.append(paper)

        logger.info("Total Zotero papers: %d", len(papers))
        return papers

    def get_library_version(self) -> int:
        """Return the current Zotero library version number."""
        # Trigger a lightweight request so pyzotero updates its internal version.
        self.library.items(limit=1)
        return self.library.last_modified_version()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_item(self, item: dict) -> Optional[ZoteroPaper]:
        data = item.get("data", {})

        # Skip non-paper item types
        if data.get("itemType") in ("attachment", "note", "annotation"):
            return None

        item_key: str = item.get("key") or data.get("key", "")
        if not item_key:
            return None

        title: str = data.get("title", "")
        abstract: str = data.get("abstractNote", "")

        # Build author list
        authors: List[str] = []
        for creator in data.get("creators", []):
            if creator.get("creatorType") not in ("author", "editor"):
                continue
            first = creator.get("firstName", "")
            last = creator.get("lastName", "")
            name = creator.get("name", "")  # single-field name
            if first or last:
                authors.append(f"{first} {last}".strip())
            elif name:
                authors.append(name)

        doi: str = data.get("DOI", "")
        url: str = data.get("url", "")
        if not url and doi:
            url = f"https://doi.org/{doi}"

        date_published: str = data.get("date", "")
        date_added: str = item.get("meta", {}).get(
            "dateAdded", data.get("dateAdded", "")
        )

        return ZoteroPaper(
            item_key=item_key,
            title=title,
            authors=authors,
            abstract=abstract,
            url=url,
            doi=doi or None,
            date_published=date_published or None,
            date_added=date_added or None,
        )
