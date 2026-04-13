"""SQLite database for storing papers and their embeddings."""

import json
import sqlite3
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class PaperDatabase:
    """SQLite database for storing papers and their embeddings."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialize database schema."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS papers (
                    id          TEXT PRIMARY KEY,
                    source      TEXT NOT NULL,
                    title       TEXT,
                    authors     TEXT,
                    abstract    TEXT,
                    url         TEXT,
                    date_added  TEXT,
                    date_published TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    paper_id        TEXT NOT NULL,
                    model           TEXT NOT NULL,
                    embedding       TEXT NOT NULL,
                    date_embedded   TEXT NOT NULL,
                    PRIMARY KEY (paper_id, model),
                    FOREIGN KEY (paper_id) REFERENCES papers(id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS zotero_collections (
                    key         TEXT PRIMARY KEY,
                    name        TEXT NOT NULL,
                    parent_key  TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS paper_collections (
                    paper_id        TEXT NOT NULL,
                    collection_key  TEXT NOT NULL,
                    PRIMARY KEY (paper_id, collection_key)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS collection_centroids (
                    collection_path TEXT NOT NULL,
                    model           TEXT NOT NULL,
                    centroid        TEXT NOT NULL,
                    updated_at      TEXT NOT NULL,
                    PRIMARY KEY (collection_path, model)
                )
            """)
            conn.commit()

    # ------------------------------------------------------------------
    # Paper CRUD
    # ------------------------------------------------------------------

    def upsert_paper(
        self,
        paper_id: str,
        source: str,
        title: str = None,
        authors: List[str] = None,
        abstract: str = None,
        url: str = None,
        date_added: str = None,
        date_published: str = None,
    ):
        """Insert or update a paper record."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO papers
                    (id, source, title, authors, abstract, url, date_added, date_published)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    paper_id,
                    source,
                    title,
                    json.dumps(authors or []),
                    abstract,
                    url,
                    date_added,
                    date_published,
                ),
            )
            conn.commit()

    def get_paper(self, paper_id: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM papers WHERE id = ?", (paper_id,)
            ).fetchone()
            return self._deserialize_paper(row) if row else None

    def get_papers_by_source(self, source: str) -> List[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM papers WHERE source = ?", (source,)
            ).fetchall()
            return [self._deserialize_paper(row) for row in rows]

    def get_papers_without_embedding(self, source: str, model: str) -> List[dict]:
        """Return papers from *source* that do not yet have an embedding for *model*."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT p.* FROM papers p
                WHERE p.source = ?
                  AND NOT EXISTS (
                      SELECT 1 FROM embeddings e
                      WHERE e.paper_id = p.id AND e.model = ?
                  )
                """,
                (source, model),
            ).fetchall()
            return [self._deserialize_paper(row) for row in rows]

    # ------------------------------------------------------------------
    # Embedding CRUD
    # ------------------------------------------------------------------

    def store_embedding(self, paper_id: str, model: str, embedding: List[float]):
        """Store or overwrite an embedding for a paper."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO embeddings (paper_id, model, embedding, date_embedded)
                VALUES (?, ?, ?, ?)
                """,
                (
                    paper_id,
                    model,
                    json.dumps(embedding),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()

    def get_embedding(self, paper_id: str, model: str) -> Optional[List[float]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT embedding FROM embeddings WHERE paper_id = ? AND model = ?",
                (paper_id, model),
            ).fetchone()
            return json.loads(row["embedding"]) if row else None

    def has_embedding(self, paper_id: str, model: str) -> bool:
        return self.get_embedding(paper_id, model) is not None

    def get_all_embeddings_for_source(
        self, source: str, model: str
    ) -> List[Tuple[str, List[float]]]:
        """Return [(paper_id, embedding), ...] for all papers from *source* with *model*."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT e.paper_id, e.embedding
                FROM embeddings e
                JOIN papers p ON e.paper_id = p.id
                WHERE p.source = ? AND e.model = ?
                """,
                (source, model),
            ).fetchall()
            return [(row["paper_id"], json.loads(row["embedding"])) for row in rows]

    def get_all_embeddings_with_titles_for_source(
        self, source: str, model: str
    ) -> List[Tuple[str, str, List[float]]]:
        """Return [(paper_id, title, embedding), ...] for papers from *source* with *model*."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT e.paper_id, p.title, e.embedding
                FROM embeddings e
                JOIN papers p ON e.paper_id = p.id
                WHERE p.source = ? AND e.model = ?
                """,
                (source, model),
            ).fetchall()
            return [
                (row["paper_id"], row["title"] or "", json.loads(row["embedding"]))
                for row in rows
            ]

    # ------------------------------------------------------------------
    # Collection CRUD
    # ------------------------------------------------------------------

    def replace_all_collections(self, collections: List[dict]) -> None:
        """Atomically replace all stored Zotero collection records.

        Each element of *collections* should have ``key``, ``name``, and
        optionally ``parent_key``.
        """
        with self._connect() as conn:
            conn.execute("DELETE FROM zotero_collections")
            for col in collections:
                conn.execute(
                    "INSERT INTO zotero_collections (key, name, parent_key) VALUES (?, ?, ?)",
                    (col["key"], col["name"], col.get("parent_key")),
                )
            conn.commit()

    def get_all_collections(self) -> List[dict]:
        """Return [{key, name, parent_key}, ...] for all stored collections."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT key, name, parent_key FROM zotero_collections"
            ).fetchall()
            return [{"key": r["key"], "name": r["name"], "parent_key": r["parent_key"]} for r in rows]

    def set_paper_collections(self, paper_id: str, collection_keys: List[str]) -> None:
        """Set which collections *paper_id* belongs to (replaces prior assignments)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM paper_collections WHERE paper_id = ?", (paper_id,))
            for key in collection_keys:
                conn.execute(
                    "INSERT OR IGNORE INTO paper_collections (paper_id, collection_key) VALUES (?, ?)",
                    (paper_id, key),
                )
            conn.commit()

    def get_paper_ids_in_collection(self, collection_key: str) -> List[str]:
        """Return all paper_ids directly assigned to *collection_key*."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT paper_id FROM paper_collections WHERE collection_key = ?",
                (collection_key,),
            ).fetchall()
            return [r["paper_id"] for r in rows]

    def get_all_paper_collection_keys(self) -> dict:
        """Return ``{paper_id: [collection_key, ...]}`` for every paper that has
        at least one collection assignment."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT paper_id, collection_key FROM paper_collections"
            ).fetchall()
        result: dict = {}
        for r in rows:
            result.setdefault(r["paper_id"], []).append(r["collection_key"])
        return result

    def upsert_collection_centroid(
        self, collection_path: str, model: str, centroid: List[float]
    ) -> None:
        """Store or overwrite the centroid for a collection path and model."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO collection_centroids
                    (collection_path, model, centroid, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    collection_path,
                    model,
                    json.dumps(centroid),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()

    def delete_all_collection_centroids_for_model(self, model: str) -> None:
        """Remove all stored collection centroids for *model* (before recomputing)."""
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM collection_centroids WHERE model = ?", (model,)
            )
            conn.commit()

    def get_collection_centroids(
        self, model: str
    ) -> List[Tuple[str, List[float]]]:
        """Return [(collection_path, centroid), ...] for all stored centroids."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT collection_path, centroid FROM collection_centroids WHERE model = ?",
                (model,),
            ).fetchall()
            return [(r["collection_path"], json.loads(r["centroid"])) for r in rows]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def build_collection_paths(collections: List[dict]) -> dict:
        """Build a ``{key: path_string}`` map from a list of collection dicts.

        Paths are formatted as ``"Grandparent/Parent/Child/"``.  Cycles and
        missing parent references are handled gracefully.
        """
        by_key = {c["key"]: c for c in collections}
        cache: dict = {}

        def path_for(key: str, visiting: frozenset = frozenset()) -> str:
            if key in cache:
                return cache[key]
            if key not in by_key or key in visiting:
                return ""
            col = by_key[key]
            name = (col["name"] or key).strip()
            parent_key = col.get("parent_key")
            if parent_key:
                parent_path = path_for(parent_key, visiting | {key})
                result = f"{parent_path}{name}/"
            else:
                result = f"{name}/"
            cache[key] = result
            return result

        return {c["key"]: path_for(c["key"]) for c in collections}

    @staticmethod
    def _deserialize_paper(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["authors"] = json.loads(d["authors"]) if d.get("authors") else []
        return d
