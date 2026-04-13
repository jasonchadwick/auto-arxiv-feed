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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _deserialize_paper(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["authors"] = json.loads(d["authors"]) if d.get("authors") else []
        return d
