"""Helpers for reading article versions from the SQLite snapshots."""

from __future__ import annotations

import gzip
import os
import shutil
import sqlite3
import tempfile
from contextlib import contextmanager
from typing import Dict, Generator, Iterable, List, Tuple


@contextmanager
def _resolved_db_path(db_path: str) -> Iterable[str]:
    """Yield a usable SQLite path, expanding `.db.gz` files into a temp copy."""
    if db_path.endswith(".gz"):
        with gzip.open(db_path, "rb") as source:
            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
                shutil.copyfileobj(source, tmp)
                tmp_path = tmp.name
        try:
            yield tmp_path
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    else:
        yield db_path


def iter_article_counts(db_path: str) -> Generator[Tuple[int, int], None, None]:
    """Yield (entry_id, version_count) pairs for the input SQLite database."""
    with _resolved_db_path(db_path) as resolved:
        conn = sqlite3.connect(resolved)
        try:
            cursor = conn.execute(
                """
                SELECT entry_id, COUNT(*) AS version_count
                FROM entryversion
                GROUP BY entry_id
                ORDER BY entry_id
                """
            )
            for row in cursor:
                yield int(row[0]), int(row[1])
        finally:
            conn.close()


def count_articles(db_path: str) -> int:
    """Return the total number of distinct entry_ids in the database."""
    with _resolved_db_path(db_path) as resolved:
        conn = sqlite3.connect(resolved)
        try:
            cursor = conn.execute("SELECT COUNT(DISTINCT entry_id) FROM entryversion")
            row = cursor.fetchone()
            return int(row[0] or 0)
        finally:
            conn.close()


def load_versions(db_path: str, entry_id: int) -> List[Dict]:
    """Load and order every version for the requested article entry_id."""
    with _resolved_db_path(db_path) as resolved:
        conn = sqlite3.connect(resolved)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                """
                SELECT id, entry_id, version, created, title, url, num_versions, summary
                FROM entryversion
                WHERE entry_id = ?
                ORDER BY version ASC
                """,
                (entry_id,),
            )
            versions = []
            for row in cursor:
                versions.append({
                    "id": row["id"],
                    # "joint_key": row["joint_key"],
                    "source": db_path.split('/')[-1].replace('.db.gz', '').replace('newssniffer', ''),
                    "entry_id": row["entry_id"],
                    "version": row["version"],
                    "created": row["created"],
                    "title": row["title"],
                    "url": row["url"],
                    "num_versions": row["num_versions"],
                    "summary": row["summary"],
                })
            return versions
        finally:
            conn.close()
