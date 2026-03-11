"""Helpers for reading article versions from the SQLite snapshots."""

from __future__ import annotations

import atexit
import gzip
import os
from pathlib import Path
import shutil
import sqlite3
import threading
from contextlib import contextmanager
from typing import Dict, Generator, Iterable, List, Optional, Tuple


_RESOLVED_DB_CACHE: Dict[str, str] = {}
_RESOLVED_DB_CACHE_LOCK = threading.Lock()
_UNZIPPED_CACHE_DIR = Path(__file__).resolve().parents[1] / "unzipped-article-versions"


def _has_entryversion_table(db_path: Path) -> bool:
    if not db_path.exists():
        return False
    try:
        with sqlite3.connect(str(db_path)) as conn:
            row = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='entryversion' LIMIT 1"
            ).fetchone()
        return row is not None
    except sqlite3.Error:
        return False


def _cleanup_resolved_db_cache() -> None:
    with _RESOLVED_DB_CACHE_LOCK:
        cached_paths = list(_RESOLVED_DB_CACHE.values())
        _RESOLVED_DB_CACHE.clear()
    for path in cached_paths:
        path_obj = Path(path)
        # Keep repo-local unzipped copies; clean only legacy temp artifacts.
        if path_obj.parent == _UNZIPPED_CACHE_DIR:
            continue
        try:
            os.remove(path)
        except OSError:
            pass


atexit.register(_cleanup_resolved_db_cache)


@contextmanager
def _resolved_db_path(db_path: str) -> Iterable[str]:
    """Yield a usable SQLite path, expanding `.db.gz` files into a repo-local copy."""
    if db_path.endswith(".gz"):
        with _RESOLVED_DB_CACHE_LOCK:
            cached_path = _RESOLVED_DB_CACHE.get(db_path)
        if cached_path and os.path.exists(cached_path):
            yield cached_path
            return
        with _RESOLVED_DB_CACHE_LOCK:
            cached_path = _RESOLVED_DB_CACHE.get(db_path)
            if cached_path and os.path.exists(cached_path):
                yield cached_path
                return
            source_path = Path(db_path)
            _UNZIPPED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            target_path = _UNZIPPED_CACHE_DIR / source_path.stem
            needs_refresh = (
                (not target_path.exists())
                or (target_path.stat().st_mtime < source_path.stat().st_mtime)
                or (not _has_entryversion_table(target_path))
            )
            if needs_refresh:
                with gzip.open(db_path, "rb") as source:
                    with target_path.open("wb") as dst:
                        shutil.copyfileobj(source, dst)
                if not _has_entryversion_table(target_path):
                    target_path.unlink(missing_ok=True)
                    raise sqlite3.OperationalError(
                        f"Materialized DB is invalid (missing entryversion): {target_path}"
                    )
            tmp_path = str(target_path)
            _RESOLVED_DB_CACHE[db_path] = tmp_path
        yield tmp_path
    else:
        yield db_path


def source_name_from_db_path(db_path: str) -> str:
    name = Path(db_path).name
    if name.endswith(".gz"):
        name = name[:-3]
    if name.endswith(".db"):
        name = name[:-3]
    return name


def iter_article_counts(db_path: str) -> Generator[Tuple[int, int, Optional[str]], None, None]:
    """Yield (entry_id, version_count, first_created) rows for the SQLite database."""
    with _resolved_db_path(db_path) as resolved:
        conn = sqlite3.connect(resolved)
        try:
            cursor = conn.execute(
                """
                SELECT entry_id, COUNT(*) AS version_count, MIN(created) AS first_created
                FROM entryversion
                GROUP BY entry_id
                ORDER BY entry_id
                """
            )
            for row in cursor:
                yield int(row[0]), int(row[1]), row[2]
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
            source_name = source_name_from_db_path(db_path)
            for row in cursor:
                versions.append({
                    "id": row["id"],
                    # "joint_key": row["joint_key"],
                    "source": source_name,
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
