#!/usr/bin/env python3
"""Clean invalid/stale files in unzipped-article-versions.

Rules:
- Invalid DB: file cannot be opened as SQLite or lacks `entryversion` table.
- Stale DB: matching source `.db.gz` exists in `article-versions/` and is newer.

Usage:
  python scripts/cleanup_unzipped_versions.py            # dry run
  python scripts/cleanup_unzipped_versions.py --apply    # delete matches
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import List, Tuple


ROOT = Path(__file__).resolve().parents[1]
UNZIPPED_DIR = ROOT / "unzipped-article-versions"
SOURCE_DIR = ROOT / "article-versions"


def _is_sqlite_with_entryversion(path: Path) -> bool:
    try:
        with sqlite3.connect(str(path)) as conn:
            row = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='entryversion' LIMIT 1"
            ).fetchone()
        return row is not None
    except sqlite3.Error:
        return False


def _source_gz_for_unzipped(path: Path) -> Path:
    # newssniffer-guardian.db -> newssniffer-guardian.db.gz
    # newssniffer-guardian.db.db -> newssniffer-guardian.db.gz
    name = path.name
    if name.endswith(".db.db"):
        stem = name[:-3]
    else:
        stem = name
    return SOURCE_DIR / f"{stem}.gz"


def classify(path: Path) -> Tuple[bool, str]:
    if path.suffix != ".db":
        return False, "skip: not .db"

    if not _is_sqlite_with_entryversion(path):
        return True, "invalid: missing/broken SQLite or no entryversion"

    source_gz = _source_gz_for_unzipped(path)
    if source_gz.exists() and path.stat().st_mtime < source_gz.stat().st_mtime:
        return True, f"stale: older than source {source_gz.name}"

    return False, "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean invalid/stale unzipped article DB files")
    parser.add_argument("--apply", action="store_true", help="Delete matched files (default is dry-run)")
    args = parser.parse_args()

    if not UNZIPPED_DIR.exists():
        print(f"No directory found: {UNZIPPED_DIR}")
        return 0

    files = sorted([p for p in UNZIPPED_DIR.iterdir() if p.is_file()])
    to_delete: List[Tuple[Path, str]] = []
    kept = 0

    for path in files:
        should_delete, reason = classify(path)
        if should_delete:
            to_delete.append((path, reason))
        else:
            kept += 1

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] scanned={len(files)} keep={kept} delete={len(to_delete)}")
    for path, reason in to_delete:
        print(f" - {path.name}: {reason}")

    if args.apply:
        deleted = 0
        for path, _reason in to_delete:
            try:
                path.unlink(missing_ok=True)
                deleted += 1
            except OSError as exc:
                print(f" ! failed to delete {path.name}: {exc}")
        print(f"[APPLY] deleted={deleted}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

