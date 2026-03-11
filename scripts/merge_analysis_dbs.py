#!/usr/bin/env python3
"""Merge two analysis.db files (SQLite) additively.

Usage:
    python scripts/merge_analysis_dbs.py <target.db> <other.db>

Merges rows from <other.db> into <target.db> without losing any local data:
  - Tables WITH a PRIMARY KEY: INSERT OR IGNORE (local wins on conflicts).
  - Tables WITHOUT a PK: inserts rows from other only for (article_id, news_org)
    pairs that have NO rows in the target table for that table.  This avoids
    duplicates while filling in data that only exists in the other DB.

Typical use: you ran --matt-features-only on sk3 (which populates
pair_edit_actions), and have a local DB with everything else.  This script
adds the missing pair_edit_actions rows from the remote DB into your local one.

Pass --prefer-other to reverse the conflict resolution: remote wins on PK
conflicts, and for non-PK tables overlapping articles are replaced with
remote data.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path


def get_tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    return [r[0] for r in rows]


def get_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info([{table}])").fetchall()
    return [r[1] for r in rows]


def get_pk_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info([{table}])").fetchall()
    # pk field is index 5; nonzero means part of the primary key
    return [r[1] for r in rows if r[5] > 0]


def merge(target_path: str, other_path: str, prefer_other: bool = False) -> None:
    target = sqlite3.connect(target_path)
    target.execute("PRAGMA journal_mode=WAL")
    target.execute("PRAGMA cache_size=-64000")  # 64 MB cache
    target.execute(f"ATTACH DATABASE ? AS other", (other_path,))

    tables = get_tables(target)
    other_tables = [
        r[0]
        for r in target.execute(
            "SELECT name FROM other.sqlite_master WHERE type='table'"
        ).fetchall()
    ]

    for table in other_tables:
        if table not in tables:
            # Table only in other — create it and copy everything.
            create_sql = target.execute(
                "SELECT sql FROM other.sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()[0]
            target.execute(create_sql)
            target.execute(f"INSERT INTO main.[{table}] SELECT * FROM other.[{table}]")
            cnt = target.execute(f"SELECT changes()").fetchone()[0]
            print(f"    {table}: created, copied {cnt} rows")
            continue

        cols = get_columns(target, table)
        pk_cols = get_pk_columns(target, table)
        col_list = ", ".join(f"[{c}]" for c in cols)

        if pk_cols:
            # Table has a PRIMARY KEY.
            if prefer_other:
                target.execute(
                    f"INSERT OR REPLACE INTO main.[{table}] "
                    f"SELECT {col_list} FROM other.[{table}]"
                )
            else:
                target.execute(
                    f"INSERT OR IGNORE INTO main.[{table}] "
                    f"SELECT {col_list} FROM other.[{table}]"
                )
            cnt = target.execute("SELECT changes()").fetchone()[0]
            print(f"    {table}: {cnt} rows added (PK, {'other wins' if prefer_other else 'local wins'})")
        else:
            # No PK — merge at the article level using a temp table of
            # "new" article IDs to avoid slow per-row NOT EXISTS scans.
            has_article_key = "article_id" in cols and "news_org" in cols
            if not has_article_key:
                target.execute(
                    f"INSERT INTO main.[{table}] SELECT {col_list} FROM other.[{table}]"
                )
                cnt = target.execute("SELECT changes()").fetchone()[0]
                print(f"    {table}: {cnt} rows added (no key, bulk copy)")
                continue

            if prefer_other:
                # Delete local rows for articles that exist in other, then copy.
                target.execute(
                    f"CREATE TEMP TABLE _ids AS "
                    f"SELECT DISTINCT article_id, news_org FROM other.[{table}]"
                )
                target.execute(
                    f"DELETE FROM main.[{table}] WHERE EXISTS ("
                    f"  SELECT 1 FROM _ids WHERE _ids.article_id = main.[{table}].article_id "
                    f"  AND _ids.news_org = main.[{table}].news_org)"
                )
                deleted = target.execute("SELECT changes()").fetchone()[0]
                target.execute(
                    f"INSERT INTO main.[{table}] SELECT {col_list} FROM other.[{table}]"
                )
                added = target.execute("SELECT changes()").fetchone()[0]
                target.execute("DROP TABLE _ids")
                print(f"    {table}: replaced {deleted} local rows with {added} from other")
            else:
                # Build set of (article_id, news_org) in other but NOT in main.
                target.execute(
                    f"CREATE TEMP TABLE _new_ids AS "
                    f"SELECT DISTINCT article_id, news_org FROM other.[{table}] "
                    f"EXCEPT "
                    f"SELECT DISTINCT article_id, news_org FROM main.[{table}]"
                )
                target.execute(
                    "CREATE INDEX _new_ids_idx ON _new_ids(article_id, news_org)"
                )
                target.execute(
                    f"INSERT INTO main.[{table}] "
                    f"SELECT {col_list} FROM other.[{table}] o "
                    f"WHERE EXISTS ("
                    f"  SELECT 1 FROM _new_ids n "
                    f"  WHERE n.article_id = o.article_id AND n.news_org = o.news_org)"
                )
                cnt = target.execute("SELECT changes()").fetchone()[0]
                target.execute("DROP TABLE _new_ids")
                print(f"    {table}: {cnt} rows added (articles not in local)")

    target.commit()
    target.execute("DETACH DATABASE other")
    target.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge two analysis.db files.")
    parser.add_argument("target", help="DB to merge INTO (modified in-place).")
    parser.add_argument("other", help="DB to merge FROM (read-only).")
    parser.add_argument(
        "--prefer-other",
        action="store_true",
        help="On conflicts, prefer the other DB's data instead of local.",
    )
    args = parser.parse_args()

    if not Path(args.target).exists():
        print(f"Target DB not found: {args.target}")
        return 1
    if not Path(args.other).exists():
        print(f"Other DB not found: {args.other}")
        return 1

    print(f"Merging {args.other} → {args.target}")
    merge(args.target, args.other, prefer_other=args.prefer_other)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
