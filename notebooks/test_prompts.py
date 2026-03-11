from __future__ import annotations

import argparse
import gzip
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

from tqdm.auto import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PIPELINE_DIR = PROJECT_ROOT / "news-edits-pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from article_processor import (  # type: ignore
    _ALEX_VERSION_TOKEN_LIMIT,
    _select_alex_pairs,
    _truncate_text_for_prompt,
)
from loader import _resolved_db_path, source_name_from_db_path  # type: ignore
from pipeline import _load_alex_pair_list  # type: ignore
from pipeline_utils import load_prompt_template, render_prompt  # type: ignore


def _maybe_decompress_pair_list_csv(path: Path) -> Path:
    if path.suffix != ".gz":
        return path
    tmp = tempfile.NamedTemporaryFile(prefix="alex_pairs_", suffix=".csv", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    with gzip.open(path, "rb") as src, tmp_path.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    return tmp_path


def _resolve_input_path(path: Union[str, Path]) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate

    fallback_root = PROJECT_ROOT / candidate
    if fallback_root.exists():
        return fallback_root

    fallback_notebooks = SCRIPT_DIR / candidate
    if fallback_notebooks.exists():
        return fallback_notebooks

    raise FileNotFoundError(f"Input path not found: {path}")


def _canonical_source_label(value: str) -> str:
    label = (value or "").strip().lower()
    aliases = {
        "guardian": "newssniffer-guardian",
        "newssniffer-guardian": "newssniffer-guardian",
        "nytimes": "newssniffer-nytimes",
        "new york times": "newssniffer-nytimes",
        "newssniffer-nytimes": "newssniffer-nytimes",
        "washpo": "newssniffer-washpo",
        "washington post": "newssniffer-washpo",
        "washington-post": "newssniffer-washpo",
        "newssniffer-washpo": "newssniffer-washpo",
        "bbc": "newssniffer-bbc",
        "newssniffer-bbc": "newssniffer-bbc",
        "independent": "newssniffer-independent",
        "newssniffer-independent": "newssniffer-independent",
        "ap": "ap",
        "associated press": "ap",
        "reuters": "reuters",
    }
    return aliases.get(label, label)


def _resolve_source_db(news_org: str, article_versions_root: Path) -> Path:
    normalized = _canonical_source_label(news_org)

    candidates = [
        article_versions_root / f"{normalized}.db",
        article_versions_root / f"{normalized}.db.gz",
    ]

    unzipped_root = PROJECT_ROOT / "unzipped-article-versions"
    if normalized.startswith("newssniffer-"):
        stem = normalized[len("newssniffer-") :]
    else:
        stem = normalized

    candidates.extend(
        [
            article_versions_root / f"newssniffer-{stem}.db",
            article_versions_root / f"newssniffer-{stem}.db.gz",
            unzipped_root / f"newssniffer-{stem}.db.db",
            unzipped_root / f"newssniffer-{stem}.db",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"No source DB found for news_org={news_org!r} (normalized={normalized!r})")


def _chunked(items: List[int], size: int) -> List[List[int]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _load_versions_for_entry_ids(
    db_path: Path,
    entry_ids: List[int],
    *,
    chunk_size: int = 500,
) -> Dict[int, List[Dict[str, object]]]:
    """Batch-load all versions for the supplied entry_ids from one source DB."""
    result: Dict[int, List[Dict[str, object]]] = {}
    if not entry_ids:
        return result

    source_name = source_name_from_db_path(str(db_path))
    with _resolved_db_path(str(db_path)) as resolved:
        conn = sqlite3.connect(resolved)
        conn.row_factory = sqlite3.Row
        try:
            for chunk in _chunked(sorted(set(entry_ids)), chunk_size):
                placeholders = ",".join("?" for _ in chunk)
                sql = f"""
                    SELECT id, entry_id, version, created, title, url, num_versions, summary
                    FROM entryversion
                    WHERE entry_id IN ({placeholders})
                    ORDER BY entry_id ASC, version ASC
                """
                rows = conn.execute(sql, chunk).fetchall()
                for row in rows:
                    entry_id = int(row["entry_id"])
                    result.setdefault(entry_id, []).append(
                        {
                            "id": row["id"],
                            "source": source_name,
                            "entry_id": row["entry_id"],
                            "version": row["version"],
                            "created": row["created"],
                            "title": row["title"],
                            "url": row["url"],
                            "num_versions": row["num_versions"],
                            "summary": row["summary"],
                        }
                    )
        finally:
            conn.close()
    return result


def dry_run_edit_action_prompts(
    pair_list_path: Union[str, Path] = "opinion-article-actions/opinion-articles-to-score.csv.gz",
    article_versions_root: Union[str, Path] = "article-versions",
    random_pairs_per_article: int = 0,
    limit_articles: Optional[int] = None,
    limit_prompts: Optional[int] = None,
    include_metadata: bool = False,
    show_progress: bool = True,
    verbose: bool = True,
    log_every_articles: int = 100,
) -> Union[List[str], List[Dict[str, object]]]:
    """Generate alex-mode edit-action prompts without calling an LLM.

    Returns a list of prompt strings by default. Set include_metadata=True to return
    dict rows containing article/pair metadata plus prompt text.
    """
    pair_list_path = _resolve_input_path(pair_list_path)
    article_versions_root = _resolve_input_path(article_versions_root)
    start_ts = time.perf_counter()

    temp_csv: Optional[Path] = None
    try:
        if verbose:
            print(f"[dry-run] pair-list: {pair_list_path}")
            print(f"[dry-run] article versions root: {article_versions_root}")
        csv_path = _maybe_decompress_pair_list_csv(pair_list_path)
        if csv_path != pair_list_path:
            temp_csv = csv_path
            if verbose:
                print(f"[dry-run] decompressed pair-list to temp csv: {csv_path}")

        load_pairs_start = time.perf_counter()
        directives_by_entry = _load_alex_pair_list(csv_path)
        if verbose:
            print(
                f"[dry-run] loaded pair directives for {len(directives_by_entry):,} entries "
                f"in {time.perf_counter() - load_pairs_start:.2f}s"
            )

        template_start = time.perf_counter()
        template = load_prompt_template("ALEX_edit_actions_pair")
        if verbose:
            print(f"[dry-run] loaded template in {time.perf_counter() - template_start:.2f}s")

        outputs: List[Union[str, Dict[str, object]]] = []
        article_count = 0
        skipped_no_versions = 0
        skipped_no_pairs = 0

        entry_ids = sorted(directives_by_entry.keys())
        if verbose:
            print(f"[dry-run] starting generation across {len(entry_ids):,} entry_ids")

        # Build a per-source batch plan, then load all needed versions in bulk.
        db_to_entry_ids: Dict[str, List[int]] = {}
        for entry_id in entry_ids:
            directives = directives_by_entry.get(entry_id, [])
            if not directives:
                continue
            db_path = _resolve_source_db(directives[0][0], article_versions_root)
            db_to_entry_ids.setdefault(str(db_path), []).append(entry_id)

        versions_by_entry: Dict[int, List[Dict[str, object]]] = {}
        batch_load_start = time.perf_counter()
        for db_path_str, db_entry_ids in db_to_entry_ids.items():
            db_path = Path(db_path_str)
            if verbose:
                print(
                    f"[dry-run] batch-loading versions from {db_path} "
                    f"for {len(db_entry_ids):,} entry_ids"
                )
            db_start = time.perf_counter()
            db_versions = _load_versions_for_entry_ids(db_path, db_entry_ids, chunk_size=500)
            versions_by_entry.update(db_versions)
            if verbose:
                print(
                    f"[dry-run] loaded {sum(len(v) for v in db_versions.values()):,} versions "
                    f"across {len(db_versions):,} entries from {db_path.name} "
                    f"in {time.perf_counter() - db_start:.2f}s"
                )
        if verbose:
            print(
                f"[dry-run] finished all batch loads in {time.perf_counter() - batch_load_start:.2f}s "
                f"(entries with any versions: {len(versions_by_entry):,})"
            )

        seen_db_paths: set[str] = set()
        for i, entry_id in enumerate(
            tqdm(
                entry_ids,
                desc="Generating prompts",
                unit="article",
                dynamic_ncols=True,
                leave=True,
                disable=not show_progress,
            ),
            start=1,
        ):
            directives = directives_by_entry[entry_id]
            if not directives:
                continue

            db_path = _resolve_source_db(directives[0][0], article_versions_root)
            db_path_str = str(db_path)
            if db_path_str not in seen_db_paths and verbose:
                print(
                    f"[dry-run] first access for source DB {db_path_str} "
                    f"(already batch-loaded; now using in-memory versions)"
                )
                seen_db_paths.add(db_path_str)

            versions = versions_by_entry.get(entry_id, [])
            if len(versions) < 2:
                skipped_no_versions += 1
                continue

            news_org = versions[0].get("source") or directives[0][0]
            pair_comparisons = _select_alex_pairs(
                entry_id=entry_id,
                news_org=str(news_org),
                versions=versions,
                random_pairs_per_article=max(0, int(random_pairs_per_article)),
                pair_directives=directives,
                pair_list_supplied=True,
            )
            if not pair_comparisons:
                skipped_no_pairs += 1
                continue

            article_count += 1
            for prev, curr, pair_policy in pair_comparisons:
                rendered = render_prompt(
                    template,
                    {
                        "article_v1": _truncate_text_for_prompt(
                            prev.get("summary") or "",
                            _ALEX_VERSION_TOKEN_LIMIT,
                            "cl100k_base",
                        ),
                        "article_v2": _truncate_text_for_prompt(
                            curr.get("summary") or "",
                            _ALEX_VERSION_TOKEN_LIMIT,
                            "cl100k_base",
                        ),
                    },
                )
                if include_metadata:
                    outputs.append(
                        {
                            "news_org": news_org,
                            "entry_id": entry_id,
                            "from_version_id": prev.get("id"),
                            "to_version_id": curr.get("id"),
                            "from_version_num": prev.get("version"),
                            "to_version_num": curr.get("version"),
                            "pair_policy": pair_policy,
                            "prompt": rendered,
                        }
                    )
                else:
                    outputs.append(rendered)

                if limit_prompts is not None and len(outputs) >= limit_prompts:
                    if verbose:
                        elapsed = time.perf_counter() - start_ts
                        print(
                            f"[dry-run] reached limit_prompts={limit_prompts} after {elapsed:.2f}s "
                            f"(articles_with_pairs={article_count}, prompts={len(outputs)})"
                        )
                    return outputs  # type: ignore[return-value]

            if verbose and log_every_articles > 0 and i % log_every_articles == 0:
                elapsed = time.perf_counter() - start_ts
                print(
                    f"[dry-run] progress: scanned={i:,}/{len(entry_ids):,}, "
                    f"articles_with_pairs={article_count:,}, prompts={len(outputs):,}, "
                    f"skipped_no_versions={skipped_no_versions:,}, skipped_no_pairs={skipped_no_pairs:,}, "
                    f"elapsed={elapsed:.1f}s"
                )

            if limit_articles is not None and article_count >= limit_articles:
                if verbose:
                    elapsed = time.perf_counter() - start_ts
                    print(
                        f"[dry-run] reached limit_articles={limit_articles} after {elapsed:.2f}s "
                        f"(prompts={len(outputs)})"
                    )
                break

        if verbose:
            elapsed = time.perf_counter() - start_ts
            print(
                f"[dry-run] complete in {elapsed:.2f}s | scanned={len(entry_ids):,}, "
                f"articles_with_pairs={article_count:,}, prompts={len(outputs):,}, "
                f"skipped_no_versions={skipped_no_versions:,}, skipped_no_pairs={skipped_no_pairs:,}"
            )
        return outputs  # type: ignore[return-value]
    finally:
        if temp_csv is not None:
            try:
                temp_csv.unlink(missing_ok=True)
            except OSError:
                pass


def _main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run generator for alex edit-action prompts")
    parser.add_argument(
        "--pair-list",
        default="opinion-article-actions/opinion-articles-to-score.csv.gz",
        help="Path to pair-list CSV or CSV.GZ with columns news_org,entry_id,from_version_num,to_version_num",
    )
    parser.add_argument(
        "--article-versions-root",
        default="article-versions",
        help="Directory containing source article SQLite DBs",
    )
    parser.add_argument("--random-pairs-per-article", type=int, default=0)
    parser.add_argument("--limit-articles", type=int, default=3)
    parser.add_argument("--limit-prompts", type=int, default=10)
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar")
    parser.add_argument("--quiet", action="store_true", help="Disable dry-run logging")
    parser.add_argument("--log-every-articles", type=int, default=100)
    args = parser.parse_args()

    prompts = dry_run_edit_action_prompts(
        pair_list_path=args.pair_list,
        article_versions_root=args.article_versions_root,
        random_pairs_per_article=args.random_pairs_per_article,
        limit_articles=args.limit_articles,
        limit_prompts=args.limit_prompts,
        include_metadata=False,
        show_progress=not args.no_progress,
        verbose=not args.quiet,
        log_every_articles=max(0, args.log_every_articles),
    )

    print(f"Generated {len(prompts)} prompts")
    for idx, prompt in enumerate(prompts[:2], start=1):
        print(f"\n{'=' * 24} Prompt {idx} {'=' * 24}\n")
        print(prompt)


if __name__ == "__main__":
    _main()
