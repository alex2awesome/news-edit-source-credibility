"""End-to-end orchestration for the news edits pipeline.

Preferred alex-mode command:

```bash
bash run.sh news-edits-pipeline/config.yaml \
  --article-workers 6 \
  --max-in-flight-llm-requests 12 \
  --alex-features-only \
  --reprompt-for-structure \
  --prompt-for-structure-fix \
  --alex-pair-list opinion-article-actions/opinion-articles-to-score.csv \
  --db article-versions/newssniffer-nytimes.db.gz \
  --db article-versions/newssniffer-washpo.db.gz \
  --db article-versions/newssniffer-guardian.db.gz \
  --verbose
```
"""

from __future__ import annotations

import logging
import random
import shutil
import csv
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm.auto import tqdm

from article_processor import ArticleResult, process_article
from config import Config, parse_cli_args
from llm_client import StructuredLLMClient
from loader import iter_article_counts
from pipeline_utils import OutputWriter, ensure_dir
from writer import write_article_result

__all__ = [
    "ArticleResult",
    "StructuredLLMClient",
    "process_article",
    "write_article_result",
    "process_db",
    "main",
]


logger = logging.getLogger(__name__)
DateRange = Tuple[datetime, datetime]
AlexPairDirective = Tuple[str, Optional[int], Optional[int]]


@dataclass
class FilterBucket:
    count: int = 0
    sample_ids: List[int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.sample_ids is None:
            self.sample_ids = []

    def add(self, entry_id: int, sample_limit: int = 5) -> None:
        self.count += 1
        if len(self.sample_ids) < sample_limit:
            self.sample_ids.append(entry_id)


@dataclass
class ProgressBar:
    """Light wrapper around tqdm that guarantees visible output."""

    total: int
    desc: str

    def __post_init__(self) -> None:
        self._bar = tqdm(
            total=self.total,
            desc=self.desc,
            unit="article",
            leave=True,
            dynamic_ncols=True,
            mininterval=0.1,
            smoothing=0.2,
            disable=self.total == 0,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )

    def update(self, advance: int = 1) -> None:
        if self._bar is not None:
            self._bar.update(advance)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()


def _filter_entry_ids(
    article_counts: Iterable[tuple[int, int]],
    config: Config,
) -> Tuple[List[int], Dict[str, FilterBucket]]:
    valid_entry_ids: List[int] = []
    skipped: Dict[str, FilterBucket] = {}

    def record(reason: str, entry_id: int) -> None:
        bucket = skipped.setdefault(reason, FilterBucket())
        bucket.add(entry_id)

    for entry_id, version_count in article_counts:
        if version_count < config.min_versions:
            record(
                f"version_count < min_versions ({config.min_versions})",
                entry_id,
            )
            continue
        if config.max_versions is not None and version_count >= config.max_versions:
            record(
                f"version_count >= max_versions ({config.max_versions})",
                entry_id,
            )
            continue
        valid_entry_ids.append(entry_id)
    return valid_entry_ids, skipped


def _parse_datetime(value: str) -> datetime:
    text = value.strip()
    if not text:
        raise ValueError("Datetime value is empty")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _coerce_created_timestamp(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return _parse_datetime(text)
    except ValueError:
        logger.warning("Unable to parse created timestamp %r; excluding from date filtering", value)
        return None


def _timestamp_in_range(timestamp: Optional[datetime], date_range: DateRange) -> bool:
    if timestamp is None:
        return False
    start, end = date_range
    return start <= timestamp <= end


def _already_processed_ids(out_db_path: Path) -> set:
    """Return article_ids already written to an existing output analysis.db."""
    if not out_db_path.exists():
        return set()
    import sqlite3
    try:
        with sqlite3.connect(str(out_db_path)) as conn:
            rows = conn.execute("SELECT article_id FROM articles").fetchall()
        return {row[0] for row in rows}
    except Exception as exc:
        logger.debug("Could not read existing output DB %s: %s", out_db_path, exc)
        return set()


def _downsample_entry_ids(entry_ids: Sequence[int], percentage: float) -> List[int]:
    if percentage <= 0:
        return []
    if percentage >= 100:
        return list(entry_ids)

    rng = random.Random()
    threshold = percentage / 100.0
    return [entry_id for entry_id in entry_ids if rng.random() < threshold]


def _uniform_cap_entry_ids(entry_ids: Sequence[int], max_articles: int) -> List[int]:
    if max_articles <= 0:
        return []
    if len(entry_ids) <= max_articles:
        return list(entry_ids)
    return random.sample(list(entry_ids), k=max_articles)


def _cleanup_article_cache(config: Config, article_dir: Path) -> None:
    if config.cleanup_cached_dirs and config.cache_raw_responses and article_dir.exists():
        try:
            shutil.rmtree(article_dir)
        except OSError:
            logger.warning("Failed to remove cache directory %s", article_dir)


def _parse_optional_int(value: object) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower() in {"none", "null", "na", "n/a"}:
        return None
    return int(text)


def _load_alex_pair_list(path: Path) -> Dict[int, List[AlexPairDirective]]:
    directives: Dict[int, List[AlexPairDirective]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is empty or missing a header row")
        lower_map = {name.strip().lower(): name for name in reader.fieldnames}
        required = {"news_org", "entry_id"}
        missing = [name for name in required if name not in lower_map]
        if missing:
            raise ValueError(f"{path} missing required columns: {', '.join(missing)}")
        source_col = lower_map["news_org"]
        entry_col = lower_map["entry_id"]
        from_col = lower_map.get("from_version_num")
        to_col = lower_map.get("to_version_num")

        for row_num, row in enumerate(reader, start=2):
            news_org = str(row.get(source_col, "")).strip()
            if not news_org:
                raise ValueError(f"{path}:{row_num} has empty news_org")
            entry_raw = row.get(entry_col)
            if entry_raw is None or not str(entry_raw).strip():
                raise ValueError(f"{path}:{row_num} has empty entry_id")
            try:
                entry_id = int(str(entry_raw).strip())
            except ValueError as exc:
                raise ValueError(f"{path}:{row_num} has invalid entry_id {entry_raw!r}") from exc

            from_num = _parse_optional_int(row.get(from_col)) if from_col else None
            to_num = _parse_optional_int(row.get(to_col)) if to_col else None
            directives.setdefault(entry_id, []).append((news_org, from_num, to_num))
    return directives


def process_db(
    db_path: Path,
    config: Config,
    schema_path: Path,
    out_path: Path,
    verbose: bool = False,
    downsample_percentage: Optional[float] = None,
    max_articles_per_outlet: Optional[int] = None,
    date_range: Optional[DateRange] = None,
    alex_pair_directives: Optional[Dict[int, List[AlexPairDirective]]] = None,
) -> None:
    """Process a single SQLite database with parallel article workers."""
    db_stem = db_path.stem.replace(".db", "")
    out_root = Path(out_path) / db_stem
    ensure_dir(out_root)

    writer = OutputWriter(out_root / "analysis.db", schema_path)

    try:
        article_metadata = list(iter_article_counts(str(db_path)))
        if not article_metadata:
            logger.debug("Database %s contains no entries; skipping", db_path)
            return

        counts_for_filter: List[Tuple[int, int]] = []
        first_created_map: Dict[int, Optional[datetime]] = {}
        for record in article_metadata:
            if len(record) == 2:
                entry_id, version_count = record  # type: ignore[misc]
                first_created_raw = None
            else:
                entry_id, version_count, first_created_raw = record  # type: ignore[misc]
            counts_for_filter.append((entry_id, version_count))
            first_created_map[entry_id] = _coerce_created_timestamp(first_created_raw)
        if config.alex_features_only and alex_pair_directives is not None:
            db_entry_ids = {entry_id for entry_id, _ in counts_for_filter}
            requested_ids = set(alex_pair_directives.keys())
            valid_entry_ids = sorted(db_entry_ids & requested_ids)
            logger.debug(
                "Alex pair-list authoritative mode for %s: requested=%d matched_in_db=%d missing=%d",
                db_path.name,
                len(requested_ids),
                len(valid_entry_ids),
                len(requested_ids - db_entry_ids),
            )
            if not valid_entry_ids:
                logger.debug("No pair-list articles match entries in %s; skipping database", db_path.name)
                return
        else:
            valid_entry_ids, skipped_by_reason = _filter_entry_ids(counts_for_filter, config)
            filtered_out = len(counts_for_filter) - len(valid_entry_ids)
            if filtered_out:
                logger.debug(
                    "Pre-filtered %d of %d entries in %s based on version thresholds (min=%d, max=%s)",
                    filtered_out,
                    len(counts_for_filter),
                    db_path.name,
                    config.min_versions,
                    str(config.max_versions),
                )
                for reason, bucket in skipped_by_reason.items():
                    logger.debug(
                        "Skipping %d entries for reason: %s [sample entry_ids: %s]",
                        bucket.count,
                        reason,
                        bucket.sample_ids,
                    )

            if date_range is not None and valid_entry_ids:
                before_date_filter = len(valid_entry_ids)
                valid_entry_ids = [
                    entry_id for entry_id in valid_entry_ids if _timestamp_in_range(first_created_map.get(entry_id), date_range)
                ]
                filtered_by_date = before_date_filter - len(valid_entry_ids)
                if filtered_by_date:
                    logger.debug(
                        "Filtered %d entries in %s outside date range %s to %s",
                        filtered_by_date,
                        db_path.name,
                        date_range[0].isoformat(),
                        date_range[1].isoformat(),
                    )

            if max_articles_per_outlet is not None and valid_entry_ids:
                before_cap = len(valid_entry_ids)
                valid_entry_ids = _uniform_cap_entry_ids(valid_entry_ids, max_articles_per_outlet)
                if len(valid_entry_ids) != before_cap:
                    logger.debug(
                        "Uniformly sampled entries in %s from %d to %d using max_articles_per_outlet=%d",
                        db_path.name,
                        before_cap,
                        len(valid_entry_ids),
                        max_articles_per_outlet,
                    )

            if downsample_percentage is not None and valid_entry_ids:
                downsampled_ids = _downsample_entry_ids(valid_entry_ids, downsample_percentage)
                if len(downsampled_ids) != len(valid_entry_ids):
                    logger.debug(
                        "Downsampled entries in %s from %d to %d using %.2f%% probability",
                        db_path.name,
                        len(valid_entry_ids),
                        len(downsampled_ids),
                        downsample_percentage,
                    )
                valid_entry_ids = downsampled_ids

            if not valid_entry_ids:
                logger.debug(
                    "No articles in %s meet version count filters (min_versions=%d, max_versions=%s); skipping database",
                    db_path.name,
                    config.min_versions,
                    str(config.max_versions),
                )
                return

        done_ids = _already_processed_ids(out_root / "analysis.db")
        if done_ids:
            before_skip = len(valid_entry_ids)
            valid_entry_ids = [eid for eid in valid_entry_ids if eid not in done_ids]
            logger.info(
                "Skipping %d already-processed article(s) in %s (%d remaining)",
                before_skip - len(valid_entry_ids),
                db_path.name,
                len(valid_entry_ids),
            )
            if not valid_entry_ids:
                logger.info("All articles in %s already processed; skipping database", db_path.name)
                return

        max_workers = max(1, config.article_workers)

        futures = {}
        thread_local = threading.local()
        created_clients: List[StructuredLLMClient] = []
        created_clients_lock = threading.Lock()

        def get_thread_client() -> StructuredLLMClient:
            client = getattr(thread_local, "client", None)
            if client is None:
                client = StructuredLLMClient(config)
                thread_local.client = client
                with created_clients_lock:
                    created_clients.append(client)
            return client

        def run_article_task(entry_id: int) -> Optional[ArticleResult]:
            client = get_thread_client()
            if config.alex_features_only:
                return process_article(
                    entry_id,
                    db_path,
                    config,
                    client,
                    out_root,
                    verbose,
                    alex_pair_directives.get(entry_id) if alex_pair_directives else None,
                    alex_pair_directives is not None,
                )
            return process_article(entry_id, db_path, config, client, out_root, verbose)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for entry_id in valid_entry_ids:
                article_dir = out_root / str(entry_id)
                futures[executor.submit(run_article_task, entry_id)] = (entry_id, article_dir)
            progress = ProgressBar(total=len(valid_entry_ids), desc=f"{db_stem} articles")

            try:
                for future in as_completed(futures):
                    entry_id, article_dir = futures[future]
                    try:
                        result = future.result()
                    except Exception:
                        logger.exception("Failed to process article %s", entry_id)
                        progress.update(1)
                        _cleanup_article_cache(config, article_dir)
                        continue

                    if result is None:
                        logger.debug("Article %s returned no result; skipping persistence", entry_id)
                        progress.update(1)
                        _cleanup_article_cache(config, article_dir)
                        continue

                    logger.debug(
                        "Persisting article %s to database %s",
                        entry_id,
                        writer.db_path,
                    )
                    write_article_result(writer, result)
                    logger.debug(result.log_message)
                    progress.update(1)
                    _cleanup_article_cache(config, article_dir)
            finally:
                progress.close()
                for client in created_clients:
                    try:
                        client.close()
                    except Exception:
                        logger.debug("Failed to close thread-local client cleanly")
    finally:
        writer.close()


def main() -> None:
    args = parse_cli_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    noisy_libs = [
        "asyncio",
        "httpx",
        "httpcore",
        "urllib3",
        "openai",
    ]
    for name in noisy_libs:
        logging.getLogger(name).setLevel(logging.WARNING)

    config = Config.from_yaml(args.config)
    if args.matt_features_only:
        config.matt_features_only = True
    if args.alex_features_only:
        config.alex_features_only = True
    if config.matt_features_only and config.alex_features_only:
        raise SystemExit("--matt-features-only and --alex-features-only are mutually exclusive")
    if args.vllm_api_base:
        config.vllm_api_base = args.vllm_api_base
    if args.article_workers is not None:
        config.article_workers = max(1, args.article_workers)
    if args.max_in_flight_llm_requests is not None:
        config.max_in_flight_llm_requests = max(1, args.max_in_flight_llm_requests)
    if args.max_articles_per_outlet is not None:
        config.max_articles_per_outlet = max(1, args.max_articles_per_outlet)
    if args.disable_max_versions:
        config.max_versions = None
    if args.alex_random_pairs_per_article is not None:
        config.alex_random_pairs_per_article = max(0, args.alex_random_pairs_per_article)
    if args.alex_pair_list is not None:
        config.alex_pair_list = args.alex_pair_list
    if args.alex_dynamic_pairs:
        config.alex_dynamic_pairs = True
    if args.dynamic_only:
        config.dynamic_only = True
        config.alex_dynamic_pairs = True
    if args.dynamic_target_change is not None:
        config.dynamic_target_change = float(args.dynamic_target_change)
    if args.disable_similarity_skip:
        config.skip_similarity_threshold = None
    if args.reprompt_for_structure:
        config.reprompt_for_structure = True
    if args.prompt_for_structure_fix:
        config.prompt_for_structure_fix = True

    out_path = args.out_path if args.out_path is not None else (Path("./edit-actions") if config.alex_features_only else Path("./out"))
    config.out_root = out_path
    ensure_dir(config.out_root)

    alex_pair_directives: Optional[Dict[int, List[AlexPairDirective]]] = None
    if config.alex_features_only and config.alex_pair_list is not None:
        try:
            alex_pair_directives = _load_alex_pair_list(config.alex_pair_list)
        except ValueError as exc:
            raise SystemExit(f"Invalid --alex-pair-list CSV: {exc}") from exc

    logger.info(
        "Runtime config: article_workers=%d max_in_flight_llm_requests=%d batch_size=%d matt_features_only=%s alex_features_only=%s max_versions=%s reprompt_for_structure=%s prompt_for_structure_fix=%s structure_max_attempts=%d",
        config.article_workers,
        config.max_in_flight_llm_requests,
        config.batch_size,
        config.matt_features_only,
        config.alex_features_only,
        config.max_versions,
        config.reprompt_for_structure,
        config.prompt_for_structure_fix,
        config.structure_max_attempts,
    )
    if args.downsample_percentage is not None:
        logger.info("Runtime sampling: downsample_percentage=%.2f", args.downsample_percentage)
    if config.max_articles_per_outlet is not None:
        logger.info("Runtime sampling: max_articles_per_outlet=%d", config.max_articles_per_outlet)
    if args.date_range:
        logger.info("Runtime date filter: start=%s end=%s", args.date_range[0], args.date_range[1])
    if config.alex_features_only:
        logger.info("Runtime alex mode: random_pairs_per_article=%d", config.alex_random_pairs_per_article)
        logger.info(
            "Runtime alex mode: dynamic_pairs=%s dynamic_only=%s dynamic_target_change=%.3f",
            config.alex_dynamic_pairs,
            config.dynamic_only,
            config.dynamic_target_change,
        )
        if config.alex_pair_list is not None:
            logger.info("Runtime alex mode: pair_list=%s (%d entry keys)", config.alex_pair_list, len(alex_pair_directives or {}))
            logger.info("Runtime alex mode: pair-list authoritative filtering enabled (bypassing date/downsample/version caps)")
    logger.info("Runtime filtering: skip_similarity_threshold=%s", config.skip_similarity_threshold)

    date_range: Optional[DateRange] = None
    if args.date_range:
        try:
            start = _parse_datetime(args.date_range[0])
            end = _parse_datetime(args.date_range[1])
        except ValueError as exc:
            raise SystemExit(f"Invalid --date-range values: {exc}") from exc
        if end < start:
            raise SystemExit("--date-range END must be greater than or equal to START")
        date_range = (start, end)

    if args.downsample_percentage is not None:
        if not (0.0 <= args.downsample_percentage <= 100.0):
            raise SystemExit("--downsample-percentage must be between 0 and 100 (inclusive)")
    if not (0.0 < float(config.dynamic_target_change) < 1.0):
        raise SystemExit("--dynamic-target-change must be between 0 and 1 (exclusive)")

    for db_path in args.dbs:
        logger.info("Processing %s", db_path)
        process_db(
            db_path,
            config,
            args.schema,
            out_path,
            verbose=args.verbose,
            downsample_percentage=args.downsample_percentage,
            max_articles_per_outlet=config.max_articles_per_outlet,
            date_range=date_range,
            alex_pair_directives=alex_pair_directives,
        )


if __name__ == "__main__":
    main()
