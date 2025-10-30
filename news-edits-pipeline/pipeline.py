"""End-to-end orchestration for the news edits pipeline."""

from __future__ import annotations

import logging
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

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
) -> List[int]:
    valid_entry_ids: List[int] = []
    for entry_id, version_count in article_counts:
        if version_count < config.min_versions:
            logger.info(
                "Skipping entry %s during pre-filter: only %d version(s) < min_versions=%d",
                entry_id,
                version_count,
                config.min_versions,
            )
            continue
        if config.max_versions is not None and version_count >= config.max_versions:
            logger.info(
                "Skipping entry %s during pre-filter: %d version(s) exceeds max_versions=%d",
                entry_id,
                version_count,
                config.max_versions,
            )
            continue
        valid_entry_ids.append(entry_id)
    return valid_entry_ids


def _cleanup_article_cache(config: Config, article_dir: Path) -> None:
    if config.cleanup_cached_dirs and config.cache_raw_responses and article_dir.exists():
        try:
            shutil.rmtree(article_dir)
        except OSError:
            logger.warning("Failed to remove cache directory %s", article_dir)


def process_db(
    db_path: Path,
    config: Config,
    schema_path: Path,
    out_path: Path,
    verbose: bool = False,
) -> None:
    """Process a single SQLite database with parallel article workers."""
    db_stem = db_path.stem.replace(".db", "")
    out_root = Path(out_path) / db_stem
    ensure_dir(out_root)

    writer = OutputWriter(out_root / "analysis.db", schema_path)

    try:
        article_counts = list(iter_article_counts(str(db_path)))
        if not article_counts:
            logger.info("Database %s contains no entries; skipping", db_path)
            return

        valid_entry_ids = _filter_entry_ids(article_counts, config)
        filtered_out = len(article_counts) - len(valid_entry_ids)
        if filtered_out:
            logger.info(
                "Pre-filtered %d of %d entries in %s based on version thresholds (min=%d, max=%s)",
                filtered_out,
                len(article_counts),
                db_path.name,
                config.min_versions,
                str(config.max_versions),
            )

        if not valid_entry_ids:
            logger.info(
                "No articles in %s meet version count filters (min_versions=%d, max_versions=%s); skipping database",
                db_path.name,
                config.min_versions,
                str(config.max_versions),
            )
            return

        max_workers = max(1, config.batch_size)

        futures = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for entry_id in valid_entry_ids:
                article_dir = out_root / str(entry_id)
                futures[executor.submit(process_article, entry_id, db_path, config, None, out_root, verbose)] = (
                    entry_id,
                    article_dir,
                )

            progress = ProgressBar(total=len(valid_entry_ids), desc=f"{db_stem} articles")

            try:
                for future in as_completed(futures):
                    entry_id, article_dir = futures[future]
                    try:
                        result = future.result()
                    except Exception:
                        logger.exception("Failed to process article %s", entry_id)
                        progress.update(1)
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
                    logger.info(result.log_message)
                    progress.update(1)
                    _cleanup_article_cache(config, article_dir)
            finally:
                progress.close()
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
    ensure_dir(config.out_root)

    for db_path in args.dbs:
        logger.info("Processing %s", db_path)
        process_db(db_path, config, args.schema, args.out_path, verbose=args.verbose)


if __name__ == "__main__":
    main()
