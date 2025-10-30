"""Write helpers that persist ArticleResult objects to SQLite."""

from __future__ import annotations

import logging
from typing import List

from article_processor import ArticleResult
from pipeline_utils import OutputWriter
from pipeline_writer_utils import (
    insert_entity_mentions,
    insert_pair_claims,
    insert_pair_cues,
    insert_pair_numeric_changes,
    insert_pair_replacements,
    insert_pair_source_transitions,
    insert_pair_sources_added,
    insert_pair_sources_removed,
    insert_source_mentions,
    insert_version_metrics,
    insert_version_pairs,
    upsert_article,
    upsert_article_metrics,
    upsert_sources_agg,
    upsert_versions,
)


logger = logging.getLogger(__name__)


def write_article_result(writer: OutputWriter, result: ArticleResult) -> None:
    """Persist the supplied ArticleResult using the shared writer helpers."""
    stats_text = (
        "versions=%d sources=%d entities=%d pairs=%d pair_added=%d pair_removed=%d "
        "pair_transitions=%d replacements=%d numeric=%d claims=%d cues=%d sources_agg=%d article_metrics=%s"
        % (
            len(result.versions_rows),
            len(result.source_mentions_rows),
            len(result.entity_rows),
            len(result.pair_rows),
            len(result.pair_sources_added),
            len(result.pair_sources_removed),
            len(result.pair_source_transitions),
            len(result.pair_replacements),
            len(result.pair_numeric),
            len(result.pair_claims_rows),
            len(result.pair_cues_rows),
            len(result.sources_agg_rows),
            "yes" if result.article_metrics_row is not None else "no",
        )
    )
    logger.debug(
        "Writing article %s (%s) to %s | %s",
        result.entry_id,
        result.news_org,
        writer.db_path,
        stats_text,
    )

    try:
        upsert_article(writer, result.article_row)
        upsert_versions(writer, result.versions_rows)

        if result.live_blog_only:
            writer.commit()
            logger.debug("Committed live-blog metadata for article %s to %s", result.entry_id, writer.db_path)
            return

        if result.source_mentions_rows:
            logger.debug("Inserting %d source mention rows for article %s", len(result.source_mentions_rows), result.entry_id)
            insert_source_mentions(writer, result.source_mentions_rows)
        if result.entity_rows:
            logger.debug("Inserting %d entity rows for article %s", len(result.entity_rows), result.entry_id)
            insert_entity_mentions(writer, result.entity_rows)
        if result.version_metrics_rows:
            logger.debug(
                "Inserting %d version metrics rows for article %s", len(result.version_metrics_rows), result.entry_id
            )
            insert_version_metrics(writer, result.version_metrics_rows)
        if result.pair_rows:
            logger.debug("Inserting %d version pair rows for article %s", len(result.pair_rows), result.entry_id)
            insert_version_pairs(writer, result.pair_rows)
        if result.pair_sources_added:
            logger.debug(
                "Inserting %d pair_sources_added rows for article %s",
                len(result.pair_sources_added),
                result.entry_id,
            )
            insert_pair_sources_added(writer, result.pair_sources_added)
        if result.pair_sources_removed:
            logger.debug(
                "Inserting %d pair_sources_removed rows for article %s",
                len(result.pair_sources_removed),
                result.entry_id,
            )
            insert_pair_sources_removed(writer, result.pair_sources_removed)
        if result.pair_source_transitions:
            logger.debug(
                "Inserting %d pair_source_transitions rows for article %s",
                len(result.pair_source_transitions),
                result.entry_id,
            )
            insert_pair_source_transitions(writer, result.pair_source_transitions)
        if result.pair_replacements:
            logger.debug(
                "Inserting %d pair_replacements rows for article %s",
                len(result.pair_replacements),
                result.entry_id,
            )
            insert_pair_replacements(writer, result.pair_replacements)
        if result.pair_numeric:
            logger.debug(
                "Inserting %d pair_numeric rows for article %s", len(result.pair_numeric), result.entry_id
            )
            insert_pair_numeric_changes(writer, result.pair_numeric)
        if result.pair_claims_rows:
            logger.debug(
                "Inserting %d pair_claims rows for article %s",
                len(result.pair_claims_rows),
                result.entry_id,
            )
            insert_pair_claims(writer, result.pair_claims_rows)
        if result.pair_cues_rows:
            logger.debug(
                "Inserting %d pair_cues rows for article %s", len(result.pair_cues_rows), result.entry_id
            )
            insert_pair_cues(writer, result.pair_cues_rows)
        if result.sources_agg_rows:
            logger.debug(
                "Upserting %d sources_agg rows for article %s", len(result.sources_agg_rows), result.entry_id
            )
            upsert_sources_agg(writer, result.sources_agg_rows)

        if result.article_metrics_row is not None:
            logger.debug("Upserting article_metrics row for article %s", result.entry_id)
            upsert_article_metrics(writer, result.article_metrics_row)

        writer.commit()
        logger.debug("Committed article %s to %s", result.entry_id, writer.db_path)
    except Exception:
        logger.exception("Failed while writing article %s (%s) to %s", result.entry_id, result.news_org, writer.db_path)
        raise
