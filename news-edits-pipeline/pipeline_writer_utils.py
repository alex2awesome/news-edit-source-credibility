"""Helpers for writing pipeline outputs to SQLite."""

from __future__ import annotations

from typing import Sequence, Tuple, Any

from pipeline_utils import OutputWriter

Rows = Sequence[Tuple[Any, ...]]


def upsert_article(writer: OutputWriter, row: Tuple[Any, ...]) -> None:
    writer.execute(
        """
        INSERT OR REPLACE INTO articles (
            article_id, news_org, url, title_first, title_final,
            original_publication_time, total_edits, is_live_blog
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        row,
    )


def upsert_versions(writer: OutputWriter, rows: Rows) -> None:
    writer.executemany(
        """
        INSERT OR REPLACE INTO versions (
            version_id, article_id, news_org, version_num, timestamp_utc, title, char_len
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        list(rows),
    )


def insert_source_mentions(writer: OutputWriter, rows: Rows) -> None:
    writer.executemany(
        """
        INSERT INTO source_mentions (
            version_id, article_id, news_org, source_id_within_article,
            source_canonical, source_type, speech_style, attribution_verb,
            char_start, char_end, sentence_index, paragraph_index,
            is_in_title, is_in_lede, attributed_text, prominence_lead_pct, confidence
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        list(rows),
    )


def upsert_sources_agg(writer: OutputWriter, rows: Rows) -> None:
    writer.executemany(
        """
        INSERT OR REPLACE INTO sources_agg (
            article_id, news_org, source_id_within_article, source_canonical,
            source_type, first_seen_version, first_seen_time, last_seen_version,
            last_seen_time, num_mentions_total, num_versions_present,
            total_attributed_words, voice_retention_index, mean_prominence,
            lead_appearance_count, title_appearance_count,
            doubted_any, deemphasized_any, disappeared_any
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        list(rows),
    )


def insert_entity_mentions(writer: OutputWriter, rows: Rows) -> None:
    writer.executemany(
        """
        INSERT INTO entity_mentions (
            version_id, article_id, news_org, entity_id_within_article, entity_type,
            canonical_name, char_start, char_end, sentence_index, paragraph_index
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        list(rows),
    )


def insert_version_metrics(writer: OutputWriter, rows: Rows) -> None:
    writer.executemany(
        """
        INSERT INTO version_metrics (
            version_id, article_id, news_org, distinct_sources,
            institutional_share_words, anonymous_source_share_words,
            hedge_density_per_1k
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        list(rows),
    )


def insert_version_pairs(writer: OutputWriter, rows: Rows) -> None:
    writer.executemany(
        """
        INSERT INTO version_pairs (
            article_id, news_org, from_version_id, to_version_id,
            from_version_num, to_version_num, delta_minutes, tokens_added,
            tokens_deleted, percent_text_new, movement_index,
            moved_into_top20pct_tokens, edit_type, angle_changed,
            title_jaccard_prev, title_jaccard_curr
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        list(rows),
    )


def insert_pair_sources_added(writer: OutputWriter, rows: Rows) -> None:
    writer.executemany("INSERT INTO pair_sources_added VALUES (?, ?, ?, ?)", list(rows))


def insert_pair_sources_removed(writer: OutputWriter, rows: Rows) -> None:
    writer.executemany("INSERT INTO pair_sources_removed VALUES (?, ?, ?, ?)", list(rows))


def insert_pair_title_events(writer: OutputWriter, rows: Rows) -> None:
    writer.executemany("INSERT INTO pair_title_events VALUES (?, ?, ?, ?)", list(rows))


def insert_pair_replacements(writer: OutputWriter, rows: Rows) -> None:
    writer.executemany("INSERT INTO pair_anon_named_replacements VALUES (?, ?, ?, ?, ?, ?)", list(rows))


def insert_pair_numeric_changes(writer: OutputWriter, rows: Rows) -> None:
    writer.executemany("INSERT INTO pair_numeric_changes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", list(rows))


def insert_pair_claims(writer: OutputWriter, rows: Rows) -> None:
    writer.executemany("INSERT INTO pair_claims VALUES (?, ?, ?, ?, ?, ?, ?)", list(rows))


def insert_pair_cues(writer: OutputWriter, rows: Rows) -> None:
    writer.executemany("INSERT INTO pair_frame_cues VALUES (?, ?, ?, ?, ?, ?)", list(rows))


def upsert_article_metrics(writer: OutputWriter, row: Tuple[Any, ...]) -> None:
    writer.execute(
        """
        INSERT OR REPLACE INTO article_metrics (
            article_id, news_org, overstate_institutional_share,
            distinct_sources_delta, anonymity_rate_delta, hedge_density_delta
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        row,
    )
