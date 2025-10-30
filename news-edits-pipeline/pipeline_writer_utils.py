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
            source_canonical, source_surface, source_type, speech_style, attribution_verb,
            char_start, char_end, sentence_index, paragraph_index,
            is_in_title, is_in_lede, attributed_text, is_anonymous, anonymous_description,
            anonymous_domain, evidence_type, evidence_text, narrative_function,
            centrality, perspective, doubted, hedge_count, hedge_markers,
            epistemic_verbs, hedge_stance, hedge_confidence, prominence_lead_pct,
            confidence
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?
        )
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
        INSERT OR REPLACE INTO version_metrics (
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
            article_id,
            news_org,
            from_version_id,
            to_version_id,
            from_version_num,
            to_version_num,
            delta_minutes,
            tokens_added,
            tokens_deleted,
            percent_text_new,
            movement_upweighted_summary,
            movement_downweighted_summary,
            movement_notes,
            movement_confidence,
            movement_notable_shifts,
            edit_type,
            edit_summary,
            edit_confidence,
            angle_changed,
            angle_change_category,
            angle_summary,
            title_alignment_notes,
            angle_confidence,
            angle_evidence,
            title_jaccard_prev,
            title_jaccard_curr,
            summary_jaccard
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        list(rows),
    )


def insert_pair_sources_added(writer: OutputWriter, rows: Rows) -> None:
    writer.executemany(
        """
        INSERT INTO pair_sources_added (
            article_id, news_org, from_version_id, to_version_id, surface, canonical, type
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        list(rows),
    )


def insert_pair_sources_removed(writer: OutputWriter, rows: Rows) -> None:
    writer.executemany(
        """
        INSERT INTO pair_sources_removed (
            article_id, news_org, from_version_id, to_version_id, surface, canonical, type
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        list(rows),
    )


def insert_pair_source_transitions(writer: OutputWriter, rows: Rows) -> None:
    writer.executemany(
        """
        INSERT INTO pair_source_transitions (
            article_id, news_org, from_version_id, to_version_id,
            canonical, transition_type, reason_category, reason_detail
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        list(rows),
    )


def insert_pair_replacements(writer: OutputWriter, rows: Rows) -> None:
    writer.executemany(
        """
        INSERT INTO pair_anon_named_replacements (
            article_id, news_org, from_version_id, to_version_id,
            src, dst, direction, likelihood
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        list(rows),
    )


def insert_pair_numeric_changes(writer: OutputWriter, rows: Rows) -> None:
    writer.executemany(
        """
        INSERT INTO pair_numeric_changes (
            article_id, news_org, from_version_id, to_version_id,
            item, prev, curr, delta, unit, source, change_type, confidence
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        list(rows),
    )


def insert_pair_claims(writer: OutputWriter, rows: Rows) -> None:
    writer.executemany(
        """
        INSERT INTO pair_claims (
            article_id, news_org, from_version_id, to_version_id,
            claim_id, proposition, status, change_note, confidence
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        list(rows),
    )


def insert_pair_cues(writer: OutputWriter, rows: Rows) -> None:
    writer.executemany(
        """
        INSERT INTO pair_frame_cues (
            article_id, news_org, from_version_id, to_version_id,
            cue, prev, curr, direction
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        list(rows),
    )


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
