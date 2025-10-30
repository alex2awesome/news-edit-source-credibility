"""Shared helpers for the news edits pipeline."""

from __future__ import annotations

import logging
import re
import sqlite3
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import orjson


logger = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).parent / "prompts"
JSON_ONLY_FOOTER = (
    "\n\nFollow the schema exactly and return ONLY the JSON payload. "
    "Do not repeat the schema, include explanations, or add any extra text."
)


def ensure_dir(path: Path) -> None:
    """Create directories as needed."""
    path.mkdir(parents=True, exist_ok=True)


def prompt_path(key: str) -> Path:
    return PROMPT_DIR / f"{key}.prompt"


def schema_path(key: str) -> Path:
    return PROMPT_DIR / f"{key}.output.json"


@lru_cache(maxsize=None)
def load_prompt_template(key: str) -> str:
    return prompt_path(key).read_text(encoding="utf-8")


def render_prompt(template: str, variables: Dict[str, Any]) -> str:
    pattern = re.compile(r"\{\{\s*(\w+)\s*\}\}")
    prompt = pattern.sub(lambda match: str(variables.get(match.group(1), "")), template)
    lower_prompt = prompt.lower()
    if "return only" not in lower_prompt and "output only" not in lower_prompt:
        prompt = prompt.rstrip() + JSON_ONLY_FOOTER
    return prompt


def _schema_name(key: str) -> str:
    return "".join(part.capitalize() for part in key.split("_"))


@lru_cache(maxsize=None)
def load_schema(key: str) -> Dict[str, Any]:
    return orjson.loads(schema_path(key).read_bytes())


@lru_cache(maxsize=None)
def response_format(key: str) -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": _schema_name(key),
            "schema": load_schema(key),
        },
    }


def schema_text(key: str) -> str:
    return orjson.dumps(load_schema(key), option=orjson.OPT_INDENT_2).decode("utf-8")


def _normalize_confidence_value(value: Any, default: int = 3) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if numeric <= 0:
        return 1
    integer = int(round(numeric))
    return max(1, min(5, integer))


def normalize_confidence(obj: Any, default: int = 3) -> Any:
    if isinstance(obj, dict):
        normalized: Dict[str, Any] = {}
        for key, value in obj.items():
            if key == "confidence":
                normalized[key] = _normalize_confidence_value(value, default=default)
            else:
                normalized[key] = normalize_confidence(value, default=default)
        return normalized
    if isinstance(obj, list):
        return [normalize_confidence(item, default=default) for item in obj]
    return obj


def _prune_low_confidence(obj: Any, threshold: float) -> Any:
    if isinstance(obj, dict):
        return {key: _prune_low_confidence(value, threshold) for key, value in obj.items()}
    if isinstance(obj, list):
        pruned = []
        for item in obj:
            if isinstance(item, dict) and item.get("confidence", threshold) < threshold:
                continue
            pruned.append(_prune_low_confidence(item, threshold))
        return pruned
    return obj


def prune_low_confidence(obj: Any, threshold: float) -> Any:
    normalized = normalize_confidence(obj, default=int(round(threshold)))
    return _prune_low_confidence(normalized, threshold)


def find_index_by_offset(spans: List[Dict[str, Any]], char_start: int) -> int:
    for span in spans:
        if span["start"] <= char_start < span["end"]:
            return span["index"]
    return -1


def extract_context_window(
    text: str,
    tokens: List[Dict[str, Any]],
    char_start: int,
    char_end: int,
    window_tokens: int,
) -> str:
    if not tokens:
        return text[max(0, char_start - 200) : char_end + 200]
    center = 0
    for idx, token in enumerate(tokens):
        if token["start"] <= char_start < token["end"]:
            center = idx
            break
    start_idx = max(0, center - window_tokens)
    end_idx = min(len(tokens), center + window_tokens)
    start_char = tokens[start_idx]["start"]
    end_char = tokens[end_idx - 1]["end"] if end_idx > start_idx else tokens[start_idx]["end"]
    return text[start_char:end_char]


def parse_json_response(raw: str) -> Any:
    try:
        return orjson.loads(raw)
    except orjson.JSONDecodeError:
        candidates: List[Tuple[Any, int]] = []
        decoder = json.JSONDecoder()
        idx = 0
        while idx < len(raw):
            try:
                obj, end = decoder.raw_decode(raw, idx)
            except json.JSONDecodeError:
                idx += 1
                continue
            if isinstance(obj, (dict, list)):
                candidates.append((obj, idx))
            idx = max(end, idx + 1)

        def is_schema(obj: Any) -> bool:
            return isinstance(obj, dict) and "type" in obj and "properties" in obj and "required" in obj

        non_schema = [obj for obj, _ in candidates if not is_schema(obj)]
        if non_schema:
            return non_schema[-1]
        logger.warning("Model response contained only schema definitions; returning empty object.")
        return {}
    except Exception:  # pragma: no cover - defensive fallback
        logger.exception("Failed to parse model response; returning empty object.")
        return {}


class OutputWriter:
    def __init__(self, db_path: Path, schema_path: Path):
        ensure_dir(db_path.parent)
        self.db_path = Path(db_path)
        schema_sql = schema_path.read_text(encoding="utf-8")
        needs_rebuild = False

        if self.db_path.exists() and self.db_path.stat().st_size > 0:
            logger.debug("Existing database found at %s; checking schema", self.db_path)
            with sqlite3.connect(self.db_path) as temp_conn:
                if _schema_requires_rebuild(temp_conn):
                    needs_rebuild = True
                    logger.warning("Existing database %s has an outdated schema; it will be rebuilt.", self.db_path)

            if needs_rebuild:
                self._recreate_database_files()

        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(schema_sql)

    def execute(self, sql: str, params: Tuple[Any, ...]) -> None:
        self.conn.execute(sql, params)

    def executemany(self, sql: str, params: List[Tuple[Any, ...]]) -> None:
        if params:
            self.conn.executemany(sql, params)

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def _recreate_database_files(self) -> None:
        try:
            self.db_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Failed to remove database file %s during rebuild; continuing", self.db_path)
        for suffix in ("-wal", "-shm"):
            try:
                aux_path = self.db_path.with_name(f"{self.db_path.name}{suffix}")
                aux_path.unlink(missing_ok=True)
            except OSError:
                logger.debug("Failed to remove auxiliary SQLite file %s; continuing", aux_path)


EXPECTED_TABLE_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "articles": (
        "article_id",
        "news_org",
        "url",
        "title_first",
        "title_final",
        "original_publication_time",
        "total_edits",
        "is_live_blog",
    ),
    "versions": (
        "version_id",
        "article_id",
        "news_org",
        "version_num",
        "timestamp_utc",
        "title",
        "char_len",
    ),
    "source_mentions": (
        "version_id",
        "article_id",
        "news_org",
        "source_id_within_article",
        "source_canonical",
        "source_type",
        "speech_style",
        "attribution_verb",
        "char_start",
        "char_end",
        "sentence_index",
        "paragraph_index",
        "is_in_title",
        "is_in_lede",
        "attributed_text",
        "is_anonymous",
        "anonymous_description",
        "anonymous_domain",
        "evidence_type",
        "evidence_text",
        "prominence_lead_pct",
        "confidence",
    ),
    "sources_agg": (
        "article_id",
        "news_org",
        "source_id_within_article",
        "source_canonical",
        "source_type",
        "first_seen_version",
        "first_seen_time",
        "last_seen_version",
        "last_seen_time",
        "num_mentions_total",
        "num_versions_present",
        "total_attributed_words",
        "voice_retention_index",
        "mean_prominence",
        "lead_appearance_count",
        "title_appearance_count",
        "doubted_any",
        "deemphasized_any",
        "disappeared_any",
    ),
    "entity_mentions": (
        "version_id",
        "article_id",
        "news_org",
        "entity_id_within_article",
        "entity_type",
        "canonical_name",
        "char_start",
        "char_end",
        "sentence_index",
        "paragraph_index",
    ),
    "version_pairs": (
        "article_id",
        "news_org",
        "from_version_id",
        "to_version_id",
        "from_version_num",
        "to_version_num",
        "delta_minutes",
        "tokens_added",
        "tokens_deleted",
        "percent_text_new",
        "movement_upweighted_summary",
        "movement_downweighted_summary",
        "movement_notes",
        "edit_type",
        "angle_changed",
        "angle_change_category",
        "angle_summary",
        "title_alignment_notes",
        "title_jaccard_prev",
        "title_jaccard_curr",
        "summary_jaccard",
    ),
    "pair_sources_added": (
        "from_version_id",
        "to_version_id",
        "canonical",
        "type",
    ),
    "pair_sources_removed": (
        "from_version_id",
        "to_version_id",
        "canonical",
        "type",
    ),
    "pair_source_transitions": (
        "from_version_id",
        "to_version_id",
        "canonical",
        "transition_type",
        "reason_category",
        "reason_detail",
    ),
    "pair_anon_named_replacements": (
        "from_version_id",
        "to_version_id",
        "src",
        "dst",
        "direction",
        "likelihood",
    ),
    "pair_numeric_changes": (
        "from_version_id",
        "to_version_id",
        "item",
        "prev",
        "curr",
        "delta",
        "unit",
        "source",
        "change_type",
        "confidence",
    ),
    "pair_claims": (
        "from_version_id",
        "to_version_id",
        "claim_id",
        "proposition",
        "status",
        "change_note",
        "confidence",
    ),
    "pair_frame_cues": (
        "from_version_id",
        "to_version_id",
        "cue",
        "prev",
        "curr",
        "direction",
    ),
    "version_metrics": (
        "version_id",
        "article_id",
        "news_org",
        "distinct_sources",
        "institutional_share_words",
        "anonymous_source_share_words",
        "hedge_density_per_1k",
    ),
    "article_metrics": (
        "article_id",
        "news_org",
        "overstate_institutional_share",
        "distinct_sources_delta",
        "anonymity_rate_delta",
        "hedge_density_delta",
    ),
}


def _schema_requires_rebuild(conn: sqlite3.Connection) -> bool:
    for table, expected_columns in EXPECTED_TABLE_COLUMNS.items():
        try:
            cursor = conn.execute(f"PRAGMA table_info({table})")
        except sqlite3.DatabaseError:
            logger.debug("Failed to inspect table %s; forcing schema rebuild.", table)
            return True
        existing_columns = [row[1] for row in cursor.fetchall()]
        if not existing_columns:
            continue
        missing = [column for column in expected_columns if column not in existing_columns]
        if missing:
            logger.debug("Table %s is missing columns %s", table, missing)
            return True
    return False
