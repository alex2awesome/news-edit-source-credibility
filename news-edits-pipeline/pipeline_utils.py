"""Shared helpers for the news edits pipeline."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from functools import lru_cache
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
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        schema_sql = schema_path.read_text(encoding="utf-8")
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
