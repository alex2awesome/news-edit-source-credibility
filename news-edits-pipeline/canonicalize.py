"""Utilities for canonicalizing and matching source names."""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, Optional

from rapidfuzz import fuzz

_ABBREV_MAP = {
    "u.s.": "united states",
    "u.s": "united states",
    "us": "united states",
    "u.k.": "united kingdom",
    "uk": "united kingdom",
    "u.n.": "united nations",
    "nyc": "new york city",
    "nypd": "new york police department",
    "lapd": "los angeles police department",
}

_PUNCT_RE = re.compile(r"[^\w\s-]")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_source(surface: str) -> str:
    """Return a normalized representation of a source surface string."""
    if not surface:
        return ""

    text = unicodedata.normalize("NFKC", surface).lower().strip()
    text = _ABBREV_MAP.get(text, text)
    text = _PUNCT_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def fuzzy_match_source(surface: str, known: Dict[str, Dict[str, str]], threshold: int = 92) -> Optional[str]:
    """Attempt to fuzzy-match a surface form to an existing canonical source."""
    norm = normalize_source(surface)
    if not norm or not known:
        return None

    best_key: Optional[str] = None
    best_score = 0
    for key, meta in known.items():
        candidate = meta.get("normalized")
        if not candidate:
            continue
        score = fuzz.ratio(norm, candidate)
        if score > best_score and score >= threshold:
            best_score = score
            best_key = key
    return best_key


__all__ = ["normalize_source", "fuzzy_match_source"]
