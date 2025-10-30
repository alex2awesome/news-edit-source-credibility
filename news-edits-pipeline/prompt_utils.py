"""Prompt-centric helpers for the news edits pipeline."""

from __future__ import annotations

import re
from typing import Any, Dict


def text_jaccard_similarity(text_a: str, text_b: str) -> float:
    """Compute case-insensitive Jaccard similarity between token sets."""
    tokens_a = {tok.lower() for tok in re.findall(r"\w+", text_a or "")}
    tokens_b = {tok.lower() for tok in re.findall(r"\w+", text_b or "")}
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    return len(intersection) / len(union)


def build_pair_prompt_payloads(
    prev_version: Dict[str, Any],
    curr_version: Dict[str, Any],
    prev_payload: Dict[str, str],
    curr_payload: Dict[str, str],
) -> Dict[str, Dict[str, Any]]:
    """Prepare variable payloads for all pairwise prompts."""
    prev_text = prev_version.get("summary") or ""
    curr_text = curr_version.get("summary") or ""
    prev_title = prev_version.get("title") or ""
    curr_title = curr_version.get("title") or ""

    prev_sources_brief = prev_payload.get("source_mentions_brief") or prev_payload.get("source_mentions") or "[]"
    curr_sources_brief = curr_payload.get("source_mentions_brief") or curr_payload.get("source_mentions") or "[]"
    prev_sources_identity = (
        prev_payload.get("source_mentions_identity") or prev_sources_brief
    )
    curr_sources_identity = (
        curr_payload.get("source_mentions_identity") or curr_sources_brief
    )
    prev_sources_speech = prev_payload.get("source_mentions_speech") or prev_sources_brief
    curr_sources_speech = curr_payload.get("source_mentions_speech") or curr_sources_brief
    prev_protest_json = prev_payload.get("protest", '{"frame_cues": [], "roles": [], "confidence": 1}')
    curr_protest_json = curr_payload.get("protest", '{"frame_cues": [], "roles": [], "confidence": 1}')

    return {
        "A3_edit_type_pair": {
            "prev_title": prev_title,
            "curr_title": curr_title,
            "prev_version_text": prev_text,
            "curr_version_text": curr_text,
            "prev_source_mentions": prev_sources_brief,
            "curr_source_mentions": curr_sources_brief,
        },
        "P3_anon_named_replacement_pair": {
            "v_prev": prev_text,
            "v_curr": curr_text,
            "prev_source_mentions": prev_sources_identity,
            "curr_source_mentions": curr_sources_identity,
        },
        "P4_verb_strength_pair": {
            "v_prev": prev_text,
            "v_curr": curr_text,
            "prev_source_mentions": prev_sources_speech,
            "curr_source_mentions": curr_sources_speech,
        },
        "P5_speech_style_pair": {
            "v_prev": prev_text,
            "v_curr": curr_text,
            "prev_source_mentions": prev_sources_speech,
            "curr_source_mentions": curr_sources_speech,
        },
        "P7_numeric_changes_pair": {
            "v_prev": prev_text,
            "v_curr": curr_text,
        },
        "P8_claims_pair": {
            "v_prev": prev_text,
            "v_curr": curr_text,
        },
        "P9_frame_cues_pair": {
            "v_prev": prev_text,
            "v_curr": curr_text,
            "prev_protest_cues": prev_protest_json,
            "curr_protest_cues": curr_protest_json,
        },
        "P10_movement_pair": {
            "v_prev": prev_text,
            "v_curr": curr_text,
        },
        "P16_stance_entities_pair": {
            "v_prev": prev_text,
            "v_curr": curr_text,
            "prev_source_mentions": prev_sources_brief,
            "curr_source_mentions": curr_sources_brief,
        },
        "D5_angle_change_pair": {
            "prev_version_text": prev_text,
            "curr_version_text": curr_text,
            "prev_title": prev_title,
            "curr_title": curr_title,
            "prev_sources": prev_sources_brief,
            "curr_sources": curr_sources_brief,
        },
    }
