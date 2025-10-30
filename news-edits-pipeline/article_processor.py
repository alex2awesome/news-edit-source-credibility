"""Per-article processing for the news edits pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import orjson

from analysis import (
    aggregate_sources_over_versions,
    compute_diff_magnitude,
    compute_prominence_features,
    extract_lede,
    final_version_bias,
    inter_update_timing,
    jaccard_title_body,
    ner_entities_spacy,
    segment,
)
from canonicalize import fuzzy_match_source, normalize_source
from config import Config
from llm_client import StructuredLLMClient
from loader import load_versions
from pipeline_utils import (
    ensure_dir,
    extract_context_window,
    find_index_by_offset,
    load_prompt_template,
    prune_low_confidence,
    render_prompt,
)
from prompt_utils import build_pair_prompt_payloads, text_jaccard_similarity


logger = logging.getLogger(__name__)

HEDGE_WINDOW_BATCH_SIZE = 5


@dataclass
class ArticleResult:
    entry_id: int
    news_org: str
    article_row: Tuple[Any, ...]
    versions_rows: List[Tuple[Any, ...]]
    source_mentions_rows: List[Tuple[Any, ...]]
    entity_rows: List[Tuple[Any, ...]]
    version_metrics_rows: List[Tuple[Any, ...]]
    pair_rows: List[Tuple[Any, ...]]
    pair_sources_added: List[Tuple[Any, ...]]
    pair_sources_removed: List[Tuple[Any, ...]]
    pair_source_transitions: List[Tuple[Any, ...]]
    pair_replacements: List[Tuple[Any, ...]]
    pair_numeric: List[Tuple[Any, ...]]
    pair_claims_rows: List[Tuple[Any, ...]]
    pair_cues_rows: List[Tuple[Any, ...]]
    sources_agg_rows: List[Tuple[Any, ...]]
    article_metrics_row: Optional[Tuple[Any, ...]]
    live_blog_only: bool
    log_message: str


def _assign_source_id(mention: Dict[str, Any], registry: Dict[str, Dict[str, Any]]) -> str:
    canonical = mention.get("canonical") or ""
    surface = mention.get("surface") or ""
    norm = normalize_source(canonical) or normalize_source(surface)
    if not norm:
        return ""

    for key, value in registry.items():
        if value["normalized"] == norm:
            return key
        if value["canonical"] == canonical:
            return key

    match = fuzzy_match_source(norm, registry)
    if match:
        return match

    source_id = f"s{len(registry) + 1:03d}"
    registry[source_id] = {"normalized": norm, "canonical": canonical}
    return source_id


def _assign_entity_id(surface: str, registry: Dict[str, str]) -> str:
    norm = normalize_source(surface)
    for key, value in registry.items():
        if value == norm:
            return key
    entity_id = f"e{len(registry) + 1:03d}"
    registry[entity_id] = norm
    return entity_id


def _spans_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return not (a[1] <= b[0] or a[0] >= b[1])


def _find_available_span(text: str, snippet: str, used_ranges: List[Tuple[int, int]]) -> Tuple[int, int]:
    if not snippet:
        return -1, -1
    haystack = text.lower()
    needle = snippet.lower()
    search_from = 0
    length = len(snippet)
    while True:
        idx = haystack.find(needle, search_from)
        if idx == -1:
            return -1, -1
        candidate = (idx, idx + length)
        if not any(_spans_overlap(candidate, existing) for existing in used_ranges):
            return candidate
        search_from = idx + 1


def _generate_snippet_variants(snippet: str) -> List[str]:
    variants: List[str] = []
    if not snippet:
        return variants
    variants.append(snippet)
    stripped = snippet.strip()
    if stripped and stripped not in variants:
        variants.append(stripped)
    normalized_quotes = stripped.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    if normalized_quotes and normalized_quotes not in variants:
        variants.append(normalized_quotes)
    return variants


def _locate_best_span(text: str, snippet: str, used_ranges: List[Tuple[int, int]]) -> Tuple[int, int, str]:
    variants = _generate_snippet_variants(snippet)
    for variant in variants:
        start, end = _find_available_span(text, variant, used_ranges)
        if start != -1:
            return start, end, text[start:end]

    for variant in variants:
        cleaned = variant.strip()
        if not cleaned:
            continue
        min_len = max(8, len(cleaned) // 2)
        length = len(cleaned)
        while length >= min_len:
            prefix = cleaned[:length]
            start, end = _find_available_span(text, prefix, used_ranges)
            if start != -1:
                return start, end, text[start:end]

            suffix = cleaned[-length:]
            start, end = _find_available_span(text, suffix, used_ranges)
            if start != -1:
                return start, end, text[start:end]
            length -= 1
    return -1, -1, ""


def _normalize_version_number(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError(f"Unable to coerce version number {value!r} to int") from exc


def process_article(
    entry_id: int,
    db_path: Path,
    config: Config,
    client: Optional[StructuredLLMClient],
    out_root: Path,
    verbose: bool = False,
) -> Optional[ArticleResult]:
    """Process a single article and return the structured results."""
    owns_client = False
    if client is None:
        client = StructuredLLMClient(config)
        owns_client = True

    cache_enabled = config.cache_raw_responses
    article_dir = out_root / str(entry_id)
    if cache_enabled:
        ensure_dir(article_dir)

    try:
        versions = load_versions(str(db_path), entry_id)
        if not versions:
            logger.info("Skipping entry %s: no versions returned from loader", entry_id)
            return None

        if len(versions) < config.min_versions:
            logger.info(
                "Skipping entry %s during processing: only %d version(s) < min_versions=%d",
                entry_id,
                len(versions),
                config.min_versions,
            )
            return None
        if config.max_versions is not None and len(versions) >= config.max_versions:
            logger.info(
                "Skipping entry %s during processing: %d version(s) exceeds max_versions=%d",
                entry_id,
                len(versions),
                config.max_versions,
            )
            return None

        similarity_threshold = config.skip_similarity_threshold
        if similarity_threshold is not None and len(versions) >= 2:
            first_text = versions[0].get("summary") or ""
            final_text = versions[-1].get("summary") or ""
            similarity = text_jaccard_similarity(first_text, final_text)
            if similarity_threshold > 0 and similarity >= similarity_threshold:
                logger.info(
                    "Skipping entry %s: high similarity between first and final versions "
                    "(Jaccard=%.3f, threshold=%.3f)",
                    entry_id,
                    similarity,
                    similarity_threshold,
                )
                return None

        news_org = versions[0]["source"]
        url = versions[0]["url"]
        title_first = versions[0]["title"]
        title_final = versions[-1]["title"]
        original_ts = versions[0]["created"]

        source_registry: Dict[str, Dict[str, Any]] = {}
        entity_registry: Dict[str, str] = {}
        source_mentions_records: List[Tuple[Any, ...]] = []
        entity_rows: List[Tuple[Any, ...]] = []
        version_metrics_rows: List[Tuple[Any, ...]] = []
        pair_rows: List[Tuple[Any, ...]] = []
        pair_sources_added: List[Tuple[Any, ...]] = []
        pair_sources_removed: List[Tuple[Any, ...]] = []
        pair_source_transitions: List[Tuple[Any, ...]] = []
        pair_replacements: List[Tuple[Any, ...]] = []
        pair_numeric: List[Tuple[Any, ...]] = []
        pair_claims_rows: List[Tuple[Any, ...]] = []
        pair_cues_rows: List[Tuple[Any, ...]] = []

        source_mentions_by_version: List[Dict[str, Any]] = []
        version_meta: List[Dict[str, Any]] = []
        ledes: Dict[int, Dict[str, Any]] = {}
        version_numeric_metrics: Dict[int, Dict[str, Any]] = {}
        versions_rows: List[Tuple[Any, ...]] = []
        version_payloads: Dict[str, Dict[str, str]] = {}

        live_blog_checked = False
        live_blog_flag = False
        skip_article_due_to_live_blog = False

        for version in versions:
            if skip_article_due_to_live_blog:
                break
            version_id = version["id"]
            version_num = _normalize_version_number(version["version"])
            version_dir = article_dir / f"v{version_num:03d}"
            if cache_enabled:
                ensure_dir(version_dir)

            text = version.get("summary") or ""
            title = version.get("title") or ""

            version_payloads.setdefault(version_id, {})

            if not live_blog_checked:
                rendered_d4 = render_prompt(
                    load_prompt_template("D4_live_blog_detect"),
                    {"version_text": text},
                )
                d4_result = client.run(
                    "D4_live_blog_detect",
                    rendered_d4,
                    version_dir / "D4_live_blog_detect.json",
                )
                d4_result = prune_low_confidence(d4_result, config.accept_confidence_min)
                version_payloads[version_id]["live_blog"] = orjson.dumps(
                    d4_result,
                    option=orjson.OPT_INDENT_2,
                ).decode("utf-8")
                if d4_result.get("is_live_blog"):
                    live_blog_flag = True
                    skip_article_due_to_live_blog = True
                    logger.info("Article %s flagged as live blog; skipping remaining processing", entry_id)
                    break
                live_blog_checked = True

            seg = segment(text, config.spacy_model)
            lede = extract_lede(title, text, seg["paragraphs"])
            ledes[version_num] = lede

            entity_doc = ner_entities_spacy(text, config.spacy_model)
            for ent in entity_doc.get("entities", []):
                entity_id = _assign_entity_id(ent["canonical"], entity_registry)
                sentence_index = find_index_by_offset(seg["sentences"], ent.get("char_start", -1))
                paragraph_index = find_index_by_offset(seg["paragraphs"], ent.get("char_start", -1))
                entity_rows.append(
                    (
                        version_id,
                        entry_id,
                        news_org,
                        entity_id,
                        ent.get("label"),
                        ent.get("canonical"),
                        ent.get("char_start"),
                        ent.get("char_end"),
                        sentence_index,
                        paragraph_index,
                    )
                )

            rendered_a1 = render_prompt(
                load_prompt_template("A1_source_mentions"),
                {
                    "news_article": text,
                    "title": title,
                },
            )
            a1_result = client.run(
                "A1_source_mentions",
                rendered_a1,
                version_dir / "A1_source_mentions.json",
            )
            a1_result = prune_low_confidence(a1_result, config.accept_confidence_min)
            mentions_raw = a1_result.get("source_mentions", [])

            processed_mentions: List[Dict[str, Any]] = []
            used_source_spans: List[Tuple[int, int]] = []
            title_lower = title.lower()
            char_len = seg.get("char_len", len(text))

            for mention in mentions_raw:
                enriched = dict(mention)
                surface = enriched.get("surface") or ""
                canonical = enriched.get("canonical") or ""
                match = fuzzy_match_source((canonical or surface).lower(), source_registry)
                if match:
                    enriched["canonical"] = source_registry[match]["canonical"]

                span_start = enriched.get("char_start", -1)
                span_end = enriched.get("char_end", span_start)
                matched_text = enriched.get("attributed_text") or ""

                if span_start is None or span_start < 0 or span_end is None or span_end <= span_start:
                    candidate_start, candidate_end, candidate_match = _locate_best_span(
                        text,
                        matched_text or surface,
                        used_source_spans,
                    )
                    if candidate_start != -1:
                        span_start, span_end = candidate_start, candidate_end
                        matched_text = candidate_match
                        used_source_spans.append((candidate_start, candidate_end))
                    else:
                        candidate_start, candidate_end, candidate_match = _locate_best_span(text, surface, [])
                        if candidate_start != -1:
                            span_start, span_end = candidate_start, candidate_end
                            matched_text = candidate_match
                            used_source_spans.append((candidate_start, candidate_end))

                enriched["char_start"] = span_start
                enriched["char_end"] = span_end if span_end != -1 else span_start
                if matched_text:
                    enriched["attributed_text"] = matched_text

                sentence_index = find_index_by_offset(seg["sentences"], span_start) if span_start >= 0 else -1
                paragraph_index = find_index_by_offset(seg["paragraphs"], span_start) if span_start >= 0 else -1
                enriched["sentence_index"] = sentence_index
                enriched["paragraph_index"] = paragraph_index
                surface_lower = surface.lower()
                enriched["is_in_title"] = bool(surface_lower and surface_lower in title_lower)
                enriched["is_in_lede"] = paragraph_index == 0 and paragraph_index != -1
                prominence = compute_prominence_features(
                    span_start if span_start >= 0 else 0,
                    char_len,
                    enriched["is_in_title"],
                    enriched["is_in_lede"],
                )
                enriched["prominence"] = prominence
                processed_mentions.append(enriched)

            source_catalog_for_prompt: List[Dict[str, Any]] = []
            for idx, catalog_item in enumerate(processed_mentions, start=1):
                prompt_source_id = f"source_{idx:03d}"
                catalog_item["_prompt_source_id"] = prompt_source_id
                source_catalog_for_prompt.append(
                    {
                        "source_id": prompt_source_id,
                        "canonical": catalog_item.get("canonical")
                        or catalog_item.get("surface")
                        or "",
                        "type": catalog_item.get("type"),
                        "attributed_text": catalog_item.get("attributed_text") or "",
                    }
                )
            source_catalog_json = orjson.dumps(
                source_catalog_for_prompt,
                option=orjson.OPT_INDENT_2,
            ).decode("utf-8")

            version_payloads[version_id]["source_catalog"] = source_catalog_json

            batch_entries: List[Dict[str, Any]] = []
            batch_requests: List[Tuple[str, str, Path, int]] = []
            mention_lookup: Dict[str, Dict[str, Any]] = {}
            batch_index = 1

            for idx, mention in enumerate(processed_mentions, start=1):
                prompt_source_id = mention.get("_prompt_source_id")
                if not prompt_source_id:
                    prompt_source_id = f"source_{idx:03d}"
                    mention["_prompt_source_id"] = prompt_source_id
                mention_lookup[prompt_source_id] = mention

                span_start = mention.get("char_start", -1)
                span_end = mention.get("char_end", span_start)
                if span_start is None or span_start < 0:
                    span_start = 0
                if span_end is None or span_end < span_start:
                    span_end = span_start + len(mention.get("attributed_text", ""))
                span_end = min(len(text), span_end)

                context_window = extract_context_window(
                    text,
                    seg["tokens"],
                    span_start,
                    span_end,
                    config.hedge_window_tokens,
                )
                target_source_payload = {
                    "source_id": prompt_source_id,
                    "canonical": mention.get("canonical") or mention.get("surface") or "",
                    "type": mention.get("type"),
                    "attributed_text": mention.get("attributed_text") or "",
                }
                batch_entries.append(
                    {
                        "source_id": prompt_source_id,
                        "context_window_text": context_window,
                        "target_source": target_source_payload,
                    }
                )

                if len(batch_entries) == HEDGE_WINDOW_BATCH_SIZE or idx == len(processed_mentions):
                    target_sources_json = orjson.dumps(
                        batch_entries,
                        option=orjson.OPT_INDENT_2,
                    ).decode("utf-8")
                    rendered_a2 = render_prompt(
                        load_prompt_template("A2_hedge_window"),
                        {
                            "target_sources": target_sources_json,
                            "source_catalog": source_catalog_json,
                        },
                    )
                    cache_path = version_dir / f"A2_hedge_batch_{batch_index:03d}.json"
                    batch_requests.append(("A2_hedge_window", rendered_a2, cache_path, config.llm_retries))
                    batch_entries = []
                    batch_index += 1

            a2_responses: List[Dict[str, Any]] = []
            if batch_requests:
                a2_responses = client.run_many(batch_requests)
                if len(a2_responses) != len(batch_requests):
                    raise RuntimeError(
                        f"Expected {len(batch_requests)} A2 responses but received {len(a2_responses)}"
                    )

            for a2_raw in a2_responses:
                a2_result = prune_low_confidence(a2_raw, config.accept_confidence_min)
                for source_result in a2_result.get("sources", []):
                    source_key = source_result.get("source_id")
                    if not source_key:
                        continue
                    mention = mention_lookup.get(source_key)
                    if not mention:
                        continue
                    mention["hedge_analysis"] = source_result
                    mention["doubted"] = source_result.get("stance_toward_source") == "skeptical"
                    mention["hedge_count"] = source_result.get("hedge_count", 0)

            roles_result: Dict[str, Any] = {}
            if processed_mentions:
                target_sources_for_prompts = []
                for mention in processed_mentions:
                    prompt_source_id = mention.get("_prompt_source_id")
                    if not prompt_source_id:
                        continue
                    target_sources_for_prompts.append(
                        {
                            "source_id": prompt_source_id,
                            "name": mention.get("canonical") or mention.get("surface") or "",
                            "surface": mention.get("surface"),
                            "attributed_text": mention.get("attributed_text"),
                            "sentence_index": mention.get("sentence_index"),
                            "paragraph_index": mention.get("paragraph_index"),
                        }
                    )

                if target_sources_for_prompts:
                    target_sources_json = orjson.dumps(
                        target_sources_for_prompts,
                        option=orjson.OPT_INDENT_2,
                    ).decode("utf-8")

                    roles_rendered = render_prompt(
                        load_prompt_template("N1_narrative_keywords"),
                        {
                            "news_article": text,
                            "target_sources": target_sources_json,
                        },
                    )

                    roles_result = client.run(
                        "N1_narrative_keywords",
                        roles_rendered,
                        version_dir / "N1_narrative_keywords.json",
                    )

                    roles_result = prune_low_confidence(roles_result, config.accept_confidence_min)

                    roles_lookup = {
                        item.get("source_id"): item
                        for item in roles_result.get("sources", [])
                        if item.get("source_id")
                    }

                    for mention in processed_mentions:
                        prompt_source_id = mention.get("_prompt_source_id")
                        if not prompt_source_id:
                            continue
                        roles_entry = roles_lookup.get(prompt_source_id)
                        if roles_entry:
                            mention["narrative_function"] = roles_entry.get("narrative_function")
                            perspectives = roles_entry.get("perspective") or []
                            if isinstance(perspectives, list):
                                mention["perspective"] = perspectives
                            else:
                                mention["perspective"] = [str(perspectives)]
                            mention["centrality"] = roles_entry.get("centrality")

                    version_payloads[version_id]["source_roles"] = orjson.dumps(
                        roles_result,
                        option=orjson.OPT_INDENT_2,
                    ).decode("utf-8")

            mentions: List[Dict[str, Any]] = []
            anonymous_words = 0
            total_attributed_words = 0
            institutional_words = 0
            hedge_total = 0
            distinct_source_ids = set()
            institutional_types = {"government", "corporate", "law_enforcement"}

            for mention in processed_mentions:
                if "hedge_analysis" not in mention:
                    mention["hedge_analysis"] = {}
                mention["doubted"] = bool(mention.get("doubted", False))
                hedge_count_val = mention.get("hedge_count", 0)
                try:
                    mention["hedge_count"] = int(hedge_count_val)
                except (TypeError, ValueError):
                    mention["hedge_count"] = 0

                mention.pop("_prompt_source_id", None)

                is_anonymous = bool(mention.get("is_anonymous"))
                mention["is_anonymous"] = is_anonymous
                anonymous_description = (mention.get("anonymous_description") or "").strip()
                anonymous_domain = (mention.get("anonymous_domain") or "unknown").lower()
                if anonymous_domain not in {"government", "corporate", "law_enforcement", "individual", "unknown"}:
                    anonymous_domain = "unknown"
                mention["anonymous_description"] = anonymous_description
                mention["anonymous_domain"] = anonymous_domain
                evidence_type = (mention.get("evidence_type") or "other").lower()
                if evidence_type not in {
                    "official_statement",
                    "press_release",
                    "eyewitness",
                    "document",
                    "statistic",
                    "prior_reporting",
                    "social_media",
                    "court_filing",
                    "other",
                }:
                    evidence_type = "other"
                evidence_text = mention.get("evidence_text") or mention.get("attributed_text") or ""
                mention["evidence_type"] = evidence_type
                mention["evidence_text"] = evidence_text

                source_id = _assign_source_id(mention, source_registry)
                mention["source_id_within_article"] = source_id

                attributed_text = mention.get("attributed_text") or ""
                words = len(attributed_text.split())
                total_attributed_words += words
                if mention.get("type") in institutional_types:
                    institutional_words += words
                hedge_total += mention.get("hedge_count", 0)
                if is_anonymous:
                    if attributed_text:
                        anonymous_words += len(attributed_text.split())
                    elif anonymous_description:
                        anonymous_words += len(anonymous_description.split())
                if source_id:
                    distinct_source_ids.add(source_id)

                source_mentions_records.append(
                    (
                        version_id,
                        entry_id,
                        news_org,
                        source_id,
                        mention.get("canonical") or mention.get("surface"),
                        mention.get("type"),
                        mention.get("speech_style"),
                        mention.get("attribution_verb"),
                        mention.get("char_start"),
                        mention.get("char_end"),
                        mention.get("sentence_index"),
                        mention.get("paragraph_index"),
                        int(mention.get("is_in_title", False)),
                        int(mention.get("is_in_lede", False)),
                        attributed_text,
                        int(is_anonymous),
                        anonymous_description,
                        anonymous_domain,
                        evidence_type,
                        evidence_text,
                        mention["prominence"]["lead_percentile"],
                        mention.get("confidence", 5),
                    )
                )
                mentions.append(mention)

            brief_mentions = [
                {
                    "canonical": m.get("canonical") or m.get("surface") or "",
                    "type": m.get("type"),
                    "centrality": m.get("centrality"),
                    "narrative_function": m.get("narrative_function"),
                    "attributed_text": m.get("attributed_text") or "",
                }
                for m in mentions
            ]

            identity_mentions = [
                {
                    **base,
                    "is_anonymous": bool(m.get("is_anonymous")),
                    "anonymous_description": (m.get("anonymous_description") or "").strip(),
                    "anonymous_domain": m.get("anonymous_domain"),
                }
                for base, m in zip(brief_mentions, mentions)
            ]

            speech_mentions = [
                {
                    **base,
                    "speech_style": m.get("speech_style"),
                    "attribution_verb": m.get("attribution_verb"),
                }
                for base, m in zip(brief_mentions, mentions)
            ]

            brief_mentions_json = orjson.dumps(
                brief_mentions,
                option=orjson.OPT_INDENT_2,
            ).decode("utf-8")

            identity_mentions_json = orjson.dumps(
                identity_mentions,
                option=orjson.OPT_INDENT_2,
            ).decode("utf-8")

            speech_mentions_json = orjson.dumps(
                speech_mentions,
                option=orjson.OPT_INDENT_2,
            ).decode("utf-8")

            mentions_json = orjson.dumps(
                mentions,
                option=orjson.OPT_INDENT_2,
            ).decode("utf-8")
            version_payloads[version_id]["source_mentions"] = mentions_json
            version_payloads[version_id]["source_mentions_brief"] = brief_mentions_json
            version_payloads[version_id]["source_mentions_identity"] = identity_mentions_json
            version_payloads[version_id]["source_mentions_speech"] = speech_mentions_json

            classifier_specs: List[Tuple[str, str, Path, int]] = []
            if "correction" in text.lower():
                classifier_specs.append(
                    (
                        "D3_corrections",
                        render_prompt(load_prompt_template("D3_corrections"), {"version_text": text}),
                        version_dir / "D3_corrections.json",
                        config.llm_retries,
                    )
                )
            classifier_specs.append(
                (
                    "B1_version_summary_sources",
                    render_prompt(
                        load_prompt_template("B1_version_summary_sources"),
                        {
                            "version_id": version_id,
                            "source_mentions": brief_mentions_json,
                        },
                    ),
                    version_dir / "B1_version_summary_sources.json",
                    config.llm_retries,
                )
            )
            classifier_specs.append(
                (
                    "C1_protest_frame_cues",
                    render_prompt(
                        load_prompt_template("C1_protest_frame_cues"),
                        {
                            "version_id": version_id,
                            "version_text": text,
                            "source_mentions": mentions_json,
                        },
                    ),
                    version_dir / "C1_protest_frame_cues.json",
                    config.llm_retries,
                )
            )

            classifier_outputs: Dict[str, Dict[str, Any]] = {}
            if classifier_specs:
                classifier_results = client.run_many(classifier_specs)
                if len(classifier_results) != len(classifier_specs):
                    raise RuntimeError(
                        f"Expected {len(classifier_specs)} classifier responses but received {len(classifier_results)}"
                    )
                classifier_outputs = {
                    key: prune_low_confidence(result, config.accept_confidence_min)
                    for (key, _, _, _), result in zip(classifier_specs, classifier_results)
                }

            d3 = classifier_outputs.get("D3_corrections", {})
            b1 = classifier_outputs.get("B1_version_summary_sources", {})
            c1 = classifier_outputs.get("C1_protest_frame_cues", {})

            if d3:
                version_payloads[version_id]["corrections"] = orjson.dumps(
                    d3,
                    option=orjson.OPT_INDENT_2,
                ).decode("utf-8")
            version_payloads[version_id]["protest"] = orjson.dumps(
                c1,
                option=orjson.OPT_INDENT_2,
            ).decode("utf-8")
            version_payloads[version_id]["version_summary"] = orjson.dumps(
                b1,
                option=orjson.OPT_INDENT_2,
            ).decode("utf-8")

            token_count = max(len(seg["tokens"]), 1)
            total_attributed_words = max(total_attributed_words, 0)
            institutional_share = (
                institutional_words / total_attributed_words if total_attributed_words else 0.0
            )
            anonymous_share = (
                anonymous_words / total_attributed_words if total_attributed_words else 0.0
            )
            hedge_density = (hedge_total / token_count) * 1000 if token_count else 0.0

            version_numeric_metrics[version_num] = {
                "distinct_sources": len(distinct_source_ids),
                "institutional_share_words": institutional_share,
                "anonymous_source_share_words": anonymous_share,
                "hedge_density_per_1k_tokens": hedge_density,
            }

            version_meta.append(
                {
                    "version_id": version_id,
                    "version_num": version_num,
                    "timestamp_utc": version.get("created"),
                    "char_len": seg["char_len"],
                }
            )
            source_mentions_by_version.append(
                {
                    "version_id": version_id,
                    "version_num": version_num,
                    "timestamp_utc": version.get("created"),
                    "mentions": mentions,
                }
            )

            versions_rows.append(
                (
                    version_id,
                    entry_id,
                    news_org,
                    version_num,
                    version.get("created"),
                    title,
                    seg["char_len"],
                )
            )

            version_metrics_rows.append(
                (
                    version_id,
                    entry_id,
                    news_org,
                    version_numeric_metrics.get(version_num, {}).get("distinct_sources", 0),
                    version_numeric_metrics.get(version_num, {}).get("institutional_share_words", 0.0),
                    version_numeric_metrics.get(version_num, {}).get("anonymous_source_share_words", 0.0),
                    version_numeric_metrics.get(version_num, {}).get("hedge_density_per_1k_tokens", 0.0),
                )
            )

        if skip_article_due_to_live_blog:
            versions_rows = [
                (
                    version_entry["id"],
                    entry_id,
                    news_org,
                    _normalize_version_number(version_entry["version"]),
                    version_entry.get("created"),
                    version_entry.get("title"),
                    len(version_entry.get("summary") or ""),
                )
                for version_entry in versions
            ]

            article_row = (
                entry_id,
                news_org,
                url,
                title_first,
                title_final,
                original_ts,
                len(versions) - 1,
                int(live_blog_flag),
            )

            log_message = f"Recorded live-blog metadata for article {entry_id}"
            return ArticleResult(
                entry_id=entry_id,
                news_org=news_org,
                article_row=article_row,
                versions_rows=versions_rows,
                source_mentions_rows=[],
                entity_rows=[],
                version_metrics_rows=[],
                pair_rows=[],
                pair_sources_added=[],
                pair_sources_removed=[],
                pair_source_transitions=[],
                pair_replacements=[],
                pair_numeric=[],
                pair_claims_rows=[],
                pair_cues_rows=[],
                sources_agg_rows=[],
                article_metrics_row=None,
                live_blog_only=True,
                log_message=log_message,
            )

        if not version_numeric_metrics:
            logger.info(
                "Skipping article %s after processing: no version metrics were generated; likely no usable source mentions",
                entry_id,
            )
            return None

        pair_comparisons: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        if len(versions) >= 2:
            pair_comparisons.append((versions[0], versions[-1]))

        for prev, curr in pair_comparisons:
            prev_id = prev["id"]
            curr_id = curr["id"]
            prev_num = _normalize_version_number(prev["version"])
            curr_num = _normalize_version_number(curr["version"])
            pair_dir = article_dir / f"v{prev_num:03d}_to_v{curr_num:03d}"
            if cache_enabled:
                ensure_dir(pair_dir)

            diff_stats = compute_diff_magnitude(prev.get("summary") or "", curr.get("summary") or "")
            delta_minutes = inter_update_timing(prev.get("created"), curr.get("created"))
            title_jacc_prev = jaccard_title_body(
                prev.get("title") or "",
                ledes[prev_num]["text"] if prev_num in ledes else "",
            )
            title_jacc_curr = jaccard_title_body(
                curr.get("title") or "",
                ledes[curr_num]["text"] if curr_num in ledes else "",
            )

            prev_payload = version_payloads.get(prev_id, {})
            curr_payload = version_payloads.get(curr_id, {})

            pair_prompts = build_pair_prompt_payloads(prev, curr, prev_payload, curr_payload)

            pair_specs: List[Tuple[str, str, Path, int]] = []
            for key, payload in pair_prompts.items():
                rendered = render_prompt(load_prompt_template(key), payload)
                pair_specs.append((key, rendered, pair_dir / f"{key}.json", config.llm_retries))

            pair_results = client.run_many(pair_specs)
            if len(pair_results) != len(pair_specs):
                raise RuntimeError(
                    f"Expected {len(pair_specs)} pair prompt responses but received {len(pair_results)}"
                )

            results = {
                key: prune_low_confidence(result, config.accept_confidence_min)
                for (key, _, _, _), result in zip(pair_specs, pair_results)
            }

            movement = results["P10_movement_pair"]
            movement_up = movement.get("movement_summary_upweighted") or ""
            movement_down = movement.get("movement_summary_downweighted") or ""
            movement_notes_parts = movement.get("movement_notes") or []
            if isinstance(movement_notes_parts, list):
                movement_notes = "\n".join(str(part) for part in movement_notes_parts if part)
            else:
                movement_notes = str(movement_notes_parts)

            a3 = results["A3_edit_type_pair"]
            d5 = results["D5_angle_change_pair"]

            angle_changed_flag = int(bool(d5.get("angle_changed", False)))
            angle_category = d5.get("angle_change_category", "no_change")
            angle_summary = d5.get("angle_summary", "")
            title_alignment_notes = d5.get("title_alignment_notes", "")
            summary_jaccard = text_jaccard_similarity(prev.get("summary") or "", curr.get("summary") or "")

            pair_rows.append(
                (
                    entry_id,
                    news_org,
                    prev_id,
                    curr_id,
                    prev_num,
                    curr_num,
                    delta_minutes,
                    diff_stats.get("tokens_added", 0),
                    diff_stats.get("tokens_deleted", 0),
                    diff_stats.get("percent_text_new", 0.0),
                    movement_up,
                    movement_down,
                    movement_notes,
                    a3.get("edit_type"),
                    angle_changed_flag,
                    angle_category,
                    angle_summary,
                    title_alignment_notes,
                    title_jacc_prev,
                    title_jacc_curr,
                    summary_jaccard,
                )
            )

            for src in a3.get("sources_added", []):
                pair_sources_added.append((prev_id, curr_id, src.get("canonical"), src.get("type")))
            for src in a3.get("sources_removed", []):
                pair_sources_removed.append((prev_id, curr_id, src.get("canonical"), src.get("type")))
            for transition in d5.get("source_transitions", []) or []:
                pair_source_transitions.append(
                    (
                        prev_id,
                        curr_id,
                        transition.get("canonical"),
                        transition.get("transition_type"),
                        transition.get("reason_category"),
                        transition.get("reason_detail"),
                    )
                )
            for repl in results["P3_anon_named_replacement_pair"].get("replacements", []):
                pair_replacements.append(
                    (
                        prev_id,
                        curr_id,
                        repl.get("from"),
                        repl.get("to"),
                        repl.get("direction"),
                        repl.get("likelihood", 0.0),
                    )
                )
            for numeric in results["P7_numeric_changes_pair"].get("numeric_changes", []):
                pair_numeric.append(
                    (
                        prev_id,
                        curr_id,
                        numeric.get("item"),
                        numeric.get("prev"),
                        numeric.get("curr"),
                        numeric.get("delta"),
                        numeric.get("unit"),
                        numeric.get("source"),
                        numeric.get("change_type"),
                        numeric.get("confidence", 1),
                    )
                )
            for claim in results["P8_claims_pair"].get("claims", []):
                pair_claims_rows.append(
                    (
                        prev_id,
                        curr_id,
                        claim.get("id"),
                        claim.get("proposition"),
                        claim.get("status"),
                        claim.get("change_note"),
                        claim.get("confidence", 1),
                    )
                )
            for cue in results["P9_frame_cues_pair"].get("cues", []):
                pair_cues_rows.append(
                    (
                        prev_id,
                        curr_id,
                        cue.get("cue"),
                        int(cue.get("prev", False)),
                        int(cue.get("curr", False)),
                        cue.get("direction"),
                    )
                )

        first_version_num = min(version_numeric_metrics.keys())
        final_version_num = max(version_numeric_metrics.keys())

        b2_result = client.run(
            "B2_first_final_framing_compare",
            render_prompt(
                load_prompt_template("B2_first_final_framing_compare"),
                {
                    "title_v0": versions[0]["title"],
                    "lede_v0": ledes.get(first_version_num, {}).get("text", ""),
                    "title_vf": versions[-1]["title"],
                    "lede_vf": ledes.get(final_version_num, {}).get("text", ""),
                },
            ),
            article_dir / "B2_first_final_framing_compare.json",
        )
        b2_result = prune_low_confidence(b2_result, config.accept_confidence_min)

        agg = aggregate_sources_over_versions(source_mentions_by_version, version_meta)

        metrics_first = version_numeric_metrics.get(first_version_num, {})
        metrics_final = version_numeric_metrics.get(final_version_num, {})
        bias = final_version_bias(metrics_first, metrics_final)
        distinct_delta = metrics_final.get("distinct_sources", 0) - metrics_first.get("distinct_sources", 0)
        anonymity_delta = metrics_final.get("anonymous_source_share_words", 0.0) - metrics_first.get(
            "anonymous_source_share_words", 0.0
        )
        hedge_delta = metrics_final.get("hedge_density_per_1k_tokens", 0.0) - metrics_first.get(
            "hedge_density_per_1k_tokens", 0.0
        )

        sources_agg_rows = [
            (
                entry_id,
                news_org,
                src.get("source_id_within_article"),
                src.get("source_canonical"),
                src.get("source_type"),
                src.get("first_seen_version"),
                src.get("first_seen_time"),
                src.get("last_seen_version"),
                src.get("last_seen_time"),
                src.get("num_mentions_total"),
                src.get("num_versions_present"),
                src.get("total_attributed_words"),
                src.get("voice_retention_index"),
                src.get("mean_prominence"),
                src.get("lead_appearance_count"),
                src.get("title_appearance_count"),
                int(src.get("doubted_any", False)),
                int(src.get("deemphasized_any", False)),
                int(src.get("disappeared_any", False)),
            )
            for src in agg.get("sources", [])
        ]

        article_row = (
            entry_id,
            news_org,
            url,
            title_first,
            title_final,
            original_ts,
            len(versions) - 1,
            int(live_blog_flag),
        )

        article_metrics_row = (
            entry_id,
            news_org,
            bias.get("overstate_institutional_share", 0.0),
            distinct_delta,
            anonymity_delta,
            hedge_delta,
        )

        log_message = f"Processed article {entry_id} ({len(versions)} versions)"
        if verbose:
            logger.debug(log_message)

        return ArticleResult(
            entry_id=entry_id,
            news_org=news_org,
            article_row=article_row,
            versions_rows=versions_rows,
            source_mentions_rows=source_mentions_records,
            entity_rows=entity_rows,
            version_metrics_rows=version_metrics_rows,
            pair_rows=pair_rows,
            pair_sources_added=pair_sources_added,
            pair_sources_removed=pair_sources_removed,
            pair_source_transitions=pair_source_transitions,
            pair_replacements=pair_replacements,
            pair_numeric=pair_numeric,
            pair_claims_rows=pair_claims_rows,
            pair_cues_rows=pair_cues_rows,
            sources_agg_rows=sources_agg_rows,
            article_metrics_row=article_metrics_row,
            live_blog_only=False,
            log_message=log_message,
        )
    finally:
        if owns_client and client is not None:
            client.close()
