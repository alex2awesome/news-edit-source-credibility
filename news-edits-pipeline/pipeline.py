"""End-to-end orchestration for the news edits pipeline."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import orjson
import requests
import yaml
from tqdm.auto import tqdm

from analysis import (
    aggregate_sources_over_versions,
    align_sentences,
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
from loader import iter_articles, load_versions
from pipeline_utils import (
    OutputWriter,
    ensure_dir,
    extract_context_window,
    find_index_by_offset,
    load_prompt_template,
    prune_low_confidence,
    render_prompt,
    response_format,
    schema_text,
    parse_json_response,
)
from pipeline_writer_utils import (
    insert_entity_mentions,
    insert_pair_claims,
    insert_pair_cues,
    insert_pair_numeric_changes,
    insert_pair_replacements,
    insert_pair_sources_added,
    insert_pair_sources_removed,
    insert_pair_title_events,
    insert_source_mentions,
    insert_version_metrics,
    insert_version_pairs,
    upsert_article,
    upsert_article_metrics,
    upsert_sources_agg,
    upsert_versions,
)


logger = logging.getLogger(__name__)
SYSTEM_PROMPT = (
    "You are a meticulous news-analysis assistant. Always emit valid JSON that matches the requested schema. "
    "Follow the schema exactly, but do NOT repeat the schema text."
)

@dataclass
class Config:
    model: str
    vllm_api_base: str
    temperature: float
    max_tokens: int
    batch_size: int
    hedge_window_tokens: int
    accept_confidence_min: float
    out_root: Path
    cache_raw_responses: bool
    skip_if_cached: bool
    backend: str
    ollama_api_base: Optional[str]
    min_versions: int
    max_versions: Optional[int]
    cleanup_cached_dirs: bool

    @staticmethod
    def from_yaml(path: Path) -> "Config":
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        backend = data.get("backend", "vllm").lower()
        if backend not in {"vllm", "ollama"}:
            raise ValueError(f"Unsupported backend '{backend}'. Expected 'vllm' or 'ollama'.")
        min_versions = max(1, int(data.get("min_versions", 2)))
        max_versions_value = data.get("max_versions", 20)
        if max_versions_value is None:
            max_versions = None
        else:
            max_versions = int(max_versions_value)
        return Config(
            model=data.get("model"),
            vllm_api_base=data.get("vllm_api_base"),
            temperature=float(data.get("temperature", 0.0)),
            max_tokens=int(data.get("max_tokens", 2048)),
            batch_size=int(data.get("batch_size", 1)),
            hedge_window_tokens=int(data.get("hedge_window_tokens", 80)),
            accept_confidence_min=float(data.get("accept_confidence_min", 3.0)),
            out_root=Path(data.get("out_root", "./out")),
            cache_raw_responses=bool(data.get("cache_raw_responses", True)),
            skip_if_cached=bool(data.get("skip_if_cached", True)),
            backend=backend,
            ollama_api_base=data.get("ollama_api_base"),
            min_versions=min_versions,
            max_versions=max_versions,
            cleanup_cached_dirs=bool(data.get("cleanup_cached_dirs", False)),
        )


class StructuredLLMClient:
    def __init__(self, config: Config):
        self.config = config
        self.session = requests.Session()
        self.api_key = os.getenv("OPENAI_API_KEY") if config.backend == "vllm" else None

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _request_payload(self, prompt: str, response_format: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "response_format": response_format,
        }

    def _run_vllm(
        self,
        prompt_key: str,
        rendered_prompt: str,
        cache_path: Path,
        retries: int,
    ) -> Dict[str, Any]:
        if self.config.cache_raw_responses:
            ensure_dir(cache_path.parent)
        url = self.config.vllm_api_base.rstrip("/") + "/chat/completions"
        payload = self._request_payload(rendered_prompt, response_format(prompt_key))

        last_error: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                response = self.session.post(url, headers=self._headers(), json=payload, timeout=120)
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                try:
                    parsed = parse_json_response(content)
                except (orjson.JSONDecodeError, ValueError) as exc:
                    last_error = exc
                    logger.warning("Invalid JSON from %s attempt %d: %s", prompt_key, attempt + 1, exc)
                    if attempt < retries:
                        continue
                    raise
                if self.config.cache_raw_responses:
                    cache_path.write_bytes(orjson.dumps(parsed, option=orjson.OPT_INDENT_2))
                return parsed
            except requests.RequestException as exc:
                last_error = exc
                logger.error("Request failed for %s: %s", prompt_key, exc)
                if attempt < retries:
                    continue
                raise
        if last_error:
            raise last_error
        raise RuntimeError("Failed to obtain response")

    def _run_ollama(
        self,
        prompt_key: str,
        rendered_prompt: str,
        cache_path: Path,
        retries: int,
    ) -> Dict[str, Any]:
        if self.config.cache_raw_responses:
            ensure_dir(cache_path.parent)
        base_url = self.config.ollama_api_base or self.config.vllm_api_base or "http://localhost:11434"
        url = base_url.rstrip("/") + "/api/chat"
        user_prompt = (
            f"{rendered_prompt}\n\nReturn ONLY valid JSON that matches this schema:\n{schema_text(prompt_key)}\n"
            "Do not include any additional text."
        )
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }

        last_error: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                response = self.session.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=120)
                response.raise_for_status()
                data = response.json()
                # Ollama returns either 'message' or top-level 'response'
                content = ""
                if isinstance(data, dict):
                    if "message" in data and isinstance(data["message"], dict):
                        content = data["message"].get("content", "")
                    elif "response" in data:
                        content = data.get("response", "")
                if not content:
                    raise ValueError(f"Ollama response missing content: {data}")
                try:
                    parsed = parse_json_response(content)
                except (orjson.JSONDecodeError, ValueError) as exc:
                    last_error = exc
                    logger.warning("Invalid JSON from Ollama for %s attempt %d: %s", prompt_key, attempt + 1, exc)
                    if attempt < retries:
                        continue
                    raise
                if self.config.cache_raw_responses:
                    cache_path.write_bytes(orjson.dumps(parsed, option=orjson.OPT_INDENT_2))
                return parsed
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                logger.error("Ollama request failed for %s: %s", prompt_key, exc)
                if attempt < retries:
                    continue
                raise
        if last_error:
            raise last_error
        raise RuntimeError("Failed to obtain response")

    def run(
        self,
        prompt_key: str,
        rendered_prompt: str,
        cache_path: Path,
        retries: int = 1,
    ) -> Dict[str, Any]:
        if self.config.skip_if_cached and cache_path.exists():
            try:
                return orjson.loads(cache_path.read_bytes())
            except orjson.JSONDecodeError:
                logger.warning("Failed to parse cache %s, recomputing", cache_path)

        if self.config.backend == "ollama":
            return self._run_ollama(prompt_key, rendered_prompt, cache_path, retries)
        return self._run_vllm(prompt_key, rendered_prompt, cache_path, retries)


def assign_source_id(
    mention: Dict[str, Any],
    registry: Dict[str, Dict[str, Any]],
    threshold: int = 92,
) -> str:
    canonical = mention.get("canonical") or mention.get("surface")
    norm = normalize_source(canonical)
    match = fuzzy_match_source(canonical, registry, threshold=threshold)
    if match is None:
        source_id = f"s{len(registry) + 1:03d}"
        registry[source_id] = {"normalized": norm, "canonical": canonical}
        return source_id
    return match


def assign_entity_id(surface: str, registry: Dict[str, str]) -> str:
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


def locate_best_span(text: str, snippet: str, used_ranges: List[Tuple[int, int]]) -> Tuple[int, int, str]:
    variants = _generate_snippet_variants(snippet)
    for variant in variants:
        start, end = _find_available_span(text, variant, used_ranges)
        if start != -1:
            return start, end, text[start:end]

    # Attempt partial matches by progressively trimming ends.
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


def normalize_version_number(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            raise ValueError(f"Unable to coerce version number {value!r} to int")


def process_db(
    db_path: Path,
    config: Config,
    client: StructuredLLMClient,
    schema_path: Path,
    verbose: bool = False,
) -> None:
    db_stem = db_path.stem.replace(".db", "")
    out_root = config.out_root / db_stem
    ensure_dir(out_root)
    writer = OutputWriter(out_root / "analysis.db", schema_path)
    cache_enabled = config.cache_raw_responses

    try:
        article_iter = iter_articles(str(db_path))
        iterator = tqdm(
            article_iter,
            desc=f"{db_stem} articles",
            unit="article",
            leave=False,
            disable=not sys.stdout.isatty(),
        )
        try:
            for entry_id in iterator:
                versions = load_versions(str(db_path), entry_id)
                if not versions:
                    continue

                num_versions = len(versions)
                if num_versions < config.min_versions:
                    if verbose:
                        logger.debug("Skipping entry %s: only %d version(s)", entry_id, num_versions)
                    continue
                if config.max_versions is not None and num_versions >= config.max_versions:
                    if verbose:
                        logger.debug(
                            "Skipping entry %s: %d versions exceeds max %s",
                            entry_id,
                            num_versions,
                            config.max_versions,
                        )
                    continue

                article_dir = out_root / str(entry_id)
                if cache_enabled:
                    ensure_dir(article_dir)

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
                pair_title_events: List[Tuple[Any, ...]] = []
                pair_replacements: List[Tuple[Any, ...]] = []
                pair_numeric: List[Tuple[Any, ...]] = []
                pair_claims_rows: List[Tuple[Any, ...]] = []
                pair_cues_rows: List[Tuple[Any, ...]] = []

                source_mentions_by_version: List[Dict[str, Any]] = []
                version_meta: List[Dict[str, Any]] = []
                ledes: Dict[int, Dict[str, Any]] = {}
                live_blog_flag = False
                version_numeric_metrics: Dict[int, Dict[str, Any]] = {}
                versions_rows: List[Tuple[Any, ...]] = []
                version_payloads: Dict[str, Dict[str, str]] = {}

                for version in versions:
                    version_id = version["id"]
                    version_num = normalize_version_number(version["version"])
                    version_dir = article_dir / f"v{version_num:03d}"
                    if cache_enabled:
                        ensure_dir(version_dir)

                    text = version.get("summary") or ""
                    title = version.get("title") or ""

                    seg = segment(text)
                    lede = extract_lede(title, text, seg["paragraphs"])
                    ledes[version_num] = lede

                    entity_doc = ner_entities_spacy(text)
                    for ent in entity_doc.get("entities", []):
                        entity_id = assign_entity_id(ent["canonical"], entity_registry)
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
                            "article_id": entry_id,
                            "version_id": version_id,
                            "timestamp_utc": version.get("created"),
                            "title": title,
                            "version_text": text,
                        },
                    )
                    a1_result = client.run(
                        "A1_source_mentions",
                        rendered_a1,
                        version_dir / "A1_source_mentions.json",
                    )
                    a1_result = prune_low_confidence(a1_result, config.accept_confidence_min)

                    raw_mentions: List[Dict[str, Any]] = []
                    for item in a1_result.get("source_mentions", []):
                        try:
                            conf_value = float(item.get("confidence", config.accept_confidence_min))
                        except (TypeError, ValueError):
                            conf_value = config.accept_confidence_min
                        if conf_value < config.accept_confidence_min:
                            continue
                        raw_mentions.append(dict(item))

                    used_source_spans: List[Tuple[int, int]] = []
                    processed_mentions: List[Dict[str, Any]] = []
                    title_lower = title.lower()
                    char_len = seg["char_len"]

                    for mention in raw_mentions:
                        enriched = dict(mention)
                        attributed_text = enriched.get("attributed_text") or ""
                        surface = enriched.get("surface") or ""

                        candidate_strings: List[str] = []
                        if attributed_text:
                            candidate_strings.append(attributed_text)
                            stripped_attr = attributed_text.strip()
                            if stripped_attr and stripped_attr not in candidate_strings:
                                candidate_strings.append(stripped_attr)
                            normalized_attr = stripped_attr.replace("“", '"').replace("”", '"')
                            if normalized_attr and normalized_attr not in candidate_strings:
                                candidate_strings.append(normalized_attr)
                        if surface and surface not in candidate_strings:
                            candidate_strings.append(surface)

                        span_start = -1
                        span_end = -1
                        matched_text = ""
                        for candidate in candidate_strings:
                            candidate_start, candidate_end, candidate_match = locate_best_span(
                                text,
                                candidate,
                                used_source_spans,
                            )
                            if candidate_start != -1:
                                span_start, span_end = candidate_start, candidate_end
                                matched_text = candidate_match
                                used_source_spans.append((candidate_start, candidate_end))
                                break
                        if span_start == -1 and surface:
                            candidate_start, candidate_end, candidate_match = locate_best_span(text, surface, [])
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

                    source_catalog_for_prompt = [
                        {
                            "index": idx,
                            "surface": catalog_item.get("surface"),
                            "canonical": catalog_item.get("canonical"),
                            "type": catalog_item.get("type"),
                            "speech_style": catalog_item.get("speech_style"),
                            "attribution_verb": catalog_item.get("attribution_verb"),
                            "char_start": catalog_item.get("char_start", -1),
                            "char_end": catalog_item.get("char_end", -1),
                            "sentence_index": catalog_item.get("sentence_index", -1),
                            "paragraph_index": catalog_item.get("paragraph_index", -1),
                            "confidence": catalog_item.get("confidence"),
                        }
                        for idx, catalog_item in enumerate(processed_mentions, start=1)
                    ]
                    source_catalog_json = orjson.dumps(
                        source_catalog_for_prompt,
                        option=orjson.OPT_INDENT_2,
                    ).decode("utf-8")

                    version_payloads.setdefault(version_id, {})
                    version_payloads[version_id]["source_catalog"] = source_catalog_json

                    mentions: List[Dict[str, Any]] = []
                    for idx, mention in enumerate(processed_mentions, start=1):
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
                            "index": idx,
                            "surface": mention.get("surface"),
                            "canonical": mention.get("canonical"),
                            "type": mention.get("type"),
                            "speech_style": mention.get("speech_style"),
                            "attribution_verb": mention.get("attribution_verb"),
                            "char_start": mention.get("char_start", -1),
                            "char_end": mention.get("char_end", -1),
                            "sentence_index": mention.get("sentence_index", -1),
                            "paragraph_index": mention.get("paragraph_index", -1),
                        }
                        target_source_json = orjson.dumps(
                            target_source_payload,
                            option=orjson.OPT_INDENT_2,
                        ).decode("utf-8")
                        rendered_a2 = render_prompt(
                            load_prompt_template("A2_hedge_window"),
                            {
                                "context_window_text": context_window,
                                "target_source": target_source_json,
                                "source_catalog": source_catalog_json,
                            },
                        )
                        a2_result = client.run(
                            "A2_hedge_window",
                            rendered_a2,
                            version_dir / f"A2_hedge_{idx:03d}.json",
                        )
                        a2_result = prune_low_confidence(a2_result, config.accept_confidence_min)
                        mention["hedge_analysis"] = a2_result
                        mention["doubted"] = a2_result.get("stance_toward_source") == "skeptical"
                        mention["hedge_count"] = a2_result.get("hedge_count", 0)

                        source_id = assign_source_id(mention, source_registry)
                        mention["source_id_within_article"] = source_id

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
                                mention.get("attributed_text"),
                                mention["prominence"]["lead_percentile"],
                                mention.get("confidence", 5),
                            )
                        )
                        mentions.append(mention)

                    mentions_json = orjson.dumps(
                        mentions,
                        option=orjson.OPT_INDENT_2,
                    ).decode("utf-8")
                    version_payloads[version_id]["source_mentions"] = mentions_json

                    d1 = client.run(
                        "D1_anonymous_sources",
                        render_prompt(
                            load_prompt_template("D1_anonymous_sources"),
                            {"version_text": text, "source_catalog": source_catalog_json},
                        ),
                        version_dir / "D1_anonymous_sources.json",
                    )
                    d2 = client.run(
                        "D2_evidence_types",
                        render_prompt(
                            load_prompt_template("D2_evidence_types"),
                            {"version_text": text, "source_mentions": mentions_json},
                        ),
                        version_dir / "D2_evidence_types.json",
                    )
                    d3 = client.run(
                        "D3_corrections",
                        render_prompt(load_prompt_template("D3_corrections"), {"version_text": text}),
                        version_dir / "D3_corrections.json",
                    )
                    d4 = client.run(
                        "D4_live_blog_detect",
                        render_prompt(load_prompt_template("D4_live_blog_detect"), {"version_text": text}),
                        version_dir / "D4_live_blog_detect.json",
                    )
                    b1 = client.run(
                        "B1_version_summary_sources",
                        render_prompt(
                            load_prompt_template("B1_version_summary_sources"),
                            {
                                "version_id": version_id,
                                "version_text": text,
                                "source_mentions": mentions_json,
                            },
                        ),
                        version_dir / "B1_version_summary_sources.json",
                    )
                    c1 = client.run(
                        "C1_protest_frame_cues",
                        render_prompt(load_prompt_template("C1_protest_frame_cues"), {"version_text": text}),
                        version_dir / "C1_protest_frame_cues.json",
                    )

                    d1 = prune_low_confidence(d1, config.accept_confidence_min)
                    d2 = prune_low_confidence(d2, config.accept_confidence_min)
                    d3 = prune_low_confidence(d3, config.accept_confidence_min)
                    d4 = prune_low_confidence(d4, config.accept_confidence_min)
                    b1 = prune_low_confidence(b1, config.accept_confidence_min)
                    c1 = prune_low_confidence(c1, config.accept_confidence_min)

                    version_payloads[version_id]["anonymous"] = orjson.dumps(
                        d1,
                        option=orjson.OPT_INDENT_2,
                    ).decode("utf-8")
                    version_payloads[version_id]["evidence"] = orjson.dumps(
                        d2,
                        option=orjson.OPT_INDENT_2,
                    ).decode("utf-8")
                    version_payloads[version_id]["corrections"] = orjson.dumps(
                        d3,
                        option=orjson.OPT_INDENT_2,
                    ).decode("utf-8")
                    version_payloads[version_id]["live_blog"] = orjson.dumps(
                        d4,
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

                    total_attributed_words = 0
                    institutional_words = 0
                    hedge_total = 0
                    distinct_source_ids = set()
                    institutional_types = {"government", "corporate", "law_enforcement"}
                    for mention in mentions:
                        source_id = mention.get("source_id_within_article")
                        if source_id:
                            distinct_source_ids.add(source_id)
                        attributed_text = mention.get("attributed_text") or ""
                        words = len(attributed_text.split())
                        total_attributed_words += words
                        if mention.get("type") in institutional_types:
                            institutional_words += words
                        hedge_total += mention.get("hedge_count", 0)

                    anonymous_words = 0
                    for anon in d1.get("anonymous_mentions", []):
                        snippet = anon.get("verbatim_text") or ""
                        if snippet:
                            anonymous_words += len(snippet.split())
                            continue
                        start = anon.get("char_start", 0) or 0
                        end = anon.get("char_end", start) or start
                        if not isinstance(start, int):
                            try:
                                start = int(start)
                            except (TypeError, ValueError):
                                start = 0
                        if not isinstance(end, int):
                            try:
                                end = int(end)
                            except (TypeError, ValueError):
                                end = start
                        start = max(0, start)
                        end = max(start, end)
                        span = text[start:end]
                        anonymous_words += len(span.split())

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

                    if d4.get("is_live_blog"):
                        live_blog_flag = True

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

                for prev, curr in zip(versions, versions[1:]):
                    prev_id = prev["id"]
                    curr_id = curr["id"]
                    prev_num = normalize_version_number(prev["version"])
                    curr_num = normalize_version_number(curr["version"])
                    pair_dir = article_dir / f"v{prev_num:03d}_to_v{curr_num:03d}"
                    if cache_enabled:
                        ensure_dir(pair_dir)

                    prev_text = prev.get("summary") or ""
                    curr_text = curr.get("summary") or ""
                    prev_title = prev.get("title") or ""
                    curr_title = curr.get("title") or ""

                    diff_stats = compute_diff_magnitude(prev_text, curr_text)
                    alignment = align_sentences(prev_text, curr_text)
                    delta_minutes = inter_update_timing(prev.get("created"), curr.get("created"))
                    title_jacc_prev = jaccard_title_body(
                        prev_title,
                        ledes[prev_num]["text"] if prev_num in ledes else "",
                    )
                    title_jacc_curr = jaccard_title_body(
                        curr_title,
                        ledes[curr_num]["text"] if curr_num in ledes else "",
                    )

                    prev_payload = version_payloads.get(prev_id, {})
                    curr_payload = version_payloads.get(curr_id, {})
                    prev_sources_json = prev_payload.get("source_mentions", "[]")
                    curr_sources_json = curr_payload.get("source_mentions", "[]")
                    prev_anonymous_json = prev_payload.get("anonymous", '{"anonymous_mentions": []}')
                    curr_anonymous_json = curr_payload.get("anonymous", '{"anonymous_mentions": []}')
                    prev_protest_json = prev_payload.get("protest", '{"frame_cues": [], "roles": [], "confidence": 1}')
                    curr_protest_json = curr_payload.get("protest", '{"frame_cues": [], "roles": [], "confidence": 1}')

                    pair_prompts = {
                        "A3_edit_type_pair": {
                            "prev_title": prev_title,
                            "curr_title": curr_title,
                            "prev_version_text": prev_text,
                            "curr_version_text": curr_text,
                            "prev_source_mentions": prev_sources_json,
                            "curr_source_mentions": curr_sources_json,
                        },
                        "P2_title_events_pair": {
                            "title_prev": prev_title,
                            "title_curr": curr_title,
                            "v_prev": prev_text,
                            "v_curr": curr_text,
                            "prev_source_mentions": prev_sources_json,
                            "curr_source_mentions": curr_sources_json,
                        },
                        "P3_anon_named_replacement_pair": {
                            "v_prev": prev_text,
                            "v_curr": curr_text,
                            "prev_source_mentions": prev_sources_json,
                            "curr_source_mentions": curr_sources_json,
                            "prev_anonymous": prev_anonymous_json,
                            "curr_anonymous": curr_anonymous_json,
                        },
                        "P4_verb_strength_pair": {
                            "v_prev": prev_text,
                            "v_curr": curr_text,
                            "prev_source_mentions": prev_sources_json,
                            "curr_source_mentions": curr_sources_json,
                        },
                        "P5_speech_style_pair": {
                            "v_prev": prev_text,
                            "v_curr": curr_text,
                            "prev_source_mentions": prev_sources_json,
                            "curr_source_mentions": curr_sources_json,
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
                            "prev_source_mentions": prev_sources_json,
                            "curr_source_mentions": curr_sources_json,
                        },
                        "P17_title_body_alignment_pair": {
                            "title_prev": prev_title,
                            "title_curr": curr_title,
                            "v_prev_first_paragraph": ledes[prev_num]["text"] if prev_num in ledes else "",
                            "v_curr_first_paragraph": ledes[curr_num]["text"] if curr_num in ledes else "",
                        },
                        "D5_angle_change_pair": {
                            "prev_version_text": prev_text,
                            "curr_version_text": curr_text,
                        },
                    }

                    results: Dict[str, Dict[str, Any]] = {}
                    for key, variables in pair_prompts.items():
                        rendered = render_prompt(load_prompt_template(key), variables)
                        result = client.run(key, rendered, pair_dir / f"{key}.json")
                        results[key] = prune_low_confidence(result, config.accept_confidence_min)

                    a3 = results["A3_edit_type_pair"]
                    p10 = results["P10_movement_pair"]
                    d5 = results["D5_angle_change_pair"]

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
                            p10.get("movement_index", alignment.get("movement_index", 0.0)),
                            p10.get("moved_into_top20pct_tokens", 0.0),
                            a3.get("edit_type"),
                            int(d5.get("angle_changed", False)),
                            title_jacc_prev,
                            title_jacc_curr,
                        )
                    )

                    for src in a3.get("sources_added", []):
                        pair_sources_added.append((prev_id, curr_id, src.get("canonical"), src.get("type")))
                    for src in a3.get("sources_removed", []):
                        pair_sources_removed.append((prev_id, curr_id, src.get("canonical"), src.get("type")))
                    for event in results["P2_title_events_pair"].get("title_events", []):
                        pair_title_events.append((prev_id, curr_id, event.get("canonical"), event.get("event")))
                    for event in results["P2_title_events_pair"].get("lede_events", []):
                        pair_title_events.append((prev_id, curr_id, event.get("canonical"), event.get("event")))
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

                if not version_numeric_metrics:
                    continue

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

                upsert_article(
                    writer,
                    (
                        entry_id,
                        news_org,
                        url,
                        title_first,
                        title_final,
                        original_ts,
                        len(versions) - 1,
                        int(live_blog_flag),
                    ),
                )

                upsert_versions(writer, versions_rows)
                insert_source_mentions(writer, source_mentions_records)

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

                upsert_sources_agg(writer, sources_agg_rows)
                insert_entity_mentions(writer, entity_rows)
                insert_version_metrics(writer, version_metrics_rows)
                insert_version_pairs(writer, pair_rows)
                insert_pair_sources_added(writer, pair_sources_added)
                insert_pair_sources_removed(writer, pair_sources_removed)
                insert_pair_title_events(writer, pair_title_events)
                insert_pair_replacements(writer, pair_replacements)
                insert_pair_numeric_changes(writer, pair_numeric)
                insert_pair_claims(writer, pair_claims_rows)
                insert_pair_cues(writer, pair_cues_rows)

                upsert_article_metrics(
                    writer,
                    (
                        entry_id,
                        news_org,
                        bias.get("overstate_institutional_share", 0.0),
                        distinct_delta,
                        anonymity_delta,
                        hedge_delta,
                    ),
                )

                writer.commit()
                if verbose:
                    logger.debug("Committed article %s (%d versions)", entry_id, num_versions)
                logger.info("Processed article %s in %s", entry_id, db_path.name)
                if config.cleanup_cached_dirs and cache_enabled:
                    try:
                        shutil.rmtree(article_dir)
                    except OSError:
                        logger.warning("Failed to remove cache directory %s", article_dir)
        finally:
            if hasattr(iterator, "close"):
                iterator.close()
    finally:
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run news edits pipeline over SQLite databases")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--db", action="append", dest="dbs", required=True, type=Path)
    parser.add_argument("--schema", default=Path(__file__).with_name("schema_out.sql"), type=Path)
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    config = Config.from_yaml(args.config)
    ensure_dir(config.out_root)
    client = StructuredLLMClient(config)

    for db_path in args.dbs:
        logger.info("Processing %s", db_path)
        process_db(db_path, config, client, args.schema, verbose=args.verbose)


if __name__ == "__main__":
    main()
