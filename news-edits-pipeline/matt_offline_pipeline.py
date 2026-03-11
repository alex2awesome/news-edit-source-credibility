"""Offline batch pipeline for --matt-features-only using vllm.LLM directly.

Instead of issuing HTTP requests to a running vLLM server one prompt at a time,
this script:
  1. Loads all eligible articles from all SQLite databases.
  2. Pre-renders every required prompt for every article.
  3. Runs all prompts through vllm.LLM in large offline batches (2 waves).
  4. Post-processes the LLM outputs into ArticleResult objects.
  5. Writes results to per-DB SQLite output files.

Wave structure (to respect data dependencies):
  Wave 1a — A1_source_mentions  : 2 × N prompts (v0 + v_final per article)
  Wave 1b — P10_movement_pair   : N prompts     (independent of A1)
  Wave 2a — A3_edit_type_pair   : N prompts     (embeds A1 source-mention briefs)
  Wave 2b — D5_angle_change_pair: N prompts     (embeds A1 source-mention briefs)

Total: 5 × N LLM calls, 4 sequential vllm.chat() invocations.

Usage:
  python matt_offline_pipeline.py \\
      --config  news-edits-pipeline/config.yaml \\
      --db      article-versions/ap.db \\
      --db      article-versions/newssniffer-nytimes.db.gz \\
      [--tensor-parallel-size 4] \\
      [--max-model-len 4096] \\
      [--gpu-memory-utilization 0.90] \\
      [--out-path ./out-matt]
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import orjson
from tqdm.auto import tqdm

# vLLM imports — guarded so the module is importable without a GPU for testing.
try:
    from vllm import LLM, SamplingParams  # type: ignore[import]
    # vllm 0.16+ renamed GuidedDecodingParams → StructuredOutputsParams and
    # changed the SamplingParams field from guided_decoding= to structured_outputs=.
    try:
        from vllm.sampling_params import StructuredOutputsParams as _StructuredOutputsParams  # type: ignore[import]
        _GUIDED_FIELD = "structured_outputs"
        _GuidedParams = _StructuredOutputsParams
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams as _GuidedDecodingParams  # type: ignore[import]
        _GUIDED_FIELD = "guided_decoding"
        _GuidedParams = _GuidedDecodingParams
    _VLLM_AVAILABLE = True
except ImportError as _vllm_err:
    import warnings
    warnings.warn(f"vllm import failed: {_vllm_err}")
    _VLLM_AVAILABLE = False

from config import Config
from loader import iter_article_counts
from pipeline_utils import (
    OutputWriter,
    ensure_dir,
    load_prompt_template,
    load_schema,
    parse_json_response,
    prune_low_confidence,
    render_prompt,
)
from prompt_utils import build_pair_prompt_payloads

# Deferred imports: analysis → spacy → thinc → torch, which initializes a CUDA
# context in the main process and corrupts the vllm subprocess.  Import only
# after LLM is initialized.
_analysis_imported = False
def _ensure_analysis_imports():
    global inter_update_timing, ArticleResult, _assign_source_id, _normalize_version_number, write_article_result, _analysis_imported
    if _analysis_imported:
        return
    from analysis import inter_update_timing as _itu
    from article_processor import (
        ArticleResult as _AR,
        _assign_source_id as _asi,
        _normalize_version_number as _nvn,
    )
    from writer import write_article_result as _war
    inter_update_timing = _itu
    ArticleResult = _AR
    _assign_source_id = _asi
    _normalize_version_number = _nvn
    write_article_result = _war
    _analysis_imported = True

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a meticulous news-analysis assistant. Always emit valid JSON that matches "
    "the requested schema. Follow the schema exactly, but do NOT repeat the schema text."
)

# Prompts used in matt-features-only mode, in dependency order.
_WAVE1_KEYS = ("A1_source_mentions", "P10_movement_pair")
_WAVE2_KEYS = ("A3_edit_type_pair", "D5_angle_change_pair")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ArticleSpec:
    """All data for one article, plus slots for LLM results."""
    entry_id: int
    db_path: Path
    db_stem: str           # e.g. "ap" or "newssniffer-nytimes"
    news_org: str
    v0: Dict[str, Any]     # first version row
    vfinal: Dict[str, Any] # last version row
    all_versions: List[Dict[str, Any]]
    out_dir: Path          # per-article cache directory

    # Filled after each inference wave.
    a1_v0_result: Optional[Dict[str, Any]] = field(default=None, repr=False)
    a1_vfinal_result: Optional[Dict[str, Any]] = field(default=None, repr=False)
    p10_result: Optional[Dict[str, Any]] = field(default=None, repr=False)
    a3_result: Optional[Dict[str, Any]] = field(default=None, repr=False)
    d5_result: Optional[Dict[str, Any]] = field(default=None, repr=False)


@dataclass
class _BatchItem:
    """One prompt to be run through vllm, with its cache path."""
    messages: List[Dict[str, str]]
    prompt_key: str
    cache_path: Path
    result: Optional[Dict[str, Any]] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

# Conservative character budget for article text.
# Pair prompts include TWO article texts + source mentions + template, so the
# budget per text must be at most ~3,000 tokens.  At ~4 chars/token → 12k chars.
# Use 10k to leave headroom for source-mention JSON and template overhead.
_MAX_ARTICLE_CHARS = 10_000


def _a1_messages(version: Dict[str, Any], entry_id: int) -> List[Dict[str, str]]:
    text = version.get("summary") or ""
    title = version.get("title") or ""
    rendered = render_prompt(
        load_prompt_template("A1_source_mentions"),
        {
            "news_article": text,
            "version_text": text,
            "title": title,
            "article_id": str(entry_id),
            "version_id": str(version["id"]),
            "timestamp_utc": version.get("created") or "",
        },
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": rendered},
    ]


def _pair_messages(key: str, payload: Dict[str, Any]) -> List[Dict[str, str]]:
    rendered = render_prompt(load_prompt_template(key), payload)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": rendered},
    ]


def _version_payload_from_a1(
    a1_result: Optional[Dict[str, Any]],
    config: Config,
) -> Dict[str, str]:
    """Build the version_payload dict that build_pair_prompt_payloads expects."""
    raw = a1_result or {}
    # LLM sometimes returns a bare list instead of {"source_mentions": [...]}.
    if isinstance(raw, list):
        raw = {"source_mentions": raw}
    pruned = prune_low_confidence(raw, config.accept_confidence_min)
    mentions = pruned.get("source_mentions", [])
    brief = [
        {
            "canonical": m.get("canonical") or m.get("surface") or "",
            "type": m.get("type"),
            "centrality": None,
            "narrative_function": None,
            "attributed_text": m.get("attributed_text") or "",
        }
        for m in mentions
        if isinstance(m, dict)
    ]
    full_json = orjson.dumps(mentions, option=orjson.OPT_INDENT_2).decode("utf-8")
    brief_json = orjson.dumps(brief, option=orjson.OPT_INDENT_2).decode("utf-8")
    return {"source_mentions": full_json, "source_mentions_brief": brief_json}


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _already_processed_ids(out_db_path: Path) -> set:
    """Return the set of article_ids already written to an output analysis.db."""
    if not out_db_path.exists():
        return set()
    import sqlite3
    try:
        with sqlite3.connect(str(out_db_path)) as conn:
            rows = conn.execute("SELECT article_id FROM articles").fetchall()
        return {row[0] for row in rows}
    except Exception as exc:
        logger.warning("Could not read existing output DB %s: %s", out_db_path, exc)
        return set()


def _load_cache(path: Path) -> Optional[Dict[str, Any]]:
    if path.exists():
        try:
            return orjson.loads(path.read_bytes())
        except orjson.JSONDecodeError:
            logger.warning("Corrupt cache at %s; will re-infer", path)
    return None


def _save_cache(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------

def _run_batch(
    llm: "LLM",
    items: List["_BatchItem"],
    config: Config,
    desc: str,
) -> None:
    """Run all items through llm.chat().  Cached items are skipped.
    Results are written back to item.result in-place.
    """
    pending: List[_BatchItem] = []
    for item in items:
        if config.skip_if_cached:
            cached = _load_cache(item.cache_path)
            if cached is not None:
                item.result = cached
                continue
        pending.append(item)

    n_cached = len(items) - len(pending)
    if n_cached:
        logger.info("%s: %d/%d served from cache", desc, n_cached, len(items))
    if not pending:
        return

    logger.info("%s: running %d prompts through vllm", desc, len(pending))

    schema = load_schema(pending[0].prompt_key)  # same key for all items in a batch
    gd = _GuidedParams(json=schema)
    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        **{_GUIDED_FIELD: gd},
    )

    outputs = llm.chat(
        messages=[item.messages for item in pending],
        sampling_params=sampling_params,
        use_tqdm=True,
    )

    for item, output in zip(pending, outputs):
        raw = output.outputs[0].text
        try:
            parsed = parse_json_response(raw)
        except Exception as exc:
            logger.warning("Parse error for %s (cache=%s): %s", item.prompt_key, item.cache_path, exc)
            parsed = {}
        item.result = parsed
        if config.cache_raw_responses:
            _save_cache(item.cache_path, parsed)


# ---------------------------------------------------------------------------
# Per-article post-processing
# ---------------------------------------------------------------------------

def _postprocess_article(spec: ArticleSpec, config: Config) -> Optional[ArticleResult]:
    """Mirror the features_only branch of article_processor.process_article()
    using pre-computed LLM results instead of live calls."""
    institutional_types = {"government", "corporate", "law_enforcement"}
    source_registry: Dict[str, Dict[str, Any]] = {}
    versions_rows: List[Tuple[Any, ...]] = []
    source_mentions_records: List[Tuple[Any, ...]] = []
    version_metrics_rows: List[Tuple[Any, ...]] = []

    entry_id = spec.entry_id
    news_org = spec.news_org
    url = spec.v0.get("url", "")
    title_first = spec.v0.get("title", "")
    title_final = spec.vfinal.get("title", "")
    original_ts = spec.v0.get("created")

    for version, raw_a1 in [(spec.v0, spec.a1_v0_result), (spec.vfinal, spec.a1_vfinal_result)]:
        version_id = version["id"]
        version_num = _normalize_version_number(version["version"])
        text = version.get("summary") or ""
        title = version.get("title") or ""

        _a1 = raw_a1 or {}
        if isinstance(_a1, list):
            _a1 = {"source_mentions": _a1}
        a1_result = prune_low_confidence(_a1, config.accept_confidence_min)
        mentions_raw = a1_result.get("source_mentions", [])

        mentions: List[Dict[str, Any]] = []
        brief_mentions: List[Dict[str, Any]] = []
        distinct_source_ids: set = set()
        total_attributed_words = 0
        institutional_words = 0
        anonymous_words = 0

        for mention_raw in mentions_raw:
            if not isinstance(mention_raw, dict):
                continue
            mention = dict(mention_raw)
            source_id = _assign_source_id(mention, source_registry)
            mention["source_id_within_article"] = source_id

            attributed_text = mention.get("attributed_text") or ""
            words = len(attributed_text.split())
            total_attributed_words += words

            source_type = mention.get("type")
            if source_type in institutional_types:
                institutional_words += words

            is_anonymous = bool(mention.get("is_anonymous"))
            anonymous_description = (mention.get("anonymous_description") or "").strip()
            anonymous_domain = (mention.get("anonymous_domain") or "unknown").lower()
            if anonymous_domain not in {
                "government", "corporate", "law_enforcement", "individual", "unknown"
            }:
                anonymous_domain = "unknown"

            if is_anonymous:
                if attributed_text:
                    anonymous_words += words
                elif anonymous_description:
                    anonymous_words += len(anonymous_description.split())

            if source_id:
                distinct_source_ids.add(source_id)

            evidence_type = (mention.get("evidence_type") or "other").lower()
            if evidence_type not in {
                "official_statement", "press_release", "eyewitness", "document",
                "statistic", "prior_reporting", "social_media", "court_filing", "other",
            }:
                evidence_type = "other"

            hedge_markers_json = orjson.dumps([], option=orjson.OPT_INDENT_2).decode("utf-8")
            epistemic_verbs_json = orjson.dumps([], option=orjson.OPT_INDENT_2).decode("utf-8")
            perspective_json = orjson.dumps([], option=orjson.OPT_INDENT_2).decode("utf-8")

            source_mentions_records.append((
                version_id, entry_id, news_org,
                source_id,
                mention.get("canonical") or mention.get("surface"),
                mention.get("surface"),
                source_type,
                mention.get("speech_style"),
                mention.get("attribution_verb"),
                mention.get("char_start", -1),
                mention.get("char_end", -1),
                -1, -1,
                int(False), int(False),
                attributed_text,
                int(is_anonymous),
                anonymous_description,
                anonymous_domain,
                evidence_type,
                mention.get("evidence_text") or attributed_text,
                None, None,
                perspective_json,
                int(False), 0,
                hedge_markers_json,
                epistemic_verbs_json,
                None, None, None,
                mention.get("confidence", 5),
            ))

            brief_mentions.append({
                "canonical": mention.get("canonical") or mention.get("surface") or "",
                "type": source_type,
                "centrality": None,
                "narrative_function": None,
                "attributed_text": attributed_text,
            })
            mentions.append(mention)

        institutional_share = institutional_words / max(total_attributed_words, 1) if total_attributed_words else 0.0
        anonymous_share = anonymous_words / max(total_attributed_words, 1) if total_attributed_words else 0.0

        version_metrics_rows.append((
            version_id, entry_id, news_org,
            len(distinct_source_ids),
            institutional_share,
            anonymous_share,
            0.0,
        ))
        versions_rows.append((
            version_id, entry_id, news_org,
            version_num,
            version.get("created"),
            title,
            len(text),
        ))

    # --- Pair row ---
    prev, curr = spec.v0, spec.vfinal
    prev_id, curr_id = prev["id"], curr["id"]
    prev_num = _normalize_version_number(prev["version"])
    curr_num = _normalize_version_number(curr["version"])
    delta_minutes = inter_update_timing(prev.get("created"), curr.get("created"))

    _a3 = spec.a3_result or {}
    _d5 = spec.d5_result or {}
    _p10 = spec.p10_result or {}
    if isinstance(_a3, list): _a3 = {}
    if isinstance(_d5, list): _d5 = {}
    if isinstance(_p10, list): _p10 = {}
    a3 = prune_low_confidence(_a3, config.accept_confidence_min)
    d5 = prune_low_confidence(_d5, config.accept_confidence_min)
    p10 = prune_low_confidence(_p10, config.accept_confidence_min)

    movement_notes_parts = p10.get("movement_notes") or []
    movement_notes = (
        "\n".join(str(part) for part in movement_notes_parts if part)
        if isinstance(movement_notes_parts, list)
        else str(movement_notes_parts)
    )

    pair_rows = [(
        entry_id, news_org, prev_id, curr_id, prev_num, curr_num,
        delta_minutes,
        None, None, None,   # tokens_added, tokens_deleted, percent_text_new
        p10.get("movement_summary_upweighted") or "",
        p10.get("movement_summary_downweighted") or "",
        movement_notes,
        p10.get("confidence"),
        orjson.dumps(p10.get("notable_shifts") or [], option=orjson.OPT_INDENT_2).decode("utf-8"),
        a3.get("edit_type"),
        a3.get("summary_of_change", ""),
        a3.get("confidence"),
        int(bool(d5.get("angle_changed", False))),
        d5.get("angle_change_category", "no_change"),
        d5.get("angle_summary", ""),
        d5.get("title_alignment_notes", ""),
        d5.get("confidence"),
        orjson.dumps(d5.get("evidence_snippets") or [], option=orjson.OPT_INDENT_2).decode("utf-8"),
        None, None, None,   # title_jaccard_prev, title_jaccard_curr, summary_jaccard
    )]

    return ArticleResult(
        entry_id=entry_id,
        news_org=news_org,
        article_row=(
            entry_id, news_org, url, title_first, title_final,
            original_ts, len(spec.all_versions) - 1, 0,
        ),
        versions_rows=versions_rows,
        source_mentions_rows=source_mentions_records,
        entity_rows=[],
        version_metrics_rows=version_metrics_rows,
        pair_rows=pair_rows,
        pair_sources_added=[],
        pair_sources_removed=[],
        pair_source_transitions=[],
        pair_replacements=[],
        pair_numeric=[],
        pair_claims_rows=[],
        pair_cues_rows=[],
        sources_agg_rows=[],
        article_metrics_row=None,
        live_blog_only=False,
        log_message=f"Processed article {entry_id} (offline, {len(spec.all_versions)} versions)",
    )


# ---------------------------------------------------------------------------
# Article loading
# ---------------------------------------------------------------------------

def _parse_ts(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
    except ValueError:
        return None


def _load_specs(
    db_path: Path,
    config: Config,
    out_root: Path,
    date_range: Optional[Tuple[datetime, datetime]] = None,
) -> List[ArticleSpec]:
    import sqlite3 as _sqlite3
    db_stem = db_path.stem.replace(".db", "")

    # Pass 1: scan article counts and collect matching entry_ids (fast, single query).
    logger.info("  Pass 1: scanning article counts in %s", db_path.name)
    matched_ids: List[int] = []
    total = skipped_versions = skipped_date = 0
    for entry_id, version_count, first_created in tqdm(
        iter_article_counts(str(db_path)), desc=f"  Scanning {db_path.name}", unit="art"
    ):
        total += 1
        if version_count < config.min_versions:
            skipped_versions += 1
            continue
        if config.max_versions is not None and version_count >= config.max_versions:
            skipped_versions += 1
            continue
        if date_range is not None:
            ts = _parse_ts(first_created)
            if ts is None or not (date_range[0] <= ts <= date_range[1]):
                skipped_date += 1
                continue
        matched_ids.append(entry_id)

    logger.info(
        "  Pass 1 done: total=%d  matched=%d  skipped_versions=%d  skipped_date=%d",
        total, len(matched_ids), skipped_versions, skipped_date,
    )
    if not matched_ids:
        return []

    # Pass 2: full table scan — load ALL entryversion rows, group in Python.
    # Faster than WHERE entry_id IN (...) for large match sets.
    logger.info("  Pass 2: full table scan of entryversion in %s...", db_path.name)
    matched_set = set(matched_ids)
    source_name = db_stem

    from loader import _resolved_db_path  # type: ignore[import]
    with _resolved_db_path(str(db_path)) as resolved:
        conn = _sqlite3.connect(resolved)
        conn.row_factory = _sqlite3.Row
        try:
            cursor = conn.execute(
                """
                SELECT id, entry_id, version, created, title, url, num_versions, summary
                FROM entryversion
                ORDER BY entry_id ASC, version ASC
                """
            )
            rows_by_entry: Dict[int, List[Dict]] = {}
            for row in tqdm(cursor, desc="  Reading rows", unit="row"):
                eid = int(row["entry_id"])
                if eid not in matched_set:
                    continue
                rows_by_entry.setdefault(eid, []).append({
                    "id": row["id"],
                    "source": source_name,
                    "entry_id": eid,
                    "version": row["version"],
                    "created": row["created"],
                    "title": row["title"],
                    "url": row["url"],
                    "num_versions": row["num_versions"],
                    "summary": (row["summary"] or "")[:_MAX_ARTICLE_CHARS],
                })
        finally:
            conn.close()

    logger.info("  Pass 2 done: loaded versions for %d articles.", len(rows_by_entry))

    specs: List[ArticleSpec] = []
    for entry_id in matched_ids:
        versions = rows_by_entry.get(entry_id, [])
        if len(versions) < 2:
            continue
        specs.append(ArticleSpec(
            entry_id=entry_id,
            db_path=db_path,
            db_stem=db_stem,
            news_org=versions[0]["source"],
            v0=versions[0],
            vfinal=versions[-1],
            all_versions=versions,
            out_dir=out_root / db_stem / str(entry_id),
        ))
    return specs


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_offline_pipeline(
    db_paths: List[Path],
    config: Config,
    schema_sql_path: Path,
    out_path: Path,
    tensor_parallel_size: int = 1,
    max_model_len: Optional[int] = None,
    gpu_memory_utilization: float = 0.95,
    dtype: str = "auto",
    enforce_eager: bool = False,
    max_articles_per_outlet: Optional[int] = None,
    date_range: Optional[Tuple[datetime, datetime]] = None,
) -> None:
    if not _VLLM_AVAILABLE:
        raise RuntimeError(
            "vllm is not installed. Install it with: pip install vllm\n"
            "For online (server) mode, use pipeline.py instead."
        )

    ensure_dir(out_path)

    # ---- Phase 1: Initialize vllm once, then process one DB at a time -------
    logger.info(
        "Loading model %s (tensor_parallel_size=%d, max_model_len=%s, gpu_mem=%.2f) — this takes several minutes...",
        config.model, tensor_parallel_size, max_model_len, gpu_memory_utilization,
    )
    llm_kwargs: Dict[str, Any] = {
        "model": config.model,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "dtype": dtype,
        "trust_remote_code": True,
        "enforce_eager": enforce_eager,
    }
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len
    llm = LLM(**llm_kwargs)
    logger.info("Model loaded.")
    _ensure_analysis_imports()

    total_ok = total_fail = 0

    for db_path in db_paths:
        logger.info("=== Processing %s ===", db_path.name)
        n_ok = n_fail = 0
        db_stem = db_path.stem.replace(".db", "")
        out_db_path = out_path / db_stem / "analysis.db"
        done_ids = _already_processed_ids(out_db_path)
        if done_ids:
            logger.info("  Skipping %d article(s) already processed", len(done_ids))

        specs = _load_specs(db_path, config, out_path, date_range=date_range)
        specs = [s for s in specs if s.entry_id not in done_ids]

        if max_articles_per_outlet is not None and len(specs) > max_articles_per_outlet:
            import random
            specs = random.sample(specs, max_articles_per_outlet)

        logger.info("  %d article(s) to process", len(specs))
        if not specs:
            logger.info("  Nothing to do for %s; skipping.", db_path.name)
            continue

        if config.cache_raw_responses:
            for spec in specs:
                ensure_dir(spec.out_dir)

        logger.info("  LLM calls: %d", len(specs) * 5)

        # Wave 1a — A1_source_mentions (2N prompts)
        a1_items: List[_BatchItem] = []
        for spec in specs:
            a1_items.append(_BatchItem(
                messages=_a1_messages(spec.v0, spec.entry_id),
                prompt_key="A1_source_mentions",
                cache_path=spec.out_dir / "A1_v0.json",
            ))
            a1_items.append(_BatchItem(
                messages=_a1_messages(spec.vfinal, spec.entry_id),
                prompt_key="A1_source_mentions",
                cache_path=spec.out_dir / "A1_vfinal.json",
            ))
        _run_batch(llm, a1_items, config, f"{db_path.name} A1_source_mentions")
        for i, spec in enumerate(specs):
            spec.a1_v0_result = a1_items[i * 2].result
            spec.a1_vfinal_result = a1_items[i * 2 + 1].result

        # Wave 1b — P10_movement_pair (N prompts, independent)
        p10_items: List[_BatchItem] = []
        for spec in specs:
            payload = build_pair_prompt_payloads(spec.v0, spec.vfinal, {}, {})["P10_movement_pair"]
            p10_items.append(_BatchItem(
                messages=_pair_messages("P10_movement_pair", payload),
                prompt_key="P10_movement_pair",
                cache_path=spec.out_dir / "P10_movement_pair.json",
            ))
        _run_batch(llm, p10_items, config, f"{db_path.name} P10_movement_pair")
        for spec, item in zip(specs, p10_items):
            spec.p10_result = item.result

        # Wave 2 — A3 + D5 (depend on A1 source-mention briefs)
        a3_items: List[_BatchItem] = []
        d5_items: List[_BatchItem] = []
        for spec in specs:
            prev_payload = _version_payload_from_a1(spec.a1_v0_result, config)
            curr_payload = _version_payload_from_a1(spec.a1_vfinal_result, config)
            pair_payloads = build_pair_prompt_payloads(spec.v0, spec.vfinal, prev_payload, curr_payload)
            a3_items.append(_BatchItem(
                messages=_pair_messages("A3_edit_type_pair", pair_payloads["A3_edit_type_pair"]),
                prompt_key="A3_edit_type_pair",
                cache_path=spec.out_dir / "A3_edit_type_pair.json",
            ))
            d5_items.append(_BatchItem(
                messages=_pair_messages("D5_angle_change_pair", pair_payloads["D5_angle_change_pair"]),
                prompt_key="D5_angle_change_pair",
                cache_path=spec.out_dir / "D5_angle_change_pair.json",
            ))
        _run_batch(llm, a3_items, config, f"{db_path.name} A3_edit_type_pair")
        _run_batch(llm, d5_items, config, f"{db_path.name} D5_angle_change_pair")
        for spec, a3, d5 in zip(specs, a3_items, d5_items):
            spec.a3_result = a3.result
            spec.d5_result = d5.result

        # Post-process and write
        db_out_dir = out_path / db_stem
        ensure_dir(db_out_dir)
        writer = OutputWriter(db_out_dir / "analysis.db", schema_sql_path)
        for spec in tqdm(specs, desc=f"{db_path.name} writing"):
            try:
                result = _postprocess_article(spec, config)
                if result is None:
                    n_fail += 1
                    continue
                write_article_result(writer, result)
                n_ok += 1
            except Exception:
                logger.exception("Failed to post-process article %s", spec.entry_id)
                n_fail += 1
        writer.close()
        logger.info("  Done %s: success=%d failed=%d", db_path.name, n_ok, n_fail)
        total_ok += n_ok
        total_fail += n_fail

    logger.info("All DBs done. total_success=%d  total_failed=%d", total_ok, total_fail)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline batch vllm pipeline for --matt-features-only"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--db", action="append", dest="dbs", required=True, type=Path)
    parser.add_argument(
        "--schema",
        default=Path(__file__).with_name("schema_out.sql"),
        type=Path,
    )
    parser.add_argument("--out-path", default=Path("./out"), type=Path)
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1,
        help="Number of GPUs for tensor parallelism (vllm --tensor-parallel-size).",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=None,
        help="Override max sequence length.  Lower = more batch slots.  Default: model config.",
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.95,
        help="Fraction of GPU memory to use for KV cache (default: 0.95).",
    )
    parser.add_argument(
        "--dtype", default="auto",
        help="Model dtype passed to vllm (default: auto).",
    )
    parser.add_argument(
        "--enforce-eager", action="store_true",
        help="Disable CUDA graphs (enforce_eager=True). Fixes flashinfer cubin mismatches.",
    )
    parser.add_argument(
        "--max-articles-per-outlet", type=int, default=None,
        help="Cap articles per DB before batching.",
    )
    parser.add_argument(
        "--date-range",
        nargs=2,
        metavar=("START", "END"),
        default=None,
        help="Only process articles whose first version falls in [START, END]. "
             "Accepts ISO-8601 dates, e.g. --date-range 2020-01-01 2022-12-31",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    for noisy in ("asyncio", "httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    config = Config.from_yaml(args.config)
    config.matt_features_only = True

    date_range = None
    if args.date_range:
        try:
            start = _parse_ts(args.date_range[0])
            end = _parse_ts(args.date_range[1])
        except Exception as exc:
            raise SystemExit(f"Invalid --date-range values: {exc}") from exc
        if start is None or end is None:
            raise SystemExit("Could not parse --date-range values as dates.")
        if end < start:
            raise SystemExit("--date-range END must be >= START.")
        date_range = (start, end)
        logger.info("Date filter: %s to %s", start.isoformat(), end.isoformat())

    run_offline_pipeline(
        db_paths=args.dbs,
        config=config,
        schema_sql_path=args.schema,
        out_path=args.out_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        enforce_eager=args.enforce_eager,
        max_articles_per_outlet=args.max_articles_per_outlet,
        date_range=date_range,
    )


if __name__ == "__main__":
    main()
