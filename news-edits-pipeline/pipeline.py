"""End-to-end orchestration for the news edits pipeline."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
import asyncio
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast
import tiktoken
import warnings
warnings.filterwarnings("ignore", message="Using `TRANSFORMERS_CACHE` is deprecated", category=FutureWarning)

import orjson
import httpx
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
import requests
import yaml
from tqdm.auto import tqdm
import re

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
from loader import iter_article_counts, load_versions
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
    insert_pair_source_transitions,
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
HEDGE_WINDOW_BATCH_SIZE = 5  # number of source mentions evaluated per A2 hedging request


def text_jaccard_similarity(text_a: str, text_b: str) -> float:
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


@dataclass
class Config:
    model: str
    vllm_api_base: str
    spacy_model: str
    tiktoken_encoding: Optional[str]
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
    skip_similarity_threshold: Optional[float]
    llm_retries: int
    llm_retry_backoff_seconds: float

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
            spacy_model=data.get("spacy_model"),
            tiktoken_encoding=data.get("tiktoken_encoding", "cl100k_base"),
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
            skip_similarity_threshold=(
                float(data["skip_similarity_threshold"])
                if data.get("skip_similarity_threshold") is not None
                else 0.95
            ),
            llm_retries=int(data.get("llm_retries", 2)),
            llm_retry_backoff_seconds=float(data.get("llm_retry_backoff_seconds", 2.0)),
        )


class StructuredLLMClient:
    _TIMEOUT_SECONDS = 120

    def __init__(self, config: Config):
        self.config = config
        self.session = requests.Session()
        self.api_key = os.getenv("OPENAI_API_KEY") if config.backend == "vllm" else None
        self.tokenizer = tiktoken.get_encoding(config.tiktoken_encoding or "cl100k_base")
        self.system_prompt_length = len(self.tokenizer.encode(SYSTEM_PROMPT))
        self._retry_backoff = max(0.0, config.llm_retry_backoff_seconds)
        logger.info(f"System prompt length: {self.system_prompt_length}")

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _request_payload(self, prompt: str, response_format: Dict[str, Any]) -> Dict[str, Any]:
        too_long = False
        input_tokens = self.tokenizer.encode(prompt)
        max_tokens = max(0, self.config.max_tokens - len(input_tokens) - self.system_prompt_length - 50)
        if max_tokens <= 0:
            logger.warning(f"Prompt is too long. Max tokens is {self.config.max_tokens} but length of input tokens is {len(input_tokens)}.")
            logger.warning(f"Prompt: {prompt}")
            too_long = True
            max_tokens = 0
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": max_tokens,
            "response_format": response_format,
        }
        return payload, too_long

    def _dump_payload_for_logging(self, payload: Dict[str, Any]) -> str:
        try:
            return orjson.dumps(payload, option=orjson.OPT_INDENT_2).decode("utf-8")
        except orjson.JSONEncodeError:
            return str(payload)

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
        payload, too_long = self._request_payload(rendered_prompt, response_format(prompt_key))
        if too_long:
            return {}

        max_attempts = max(1, retries)
        last_error: Optional[Exception] = None
        for attempt in range(max_attempts):
            attempt_num = attempt + 1
            try:
                response = self.session.post(
                    url,
                    headers=self._headers(),
                    json=payload,
                    timeout=self._TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                try:
                    parsed = parse_json_response(content)
                except (orjson.JSONDecodeError, ValueError) as exc:
                    last_error = exc
                    logger.warning("Invalid JSON from %s attempt %d: %s", prompt_key, attempt_num, exc)
                    if attempt_num < max_attempts:
                        wait = self._retry_backoff * attempt_num
                        if wait > 0:
                            logger.info("Retrying %s after %.1fs due to JSON parse error", prompt_key, wait)
                            time.sleep(wait)
                        continue
                    logger.error("Giving up on %s after %d attempts due to JSON parsing errors", prompt_key, max_attempts)
                    return {}
                if self.config.cache_raw_responses:
                    cache_path.write_bytes(orjson.dumps(parsed, option=orjson.OPT_INDENT_2))
                return parsed
            except requests.RequestException as exc:
                last_error = exc
                logger.error(
                    "Request failed for %s (attempt %d/%d): %s",
                    prompt_key,
                    attempt_num,
                    max_attempts,
                    exc,
                )
                if attempt_num >= max_attempts:
                    try:
                        payload_dump = orjson.dumps(payload, option=orjson.OPT_INDENT_2).decode("utf-8")
                    except orjson.JSONEncodeError:
                        payload_dump = str(payload)
                    logger.error("Request payload for %s:\n%s", prompt_key, payload_dump)
                    response_text: Optional[str] = None
                    status_code: Optional[int] = None
                    if isinstance(exc, requests.HTTPError) and exc.response is not None:
                        status_code = exc.response.status_code
                        try:
                            response_text = exc.response.text
                        except Exception:
                            response_text = None
                    if response_text:
                        logger.error(
                            "Response body for %s (status %s):\\n%s",
                            prompt_key,
                            status_code,
                            response_text,
                        )
                    return {}
                wait = self._retry_backoff * attempt_num
                if wait > 0:
                    logger.info("Retrying %s after %.1fs due to request error", prompt_key, wait)
                    time.sleep(wait)
                continue
        if last_error:
            logger.error("Failed to obtain response for %s after %d attempts: %s", prompt_key, max_attempts, last_error)
        return {}

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
            f"{rendered_prompt}\n\n"
            "Return ONLY valid JSON that matches this schema:\n"
            f"{schema_text(prompt_key)}\n"
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

        max_attempts = max(1, retries)
        last_error: Optional[Exception] = None
        for attempt in range(max_attempts):
            attempt_num = attempt + 1
            try:
                response = self.session.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=self._TIMEOUT_SECONDS,
                )
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
                    logger.warning("Invalid JSON from Ollama for %s attempt %d: %s", prompt_key, attempt_num, exc)
                    if attempt_num < max_attempts:
                        wait = self._retry_backoff * attempt_num
                        if wait > 0:
                            logger.info("Retrying %s after %.1fs due to Ollama JSON error", prompt_key, wait)
                            time.sleep(wait)
                        continue
                    logger.error("Giving up on Ollama request %s after %d attempts due to JSON errors", prompt_key, max_attempts)
                    return {}
                if self.config.cache_raw_responses:
                    cache_path.write_bytes(orjson.dumps(parsed, option=orjson.OPT_INDENT_2))
                return parsed
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                logger.error(
                    "Ollama request failed for %s (attempt %d/%d): %s",
                    prompt_key,
                    attempt_num,
                    max_attempts,
                    exc,
                )
                if attempt_num >= max_attempts:
                    try:
                        payload_dump = orjson.dumps(payload, option=orjson.OPT_INDENT_2).decode("utf-8")
                    except orjson.JSONEncodeError:
                        payload_dump = str(payload)
                    logger.error("Ollama request payload for %s:\\n%s", prompt_key, payload_dump)
                    if isinstance(exc, requests.HTTPError) and exc.response is not None:
                        try:
                            response_text = exc.response.text
                        except Exception:
                            response_text = None
                        if response_text:
                            logger.error(
                                "Ollama response body for %s (status %s):\\n%s",
                                prompt_key,
                                exc.response.status_code,
                                response_text,
                            )
                    return {}
                wait = self._retry_backoff * attempt_num
                if wait > 0:
                    logger.info("Retrying %s after %.1fs due to Ollama request error", prompt_key, wait)
                    time.sleep(wait)
                continue
        if last_error:
            logger.error("Failed to obtain Ollama response for %s after %d attempts: %s", prompt_key, max_attempts, last_error)
        return {}

    async def _run_vllm_async(
        self,
        http_client: httpx.AsyncClient,
        prompt_key: str,
        rendered_prompt: str,
        cache_path: Path,
        retries: int,
    ) -> Dict[str, Any]:
        if self.config.cache_raw_responses:
            ensure_dir(cache_path.parent)
        url = self.config.vllm_api_base.rstrip("/") + "/chat/completions"
        payload, too_long = self._request_payload(rendered_prompt, response_format(prompt_key))
        if too_long:
            return {}

        max_attempts = max(1, retries)
        last_error: Optional[Exception] = None
        for attempt in range(max_attempts):
            attempt_num = attempt + 1
            try:
                response = await http_client.post(
                    url,
                    headers=self._headers(),
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                try:
                    parsed = parse_json_response(content)
                except (orjson.JSONDecodeError, ValueError) as exc:
                    last_error = exc
                    logger.warning("Invalid JSON from %s attempt %d: %s", prompt_key, attempt_num, exc)
                    if attempt_num < max_attempts:
                        wait = self._retry_backoff * attempt_num
                        if wait > 0:
                            logger.info("Retrying %s after %.1fs due to async JSON parse error", prompt_key, wait)
                            await asyncio.sleep(wait)
                        continue
                    logger.error("Giving up on %s after %d attempts due to async JSON parsing errors", prompt_key, max_attempts)
                    return {}
                if self.config.cache_raw_responses:
                    cache_path.write_bytes(orjson.dumps(parsed, option=orjson.OPT_INDENT_2))
                return parsed
            except httpx.HTTPError as exc:
                last_error = exc
                logger.error(
                    "Async request failed for %s (attempt %d/%d): %s",
                    prompt_key,
                    attempt_num,
                    max_attempts,
                    exc,
                )
                if attempt_num >= max_attempts:
                    payload_dump = self._dump_payload_for_logging(payload)
                    logger.error("Request payload for %s:\n%s", prompt_key, payload_dump)
                    response_text: Optional[str] = None
                    status_code: Optional[int] = None
                    if exc.response is not None:
                        status_code = exc.response.status_code
                        try:
                            response_text = exc.response.text
                        except Exception:
                            response_text = None
                    if response_text:
                        logger.error(
                            "Response body for %s (status %s):\\n%s",
                            prompt_key,
                            status_code,
                            response_text,
                        )
                    logger.error("Giving up on %s after %d attempts due to async request errors", prompt_key, max_attempts)
                    return {}
                wait = self._retry_backoff * attempt_num
                if wait > 0:
                    logger.info("Retrying %s after %.1fs due to async request error", prompt_key, wait)
                    await asyncio.sleep(wait)
                continue
        if last_error:
            logger.error("Failed to obtain async response for %s after %d attempts: %s", prompt_key, max_attempts, last_error)
        return {}

    async def _run_ollama_async(
        self,
        http_client: httpx.AsyncClient,
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
            f"{rendered_prompt}\n\n"
            "Return ONLY valid JSON that matches this schema:\n"
            f"{schema_text(prompt_key)}\n"
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

        max_attempts = max(1, retries)
        last_error: Optional[Exception] = None
        for attempt in range(max_attempts):
            attempt_num = attempt + 1
            try:
                response = await http_client.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
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
                    logger.warning("Invalid JSON from Ollama for %s attempt %d: %s", prompt_key, attempt_num, exc)
                    if attempt_num < max_attempts:
                        wait = self._retry_backoff * attempt_num
                        if wait > 0:
                            logger.info("Retrying %s after %.1fs due to async Ollama JSON error", prompt_key, wait)
                            await asyncio.sleep(wait)
                        continue
                    logger.error("Giving up on Ollama request %s after %d attempts due to JSON errors", prompt_key, max_attempts)
                    return {}
                if self.config.cache_raw_responses:
                    cache_path.write_bytes(orjson.dumps(parsed, option=orjson.OPT_INDENT_2))
                return parsed
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                logger.error(
                    "Async Ollama request failed for %s (attempt %d/%d): %s",
                    prompt_key,
                    attempt_num,
                    max_attempts,
                    exc,
                )
                if attempt_num >= max_attempts:
                    payload_dump = self._dump_payload_for_logging(payload)
                    logger.error("Ollama request payload for %s:\\n%s", prompt_key, payload_dump)
                    response_text: Optional[str] = None
                    status_code: Optional[int] = None
                    if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
                        status_code = exc.response.status_code
                        try:
                            response_text = exc.response.text
                        except Exception:
                            response_text = None
                    if response_text:
                        logger.error(
                            "Ollama response body for %s (status %s):\\n%s",
                            prompt_key,
                            status_code,
                            response_text,
                        )
                    logger.error("Giving up on Ollama request %s after %d attempts due to request errors", prompt_key, max_attempts)
                    return {}
                wait = self._retry_backoff * attempt_num
                if wait > 0:
                    logger.info("Retrying %s after %.1fs due to async Ollama request error", prompt_key, wait)
                    await asyncio.sleep(wait)
                continue
        if last_error:
            logger.error("Failed to obtain async Ollama response for %s after %d attempts: %s", prompt_key, max_attempts, last_error)
        return {}
    async def run_async(
        self,
        prompt_key: str,
        rendered_prompt: str,
        cache_path: Path,
        retries: Optional[int] = None,
    ) -> Dict[str, Any]:
        effective_retries = max(1, retries if retries is not None else self.config.llm_retries)
        results = await self.run_many_async([(prompt_key, rendered_prompt, cache_path, effective_retries)])
        if not results:
            raise RuntimeError("run_async returned no results")
        return results[0]

    async def run_many_async(
        self,
        requests_to_run: Sequence[Tuple[str, str, Path, int]],
    ) -> List[Dict[str, Any]]:
        if not requests_to_run:
            return []

        results: List[Optional[Dict[str, Any]]] = [None] * len(requests_to_run)
        pending: List[Tuple[int, str, str, Path, int]] = []

        for idx, item in enumerate(requests_to_run):
            if len(item) == 3:
                prompt_key, rendered_prompt, cache_path = item
                retries = self.config.llm_retries
            else:
                prompt_key, rendered_prompt, cache_path, retries = item
            if self.config.skip_if_cached and cache_path.exists():
                try:
                    results[idx] = orjson.loads(cache_path.read_bytes())
                    continue
                except orjson.JSONDecodeError:
                    logger.warning("Failed to parse cache %s, recomputing", cache_path)
            pending.append((idx, prompt_key, rendered_prompt, cache_path, max(1, retries)))

        if pending:
            async with httpx.AsyncClient(timeout=self._TIMEOUT_SECONDS) as http_client:
                if self.config.backend == "ollama":
                    coroutines = [
                        self._run_ollama_async(http_client, prompt_key, rendered_prompt, cache_path, retries)
                        for (_, prompt_key, rendered_prompt, cache_path, retries) in pending
                    ]
                else:
                    coroutines = [
                        self._run_vllm_async(http_client, prompt_key, rendered_prompt, cache_path, retries)
                        for (_, prompt_key, rendered_prompt, cache_path, retries) in pending
                    ]
                responses = await asyncio.gather(*coroutines)
            for (idx, _, _, _, _), response in zip(pending, responses):
                results[idx] = response

        missing = [i for i, res in enumerate(results) if res is None]
        if missing:
            raise RuntimeError(f"Missing responses for request indices: {missing}")

        return [cast(Dict[str, Any], res) for res in results]

    def run_many(
        self,
        requests_to_run: Sequence[Tuple[str, str, Path, int]],
    ) -> List[Dict[str, Any]]:
        if not requests_to_run:
            return []
        return asyncio.run(self.run_many_async(requests_to_run))

    def run(
        self,
        prompt_key: str,
        rendered_prompt: str,
        cache_path: Path,
        retries: Optional[int] = None,
    ) -> Dict[str, Any]:
        if self.config.skip_if_cached and cache_path.exists():
            try:
                return orjson.loads(cache_path.read_bytes())
            except orjson.JSONDecodeError:
                logger.warning("Failed to parse cache %s, recomputing", cache_path)

        effective_retries = max(1, retries if retries is not None else self.config.llm_retries)

        if self.config.backend == "ollama":
            return self._run_ollama(prompt_key, rendered_prompt, cache_path, effective_retries)
        return self._run_vllm(prompt_key, rendered_prompt, cache_path, effective_retries)


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
    out_path: Path,
    verbose: bool = False,
) -> None:
    db_stem = db_path.stem.replace(".db", "")
    out_root = Path(out_path) / db_stem
    ensure_dir(out_root)
    writer = OutputWriter(out_root / "analysis.db", schema_path)
    cache_enabled = config.cache_raw_responses

    try:
        article_counts = list(iter_article_counts(str(db_path)))
        if not article_counts:
            logger.info("Database %s contains no entries; skipping", db_path)
            return

        valid_entry_ids: List[int] = []
        for entry_id, version_count in article_counts:
            if version_count < config.min_versions:
                logger.info(
                    "Skipping entry %s during pre-filter: only %d version(s) < min_versions=%d",
                    entry_id,
                    version_count,
                    config.min_versions,
                )
                continue
            if config.max_versions is not None and version_count >= config.max_versions:
                logger.info(
                    "Skipping entry %s during pre-filter: %d version(s) exceeds max_versions=%d",
                    entry_id,
                    version_count,
                    config.max_versions,
                )
                continue
            valid_entry_ids.append(entry_id)

        filtered_out = len(article_counts) - len(valid_entry_ids)
        if filtered_out:
            logger.info("Pre-filtered %d of %d entries in %s based on version thresholds (min=%d, max=%s)", filtered_out, len(article_counts), db_path.name, config.min_versions, str(config.max_versions))

        if not valid_entry_ids:
            logger.info(
                "No articles in %s meet version count filters (min_versions=%d, max_versions=%s); skipping database",
                db_path.name,
                config.min_versions,
                str(config.max_versions),
            )
            return

        iterator = tqdm(
            valid_entry_ids,
            total=len(valid_entry_ids),
            desc=f"{db_stem} articles",
            unit="article",
            leave=False,
            dynamic_ncols=True,
            mininterval=0.1,
            smoothing=0.1,
            disable=False,
        )
        try:
            for entry_id in iterator:
                versions = load_versions(str(db_path), entry_id)
                if not versions:
                    logger.info("Skipping entry %s: no versions returned from loader", entry_id)
                    continue

                similarity_threshold = config.skip_similarity_threshold
                if len(versions) >= 2 and similarity_threshold is not None and similarity_threshold > 0:
                    first_text = versions[0].get("summary") or ""
                    final_text = versions[-1].get("summary") or ""
                    similarity = text_jaccard_similarity(first_text, final_text)
                    if similarity >= similarity_threshold:
                        logger.info(
                            "Skipping entry %s: high similarity between first and final versions "
                            "(Jaccard=%.3f, threshold=%.3f)",
                            entry_id,
                            similarity,
                            similarity_threshold,
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
                pair_source_transitions: List[Tuple[Any, ...]] = []
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

                live_blog_checked = False
                skip_article_due_to_live_blog = False

                for version in versions:
                    if skip_article_due_to_live_blog:
                        break
                    version_id = version["id"]
                    version_num = normalize_version_number(version["version"])
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

                    source_catalog_for_prompt: List[Dict[str, Any]] = []
                    for idx, catalog_item in enumerate(processed_mentions, start=1):
                        prompt_source_id = f"source_{idx:03d}"
                        catalog_item["_prompt_source_id"] = prompt_source_id
                        source_catalog_for_prompt.append(
                            {
                                "source_id": prompt_source_id,
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
                            batch_requests.append(("A2_hedge_window", rendered_a2, cache_path, self.config.llm_retries))
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
                        if evidence_type not in {"official_statement", "press_release", "eyewitness", "document", "statistic", "prior_reporting", "social_media", "court_filing", "other"}:
                            evidence_type = "other"
                        evidence_text = mention.get("evidence_text") or mention.get("attributed_text") or ""
                        mention["evidence_type"] = evidence_type
                        mention["evidence_text"] = evidence_text

                        source_id = assign_source_id(mention, source_registry)
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

                    mentions_json = orjson.dumps(
                        mentions,
                        option=orjson.OPT_INDENT_2,
                    ).decode("utf-8")
                    version_payloads[version_id]["source_mentions"] = mentions_json

                    classifier_specs: List[Tuple[str, str, Path, int]] = []
                    if "correction" in text.lower():
                        classifier_specs.append(
                            (
                                "D3_corrections",
                                render_prompt(load_prompt_template("D3_corrections"), {"version_text": text}),
                                version_dir / "D3_corrections.json",
                                self.config.llm_retries,
                            )
                        )
                    classifier_specs.append(
                        (
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
                            self.config.llm_retries,
                        )
                    )
                    classifier_specs.append(
                        (
                            "C1_protest_frame_cues",
                            render_prompt(load_prompt_template("C1_protest_frame_cues"), {"version_text": text}),
                            version_dir / "C1_protest_frame_cues.json",
                            self.config.llm_retries,
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
                        ver["id"],
                        entry_id,
                        news_org,
                        normalize_version_number(ver["version"]),
                        ver.get("created"),
                        ver.get("title"),
                        len(ver.get("summary") or ""),
                    )
                    for ver in versions
                ]

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
                        1,
                    ),
                )
                upsert_versions(writer, versions_rows)
                writer.commit()
                logger.info("Recorded live-blog metadata for article %s and skipped pairwise processing", entry_id)
            else:
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
                        "P3_anon_named_replacement_pair": {
                            "v_prev": prev_text,
                            "v_curr": curr_text,
                            "prev_source_mentions": prev_sources_json,
                            "curr_source_mentions": curr_sources_json,
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
                        "D5_angle_change_pair": {
                            "prev_version_text": prev_text,
                            "curr_version_text": curr_text,
                            "prev_title": prev_title,
                            "curr_title": curr_title,
                            "prev_lede": ledes[prev_num]["text"] if prev_num in ledes else "",
                            "curr_lede": ledes[curr_num]["text"] if curr_num in ledes else "",
                            "prev_source_mentions": prev_sources_json,
                            "curr_source_mentions": curr_sources_json,
                        },
                    }

                    pair_specs: List[Tuple[str, str, Path, int]] = []
                    for key, variables in pair_prompts.items():
                        rendered = render_prompt(load_prompt_template(key), variables)
                        pair_specs.append((key, rendered, pair_dir / f"{key}.json", self.config.llm_retries))
                    pair_results = client.run_many(pair_specs)
                    if len(pair_results) != len(pair_specs):
                        raise RuntimeError(
                            f"Expected {len(pair_specs)} pair responses but received {len(pair_results)}"
                        )
                    results: Dict[str, Dict[str, Any]] = {
                        key: prune_low_confidence(result, config.accept_confidence_min)
                        for (key, _, _, _), result in zip(pair_specs, pair_results)
                    }

                    a3 = results["A3_edit_type_pair"]
                    p10 = results["P10_movement_pair"]
                    d5 = results["D5_angle_change_pair"]

                    movement_up = p10.get("upweighted_summary", "")
                    movement_down = p10.get("downweighted_summary", "")
                    movement_notes_parts: List[str] = []
                    for shift in p10.get("notable_shifts", []) or []:
                        direction = shift.get("direction")
                        snippet = shift.get("snippet")
                        explanation = shift.get("explanation")
                        pieces = [piece for piece in [direction, f'"{snippet}"' if snippet else None, explanation] if piece]
                        if pieces:
                            movement_notes_parts.append(" - ".join(pieces))
                    movement_notes = "; ".join(movement_notes_parts)

                    angle_changed_flag = int(bool(d5.get("angle_changed", False)))
                    angle_category = d5.get("angle_change_category", "no_change")
                    angle_summary = d5.get("angle_summary", "")
                    title_alignment_notes = d5.get("title_alignment_notes", "")

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


                if version_numeric_metrics:
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
                    insert_pair_source_transitions(writer, pair_source_transitions)
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
                else:
                    logger.info(
                        "Skipping article %s after processing: no version metrics were generated; likely no usable source mentions",
                        entry_id,
                    )
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
    parser.add_argument("--out-path", default="./out", type=Path)
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
        process_db(db_path, config, client, args.schema, args.out_path, verbose=args.verbose)


if __name__ == "__main__":
    main()
