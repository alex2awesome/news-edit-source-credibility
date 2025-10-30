"""LLM client utilities for synchronous and asynchronous prompt execution."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import httpx
import orjson
import requests
import tiktoken

from config import Config
from pipeline_utils import ensure_dir, parse_json_response, response_format, schema_text

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)
SYSTEM_PROMPT = (
    "You are a meticulous news-analysis assistant. Always emit valid JSON that matches the requested schema. "
    "Follow the schema exactly, but do NOT repeat the schema text."
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

    def _request_payload(self, prompt: str, response_format: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        too_long = False
        input_tokens = self.tokenizer.encode(prompt)
        max_tokens = max(0, self.config.max_tokens - len(input_tokens) - self.system_prompt_length - 50)
        if max_tokens <= 0:
            logger.warning(
                "Prompt is too long. Max tokens is %s but length of input tokens is %s.",
                self.config.max_tokens,
                len(input_tokens),
            )
            logger.warning("Prompt: %s", prompt)
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
                    logger.error(
                        "Giving up on %s after %d attempts due to JSON parsing errors",
                        prompt_key,
                        max_attempts,
                    )
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
                            "Response body for %s (status %s):\n%s",
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
                            logger.info("Retrying %s after %.1fs due to JSON error", prompt_key, wait)
                            time.sleep(wait)
                        continue
                    logger.error(
                        "Giving up on Ollama request %s after %d attempts due to JSON errors",
                        prompt_key,
                        max_attempts,
                    )
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
                    payload_dump = self._dump_payload_for_logging(payload)
                    logger.error("Ollama request payload for %s:\n%s", prompt_key, payload_dump)
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
                            "Ollama response body for %s (status %s):\n%s",
                            prompt_key,
                            status_code,
                            response_text,
                        )
                    return {}
                wait = self._retry_backoff * attempt_num
                if wait > 0:
                    logger.info("Retrying %s after %.1fs due to Ollama request error", prompt_key, wait)
                    time.sleep(wait)
                continue
        if last_error:
            logger.error(
                "Failed to obtain Ollama response for %s after %d attempts: %s",
                prompt_key,
                max_attempts,
                last_error,
            )
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
                    logger.error(
                        "Giving up on %s after %d attempts due to async JSON parsing errors",
                        prompt_key,
                        max_attempts,
                    )
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
                            "Response body for %s (status %s):\n%s",
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
            logger.error(
                "Failed to obtain async response for %s after %d attempts: %s",
                prompt_key,
                max_attempts,
                last_error,
            )
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
                    logger.error(
                        "Giving up on Ollama request %s after %d attempts due to JSON errors",
                        prompt_key,
                        max_attempts,
                    )
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
                    logger.error("Ollama request payload for %s:\n%s", prompt_key, payload_dump)
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
                            "Ollama response body for %s (status %s):\n%s",
                            prompt_key,
                            status_code,
                            response_text,
                        )
                    logger.error(
                        "Giving up on Ollama request %s after %d attempts due to request errors",
                        prompt_key,
                        max_attempts,
                    )
                    return {}
                wait = self._retry_backoff * attempt_num
                if wait > 0:
                    logger.info("Retrying %s after %.1fs due to async Ollama request error", prompt_key, wait)
                    await asyncio.sleep(wait)
                continue
        if last_error:
            logger.error(
                "Failed to obtain async Ollama response for %s after %d attempts: %s",
                prompt_key,
                max_attempts,
                last_error,
            )
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

    def run_many(
        self,
        requests_to_run: Sequence[Tuple[str, str, Path, int]],
    ) -> List[Dict[str, Any]]:
        if not requests_to_run:
            return []
        return asyncio.run(self.run_many_async(requests_to_run))

    def close(self) -> None:
        try:
            self.session.close()
        except Exception:
            logger.debug("Failed to close HTTP session cleanly")
