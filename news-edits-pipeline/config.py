"""Configuration and CLI helpers for the news edits pipeline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


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

        spacy_model = data.get("spacy_model")
        if not isinstance(spacy_model, str) or not spacy_model.strip():
            raise ValueError("Config must define 'spacy_model' with a valid spaCy model name (e.g. 'en_core_web_sm').")
        spacy_model = spacy_model.strip()

        min_versions = max(1, int(data.get("min_versions", 2)))
        max_versions_value = data.get("max_versions", 20)
        if max_versions_value is None:
            max_versions = None
        else:
            max_versions = int(max_versions_value)
        return Config(
            model=data.get("model"),
            vllm_api_base=data.get("vllm_api_base"),
            spacy_model=spacy_model,
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


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run news edits pipeline over SQLite databases")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--db", action="append", dest="dbs", required=True, type=Path)
    parser.add_argument("--schema", default=Path(__file__).with_name("schema_out.sql"), type=Path)
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--out-path", default=Path("./out"), type=Path)
    return parser.parse_args()
