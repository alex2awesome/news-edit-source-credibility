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
    article_workers: int
    max_in_flight_llm_requests: int
    hedge_window_tokens: int
    accept_confidence_min: float
    out_root: Path
    cache_raw_responses: bool
    skip_if_cached: bool
    backend: str
    ollama_api_base: Optional[str]
    min_versions: int
    max_versions: Optional[int]
    max_articles_per_outlet: Optional[int]
    cleanup_cached_dirs: bool
    skip_similarity_threshold: Optional[float]
    llm_retries: int
    llm_retry_backoff_seconds: float
    matt_features_only: bool
    alex_features_only: bool = False
    alex_random_pairs_per_article: int = 0
    alex_pair_list: Optional[Path] = None
    alex_dynamic_pairs: bool = False
    dynamic_only: bool = False
    dynamic_target_change: float = 0.20
    reprompt_for_structure: bool = False
    prompt_for_structure_fix: bool = False
    structure_max_attempts: int = 5

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
        max_articles_value = data.get("max_articles_per_outlet")
        if max_articles_value is None:
            max_articles_per_outlet = None
        else:
            max_articles_per_outlet = max(1, int(max_articles_value))
        return Config(
            model=data.get("model"),
            vllm_api_base=data.get("vllm_api_base"),
            spacy_model=spacy_model,
            tiktoken_encoding=data.get("tiktoken_encoding", "cl100k_base"),
            temperature=float(data.get("temperature", 0.0)),
            max_tokens=int(data.get("max_tokens", 2048)),
            batch_size=int(data.get("batch_size", 1)),
            article_workers=max(1, int(data.get("article_workers", data.get("batch_size", 1)))),
            max_in_flight_llm_requests=max(1, int(data.get("max_in_flight_llm_requests", 8))),
            hedge_window_tokens=int(data.get("hedge_window_tokens", 80)),
            accept_confidence_min=float(data.get("accept_confidence_min", 3.0)),
            out_root=Path(data.get("out_root", "./out")),
            cache_raw_responses=bool(data.get("cache_raw_responses", True)),
            skip_if_cached=bool(data.get("skip_if_cached", True)),
            backend=backend,
            ollama_api_base=data.get("ollama_api_base"),
            min_versions=min_versions,
            max_versions=max_versions,
            max_articles_per_outlet=max_articles_per_outlet,
            cleanup_cached_dirs=bool(data.get("cleanup_cached_dirs", False)),
            skip_similarity_threshold=(
                float(data["skip_similarity_threshold"])
                if data.get("skip_similarity_threshold") is not None
                else 0.95
            ),
            llm_retries=int(data.get("llm_retries", 2)),
            llm_retry_backoff_seconds=float(data.get("llm_retry_backoff_seconds", 2.0)),
            reprompt_for_structure=bool(data.get("reprompt_for_structure", False)),
            prompt_for_structure_fix=bool(data.get("prompt_for_structure_fix", False)),
            structure_max_attempts=max(1, int(data.get("structure_max_attempts", 5))),
            matt_features_only=bool(data.get("matt_features_only", False)),
            alex_features_only=bool(data.get("alex_features_only", False)),
            alex_random_pairs_per_article=max(0, int(data.get("alex_random_pairs_per_article", 0))),
            alex_pair_list=(Path(data["alex_pair_list"]) if data.get("alex_pair_list") else None),
            alex_dynamic_pairs=bool(data.get("alex_dynamic_pairs", False)),
            dynamic_only=bool(data.get("dynamic_only", False)),
            dynamic_target_change=float(data.get("dynamic_target_change", 0.20)),
        )


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run news edits pipeline over SQLite databases")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--db", action="append", dest="dbs", required=True, type=Path)
    parser.add_argument("--schema", default=Path(__file__).with_name("schema_out.sql"), type=Path)
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--out-path", default=None, type=Path)
    parser.add_argument(
        "--vllm-api-base",
        type=str,
        default=None,
        help="Override vLLM/OpenAI-compatible base URL from config (e.g. http://172.24.75.251:8000/v1).",
    )
    parser.add_argument(
        "--downsample-percentage",
        type=float,
        default=None,
        help="Randomly keep approximately this percent of eligible articles (0-100, applied per article).",
    )
    parser.add_argument(
        "--max-articles-per-outlet",
        type=int,
        default=None,
        help="Uniformly sample up to this many articles per outlet/database after filtering.",
    )
    parser.add_argument(
        "--article-workers",
        type=int,
        default=None,
        help="Override number of article workers (defaults to config article_workers or batch_size).",
    )
    parser.add_argument(
        "--max-in-flight-llm-requests",
        type=int,
        default=None,
        help="Cap concurrent LLM HTTP requests across workers.",
    )
    parser.add_argument(
        "--date-range",
        nargs=2,
        metavar=("START", "END"),
        help="Restrict processing to articles whose first version timestamp falls within the inclusive ISO 8601 range.",
    )
    parser.add_argument(
        "--matt-features-only",
        action="store_true",
        help="Skip prompts/tables outside Matt's minimal feature set to reduce runtime and cost.",
    )
    parser.add_argument(
        "--alex-features-only",
        action="store_true",
        help="Run only Alex's pair edit-action prompt suite.",
    )
    parser.add_argument(
        "--alex-random-pairs-per-article",
        type=int,
        default=None,
        help="Sample up to this many additional random version pairs per article in alex mode.",
    )
    parser.add_argument(
        "--alex-pair-list",
        type=Path,
        default=None,
        help=(
            "Optional CSV with columns news_org,entry_id,from_version_num,to_version_num. "
            "When provided in alex mode, only listed articles are processed."
        ),
    )
    parser.add_argument(
        "--alex-dynamic-pairs",
        action="store_true",
        help="Enable dynamic version stride pairing in alex mode targeting --dynamic-target-change.",
    )
    parser.add_argument(
        "--dynamic-only",
        action="store_true",
        help=(
            "In alex mode, select only dynamic pairs plus explicit pair-list rows "
            "(disables first/final, consecutive, and random generated pairs)."
        ),
    )
    parser.add_argument(
        "--dynamic-target-change",
        type=float,
        default=None,
        help="Target change ratio per dynamic stride in (0,1). Default: 0.20.",
    )
    parser.add_argument(
        "--disable-similarity-skip",
        action="store_true",
        help="Disable first-vs-final similarity pre-filter by setting skip_similarity_threshold to null.",
    )
    parser.add_argument(
        "--disable-max-versions",
        action="store_true",
        help="Disable max_versions filtering by setting max_versions to null.",
    )
    parser.add_argument(
        "--reprompt-for-structure",
        action="store_true",
        help=(
            "Disable vLLM response_format and rerun prompts up to structure_max_attempts until "
            "output parses and validates against schema."
        ),
    )
    parser.add_argument(
        "--prompt-for-structure-fix",
        action="store_true",
        help=(
            "Disable vLLM response_format and, after failed parsing/schema checks, "
            "ask the model to repair output to match schema."
        ),
    )
    return parser.parse_args()
