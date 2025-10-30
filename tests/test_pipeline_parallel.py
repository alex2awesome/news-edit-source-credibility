import asyncio
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import json
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
PIPELINE_DIR = ROOT_DIR / "news-edits-pipeline"
sys.path.insert(0, str(PIPELINE_DIR))

import types


class _DummyTokenizer:
    def encode(self, _text: str) -> list[int]:
        return []


sys.modules.setdefault(
    "tiktoken",
    types.SimpleNamespace(get_encoding=lambda _name: _DummyTokenizer()),
)

sys.modules.setdefault(
    "orjson",
    types.SimpleNamespace(
        OPT_INDENT_2=0,
        dumps=lambda obj, option=None: json.dumps(obj).encode("utf-8"),
        loads=lambda data: json.loads(data.decode("utf-8") if isinstance(data, (bytes, bytearray)) else data),
    ),
)

analysis_stub = types.SimpleNamespace(
    aggregate_sources_over_versions=lambda *_args, **_kwargs: {"sources": []},
    compute_diff_magnitude=lambda *_args, **_kwargs: {
        "tokens_added": 0,
        "tokens_deleted": 0,
        "percent_text_new": 0.0,
    },
    compute_prominence_features=lambda *_args, **_kwargs: {"lead_percentile": 0.0},
    extract_lede=lambda *_args, **_kwargs: {"text": ""},
    final_version_bias=lambda *_args, **_kwargs: {},
    inter_update_timing=lambda *_args, **_kwargs: 0.0,
    jaccard_title_body=lambda *_args, **_kwargs: 0.0,
    ner_entities_spacy=lambda *_args, **_kwargs: {"entities": []},
    segment=lambda *_args, **_kwargs: {"paragraphs": [], "sentences": [], "tokens": [], "char_len": 0},
)
sys.modules.setdefault("analysis", analysis_stub)

canonicalize_stub = types.SimpleNamespace(
    fuzzy_match_source=lambda *_args, **_kwargs: None,
    normalize_source=lambda value: (value or "").lower(),
)
sys.modules.setdefault("canonicalize", canonicalize_stub)

loader_stub = types.SimpleNamespace(
    iter_article_counts=lambda _path: [],
    load_versions=lambda *_args, **_kwargs: [],
)
sys.modules.setdefault("loader", loader_stub)

import pipeline  # type: ignore  # noqa: E402


@pytest.fixture(autouse=True)
def reset_asyncio_policy():
    # Ensure a fresh event loop for asyncio.run during tests.
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())


def build_article_result(entry_id: int) -> pipeline.ArticleResult:
    news_org = "TestOrg"
    article_row = (entry_id, news_org, "http://example.com", "Title", "Title", "2024-01-01T00:00:00Z", 1, 0)
    versions_rows = [
        (entry_id * 10 + 1, entry_id, news_org, 0, "2024-01-01T00:00:00Z", "Title", 100),
        (entry_id * 10 + 2, entry_id, news_org, 1, "2024-01-01T01:00:00Z", "Title", 120),
    ]
    empty_rows: List[Tuple] = []
    return pipeline.ArticleResult(
        entry_id=entry_id,
        news_org=news_org,
        article_row=article_row,
        versions_rows=versions_rows,
        source_mentions_rows=empty_rows,
        entity_rows=empty_rows,
        version_metrics_rows=empty_rows,
        pair_rows=empty_rows,
        pair_sources_added=empty_rows,
        pair_sources_removed=empty_rows,
        pair_source_transitions=empty_rows,
        pair_replacements=empty_rows,
        pair_numeric=empty_rows,
        pair_claims_rows=empty_rows,
        pair_cues_rows=empty_rows,
        sources_agg_rows=empty_rows,
        article_metrics_row=None,
        live_blog_only=False,
        log_message=f"Processed article {entry_id}",
    )


def test_process_db_parallel_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / "input.db"
    db_path.write_text("")

    schema_path = Path("news-edits-pipeline/schema_out.sql")

    config = pipeline.Config(
        model="test",
        vllm_api_base="http://localhost",
        spacy_model="en_core_web_sm",
        tiktoken_encoding=None,
        temperature=0.0,
        max_tokens=1024,
        batch_size=2,
        hedge_window_tokens=80,
        accept_confidence_min=3.0,
        out_root=tmp_path,
        cache_raw_responses=False,
        skip_if_cached=False,
        backend="vllm",
        ollama_api_base=None,
        min_versions=1,
        max_versions=None,
        cleanup_cached_dirs=False,
        skip_similarity_threshold=None,
        llm_retries=1,
        llm_retry_backoff_seconds=0.0,
    )

    counts = [(1, 2), (2, 2)]
    monkeypatch.setattr(pipeline, "iter_article_counts", lambda _: counts)

    processed_ids: List[int] = []

    def fake_process_article(
        entry_id: int,
        _db_path: Path,
        _config: pipeline.Config,
        _client: Optional[pipeline.StructuredLLMClient],
        _out_root: Path,
        verbose: bool = False,
    ) -> Optional[pipeline.ArticleResult]:
        # simulate staggered work to exercise concurrency ordering
        time.sleep(0.05 if entry_id == 1 else 0.01)
        return build_article_result(entry_id)

    class DummyClient:
        def __init__(self, _config: pipeline.Config):
            pass

        def close(self) -> None:
            pass

    def fake_write_article_result(_writer: pipeline.OutputWriter, result: pipeline.ArticleResult) -> None:
        processed_ids.append(result.entry_id)

    monkeypatch.setattr(pipeline, "process_article", fake_process_article)
    monkeypatch.setattr(pipeline, "StructuredLLMClient", DummyClient)
    monkeypatch.setattr(pipeline, "write_article_result", fake_write_article_result)

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    pipeline.process_db(db_path, config, schema_path, out_dir, verbose=False)

    assert sorted(processed_ids) == [1, 2]
    # Results may arrive out of order; ensure concurrency did not duplicate processing.
    assert len(processed_ids) == len(counts)
