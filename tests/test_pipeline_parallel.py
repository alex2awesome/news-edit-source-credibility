import asyncio
import sys
import time
from datetime import datetime, timezone
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
import article_processor  # type: ignore  # noqa: E402


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
        article_workers=2,
        max_in_flight_llm_requests=8,
        hedge_window_tokens=80,
        accept_confidence_min=3.0,
        out_root=tmp_path,
        cache_raw_responses=False,
        skip_if_cached=False,
        backend="vllm",
        ollama_api_base=None,
        min_versions=1,
        max_versions=None,
        max_articles_per_outlet=None,
        cleanup_cached_dirs=False,
        skip_similarity_threshold=None,
        llm_retries=1,
        llm_retry_backoff_seconds=0.0,
        matt_features_only=False,
    )

    counts = [
        (1, 2, "2024-01-01T00:00:00Z"),
        (2, 2, "2024-01-10T00:00:00Z"),
    ]
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


def test_process_db_filters_by_date_range(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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
        batch_size=1,
        article_workers=1,
        max_in_flight_llm_requests=8,
        hedge_window_tokens=80,
        accept_confidence_min=3.0,
        out_root=tmp_path,
        cache_raw_responses=False,
        skip_if_cached=False,
        backend="vllm",
        ollama_api_base=None,
        min_versions=1,
        max_versions=None,
        max_articles_per_outlet=None,
        cleanup_cached_dirs=False,
        skip_similarity_threshold=None,
        llm_retries=1,
        llm_retry_backoff_seconds=0.0,
        matt_features_only=False,
    )

    counts = [
        (1, 2, "2024-01-01T00:00:00Z"),
        (2, 2, "2024-02-01T00:00:00Z"),
    ]
    monkeypatch.setattr(pipeline, "iter_article_counts", lambda _path: counts)

    processed_ids: List[int] = []

    def fake_process_article(
        entry_id: int,
        _db_path: Path,
        _config: pipeline.Config,
        _client: Optional[pipeline.StructuredLLMClient],
        _out_root: Path,
        verbose: bool = False,
    ) -> Optional[pipeline.ArticleResult]:
        processed_ids.append(entry_id)
        return build_article_result(entry_id)

    monkeypatch.setattr(pipeline, "process_article", fake_process_article)
    monkeypatch.setattr(pipeline, "write_article_result", lambda *_args, **_kwargs: None)

    out_dir = tmp_path / "out_date"
    out_dir.mkdir()

    date_range = (
        datetime(2024, 1, 15, tzinfo=timezone.utc),
        datetime(2024, 2, 15, tzinfo=timezone.utc),
    )

    pipeline.process_db(
        db_path,
        config,
        schema_path,
        out_dir,
        verbose=False,
        date_range=date_range,
    )

    assert processed_ids == [2]


def test_process_db_downsamples_all(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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
        batch_size=1,
        article_workers=1,
        max_in_flight_llm_requests=8,
        hedge_window_tokens=80,
        accept_confidence_min=3.0,
        out_root=tmp_path,
        cache_raw_responses=False,
        skip_if_cached=False,
        backend="vllm",
        ollama_api_base=None,
        min_versions=1,
        max_versions=None,
        max_articles_per_outlet=None,
        cleanup_cached_dirs=False,
        skip_similarity_threshold=None,
        llm_retries=1,
        llm_retry_backoff_seconds=0.0,
        matt_features_only=False,
    )

    counts = [
        (1, 2, "2024-01-01T00:00:00Z"),
        (2, 2, "2024-01-02T00:00:00Z"),
    ]
    monkeypatch.setattr(pipeline, "iter_article_counts", lambda _path: counts)

    processed_ids: List[int] = []

    def fake_process_article(
        entry_id: int,
        _db_path: Path,
        _config: pipeline.Config,
        _client: Optional[pipeline.StructuredLLMClient],
        _out_root: Path,
        verbose: bool = False,
    ) -> Optional[pipeline.ArticleResult]:
        processed_ids.append(entry_id)
        return build_article_result(entry_id)

    monkeypatch.setattr(pipeline, "process_article", fake_process_article)
    monkeypatch.setattr(pipeline, "write_article_result", lambda *_args, **_kwargs: None)

    out_dir = tmp_path / "out_downsample"
    out_dir.mkdir()

    pipeline.process_db(
        db_path,
        config,
        schema_path,
        out_dir,
        verbose=False,
        downsample_percentage=0.0,
    )

    assert processed_ids == []


def test_process_db_caps_max_articles_per_outlet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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
        batch_size=1,
        article_workers=1,
        max_in_flight_llm_requests=8,
        hedge_window_tokens=80,
        accept_confidence_min=3.0,
        out_root=tmp_path,
        cache_raw_responses=False,
        skip_if_cached=False,
        backend="vllm",
        ollama_api_base=None,
        min_versions=1,
        max_versions=None,
        max_articles_per_outlet=2,
        cleanup_cached_dirs=False,
        skip_similarity_threshold=None,
        llm_retries=1,
        llm_retry_backoff_seconds=0.0,
        matt_features_only=False,
    )

    counts = [
        (1, 2, "2024-01-01T00:00:00Z"),
        (2, 2, "2024-01-02T00:00:00Z"),
        (3, 2, "2024-01-03T00:00:00Z"),
    ]
    monkeypatch.setattr(pipeline, "iter_article_counts", lambda _path: counts)
    monkeypatch.setattr(pipeline.random, "sample", lambda seq, k: list(seq)[:k])

    processed_ids: List[int] = []

    def fake_process_article(
        entry_id: int,
        _db_path: Path,
        _config: pipeline.Config,
        _client: Optional[pipeline.StructuredLLMClient],
        _out_root: Path,
        verbose: bool = False,
    ) -> Optional[pipeline.ArticleResult]:
        processed_ids.append(entry_id)
        return build_article_result(entry_id)

    monkeypatch.setattr(pipeline, "process_article", fake_process_article)
    monkeypatch.setattr(pipeline, "write_article_result", lambda *_args, **_kwargs: None)

    out_dir = tmp_path / "out_cap"
    out_dir.mkdir()

    pipeline.process_db(
        db_path,
        config,
        schema_path,
        out_dir,
        verbose=False,
        max_articles_per_outlet=2,
    )

    assert processed_ids == [1, 2]


def test_process_db_alex_pair_list_limits_articles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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
        batch_size=1,
        article_workers=1,
        max_in_flight_llm_requests=8,
        hedge_window_tokens=80,
        accept_confidence_min=3.0,
        out_root=tmp_path,
        cache_raw_responses=False,
        skip_if_cached=False,
        backend="vllm",
        ollama_api_base=None,
        min_versions=1,
        max_versions=None,
        max_articles_per_outlet=None,
        cleanup_cached_dirs=False,
        skip_similarity_threshold=None,
        llm_retries=1,
        llm_retry_backoff_seconds=0.0,
        matt_features_only=False,
        alex_features_only=True,
    )

    counts = [
        (1, 2, "2024-01-01T00:00:00Z"),
        (2, 2, "2024-01-02T00:00:00Z"),
        (3, 2, "2024-01-03T00:00:00Z"),
    ]
    monkeypatch.setattr(pipeline, "iter_article_counts", lambda _path: counts)

    processed_ids: List[int] = []
    pair_flags: List[bool] = []

    def fake_process_article(
        entry_id: int,
        _db_path: Path,
        _config: pipeline.Config,
        _client: Optional[pipeline.StructuredLLMClient],
        _out_root: Path,
        _verbose: bool = False,
        _alex_pair_directives=None,
        _alex_pair_list_supplied: bool = False,
    ) -> Optional[pipeline.ArticleResult]:
        processed_ids.append(entry_id)
        pair_flags.append(_alex_pair_list_supplied)
        return build_article_result(entry_id)

    monkeypatch.setattr(pipeline, "process_article", fake_process_article)
    monkeypatch.setattr(pipeline, "write_article_result", lambda *_args, **_kwargs: None)

    out_dir = tmp_path / "out_alex"
    out_dir.mkdir()

    pipeline.process_db(
        db_path,
        config,
        schema_path,
        out_dir,
        verbose=False,
        alex_pair_directives={2: [("TestOrg", None, None)]},
    )

    assert processed_ids == [2]
    assert pair_flags == [True]


def test_load_alex_pair_list_parses_optional_versions(tmp_path: Path):
    csv_path = tmp_path / "pairs.csv"
    csv_path.write_text(
        "news_org,entry_id,from_version_num,to_version_num\n"
        "ap,10,1,3\n"
        "ap,11,None,None\n"
        "ap,12,,\n",
        encoding="utf-8",
    )

    parsed = pipeline._load_alex_pair_list(csv_path)
    assert parsed[10] == [("ap", 1, 3)]
    assert parsed[11] == [("ap", None, None)]
    assert parsed[12] == [("ap", None, None)]


def test_alex_source_alias_matching_for_pair_list():
    versions = [
        {"id": "v0", "version": 0, "summary": "a", "title": "t", "source": "newssniffer-guardian"},
        {"id": "v1", "version": 1, "summary": "b", "title": "t", "source": "newssniffer-guardian"},
    ]
    pairs = article_processor._select_alex_pairs(
        entry_id=497799,
        news_org="newssniffer-guardian",
        versions=versions,
        random_pairs_per_article=0,
        pair_directives=[("guardian", None, None)],
        pair_list_supplied=True,
        dynamic_pairs_enabled=False,
        dynamic_only=False,
        dynamic_target_change=0.20,
    )
    assert len(pairs) == 1
    prev, curr, policy = pairs[0]
    assert prev["id"] == "v0"
    assert curr["id"] == "v1"
    assert policy == "pair_list_generated"


def test_alex_dynamic_pair_selection_adds_dynamic_policy():
    versions = [
        {"id": "v0", "version": 0, "summary": "A", "title": "t", "source": "newssniffer-guardian"},
        {"id": "v1", "version": 1, "summary": "A B", "title": "t", "source": "newssniffer-guardian"},
        {"id": "v2", "version": 2, "summary": "A B C D E", "title": "t", "source": "newssniffer-guardian"},
    ]
    pairs = article_processor._select_alex_pairs(
        entry_id=111,
        news_org="newssniffer-guardian",
        versions=versions,
        random_pairs_per_article=0,
        pair_directives=None,
        pair_list_supplied=False,
        dynamic_pairs_enabled=True,
        dynamic_only=True,
        dynamic_target_change=0.20,
    )

    assert pairs
    assert all("dynamically_generated" in policy for _, _, policy in pairs)
    assert all(policy == "dynamically_generated" for _, _, policy in pairs)


def test_alex_policy_merge_includes_dynamic_and_pair_list_generated():
    versions = [
        {"id": "v0", "version": 0, "summary": "a", "title": "t", "source": "newssniffer-guardian"},
        {"id": "v1", "version": 1, "summary": "b", "title": "t", "source": "newssniffer-guardian"},
    ]
    pairs = article_processor._select_alex_pairs(
        entry_id=7,
        news_org="newssniffer-guardian",
        versions=versions,
        random_pairs_per_article=0,
        pair_directives=[("guardian", None, None)],
        pair_list_supplied=True,
        dynamic_pairs_enabled=True,
        dynamic_only=False,
        dynamic_target_change=0.20,
    )

    assert len(pairs) == 1
    _, _, policy = pairs[0]
    assert policy == "dynamically_generated|pair_list_generated"


def test_process_db_alex_pair_list_bypasses_global_filters(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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
        batch_size=1,
        article_workers=1,
        max_in_flight_llm_requests=8,
        hedge_window_tokens=80,
        accept_confidence_min=3.0,
        out_root=tmp_path,
        cache_raw_responses=False,
        skip_if_cached=False,
        backend="vllm",
        ollama_api_base=None,
        min_versions=99,
        max_versions=1,
        max_articles_per_outlet=1,
        cleanup_cached_dirs=False,
        skip_similarity_threshold=0.0,
        llm_retries=1,
        llm_retry_backoff_seconds=0.0,
        matt_features_only=False,
        alex_features_only=True,
    )

    counts = [
        (2, 2, "2024-01-02T00:00:00Z"),
    ]
    monkeypatch.setattr(pipeline, "iter_article_counts", lambda _path: counts)

    processed_ids: List[int] = []

    def fake_process_article(
        entry_id: int,
        _db_path: Path,
        _config: pipeline.Config,
        _client: Optional[pipeline.StructuredLLMClient],
        _out_root: Path,
        _verbose: bool = False,
        _alex_pair_directives=None,
        _alex_pair_list_supplied: bool = False,
    ) -> Optional[pipeline.ArticleResult]:
        processed_ids.append(entry_id)
        return build_article_result(entry_id)

    monkeypatch.setattr(pipeline, "process_article", fake_process_article)
    monkeypatch.setattr(pipeline, "write_article_result", lambda *_args, **_kwargs: None)

    out_dir = tmp_path / "out_alex_bypass"
    out_dir.mkdir()

    pipeline.process_db(
        db_path,
        config,
        schema_path,
        out_dir,
        verbose=False,
        downsample_percentage=0.0,
        max_articles_per_outlet=1,
        date_range=(
            datetime(2025, 1, 1, tzinfo=timezone.utc),
            datetime(2025, 1, 2, tzinfo=timezone.utc),
        ),
        alex_pair_directives={2: [("guardian", None, None)]},
    )

    assert processed_ids == [2]


def test_alex_process_article_ignores_similarity_and_version_thresholds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = pipeline.Config(
        model="test",
        vllm_api_base="http://localhost",
        spacy_model="en_core_web_sm",
        tiktoken_encoding=None,
        temperature=0.0,
        max_tokens=1024,
        batch_size=1,
        article_workers=1,
        max_in_flight_llm_requests=8,
        hedge_window_tokens=80,
        accept_confidence_min=3.0,
        out_root=tmp_path,
        cache_raw_responses=False,
        skip_if_cached=False,
        backend="vllm",
        ollama_api_base=None,
        min_versions=99,
        max_versions=1,
        max_articles_per_outlet=None,
        cleanup_cached_dirs=False,
        skip_similarity_threshold=0.0,
        llm_retries=1,
        llm_retry_backoff_seconds=0.0,
        matt_features_only=False,
        alex_features_only=True,
    )

    versions = [
        {"id": "v0", "entry_id": 10, "version": 0, "created": "2024-01-01T00:00:00Z", "title": "T1", "url": "u", "summary": "same"},
        {"id": "v1", "entry_id": 10, "version": 1, "created": "2024-01-01T01:00:00Z", "title": "T2", "url": "u", "summary": "same"},
    ]
    monkeypatch.setattr(article_processor, "load_versions", lambda _db, _entry: [
        dict(item, source="newssniffer-guardian") for item in versions
    ])

    class DummyClient:
        def run(self, *_args, **_kwargs):
            return [{"request": "r", "writer action": "a", "content added": None, "content removed": None, "content changed": None}]

        def close(self):
            return None

    result = article_processor.process_article(
        entry_id=10,
        db_path=Path("article-versions/newssniffer-guardian.db.gz"),
        config=config,
        client=DummyClient(),
        out_root=tmp_path,
        verbose=False,
        alex_pair_directives=[("guardian", None, None)],
        alex_pair_list_supplied=True,
    )

    assert result is not None
    assert len(result.pair_edit_actions_rows) >= 1


def test_matt_features_only_lean_path_skips_cpu_extras(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = pipeline.Config(
        model="test",
        vllm_api_base="http://localhost",
        spacy_model="en_core_web_sm",
        tiktoken_encoding=None,
        temperature=0.0,
        max_tokens=1024,
        batch_size=1,
        article_workers=1,
        max_in_flight_llm_requests=8,
        hedge_window_tokens=80,
        accept_confidence_min=3.0,
        out_root=tmp_path,
        cache_raw_responses=False,
        skip_if_cached=False,
        backend="vllm",
        ollama_api_base=None,
        min_versions=1,
        max_versions=None,
        max_articles_per_outlet=None,
        cleanup_cached_dirs=False,
        skip_similarity_threshold=0.95,
        llm_retries=1,
        llm_retry_backoff_seconds=0.0,
        matt_features_only=True,
        alex_features_only=False,
    )

    versions = [
        {
            "id": "v0",
            "entry_id": 1,
            "version": 0,
            "created": "2024-01-01T00:00:00Z",
            "title": "T0",
            "url": "u",
            "summary": "alpha",
            "source": "newssniffer-guardian",
        },
        {
            "id": "v1",
            "entry_id": 1,
            "version": 1,
            "created": "2024-01-01T01:00:00Z",
            "title": "T1",
            "url": "u",
            "summary": "beta",
            "source": "newssniffer-guardian",
        },
    ]
    monkeypatch.setattr(article_processor, "load_versions", lambda *_args, **_kwargs: versions)
    monkeypatch.setattr(article_processor, "segment", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("segment should not run")))
    monkeypatch.setattr(
        article_processor,
        "ner_entities_spacy",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("ner_entities_spacy should not run")),
    )
    monkeypatch.setattr(
        article_processor,
        "compute_diff_magnitude",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("compute_diff_magnitude should not run")),
    )
    monkeypatch.setattr(
        article_processor,
        "jaccard_title_body",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("jaccard_title_body should not run")),
    )
    monkeypatch.setattr(
        article_processor,
        "text_jaccard_similarity",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("text_jaccard_similarity should not run")),
    )

    class DummyClient:
        def __init__(self):
            self.run_keys = []
            self.run_many_keys = []

        def run(self, key, *_args, **_kwargs):
            self.run_keys.append(key)
            if key == "A1_source_mentions":
                return {
                    "source_mentions": [
                        {
                            "surface": "Alice",
                            "canonical": "Alice",
                            "type": "individual",
                            "attributed_text": "Alice said.",
                            "confidence": 5,
                        }
                    ]
                }
            return {}

        def run_many(self, specs):
            self.run_many_keys.extend(spec[0] for spec in specs)
            mapping = {
                "A3_edit_type_pair": {"edit_type": "edit", "summary_of_change": "changed", "confidence": 5},
                "D5_angle_change_pair": {
                    "angle_changed": False,
                    "angle_change_category": "no_change",
                    "angle_summary": "",
                    "title_alignment_notes": "",
                    "confidence": 5,
                    "evidence_snippets": [],
                },
                "P10_movement_pair": {
                    "movement_summary_upweighted": "",
                    "movement_summary_downweighted": "",
                    "movement_notes": [],
                    "confidence": 5,
                    "notable_shifts": [],
                },
            }
            return [mapping[spec[0]] for spec in specs]

        def close(self):
            return None

    client = DummyClient()
    result = article_processor.process_article(
        entry_id=1,
        db_path=Path("article-versions/newssniffer-guardian.db.gz"),
        config=config,
        client=client,
        out_root=tmp_path,
        verbose=False,
    )

    assert result is not None
    assert result.entity_rows == []
    assert len(result.source_mentions_rows) >= 1
    assert len(result.version_metrics_rows) == 2
    assert len(result.pair_rows) == 1
    assert "A1_source_mentions" in client.run_keys
    assert "D4_live_blog_detect" not in client.run_keys
    assert set(client.run_many_keys) == {"A3_edit_type_pair", "D5_angle_change_pair", "P10_movement_pair"}
