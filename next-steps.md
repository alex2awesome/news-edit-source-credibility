# Current Progress & Next Steps

## Current Status
- **Retry & Robustness**: Added configurable LLM retry logic (`llm_retries`, `llm_retry_backoff_seconds`) so the pipeline gracefully handles vLLM timeouts and schema-only responses.
- **Cache-aware Similarity Skip**: Articles with high similarity now reuse cached outputs instead of being skipped outright.
- **Prompt Merges**: Narratives and perspective prompts consolidated (`N1_narrative_keywords` now returns narrative function, perspective, and centrality).
- **Bug Fix**: Corrected `self.config` usage outside the class (NameError) when batching A2 hedging requests.
- **Notebook**: Added `notebooks/ap_analysis_overview.ipynb` for quick exploration of `analysis.db`.
- **Config Update**: Documented retry/similarity thresholds in `config.yaml` and `agent-description.md`.

## Outstanding Work
- **Parallel Refactor**: Need a clean redesign of `process_db` to split per-article work from DB writes, then orchestrate articles via an async/worker pool (while keeping SQLite single-writer semantics).
- **Cache Replay Path**: Create a helper that reads cached JSON and emits row tuples, decoupled from the LLM calls.
- **Structured ArticleResult**: Finish wiring the `ArticleResult` dataclass (introduced during the attempted refactor) or remove it if we defer parallelism.

## Next Steps
1. **Restore Baseline**: Reset `news-edits-pipeline/pipeline.py` to the last working version before starting the refactor (undo the partial scaffolding).
2. **Design Parallel Flow**:
   - Define `process_article(entry_id) -> ArticleResult` that either reuses caches or runs prompts, returning all rows in memory.
   - Implement a queue to run several `process_article` tasks concurrently (e.g., with `asyncio.Semaphore`).
   - Add a writer coroutine that consumes `ArticleResult` objects and serially calls the existing `insert_*` helpers.
3. **Testing**: Spin up a small sample DB to verify concurrency logic, retry handling, and DB commits.
4. **Cleanup**: Disable `cleanup_cached_dirs` while debugging; once stable, decide whether to re-enable or expose via CLI flag.

## Notes
- Parallelism introduces SQLite locking issues; ensure the writer never runs concurrently.
- Make sure caches survive if you rely on them (set `cleanup_cached_dirs: false`).
- Aim to keep logging progress readable with multiple workers.
