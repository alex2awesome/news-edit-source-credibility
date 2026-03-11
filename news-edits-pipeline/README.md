# News Edits Pipeline

Tools for analysing versioned news articles using a mix of deterministic NLP and structured LLM calls. The pipeline produces a harmonised SQLite database capturing sourcing, framing, and change dynamics across every version of each article.

## Environment Setup

- Python 3.10+ recommended. Install dependencies:
  ```bash
  pip install -r requirements.txt
  python -m spacy download en_core_web_sm
  ```
- Launch a vLLM OpenAI-compatible server (or equivalent) that supports [Structured Outputs](https://docs.vllm.ai/en/v0.8.1/features/structured_outputs.html). Example:
  ```bash
  python -m vllm.entrypoints.openai.api_server \
    --model /models/llama-3.2-70b-instruct \
    --host 0.0.0.0 --port 8000 \
    --dtype auto
  ```

## Configuration

`config.yaml` controls runtime behaviour:

| key | description |
| --- | --- |
| `model` | Model name exposed by the chosen backend (vLLM, Ollama, etc.). |
| `vllm_api_base` | Base URL for the OpenAI-compatible endpoint (also reused as fallback Ollama host). |
| `temperature` / `max_tokens` | LLM decoding params (kept deterministic by default). |
| `batch_size` | Backward-compatible fallback for article worker count (legacy knob). |
| `article_workers` | Number of article-level workers processed in parallel. |
| `max_in_flight_llm_requests` | Global cap on concurrent LLM HTTP requests across all workers. |
| `max_articles_per_outlet` | Uniformly sample up to this many filtered articles per outlet/database before optional percentage downsampling. |
| `hedge_window_tokens` | Number of tokens to include on each side of a source mention when probing for hedging. |
| `accept_confidence_min` | Minimum confidence (0–1) below which LLM extractions are discarded. |
| `out_root` | Root directory for cached prompts and resulting SQLite databases. |
| `cache_raw_responses` / `skip_if_cached` | Control JSON cache reuse. |
| `backend` | `vllm` (default) uses Structured Outputs; `ollama` targets a local Ollama server with JSON-only responses. |
| `ollama_api_base` | Optional explicit base URL for Ollama (defaults to `vllm_api_base` or `http://localhost:11434`). |
| `matt_features_only` | When `true`, skip every prompt/table outside Matt's minimal feature set (articles, versions, source/source agg, entity mentions, version metrics, version pairs). Equivalent to passing `--matt-features-only` on the CLI. |

### Local testing with Ollama (macOS-friendly)

Install [Ollama](https://ollama.com/) and pull a structured-capable model (e.g. `ollama pull llama3`). Then set in `config.yaml`:

```yaml
backend: ollama
model: llama3
ollama_api_base: http://localhost:11434
min_versions: 2         # only process articles with at least two versions
max_versions: 20        # skip articles with 20 or more versions (set to null to disable)
cache_raw_responses: false
cleanup_cached_dirs: true
accept_confidence_min: 3   # keep responses with confidence >= 3 (1–5 scale)
```

The pipeline will route requests through `/api/chat` and still enforce schema-conformant JSON, making it easy to iterate locally on a Mac without spinning up vLLM.

## Running the Pipeline

1. Decompress the gzipped SQLite files (or pass `.db.gz` directly; the loader handles temporary extraction).
2. Execute:
   ```bash
   python pipeline.py \
     --config config.yaml \
     --db article-versions/ap.db \
     --db article-versions/newssniffer-bbc.db \
     --db article-versions/newssniffer-guardian.db \
     --db article-versions/newssniffer-independent.db \
     --db article-versions/newssniffer-nytimes.db \
     --db article-versions/newssniffer-washpo.db \
     --db article-versions/reuters.db
   ```
3. Results land in `out/<db-stem>/analysis.db`. Raw prompt responses are cached alongside per-article folders (e.g. `out/ap/1234/v000/A1_source_mentions.json`).

Shortcut wrappers are available one directory up from this README:

```bash
../run.sh             # Uses config.yaml with vLLM backend
../run_local.sh       # Creates a temp config targeting Ollama (respect OLLAMA_MODEL/OLLAMA_API_BASE)
```

### Matt feature-only mode

Matt's downstream notebook (`notebooks/ap_analysis_matt_features.ipynb`) only touches a subset of tables. Enable the streamlined mode to avoid unnecessary prompts and SQLite inserts:

```bash
python pipeline.py ... --matt-features-only
# or set matt_features_only: true inside config.yaml
```

This flag keeps the following tables up to date: `articles`, `versions`, `source_mentions`, `entity_mentions`, `sources_agg`, `version_metrics`, and `version_pairs`. Every other table (e.g., `pair_sources_added`, `pair_claims`, `article_metrics`) remains empty so the run finishes faster and cheaper. The cache directory structure is identical, so you can switch back to the full run later without re-querying previously cached prompts.

In current matt mode, the pipeline prioritizes speed: it skips live-blog detection and CPU-heavy enrichment (spaCy segmentation/NER, hedge/narrative sub-prompts, diff/jaccard metrics). Some enrichment-heavy columns may be left null/default while core matt tables remain populated.

### Alex feature-only mode

Run only the edit-action extraction prompt (new table: `pair_edit_actions`):

```bash
python pipeline.py ... --alex-features-only
```

Defaults and options:

- Output defaults to `./edit-actions` when `--alex-features-only` is set (unless you pass `--out-path`).
- Pair selection includes first-final, consecutive, and optional random extras (`--alex-random-pairs-per-article K`).
- Optional pair list CSV (`--alex-pair-list path.csv`) columns: `news_org,entry_id,from_version_num,to_version_num`.
- If `--alex-pair-list` is provided, only listed article keys are processed and pair-list becomes authoritative (bypasses date/downsample/version-cap filtering). Rows with missing `from_version_num`/`to_version_num` trigger generated pairs for that article key.
- `--alex-features-only` and `--matt-features-only` are mutually exclusive.
- Alex mode intentionally skips CPU-heavy spaCy/diff analytics and runs only pair selection + `ALEX_edit_actions_pair`.

## Outputs

`schema_out.sql` defines all tables. Highlights:

- `articles`, `versions`: metadata for each article/version (titles, timestamps, character counts, live-blog flag).
- `source_mentions`: granular attribution spans with prominence metrics.
- `sources_agg`: per-article canonicalised source lifecycle statistics (voice retention, lead/title counts, disappearance flags).
- `entity_mentions`: spaCy entity outputs with offsets.
- `version_pairs` + detail tables (`pair_numeric_changes`, `pair_claims`, etc.): pairwise features for consecutive versions.
- `pair_edit_actions`: editor-request / writer-action extraction for selected version pairs (alex mode).
- `version_metrics`: LLM summarised sourcing metrics per version.
- `article_metrics`: deltas between the first and final versions.

## Prompt Library

All prompt templates live in `prompts/`. Each prompt now pairs a `.prompt` template with a `.output.json` schema that `pipeline.py` loads at runtime, keeping the structured output definitions alongside the instructions. You can edit these files to adjust model behaviour without touching application code.

## Development Notes

- The pipeline caches every LLM call (parsed JSON). Delete individual cache files to force re-generation.
- For large corpora, set `cache_raw_responses: false` to avoid writing per-version JSON, or enable `cleanup_cached_dirs: true` to drop cached prompts once written to the database.
- Filtering knobs: `min_versions` and `max_versions` (null to disable) let you restrict which articles from each database are processed.
- Confidence scores now use an integer 1–5 scale (1 = very low, 5 = very high); set `accept_confidence_min` accordingly (defaults to 3).
- Confidence filtering happens post-hoc; you can tweak `accept_confidence_min` to be more or less conservative.
- Non-LLM analytics (segmentation, NER, token diffs, diversity metrics) live in `analysis.py` for standalone reuse or unit testing.
- `pipeline.py` is idempotent: it reuses cached responses, recreates tables if needed, and can be rerun safely on subsets of the data.

## Caching & replay tips

- Prompt caches live under `out_root/<db>/<entry_id>/v###/*.json`. Leave `cache_raw_responses: true` and `skip_if_cached: true` (defaults) to automatically replay prior outputs on reruns.
- Share the same `out_root` between matt-only and full runs so overlapping prompts are reused. The selective mode simply skips issuing extra prompts; the shared cache guarantees you never repay for work already done.
- Only enable `cleanup_cached_dirs: true` if you explicitly want to delete per-article caches after a successful write. For most workflows (especially when iterating between modes) keep it `false` so caches persist.
- To rerun a single article or DB from scratch, delete its folder under `out/<db>/<entry_id>` (or the entire `out/<db>` directory) before invoking the pipeline again.
- If you want isolated experiments, point `--out-path` (or `out_root` in config) to a fresh directory; otherwise, stick to the default so caches accumulate over time.

## Next Steps

- Integrate automated evaluation/mock responses for offline testing.
- Add CLI flags to restrict processing to specific article IDs or to resume from midway checkpoints.
