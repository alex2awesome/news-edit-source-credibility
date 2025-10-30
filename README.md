# News-Edits Pipeline

Analyse how newsroom copy shifts across revisions, with a focus on **who is quoted, how they are framed, and what factual claims change**. The project produces a structured SQLite database (`out/<source>/analysis.db`) that captures every extracted source mention, narrative role, numeric correction, framing cue, and article-level delta so downstream analysts can quantify edit decisions without re-running costly LLM calls.

---

## What You Get

- **`analysis.db`** – relational tables describing articles, per-version metrics, per-source aggregates, and pairwise change artefacts enriched with LLM outputs.
- **Prompt caches** – every LLM response is stored as prettified JSON under `out/<source>/<article>/…` so you can audit or reuse model outputs without recomputation.
- **Exploration notebook** – `notebooks/ap_analysis_overview.ipynb` tours each table, linking columns back to their prompts and running starter SQL summaries.

---

## Data Pipeline (Sketch)

```
article_versions.db
      │  load_versions
      ▼
┌─────────────────────┐
│ Text segmentation   │  spaCy sentences/paragraphs, entity recogniser
│ (`analysis.segment`)│───────────────────────────────┐
└─────────────────────┘                               │
      │                                                │
      ▼                                                │
┌─────────────────────┐          ┌─────────────────────────────────────────────┐
│ Source extraction   │◄────────►│ Prompt A1_source_mentions (quotes + evidence│
│ Re-anchoring        │          └─────────────────────────────────────────────┘
└─────────────────────┘                  │
      │                                  ▼
      │                        ┌─────────────────────────────────────────────┐
      │                        │ Prompt A2_hedge_window (stance, hedges)     │
      │                        └─────────────────────────────────────────────┘
      │                                  │
      │                                  ▼
      │                        ┌─────────────────────────────────────────────┐
      │                        │ Prompt N1_narrative_keywords (roles, view) │
      │                        └─────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────┐
│ Per-version metrics │  distinct sources, anonymity rate, hedge density
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ Pairwise prompts    │  A3 (sources added/removed), D5 (angle), P3/P7/P8/P9/P10
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ SQLite writer       │→ analysis.db (tables described below)
└─────────────────────┘
```

All prompts live in [`news-edits-pipeline/prompts/`](news-edits-pipeline/prompts/) and are referenced by ID (e.g. `A1`, `P7`). The pipeline caches every response, so reruns can populate the database directly from disk.

---

## Output SQL Schema (with prompt references)

> Tip: every prompt link below points to the JSON schema that constrains the LLM response.

### `articles`

Baseline metadata for each story. Populated entirely from the upstream CMS plus live-blog detection via [`D4_live_blog_detect`](https://github.com/alex2awesome/news-edit-source-credibility/blob/main/news-edits-pipeline/prompts/D4_live_blog_detect.output.json).

| Column | Description |
| --- | --- |
| `article_id`, `news_org` | Primary key for the story (used across all tables). |
| `url` | Canonical ingestion URL. |
| `title_first`, `title_final` | First and final headlines observed. |
| `original_publication_time` | Timestamp of first version captured. |
| `total_edits` | Count of subsequent revisions (excluding the seed). |
| `is_live_blog` | Flag raised when D4 judges the story to be a rolling live blog. |

### `versions`

Direct dump of each revision with the headline and summary length; no prompt involvement.

| Column | Description |
| --- | --- |
| `version_id` | Unique ID for the revision (foreign key used everywhere). |
| `article_id`, `news_org` | Story reference. |
| `version_num` | Sequential order of the revision. |
| `timestamp_utc` | Ingestion timestamp (UTC). |
| `title` | Headline text for the revision. |
| `char_len` | Character count of the processed summary/body. |

### `entity_mentions`

spaCy `ner_entities_spacy` output; handy for comparing quoted sources with generic entity coverage.

| Column | Description |
| --- | --- |
| `version_id`, `article_id`, `news_org` | Revision context. |
| `entity_id_within_article` | Canonical ID per entity per article. |
| `entity_type` | spaCy label (PERSON, ORG, etc.). |
| `canonical_name` | Normalised surface form. |
| `char_start`, `char_end`, `sentence_index`, `paragraph_index` | Offsets for downstream highlighting. |

### `source_mentions`

One row per quote/attribution extracted by [`A1_source_mentions`](https://github.com/alex2awesome/news-edit-source-credibility/blob/main/news-edits-pipeline/prompts/A1_source_mentions.output.json#L1-L123), enriched with hedging cues from [`A2_hedge_window`](https://github.com/alex2awesome/news-edit-source-credibility/blob/main/news-edits-pipeline/prompts/A2_hedge_window.output.json#L1-L82) and narrative roles from [`N1_narrative_keywords`](https://github.com/alex2awesome/news-edit-source-credibility/blob/main/news-edits-pipeline/prompts/N1_narrative_keywords.output.json#L1-L66).

| Column | Description |
| --- | --- |
| `version_id`, `article_id`, `news_org` | Revision context. |
| `source_id_within_article` | Stable ID assigned to the canonical source. |
| `source_canonical`, `source_surface`, `source_type` | Normalised name, surface string, and sector label. |
| `speech_style`, `attribution_verb` | How the quote is presented and the verb used. |
| `char_start`, `char_end`, `sentence_index`, `paragraph_index` | Span placement for the quote. |
| `is_in_title`, `is_in_lede` | Prominence flags derived from segmentation. |
| `attributed_text` | Verbatim snippet attributed to the speaker. |
| `is_anonymous`, `anonymous_description`, `anonymous_domain` | Anonymity metadata emitted by A1. |
| `evidence_type`, `evidence_text` | What kind of supporting evidence the source provides. |
| `narrative_function`, `centrality`, `perspective` | Role and stance annotations from N1 (perspective is JSON-encoded list). |
| `doubted`, `hedge_count`, `hedge_markers`, `epistemic_verbs`, `hedge_stance`, `hedge_confidence` | Skepticism cues from A2 (markers/verbs stored as JSON strings). |
| `prominence_lead_pct` | Normalised position of the mention within the summary. |
| `confidence` | Extraction confidence (1–5) from A1. |

### `version_metrics`

Numerical roll-ups for each revision, computed from the `source_mentions` rows plus deterministic analytics.

| Column | Description |
| --- | --- |
| `version_id`, `article_id`, `news_org` | Revision context. |
| `distinct_sources` | Count of unique canonical sources. |
| `institutional_share_words` | Share of attributed words assigned to institutional sectors. |
| `anonymous_source_share_words` | Share of attributed words spoken by anonymous sources. |
| `hedge_density_per_1k` | Hedges per 1,000 tokens, derived from A2 outputs. |

### `sources_agg`

Per-article voice persistence metrics, combining mention-level data and deterministic tracking.

| Column | Description |
| --- | --- |
| `article_id`, `news_org`, `source_id_within_article` | Aggregation keys. |
| `source_canonical`, `source_type` | Canonical speaker info. |
| `first_seen_version`, `first_seen_time`, `last_seen_version`, `last_seen_time` | Lifecycle metadata. |
| `num_mentions_total`, `num_versions_present`, `total_attributed_words` | Engagement counts. |
| `voice_retention_index`, `mean_prominence` | Persistence and average prominence metrics. |
| `lead_appearance_count`, `title_appearance_count` | Prominence counters. |
| `doubted_any`, `deemphasized_any`, `disappeared_any` | Behavioural flags derived from hedge and presence data. |

### `version_pairs`

Summaries of how the first and final versions differ, fed by multiple pairwise prompts: [`P10_movement_pair`](https://github.com/alex2awesome/news-edit-source-credibility/blob/main/news-edits-pipeline/prompts/P10_movement_pair.output.json#L1-L60), [`A3_edit_type_pair`](https://github.com/alex2awesome/news-edit-source-credibility/blob/main/news-edits-pipeline/prompts/A3_edit_type_pair.output.json#L1-L84), and [`D5_angle_change_pair`](https://github.com/alex2awesome/news-edit-source-credibility/blob/main/news-edits-pipeline/prompts/D5_angle_change_pair.output.json#L1-L90).

| Column | Description |
| --- | --- |
| `article_id`, `news_org` | Story reference. |
| `from_version_id`, `to_version_id`, `from_version_num`, `to_version_num` | Versions being compared. |
| `delta_minutes`, `tokens_added`, `tokens_deleted`, `percent_text_new` | Quantitative text deltas. |
| `movement_upweighted_summary`, `movement_downweighted_summary`, `movement_notes`, `movement_confidence`, `movement_notable_shifts` | Movement prompt outputs (note list stored as JSON). |
| `edit_type`, `edit_summary`, `edit_confidence` | Classification of the edit from A3. |
| `angle_changed`, `angle_change_category`, `angle_summary`, `title_alignment_notes`, `angle_confidence`, `angle_evidence` | Angle diagnostics from D5 (`angle_evidence` is JSON). |
| `title_jaccard_prev`, `title_jaccard_curr`, `summary_jaccard` | Jaccard overlaps for headline/lede and summary text. |

### `pair_sources_added` & `pair_sources_removed`

Outputs from A3 listing voices that entered or left the story between versions.

| Column | Description |
| --- | --- |
| `article_id`, `news_org`, `from_version_id`, `to_version_id` | Pair context. |
| `surface`, `canonical`, `type` | Source surface string, canonical name, and sector label. |

### `pair_source_transitions`

Role changes captured by D5.

| Column | Description |
| --- | --- |
| `article_id`, `news_org`, `from_version_id`, `to_version_id` | Pair context. |
| `canonical`, `transition_type` | Source name and transition label (`added`, `removed`, `promoted`, `demoted`). |
| `reason_category`, `reason_detail` | Why the transition happened (category + free text). |

### `pair_anon_named_replacements`

Transparency shifts detected by [`P3_anon_named_replacement_pair`](https://github.com/alex2awesome/news-edit-source-credibility/blob/main/news-edits-pipeline/prompts/P3_anon_named_replacement_pair.output.json#L1-L42).

| Column | Description |
| --- | --- |
| `article_id`, `news_org`, `from_version_id`, `to_version_id` | Pair context. |
| `src`, `dst` | Before/after representation of the source. |
| `direction` | `anon_to_named` or `named_to_anon`. |
| `likelihood` | Confidence score (0.0–1.0). |

### `pair_numeric_changes`

Numeric updates extracted by [`P7_numeric_changes_pair`](https://github.com/alex2awesome/news-edit-source-credibility/blob/main/news-edits-pipeline/prompts/P7_numeric_changes_pair.output.json#L1-L59).

| Column | Description |
| --- | --- |
| `article_id`, `news_org`, `from_version_id`, `to_version_id` | Pair context. |
| `item`, `prev`, `curr`, `delta`, `unit`, `source` | Description, old/new values, computed delta, units, cited source. |
| `change_type`, `confidence` | Prompt classification and confidence score. |

### `pair_claims`

Narrative claim tracking from [`P8_claims_pair`](https://github.com/alex2awesome/news-edit-source-credibility/blob/main/news-edits-pipeline/prompts/P8_claims_pair.output.json#L1-L48).

| Column | Description |
| --- | --- |
| `article_id`, `news_org`, `from_version_id`, `to_version_id` | Pair context. |
| `claim_id`, `proposition` | Stable ID and text of the claim. |
| `status`, `change_note`, `confidence` | Change classification, explanation, and confidence score. |

### `pair_frame_cues`

Framing cue movements detected by [`P9_frame_cues_pair`](https://github.com/alex2awesome/news-edit-source-credibility/blob/main/news-edits-pipeline/prompts/P9_frame_cues_pair.output.json#L1-L41).

| Column | Description |
| --- | --- |
| `article_id`, `news_org`, `from_version_id`, `to_version_id` | Pair context. |
| `cue` | Name of the framing cue (e.g., blame, victim). |
| `prev`, `curr` | Binary flags for presence in earlier/later versions. |
| `direction` | `appeared`, `disappeared`, or `unchanged`. |

### `article_metrics`

First-to-final deltas derived from version metrics plus the framing comparison prompt [`B2_first_final_framing_compare`](https://github.com/alex2awesome/news-edit-source-credibility/blob/main/news-edits-pipeline/prompts/B2_first_final_framing_compare.output.json#L1-L49).

| Column | Description |
| --- | --- |
| `article_id`, `news_org` | Story reference. |
| `overstate_institutional_share` | Change in institutional voice share from first to final version. |
| `distinct_sources_delta` | Delta in unique source count. |
| `anonymity_rate_delta` | Change in anonymous word share. |
| `hedge_density_delta` | Change in hedge density per 1k tokens. |

---

## Working With the Data

1. **Prepare article snapshots**: place source `.db` or `.db.gz` files under `article-versions/`.
2. **Configure runtime**: edit `news-edits-pipeline/config.yaml` (model endpoint, caching, filters).
3. **Run the pipeline**: `./run.sh` (remote vLLM) or `./run_local.sh` (local inference). Every prompt response is cached and replayable.
4. **Explore**: open `notebooks/ap_analysis_overview.ipynb` after a run to inspect tables, sample rows, and quick aggregates.

Because caches are always consulted first (`skip_if_cached: true`), you can rerun the pipeline to rebuild `analysis.db` without incurring new LLM costs.

---

## Implementation & Engineering Notes

- **Project layout**: all pipeline code lives in `news-edits-pipeline/`. `article_processor.py` orchestrates per-article work; `pipeline.py` handles database iteration and multithreading.
- **Caching & idempotency**: `StructuredLLMClient` writes each prompt response to disk, and `OutputWriter` performs `INSERT OR REPLACE` writes so replays stay deterministic.
- **Parallelism**: articles are processed in a thread pool sized by `batch_size`. Failures are logged per article without aborting the overall run.
- **Schema migration**: `OutputWriter` checks existing `analysis.db` files and rebuilds them automatically when a new column is expected.
- **Dependencies**: see `requirements.txt` for Python packages (spaCy, httpx/requests, tqdm, numpy). Prompt schemas rely on JSON Schema to keep LLM outputs well formed.

For additional engineering details (configuration flags, backend support, etc.), consult `news-edits-pipeline/README.md`.
