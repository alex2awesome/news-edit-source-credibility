# News Edits Pipeline Workspace

This workspace manages the full experiment stack for analysing how news stories evolve across edits. It combines deterministic Python analytics with structured large-language-model (LLM) calls to trace sourcing, framing, numerical corrections, title movements, and other editorial decisions. The root layout is:
- `news-edits-pipeline/`: all pipeline code, prompts, and schema definitions.
- `article-versions/`: gzipped SQLite snapshots of article version histories (`entryversion` table).
- `out/`: run artefacts (cached prompts, intermediate JSON, final SQLite outputs).
- `run.sh` / `run_local.sh`: convenience wrappers that call `news-edits-pipeline/pipeline.py` with the correct configuration for vLLM or Ollama backends.

## Key Features
- **LLM-backed extraction library**: Prompt families (`A*`, `B*`, `C*`, `D*`, `N*`, `P*`) capture source mentions, hedging, edit types, narrative roles, perspectives, framing cues, and claim/numeric changes. Each prompt has a paired JSON schema so `StructuredLLMClient` can enforce schema-conformant replies while keeping anonymous sourcing and evidence typing inside the core `A1_source_mentions` response.
- **Deterministic analytics**: Functions in `analysis.py` deliver segmentation, sentence alignment, token-level diffs, Jaccard title-to-lede overlap, spaCy NER, and aggregate voice-retention calculations that do not require model calls.
- **Source canonicalisation**: `canonicalize.py` and helper methods in `pipeline.py` normalise names, deduplicate sources via fuzzy matching, and assign persistent IDs (`s001`, `s002`, …) per article so lifecycle metrics stay consistent.
- **Pairwise change detectors**: Consecutive versions are compared with prompts such as `A3_edit_type_pair`, `P3_anon_named_replacement_pair`, `P7_numeric_changes_pair`, `P8_claims_pair`, `P10_movement_pair`, `P16_stance_entities_pair`, and the expanded `D5_angle_change_pair`, surfacing where edits add/remove sources, revise facts, or shift narrative framing.
- **Configurable runtime and caching**: `config.yaml` controls model endpoints (vLLM or Ollama), decoding parameters, filtering thresholds, hedging context windows, cache behaviour, similarity skips, and the LLM retry policy (`llm_retries`, `llm_retry_backoff_seconds`). Cached prompt responses live under `out/<db>/<article>/` and are reused when `skip_if_cached` is true.
- **SQLite writer with schema bootstrap**: `OutputWriter` initialises `out/<db>/analysis.db` using `schema_out.sql`, then streams inserts via batched helpers in `pipeline_writer_utils.py` so reruns are idempotent.
- **Asynchronous LLM orchestration**: `StructuredLLMClient` batches requests with `run_many_async`, enforces max token budgets with `tiktoken`, retries failures, and stores raw JSON alongside each prompt for auditing.
- **Flexible article filtering**: `config.yaml` exposes `min_versions` / `max_versions` selectors, enabling focused experiments (e.g., only articles with ≥2 snapshots) without editing code.

## Data Processing Pipeline
1. **Ingestion** – `loader.py` iterates each `.db` or `.db.gz` in `article-versions/`, decompressing gzipped files into temporary SQLite copies. `iter_article_counts` returns `(entry_id, version_count)` pairs so the pipeline can pre-filter by version thresholds, while `load_versions` returns ordered version metadata and article text for a single article.
2. **Segmentation & entity pass** – For each version, `analysis.segment` tokenises the body text, returning paragraph, sentence, and token spans with character offsets. `analysis.ner_entities_spacy` uses the configured spaCy model to tag entities, which are canonicalised and written into `entity_mentions`.
3. **Source extraction & localisation** – Prompt `A1_source_mentions` now returns canonical names, speech style, attribution verb, plus anonymity flags (`is_anonymous`, domain categories, verbatim phrasing) and evidence metadata (`evidence_type`, `evidence_text`). The pipeline re-anchors each mention into the article text (character, sentence, paragraph indices), flags whether it appears in the title or lede, attaches prominence metrics (`compute_prominence_features`), and assigns a stable `source_id_within_article` via fuzzy matching.
4. **Hedging & stance analysis** – For each mention, prompt `A2_hedge_window` examines a token window (size controlled by `hedge_window_tokens`) to count hedges, capture epistemic verbs, and judge whether the surrounding prose treats the source skeptically. Hedge counts feed per-version metrics and `doubted_any` flags in `sources_agg`.
5. **Per-version classifiers** – `D4_live_blog_detect` runs first; if a version is flagged as a live blog, the pipeline records that status and skips further prompts for the article. Otherwise `D3_corrections` (only when “correction” appears in the text), `B1_version_summary_sources`, `C1_protest_frame_cues`, and the combined narrative-role prompt (`N1_narrative_keywords`) enrich each version with framing summaries, protest cues, and source-level perspective/centrality labels. Numerical metrics (distinct sources, institutional share, anonymous share, hedge density per 1k tokens) derive directly from `A1` plus the hedging analysis.
6. **Pairwise comparisons** – Every consecutive version pair runs change prompts: `A3_edit_type_pair` (edit classification and added/removed sources), `P3_anon_named_replacement_pair` (anonymous ↔ named swaps), `P4_verb_strength_pair` & `P5_speech_style_pair` (qualitative shifts in attribution tone), `P7_numeric_changes_pair` (fact deltas), `P8_claims_pair` (status updates on propositions), `P9_frame_cues_pair` (framing elements), `P10_movement_pair` (descriptive upweight/downweight narratives), `P16_stance_entities_pair` (entity stance shifts), and the expanded `D5_angle_change_pair` (angle/tone changes, title-lede alignment notes, and source transition rationales). Classical diffs (`compute_diff_magnitude`, `align_sentences`, `inter_update_timing`, `jaccard_title_body`) supply context such as token churn and timing gaps, and the code skips the whole analysis when the first and final versions show negligible textual change (Jaccard ≥ 0.95).
7. **Aggregation & article-level metrics** – `aggregate_sources_over_versions` rolls per-version mentions into lifecycle summaries (voice retention, prominence averages, disappearance/de-emphasis flags). Prompt `B2_first_final_framing_compare` summarises framing changes between the first and final versions, and `analysis.final_version_bias` converts per-version metrics into deltas stored in `article_metrics`.
8. **Persistence & cleanup** – `pipeline_writer_utils` writes each table in dependency order, commits, and optionally prunes cached prompt folders when `cleanup_cached_dirs` is enabled. The result is an `analysis.db` per source database inside `out/`, ready for downstream notebooks.

## Output SQLite Schema

### `articles`
| Column | Type | Description |
| --- | --- | --- |
| `article_id` | INTEGER | Source `entry_id` from the input database. |
| `news_org` | TEXT | Normalised source key (e.g., `ap`, `guardian`). |
| `url` | TEXT | Canonical article URL captured in the version feed. |
| `title_first` | TEXT | Headline from the first available version. |
| `title_final` | TEXT | Headline from the most recent version analysed. |
| `original_publication_time` | TEXT | ISO-8601 timestamp of the first version (`created` field). |
| `total_edits` | INTEGER | Count of transitions between versions (`len(versions) - 1`). |
| `is_live_blog` | INTEGER | 1 if any version triggered `D4_live_blog_detect`, else 0. |

### `versions`
| Column | Type | Description |
| --- | --- | --- |
| `version_id` | TEXT | Stable version key (`entryversion.id`). |
| `article_id` | INTEGER | Foreign key to `articles.article_id`. |
| `news_org` | TEXT | Matches `articles.news_org`. |
| `version_num` | INTEGER | Normalised, zero-based revision index. |
| `timestamp_utc` | TEXT | ISO-8601 capture time (`created`). |
| `title` | TEXT | Version headline. |
| `char_len` | INTEGER | Character length of the body text determined by `analysis.segment`. |

### `source_mentions`
| Column | Type | Description |
| --- | --- | --- |
| `version_id` | TEXT | Version containing the mention. |
| `article_id` | INTEGER | Article owner of the mention. |
| `news_org` | TEXT | News organisation key. |
| `source_id_within_article` | TEXT | Stable per-article source handle (`s001`, `s002`, …). |
| `source_canonical` | TEXT | Canonicalised display name (from `A1`). |
| `source_type` | TEXT | Source taxonomy label (`government`, `civil_society`, etc.). |
| `speech_style` | TEXT | Whether the mention is `direct`, `indirect`, or `mixed`. |
| `attribution_verb` | TEXT | Dominant verb describing the attribution (e.g., `said`, `denied`). |
| `char_start` | INTEGER | Character offset for the mention start within the article body. |
| `char_end` | INTEGER | Character offset for the mention end. |
| `sentence_index` | INTEGER | Zero-based index of the sentence containing the mention (`-1` when unresolved). |
| `paragraph_index` | INTEGER | Zero-based paragraph index (`-1` when unresolved). |
| `is_in_title` | INTEGER | 1 if the canonical surface appears in the title. |
| `is_in_lede` | INTEGER | 1 if the mention resides in the lede paragraph. |
| `attributed_text` | TEXT | Extracted quote/paraphrase tied to the source. |
| `is_anonymous` | INTEGER | 1 if the attribution intentionally withholds the speaker’s identity. |
| `anonymous_description` | TEXT | Surface phrasing that describes the anonymous source. |
| `anonymous_domain` | TEXT | Domain classification for the anonymous source (`government`, `corporate`, `law_enforcement`, `individual`, `unknown`). |
| `evidence_type` | TEXT | Evidence category supplied by the source (e.g., `statistic`, `document`). |
| `evidence_text` | TEXT | Exact span demonstrating the evidence cited. |
| `prominence_lead_pct` | REAL | Fractional position of the mention within the article (0.0 = start). |
| `confidence` | REAL | Structured LLM confidence (1–5) for the extraction. |

### `sources_agg`
| Column | Type | Description |
| --- | --- | --- |
| `article_id` | INTEGER | Article identifier. |
| `news_org` | TEXT | News organisation key. |
| `source_id_within_article` | TEXT | Stable per-article source ID. |
| `source_canonical` | TEXT | Canonical label chosen after aggregation. |
| `source_type` | TEXT | Dominant source category across versions. |
| `first_seen_version` | INTEGER | Version number where the source first appeared. |
| `first_seen_time` | TEXT | Timestamp of the first appearance. |
| `last_seen_version` | INTEGER | Final version index containing the source. |
| `last_seen_time` | TEXT | Timestamp of the last appearance. |
| `num_mentions_total` | INTEGER | Total mentions across all versions. |
| `num_versions_present` | INTEGER | Number of versions in which the source appears. |
| `total_attributed_words` | INTEGER | Aggregate word count of attributed text snippets. |
| `voice_retention_index` | REAL | Continuity score (0–1) measuring sustained presence across consecutive versions. |
| `mean_prominence` | REAL | Average `prominence_lead_pct` for the source’s mentions. |
| `lead_appearance_count` | INTEGER | Number of versions where the source appears in the lede. |
| `title_appearance_count` | INTEGER | Number of versions where the source appears in the title. |
| `doubted_any` | INTEGER | 1 if any mention’s hedge analysis flagged skeptical stance. |
| `deemphasized_any` | INTEGER | 1 if the source moved from lede/title into later sections over time. |
| `disappeared_any` | INTEGER | 1 if the source vanished before the final version. |

### `entity_mentions`
| Column | Type | Description |
| --- | --- | --- |
| `version_id` | TEXT | Version where the entity was detected. |
| `article_id` | INTEGER | Article identifier. |
| `news_org` | TEXT | News organisation key. |
| `entity_id_within_article` | TEXT | Stable per-article entity ID (`e001`, `e002`, …). |
| `entity_type` | TEXT | spaCy entity label (e.g., `PERSON`, `ORG`). |
| `canonical_name` | TEXT | Normalised entity surface (using `canonicalize.normalize_source`). |
| `char_start` | INTEGER | Character offset for the entity start. |
| `char_end` | INTEGER | Character offset for the entity end. |
| `sentence_index` | INTEGER | Sentence position or `-1` if unresolved. |
| `paragraph_index` | INTEGER | Paragraph position or `-1` if unresolved. |

### `version_pairs`
| Column | Type | Description |
| --- | --- | --- |
| `article_id` | INTEGER | Article identifier. |
| `news_org` | TEXT | News organisation key. |
| `from_version_id` | TEXT | Earlier version in the pair. |
| `to_version_id` | TEXT | Later version in the pair. |
| `from_version_num` | INTEGER | Numeric index of the earlier version. |
| `to_version_num` | INTEGER | Numeric index of the later version. |
| `delta_minutes` | REAL | Minutes elapsed between captures. |
| `tokens_added` | INTEGER | Count of tokens newly introduced (`compute_diff_magnitude`). |
| `tokens_deleted` | INTEGER | Count of tokens removed. |
| `percent_text_new` | REAL | Share of later-version tokens that are new (0–1). |
| `movement_upweighted_summary` | TEXT | Free-text summary of material the editors promote. |
| `movement_downweighted_summary` | TEXT | Free-text summary of material the editors diminish. |
| `movement_notes` | TEXT | Consolidated bullet-style notes about upweighted/downweighted snippets. |
| `edit_type` | TEXT | Primary edit classification from `A3_edit_type_pair`. |
| `angle_changed` | INTEGER | 1 if `D5_angle_change_pair` flagged a narrative angle shift. |
| `angle_change_category` | TEXT | Qualitative category describing the angle shift. |
| `angle_summary` | TEXT | Free-text summary of how the dominant narrative evolves. |
| `title_alignment_notes` | TEXT | Narrative description of how title/lede alignment changed. |
| `title_jaccard_prev` | REAL | Jaccard overlap between previous title and lede. |
| `title_jaccard_curr` | REAL | Jaccard overlap between current title and lede. |
| `summary_jaccard` | REAL | Jaccard overlap between earlier and later version summaries. |

### `pair_sources_added`
| Column | Type | Description |
| --- | --- | --- |
| `from_version_id` | TEXT | Earlier version ID. |
| `to_version_id` | TEXT | Later version ID. |
| `canonical` | TEXT | Canonical name of the newly added source. |
| `type` | TEXT | Source type label provided by `A3`. |

### `pair_sources_removed`
| Column | Type | Description |
| --- | --- | --- |
| `from_version_id` | TEXT | Earlier version ID. |
| `to_version_id` | TEXT | Later version ID. |
| `canonical` | TEXT | Canonical name of the removed source. |
| `type` | TEXT | Source type label provided by `A3`. |

### `pair_source_transitions`
| Column | Type | Description |
| --- | --- | --- |
| `from_version_id` | TEXT | Earlier version ID. |
| `to_version_id` | TEXT | Later version ID. |
| `canonical` | TEXT | Canonical name of the source whose prominence changed. |
| `transition_type` | TEXT | How the source moved (`added`, `removed`, `promoted`, `demoted`). |
| `reason_category` | TEXT | Editor-inferred rationale (`new_actor`, `escalation`, `context_clarification`, `accountability`, `audience_need`, `other`). |
| `reason_detail` | TEXT | Free-text explanation describing the transition. |

### `pair_anon_named_replacements`
| Column | Type | Description |
| --- | --- | --- |
| `from_version_id` | TEXT | Earlier version ID. |
| `to_version_id` | TEXT | Later version ID. |
| `src` | TEXT | Original reference (anonymous or named). |
| `dst` | TEXT | Replacement reference in the later version. |
| `direction` | TEXT | Change direction (`anon_to_named`, `named_to_anon`). |
| `likelihood` | REAL | Confidence (0.0–1.0) from `P3`. |

### `pair_numeric_changes`
| Column | Type | Description |
| --- | --- | --- |
| `from_version_id` | TEXT | Earlier version ID. |
| `to_version_id` | TEXT | Later version ID. |
| `item` | TEXT | Quantity or fact being tracked. |
| `prev` | TEXT | Value in the earlier version. |
| `curr` | TEXT | Value in the later version. |
| `delta` | TEXT | Human-readable difference. |
| `unit` | TEXT | Unit of measure (e.g., `people`, `%`). |
| `source` | TEXT | Source responsible for the number (if known). |
| `change_type` | TEXT | Categorisation (`update`, `correction`, `refinement`). |
| `confidence` | REAL | Confidence score (1–5). |

### `pair_claims`
| Column | Type | Description |
| --- | --- | --- |
| `from_version_id` | TEXT | Earlier version ID. |
| `to_version_id` | TEXT | Later version ID. |
| `claim_id` | TEXT | Pair-specific identifier for the proposition. |
| `proposition` | TEXT | Claim text summarised by the LLM. |
| `status` | TEXT | Status label (`stable`, `elaborated`, `contradicted`, `retracted`). |
| `change_note` | TEXT | Free-form explanation describing what changed. |
| `confidence` | REAL | Confidence score (1–5). |

### `pair_frame_cues`
| Column | Type | Description |
| --- | --- | --- |
| `from_version_id` | TEXT | Earlier version ID. |
| `to_version_id` | TEXT | Later version ID. |
| `cue` | TEXT | Framing cue (e.g., protest frame). |
| `prev` | INTEGER | 1 if the cue appeared in the earlier version. |
| `curr` | INTEGER | 1 if the cue appears in the later version. |
| `direction` | TEXT | Transition label (`unchanged`, `appeared`, `disappeared`). |

### `version_metrics`
| Column | Type | Description |
| --- | --- | --- |
| `version_id` | TEXT | Version identifier. |
| `article_id` | INTEGER | Article identifier. |
| `news_org` | TEXT | News organisation key. |
| `distinct_sources` | INTEGER | Count of unique `source_id_within_article` present. |
| `institutional_share_words` | REAL | Share of attributed words tied to institutional source types. |
| `anonymous_source_share_words` | REAL | Share of attributed words coming from anonymous mentions. |
| `hedge_density_per_1k` | REAL | Hedge markers per 1,000 tokens in the version. |

### `article_metrics`
| Column | Type | Description |
| --- | --- | --- |
| `article_id` | INTEGER | Article identifier. |
| `news_org` | TEXT | News organisation key. |
| `overstate_institutional_share` | REAL | Net change in institutional share between first and final versions. |
| `distinct_sources_delta` | INTEGER | Difference in distinct source counts (final − first). |
| `anonymity_rate_delta` | REAL | Change in anonymous share of attributed words. |
| `hedge_density_delta` | REAL | Change in hedge density per 1,000 tokens. |

With these components, the directory provides a reproducible environment for studying how newsroom edits reshape sourcing, framing, and factual claims over time. Run `python news-edits-pipeline/pipeline.py --config news-edits-pipeline/config.yaml --db article-versions/<source>.db` to regenerate the outputs or adapt the prompts/configuration for new corpora.
