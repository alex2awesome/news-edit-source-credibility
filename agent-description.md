Great — here’s a single, end-to-end build plan your agent can implement. It includes: the **analyses to run**, which ones **need LLM prompts**, the **exact prompt files** to drop in `prompts/`, the **non-LLM analyses** to code in `analysis.py`, a **vLLM chaining script**, and how to **run it over all six .db.gz files**.

---

# 0) Data source & schema (all 6 DBs share this)

Each file is a **gzipped SQLite database** with a single table `entryversion`. Decompress before use.

**Files to process**

```
ap.db.gz
newssniffer-bbc.db.gz
newssniffer-guardian.db.gz
newssniffer-independent.db.gz
newssniffer-nytimes.db.gz
newssniffer-washpo.db.gz
reuters.db.gz
```

**Decompress**

```bash
gunzip -k ap.db.gz                       # leaves ap.db
# …repeat for all
```

**SQLite schema (identical across DBs)**

* Table: `entryversion` (one row = one article *version/snapshot*)
* Columns:

  * `index` INTEGER — load-order id (not unique; reused)
  * `version` INTEGER — 0-based version number within an article
  * `title` TEXT — headline at that version
  * `created` TEXT — ISO-8601 UTC when this version was captured
  * `url` TEXT — canonical (live) story URL
  * `source` TEXT — news org key (e.g., `washpo`, `reuters`, etc.)
  * `entry_id` INTEGER — **article id** (stable across its versions)
  * `archive_url` TEXT — permalink to archived version
  * `num_versions` INTEGER — total versions we saw for that `entry_id`
  * `summary` TEXT — **full article body text at that version**  ✅
  * `joint_key` TEXT — `"{entry_id}-{version}"`
  * `id` TEXT — identical to `joint_key`

**SQL access pattern**

```sql
-- enumerate articles
SELECT DISTINCT entry_id FROM entryversion;
-- enumerate versions of an article in order
SELECT id, joint_key, entry_id, version, created, title, url, source, num_versions, summary
FROM entryversion
WHERE entry_id = ?
ORDER BY version ASC;
```

---

# 1) Project layout (create this repo structure)

```
news-edits-pipeline/
├── prompts/                 # ALL LLM prompt templates (see §3)
│   ├── A1_source_mentions.prompt
│   ├── A2_hedge_window.prompt
│   ├── A3_edit_type_pair.prompt
│   ├── B1_version_summary_sources.prompt
│   ├── B2_first_final_framing_compare.prompt
│   ├── C1_protest_frame_cues.prompt
│   ├── D1_anonymous_sources.prompt
│   ├── D2_evidence_types.prompt
│   ├── D3_corrections.prompt
│   ├── D4_live_blog_detect.prompt
│   ├── D5_angle_change_pair.prompt
│   ├── P2_title_events_pair.prompt
│   ├── P3_anon_named_replacement_pair.prompt
│   ├── P4_verb_strength_pair.prompt
│   ├── P5_speech_style_pair.prompt
│   ├── P7_numeric_changes_pair.prompt
│   ├── P8_claims_pair.prompt
│   ├── P9_frame_cues_pair.prompt
│   ├── P10_movement_pair.prompt
│   ├── P16_stance_entities_pair.prompt
│   └── P17_title_body_alignment_pair.prompt
├── analysis.py              # NON-LLM analyses (see §4)
├── canonicalize.py          # Source canon & matching helpers
├── loader.py                # SQLite I/O, segmentation, char offsets
├── pipeline.py         # Orchestrates vLLM calls + analysis (see §5)
├── schema_out.sql           # Output DB schema (see §6)
├── config.yaml              # Paths, model name, batch sizes, thresholds
├── requirements.txt
└── README.md
```

**requirements.txt**

```
vllm>=0.5
openai>=1.40        # if using vLLM’s OpenAI-compatible server
tiktoken            # optional
sqlite-utils        # helper or use stdlib sqlite3
pandas
numpy
scikit-learn
spacy>=3.7
spacy-transformers  # optional
rapidfuzz
orjson
python-Levenshtein
```

Run once:

```bash
python -m spacy download en_core_web_sm
```

---

# 2) The set of analyses we will do

Below is the consolidated list, grouped by **per-version**, **pairwise (between versions)**, and **article-level aggregates**. I’ve marked **[LLM]** where a prompt is required; the rest are **[NON-LLM]** and must be implemented as functions in `analysis.py`.

### 2.1 Per-version (single snapshot)

1. **Source detection & attribution with spans** [LLM; A1]
2. **Hedging/doubt within a window around each source mention** [LLM; A2]
3. **Anonymous source detection** (“officials said…”) [LLM; D1]
4. **Evidence type classification** (press release, eyewitness, doc, stats, prior reporting…) [LLM; D2]
5. **Corrections/clarifications section detection** [LLM; D3]
6. **Live-blog classifier** [LLM; D4]
7. **Title–source extraction & matching** [LLM; part of P17 or separate simple pass]
8. **Version summary for bias metrics** (institutional share, anonymity share, hedge density, top sources) [LLM; B1]
9. **Segmentation & offsets** [NON-LLM]
10. **NER (Entities) + offsets** [NON-LLM; spaCy]
11. **Prominence features** (lead percentile, title/lede presence) [NON-LLM]

### 2.2 Pairwise (v → v+1; *between versions*)

12. **Edit type classification** (copyedit, content, source add/remove, emphasis, correction) [LLM; A3]
13. **Source lifecycle** (added/removed/moved/expanded/contracted) [LLM; covered in A3 outputs]
14. **Promotion/demotion into lede/title** [LLM; P2]
15. **Anonymous ↔ named replacement mapping** [LLM; P3]
16. **Hedging delta near persisting sources** [LLM; P4-ish via A2 diffs — provide P4 prompt]
17. **Attribution verb strength shift** [LLM; P4]
18. **Quote ↔ paraphrase transitions** [LLM; P5]
19. **Numeric fact changes** [LLM; P7]
20. **Claim tracking** (stable/elaborated/contradicted/retracted) [LLM; P8]
21. **Frame cue drift** (e.g., protest paradigm) [LLM; P9]
22. **Paragraph reordering / movement index** [LLM; P10, also [NON-LLM] alignment for features]
23. **Title–body alignment change (Jaccard), title-source delta** [LLM; P17 plus [NON-LLM] Jaccard]
24. **Angle change detector** (who/what/why/blame/impact) [LLM; D5]
25. **Inter-update timing vs edit magnitude** [NON-LLM]

### 2.3 Article-level aggregates (across all versions of an entry_id)

26. **Voice Retention Index (VRI)** by source & type [NON-LLM]
27. **Institutional share over time** (words attributed) [NON-LLM]
28. **Diversity/concentration** (Shannon/Herfindahl/Gini) of attributed words by source over time [NON-LLM]
29. **Final-version bias** (first vs final deltas on institutional share, anonymity, hedge density) [NON-LLM; uses B1 outputs]
30. **Story-level optional** (cluster URLs into larger stories) [NON-LLM; optional; out of scope for MVP]

---

# 3) Prompts to create (drop these exact files in `prompts/`)

All prompts must end with “**Return ONLY valid JSON**”. Your inference code will insert the variables in `{{double_curly}}`.

> **IMPORTANT**: Do not request or log chain-of-thought. Output should be terse JSON objects with **character offsets**, **indices**, and **confidence** fields where relevant.

### 3.1 Per-version prompts

**prompts/A1_source_mentions.prompt**

```
You are an information-extraction model. Return ONLY valid JSON.
Task: Extract all attributed source mentions and their context from the article version text.

Input metadata:
- article_id: {{article_id}}
- version_id: {{version_id}}
- timestamp_utc: {{timestamp_utc}}

Output schema:
{
  "article_id": "{{article_id}}",
  "version_id": "{{version_id}}",
  "timestamp_utc": "{{timestamp_utc}}",
  "source_mentions": [
    {
      "surface": "U.S. Navy",
      "canonical": "United States Navy",
      "type": "government|civil_society|individual|corporate|law_enforcement|unknown",
      "speech_style": "direct|indirect|mixed",
      "attribution_verb": "said|announced|claimed|denied|confirmed|ordered|estimated|unknown",
      "char_start": 123,
      "char_end": 245,
      "sentence_index": 5,
      "paragraph_index": 2,
      "is_in_title": false,
      "is_in_lede": true,
      "attributed_text": "exact quoted or paraphrased span attributed to the source",
      "confidence": 0.0
    }
  ]
}

Article title:
{{title}}

Article version text:
{{version_text}}
```

**prompts/A2_hedge_window.prompt**

```
Return ONLY JSON.
Task: Given a text window around a source mention, detect hedging/doubt.

Schema:
{
  "hedge_markers": ["reportedly","allegedly"],
  "hedge_count": 0,
  "epistemic_verbs": [],
  "stance_toward_source": "supportive|neutral|skeptical|unclear",
  "confidence": 0.0
}

Window:
{{context_window_text}}
```

**prompts/A3_edit_type_pair.prompt**

```
Return ONLY JSON.
Task: Classify the change from prev_text → curr_text.

Schema:
{
  "edit_type": "copyedit|content_update|new_source_added|source_removed|emphasis_change|correction|title_change|other",
  "sources_added": [{"surface":"...", "canonical":"...", "type":"..."}],
  "sources_removed": [{"surface":"...", "canonical":"...", "type":"..."}],
  "angle_changed": true|false,
  "summary_of_change": "1 sentence",
  "confidence": 0.0
}

prev_title: {{prev_title}}
curr_title: {{curr_title}}

prev_text:
{{prev_version_text}}

curr_text:
{{curr_version_text}}
```

**prompts/B1_version_summary_sources.prompt**

```
Return ONLY JSON.
Task: Summarize source composition and anonymity for this version.

Schema:
{
  "version_id": "{{version_id}}",
  "distinct_sources": 0,
  "institutional_share_words": 0.0,
  "anonymous_source_share_words": 0.0,
  "hedge_density_per_1k_tokens": 0.0,
  "top_sources_by_words": [{"canonical":"...", "type":"...", "word_count":0}],
  "confidence": 0.0
}

Text:
{{version_text}}
```

**prompts/B2_first_final_framing_compare.prompt**

```
Return ONLY JSON.
Task: Compare first and final versions for framing changes.

Schema:
{
  "lede_changed": true|false,
  "lede_change_type": "minor_copy|new_facts|reframing|unclear",
  "title_sources_added": ["..."],
  "title_sources_removed": ["..."],
  "angle_changed": true|false,
  "one_sentence_diff": "...",
  "confidence": 0.0
}

first_version_title: {{title_v0}}
first_version_lede: {{lede_v0}}

final_version_title: {{title_vf}}
final_version_lede: {{lede_vf}}
```

**prompts/C1_protest_frame_cues.prompt**

```
Return ONLY JSON.
Task: Identify protest-paradigm frame cues and roles present.

Schema:
{
  "frame_cues": [
    {"cue":"law_and_order_emphasis","present":false,"evidence_snippet":""},
    {"cue":"violence_highlight","present":false,"evidence_snippet":""}
  ],
  "roles": [
    {"role":"protesters","depiction":"orderly|violent|diverse|unclear","evidence_snippet":""},
    {"role":"police","depiction":"restraint|force|neutral|unclear","evidence_snippet":""}
  ],
  "confidence": 0.0
}

Text:
{{version_text}}
```

**prompts/D1_anonymous_sources.prompt**

```
Return ONLY JSON.
Task: Extract anonymous/unnamed source mentions.

Schema:
{
  "anonymous_mentions": [
    {
      "pattern": "officials said|a person familiar with ...",
      "char_start": 0,
      "char_end": 0,
      "inferred_domain": "government|corporate|law_enforcement|unknown",
      "confidence": 0.0
    }
  ]
}

Text:
{{version_text}}
```

**prompts/D2_evidence_types.prompt**

```
Return ONLY JSON.
Task: Classify evidence types used and attribute them to sources.

Schema:
{
  "evidence_items":[
    {
      "source_canonical":"CDC",
      "evidence_type":"official_statement|press_release|eyewitness|document|statistic|prior_reporting|social_media|court_filing|other",
      "evidence_span_char_start": 0,
      "evidence_span_char_end": 0
    }
  ],
  "confidence": 0.0
}

Text:
{{version_text}}
```

**prompts/D3_corrections.prompt**

```
Return ONLY JSON.
Task: Identify corrections/clarifications and whether they change sourcing.

Schema:
{
  "has_correction": false,
  "correction_affects": ["source_identity|claim_content|numbers|typo|none"],
  "correction_span": {"char_start":0, "char_end":0},
  "confidence": 0.0
}

Text:
{{version_text}}
```

**prompts/D4_live_blog_detect.prompt**

```
Return ONLY JSON.
Task: Decide if this article version is part of a live blog.

Schema: {
  "is_live_blog": false,
  "confidence": 0.0,
  "signals": []
}

Text:
{{version_text}}
```

**prompts/D5_angle_change_pair.prompt**

```
Return ONLY JSON.
Task: Did the main angle change between versions?

Schema: {
  "angle_changed": false,
  "old_angle": "",
  "new_angle": "",
  "change_type": "who|what|why|responsibility|impact|blame|unclear",
  "evidence_spans": [{"from_char":0,"to_char":0}],
  "confidence": 0.0
}

prev_text:
{{prev_version_text}}

curr_text:
{{curr_version_text}}
```

### 3.2 Pairwise prompts (P-series)

**prompts/P2_title_events_pair.prompt**

```
Return ONLY JSON.
Task: Detect if any source newly appears in or disappears from title or first paragraph (lede).

Schema: {
  "title_events":[{"canonical":"...", "event":"entered_title|left_title"}],
  "lede_events":[{"canonical":"...", "event":"entered_lede|left_lede"}]
}

prev_title: {{title_prev}}
curr_title: {{title_curr}}
prev_text: {{v_prev}}
curr_text: {{v_curr}}
```

**prompts/P3_anon_named_replacement_pair.prompt**

```
Return ONLY JSON.
Task: Identify likely replacements between anonymous and named sources.

Schema:{
  "replacements":[
    {"from":"Unnamed government official","to":"U.S. Department of Defense","direction":"anon_to_named|named_to_anon","likelihood": 0.0}
  ]
}

prev_text: {{v_prev}}
curr_text: {{v_curr}}
```

**prompts/P4_verb_strength_pair.prompt**

```
Return ONLY JSON.
Task: Compare main attribution verbs per source and rate strength change (-1 weak → +1 strong).

Schema:{
  "by_source":[
    {"canonical":"...", "prev_verbs":["say"], "curr_verbs":["confirm"], "strength_delta": 0.0}
  ]
}

prev_text: {{v_prev}}
curr_text: {{v_curr}}
```

**prompts/P5_speech_style_pair.prompt**

```
Return ONLY JSON.
Task: For each source, detect shift between direct and indirect speech.

Schema:{
  "speech_style_changes":[
    {"canonical":"...", "change":"direct_to_indirect|indirect_to_direct|unchanged", "delta_quoted_words": 0}
  ]
}

prev_text: {{v_prev}}
curr_text: {{v_curr}}
```

**prompts/P7_numeric_changes_pair.prompt**

```
Return ONLY JSON.
Task: Extract numeric facts that changed and attribute them if possible.

Schema:{
  "numeric_changes":[
    {"item":"...", "prev":"...", "curr":"...", "delta":"+2", "unit":"...", "source":"...", "change_type":"update|correction|refinement", "confidence": 0.0}
  ]
}

prev_text: {{v_prev}}
curr_text: {{v_curr}}
```

**prompts/P8_claims_pair.prompt**

```
Return ONLY JSON.
Task: Identify core claims and how they changed.

Schema:{
  "claims":[
    {"id":"C1", "proposition":"...", "status":"stable|elaborated|contradicted|retracted", "change_note":"...", "confidence": 0.0}
  ]
}

prev_text: {{v_prev}}
curr_text: {{v_curr}}
```

**prompts/P9_frame_cues_pair.prompt**

```
Return ONLY JSON.
Task: Compare presence of frame cues between versions.

Schema:{
  "cues":[
    {"cue":"law_and_order_emphasis","prev":false,"curr":false,"direction":"unchanged|appeared|disappeared"}
  ]
}

prev_text: {{v_prev}}
curr_text: {{v_curr}}
```

**prompts/P10_movement_pair.prompt**

```
Return ONLY JSON.
Task: Align paragraphs and estimate movement toward/away from the top.

Schema:{
  "movement_index": 0.0,
  "moved_into_top20pct_tokens": 0.0,
  "examples":[{"snippet":"...", "prev_para":0, "curr_para":0}]
}

prev_text: {{v_prev}}
curr_text: {{v_curr}}
```

**prompts/P16_stance_entities_pair.prompt**

```
Return ONLY JSON.
Task: Compare stance toward named entities.

Schema:{
  "entities":[
    {"canonical":"...", "prev_stance":"supportive|neutral|skeptical|unclear","curr_stance":"...", "delta": -1|0|+1}
  ]
}

prev_text: {{v_prev}}
curr_text: {{v_curr}}
```

**prompts/P17_title_body_alignment_pair.prompt**

```
Return ONLY JSON.
Task: Measure title-body alignment change and title-source change.

Schema:{
  "jaccard_overlap_prev": 0.0,
  "jaccard_overlap_curr": 0.0,
  "title_sources_added":["..."],
  "title_sources_removed":["..."]
}

prev_title: {{title_prev}}
curr_title: {{title_curr}}
prev_text: {{v_prev_first_paragraph}}
curr_text: {{v_curr_first_paragraph}}
```

---

# 4) Non-LLM analyses to implement in `analysis.py`

Implement each as a pure function. Inputs are plain dicts/strings; outputs are dicts/rows that your pipeline will persist.

```python
# analysis.py

def segment(text: str) -> dict:
    """
    Returns:
      {
        "sentences": [{"start": int, "end": int, "text": str, "index": int}],
        "paragraphs": [{"start": int, "end": int, "text": str, "index": int}],
        "tokens": [{"start": int, "end": int, "text": str, "index": int}],
        "char_len": int
      }
    """
    ...

def extract_lede(title: str, text: str, paragraphs: list) -> dict:
    """Return first paragraph text + indices/offsets."""

def compute_prominence_features(char_start: int, char_len: int, in_title: bool, in_lede: bool) -> dict:
    """Return lead_percentile, bool flags, etc."""

def ner_entities_spacy(text: str) -> dict:
    """
    Returns entity mentions with canonicalization (lowercased surface) and char offsets.
    Use spaCy en_core_web_sm by default.
    """

def align_sentences(prev_text: str, curr_text: str) -> dict:
    """
    Sentence-level alignment using cosine of tf-idf or embeddings (simple version: tf-idf).
    Returns mapping + movement stats (avg abs change in rank).
    """

def jaccard_title_body(title: str, body_first_paragraph: str) -> float:
    """Token Jaccard overlap."""

def compute_diff_magnitude(prev_text: str, curr_text: str) -> dict:
    """Percent of new tokens, tokens added/deleted, etc."""

def inter_update_timing(prev_ts: str, curr_ts: str) -> float:
    """Minutes between versions."""

def aggregate_sources_over_versions(source_mentions_by_version: list, char_lens_by_version: list) -> dict:
    """
    Build per-article canonical source table: first/last seen, words, presence by version,
    voice_retention_index, mean prominence, lead/title counts.
    """

def diversity_indices(attributed_words_by_source: dict) -> dict:
    """Shannon, Herfindahl, Gini."""

def final_version_bias(metrics_first: dict, metrics_final: dict) -> dict:
    """Compute deltas and overstatement indexes."""
```

**canonicalize.py** (helpers)

```python
def normalize_source(surface: str) -> str: ...
def fuzzy_match_source(surface: str, known: dict) -> str | None: ...
```

**loader.py**

```python
import sqlite3

def iter_articles(db_path: str):
    """yield entry_id"""

def load_versions(db_path: str, entry_id: int):
    """return ordered list of dicts with fields:
       id, entry_id, version, created, title, url, source, num_versions, summary (full text)
    """
```

---

# 5) vLLM chaining script (`pipeline.py`)

* Reads `config.yaml`:

  ```yaml
  model: "llama-3.2-70b-instruct"         # your local model name
  vllm_api_base: "http://localhost:8000/v1"
  temperature: 0.0
  max_tokens: 2048
  batch_size: 4
  hedge_window_tokens: 80
  accept_confidence_min: 0.6
  out_root: "./out"
  ```
* Starts an OpenAI-compatible client to vLLM (or vLLM Python engine).
* For each **DB file**:

  1. Open SQLite; enumerate `entry_id`.
  2. For each article:

     * Load ordered versions (v0..vN).
     * For each version:

       * Build segmentation/offsets [NON-LLM].
       * Run **A1** (source mentions).
       * For each source span, slice **± hedge_window_tokens** to call **A2**.
       * Run **D1**, **D2**, **D3**, **D4**.
       * Run **B1**.
     * For each consecutive pair (v, v+1):

       * Run **A3**, **P2**, **P3**, **P4**, **P5**, **P7**, **P8**, **P9**, **P10**, **P16**, **P17**, **D5**.
       * Compute [NON-LLM] diff magnitude and inter-update timing.
     * Aggregate per-article [NON-LLM]: VRI, diversity indices, final-version bias.
  3. Persist outputs to an **output SQLite** (see §6) under `out/{db_basename}/analysis.db`.

**Implementation notes**

* Add a tiny wrapper `llm_call(prompt_path, variables) -> dict` that:

  * loads the prompt template text
  * replaces `{{vars}}`
  * calls the vLLM server
  * parses JSON with `orjson`, retries once if invalid, and returns dict
* Cache raw LLM results on disk (JSONL) with paths:

  * `out/{db}/{entry_id}/v{version}/A1_source_mentions.json`
  * …so re-runs are incremental/idempotent.

**Batching**

* You can batch **per-version** prompts across articles in the same request stream to vLLM for throughput, but keep **pairwise** prompts batched per article to reduce context prep overhead.

**Confidence filtering**

* When a response includes `confidence`, drop items below `accept_confidence_min`, but keep the full file for debugging.

---

# 6) Output database (create with `schema_out.sql`)

```sql
-- schema_out.sql

PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS articles (
  article_id INTEGER,
  news_org TEXT,
  url TEXT,
  title_first TEXT,
  title_final TEXT,
  original_publication_time TEXT,
  total_edits INTEGER,
  is_live_blog INTEGER,
  PRIMARY KEY(article_id, news_org)
);

CREATE TABLE IF NOT EXISTS versions (
  version_id TEXT PRIMARY KEY,
  article_id INTEGER,
  news_org TEXT,
  version_num INTEGER,
  timestamp_utc TEXT,
  title TEXT,
  char_len INTEGER
);

CREATE TABLE IF NOT EXISTS source_mentions (
  version_id TEXT,
  article_id INTEGER,
  news_org TEXT,
  source_id_within_article TEXT,
  source_canonical TEXT,
  source_type TEXT,
  speech_style TEXT,
  attribution_verb TEXT,
  char_start INTEGER,
  char_end INTEGER,
  sentence_index INTEGER,
  paragraph_index INTEGER,
  is_in_title INTEGER,
  is_in_lede INTEGER,
  attributed_text TEXT,
  prominence_lead_pct REAL,
  confidence REAL
);

CREATE TABLE IF NOT EXISTS sources_agg (
  article_id INTEGER,
  news_org TEXT,
  source_id_within_article TEXT,
  source_canonical TEXT,
  source_type TEXT,
  first_seen_version INTEGER,
  first_seen_time TEXT,
  last_seen_version INTEGER,
  last_seen_time TEXT,
  num_mentions_total INTEGER,
  num_versions_present INTEGER,
  total_attributed_words INTEGER,
  voice_retention_index REAL,
  mean_prominence REAL,
  lead_appearance_count INTEGER,
  title_appearance_count INTEGER,
  doubted_any INTEGER,
  deemphasized_any INTEGER,
  disappeared_any INTEGER,
  PRIMARY KEY(article_id, news_org, source_id_within_article)
);

CREATE TABLE IF NOT EXISTS entity_mentions (
  version_id TEXT,
  article_id INTEGER,
  news_org TEXT,
  entity_id_within_article TEXT,
  entity_type TEXT,
  canonical_name TEXT,
  char_start INTEGER,
  char_end INTEGER,
  sentence_index INTEGER,
  paragraph_index INTEGER
);

CREATE TABLE IF NOT EXISTS version_pairs (
  article_id INTEGER,
  news_org TEXT,
  from_version_id TEXT,
  to_version_id TEXT,
  from_version_num INTEGER,
  to_version_num INTEGER,
  delta_minutes REAL,
  tokens_added INTEGER,
  tokens_deleted INTEGER,
  percent_text_new REAL,
  movement_index REAL,
  moved_into_top20pct_tokens REAL,
  edit_type TEXT,
  angle_changed INTEGER,
  title_jaccard_prev REAL,
  title_jaccard_curr REAL
);

-- Optional detail tables for pairwise outputs
CREATE TABLE IF NOT EXISTS pair_sources_added (from_version_id TEXT, to_version_id TEXT, canonical TEXT, type TEXT);
CREATE TABLE IF NOT EXISTS pair_sources_removed (from_version_id TEXT, to_version_id TEXT, canonical TEXT, type TEXT);
CREATE TABLE IF NOT EXISTS pair_title_events (from_version_id TEXT, to_version_id TEXT, canonical TEXT, event TEXT);
CREATE TABLE IF NOT EXISTS pair_anon_named_replacements (from_version_id TEXT, to_version_id TEXT, src TEXT, dst TEXT, direction TEXT, likelihood REAL);
CREATE TABLE IF NOT EXISTS pair_numeric_changes (from_version_id TEXT, to_version_id TEXT, item TEXT, prev TEXT, curr TEXT, delta TEXT, unit TEXT, source TEXT, change_type TEXT, confidence REAL);
CREATE TABLE IF NOT EXISTS pair_claims (from_version_id TEXT, to_version_id TEXT, claim_id TEXT, proposition TEXT, status TEXT, change_note TEXT, confidence REAL);
CREATE TABLE IF NOT EXISTS pair_frame_cues (from_version_id TEXT, to_version_id TEXT, cue TEXT, prev INTEGER, curr INTEGER, direction TEXT);
CREATE TABLE IF NOT EXISTS version_metrics (
  version_id TEXT PRIMARY KEY,
  article_id INTEGER,
  news_org TEXT,
  distinct_sources INTEGER,
  institutional_share_words REAL,
  anonymous_source_share_words REAL,
  hedge_density_per_1k REAL
);

CREATE TABLE IF NOT EXISTS article_metrics (
  article_id INTEGER,
  news_org TEXT,
  overstate_institutional_share REAL,
  distinct_sources_delta INTEGER,
  anonymity_rate_delta REAL,
  hedge_density_delta REAL,
  PRIMARY KEY(article_id, news_org)
);
```

---

# 7) Chaining logic (high level)

For each DB:

1. **Open DB** and **out DB** (`analysis.db` under `out/{db}/`).
2. For each `entry_id` (article):

   * Load all versions ordered by `version`.
   * Insert article header (news_org = first row’s `source`).
   * **Per-version loop**:

     * Build segmentation (sentences/paragraphs/tokens/offsets).
     * Run prompts: **A1, D1, D2, D3, D4, B1**.
     * For each A1 mention, compute prominence (lead percentile, title/lede flags).
     * Run **A2** on small windows around each A1 mention.
     * NER with spaCy; store entity mentions.
     * Persist `versions`, `source_mentions`, `version_metrics`.
   * **Pairwise loop** over consecutive versions:

     * Compute NON-LLM diffs (tokens added/deleted, % new; sentence alignment & movement index; delta_minutes; title-body Jaccard).
     * Run **A3, P2, P3, P4, P5, P7, P8, P9, P10, P16, P17, D5**.
     * Persist into `version_pairs` + detail tables.
   * **Aggregate** over all version rows to build `sources_agg` and `article_metrics` (VRI, diversity indices, final-version bias).
3. Repeat for the next DB.

---

# 8) Implementation tips & defaults

* **Segmentation**: use a single spaCy pass to get sentences, tokens, and per-token char offsets. Paragraphs: split on double newline or `<p>` if HTML (most `summary` strings are plain text).
* **Lede**: paragraph index 0.
* **Lead percentile**: `char_start / total_char_len`.
* **Institutional type**: use model output; if missing, rules: ORG names with government lexemes (Department, Ministry, Police, Pentagon, CDC, White House) → `government`; NGOs → `civil_society`; Co. suffixes → `corporate`.
* **Canonicalization**: lowercase, strip punctuation, expand common abbreviations (U.S. → United States), then fuzzy match (`rapidfuzz.ratio > 92`) within the article to assign a stable `source_id_within_article`.
* **Anonymous patterns**: `(officials?|authorities|spokes(wo)?man|source|person familiar with|on condition of anonymity)` as a pre-filter; then D1 prompt.
* **Confidence**: drop items with `confidence < 0.6`, but keep the raw JSON on disk for audit.
* **Live blogs**: if `is_live_blog` true at any version, mark the article; you may exclude from some aggregate stats or stratify.
* **Caching**: write raw LLM responses under `out/{db}/{entry_id}/...` and skip re-inference if the JSON exists.
* **Idempotence**: primary keys in output tables prevent duplicate inserts.

---

# 9) How to run

1. **Start vLLM** (example)

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /models/llama-3.2-70b-instruct \
  --host 0.0.0.0 --port 8000 --dtype auto --tensor-parallel-size 2
```

2. **Configure** `config.yaml` (model, API base, out_root, thresholds).

3. **Process all DBs**

```bash
python pipeline.py \
  --db ap.db \
  --db newssniffer-bbc.db \
  --db newssniffer-guardian.db \
  --db newssniffer-independent.db \
  --db newssniffer-nytimes.db \
  --db newssniffer-washpo.db \
  --db reuters.db \
  --config config.yaml
```

(Or point to the `.db.gz` and have the script auto-decompress to a temp folder.)

---

# 10) What you’ll be able to answer (directly supports the study)

* **RQ1 (first responders vs late influence)**: from per-version institutional share, promotion into lede/title (P2), source lifecycle (A3), and verb strength (P4).
* **RQ2 (final-version bias)**: `article_metrics` deltas (B1 + NON-LLM aggregation), plus B2 framing comparison.
* **RQ3 (sources vs framing)**: correlate `pair_frame_cues`, `source_mentions`/types, `voice_retention_index`, and `hedge densities`.

If you want sample starter code for `pipeline.py` stubs or `analysis.py` functions, say the word and I’ll inline minimal working versions you can paste in.

