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
  source_surface TEXT,
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
  is_anonymous INTEGER,
  anonymous_description TEXT,
  anonymous_domain TEXT,
  evidence_type TEXT,
  evidence_text TEXT,
  narrative_function TEXT,
  centrality TEXT,
  perspective TEXT,
  doubted INTEGER,
  hedge_count INTEGER,
  hedge_markers TEXT,
  epistemic_verbs TEXT,
  hedge_stance TEXT,
  hedge_confidence INTEGER,
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
  movement_upweighted_summary TEXT,
  movement_downweighted_summary TEXT,
  movement_notes TEXT,
  movement_confidence INTEGER,
  movement_notable_shifts TEXT,
  edit_type TEXT,
  edit_summary TEXT,
  edit_confidence INTEGER,
  angle_changed INTEGER,
  angle_change_category TEXT,
  angle_summary TEXT,
  title_alignment_notes TEXT,
  angle_confidence INTEGER,
  angle_evidence TEXT,
  title_jaccard_prev REAL,
  title_jaccard_curr REAL,
  summary_jaccard REAL
);

CREATE TABLE IF NOT EXISTS pair_sources_added (
  article_id INTEGER,
  news_org TEXT,
  from_version_id TEXT,
  to_version_id TEXT,
  surface TEXT,
  canonical TEXT,
  type TEXT
);

CREATE TABLE IF NOT EXISTS pair_sources_removed (
  article_id INTEGER,
  news_org TEXT,
  from_version_id TEXT,
  to_version_id TEXT,
  surface TEXT,
  canonical TEXT,
  type TEXT
);

CREATE TABLE IF NOT EXISTS pair_source_transitions (
  article_id INTEGER,
  news_org TEXT,
  from_version_id TEXT,
  to_version_id TEXT,
  canonical TEXT,
  transition_type TEXT,
  reason_category TEXT,
  reason_detail TEXT
);

CREATE TABLE IF NOT EXISTS pair_anon_named_replacements (
  article_id INTEGER,
  news_org TEXT,
  from_version_id TEXT,
  to_version_id TEXT,
  src TEXT,
  dst TEXT,
  direction TEXT,
  likelihood REAL
);

CREATE TABLE IF NOT EXISTS pair_numeric_changes (
  article_id INTEGER,
  news_org TEXT,
  from_version_id TEXT,
  to_version_id TEXT,
  item TEXT,
  prev TEXT,
  curr TEXT,
  delta TEXT,
  unit TEXT,
  source TEXT,
  change_type TEXT,
  confidence REAL
);

CREATE TABLE IF NOT EXISTS pair_claims (
  article_id INTEGER,
  news_org TEXT,
  from_version_id TEXT,
  to_version_id TEXT,
  claim_id TEXT,
  proposition TEXT,
  status TEXT,
  change_note TEXT,
  confidence REAL
);

CREATE TABLE IF NOT EXISTS pair_frame_cues (
  article_id INTEGER,
  news_org TEXT,
  from_version_id TEXT,
  to_version_id TEXT,
  cue TEXT,
  prev INTEGER,
  curr INTEGER,
  direction TEXT
);

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
