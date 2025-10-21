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

CREATE TABLE IF NOT EXISTS pair_sources_added (
  from_version_id TEXT,
  to_version_id TEXT,
  canonical TEXT,
  type TEXT
);

CREATE TABLE IF NOT EXISTS pair_sources_removed (
  from_version_id TEXT,
  to_version_id TEXT,
  canonical TEXT,
  type TEXT
);

CREATE TABLE IF NOT EXISTS pair_title_events (
  from_version_id TEXT,
  to_version_id TEXT,
  canonical TEXT,
  event TEXT
);

CREATE TABLE IF NOT EXISTS pair_anon_named_replacements (
  from_version_id TEXT,
  to_version_id TEXT,
  src TEXT,
  dst TEXT,
  direction TEXT,
  likelihood REAL
);

CREATE TABLE IF NOT EXISTS pair_numeric_changes (
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
  from_version_id TEXT,
  to_version_id TEXT,
  claim_id TEXT,
  proposition TEXT,
  status TEXT,
  change_note TEXT,
  confidence REAL
);

CREATE TABLE IF NOT EXISTS pair_frame_cues (
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
