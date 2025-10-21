#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${1:-"$ROOT_DIR/news-edits-pipeline/config.yaml"}"
DB_DIR="$ROOT_DIR/article-versions"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config file not found: $CONFIG_PATH" >&2
  exit 1
fi

declare -a DB_NAMES=(
  "ap.db"
  "newssniffer-bbc.db"
  "newssniffer-guardian.db"
  "newssniffer-independent.db"
  "newssniffer-nytimes.db"
  "newssniffer-washpo.db"
  "reuters.db"
)

declare -a DB_ARGS=()
for name in "${DB_NAMES[@]}"; do
  db_path="$DB_DIR/$name"
  if [[ -f "$db_path" ]]; then
    DB_ARGS+=("--db" "$db_path")
  elif [[ -f "$db_path.gz" ]]; then
    DB_ARGS+=("--db" "$db_path.gz")
  else
    echo "Warning: Skipping missing database $name" >&2
  fi
done

if [[ ${#DB_ARGS[@]} -eq 0 ]]; then
  echo "No database files found in $DB_DIR" >&2
  exit 1
fi

python "$ROOT_DIR/news-edits-pipeline/pipeline.py" \
  --config "$CONFIG_PATH" \
  "${DB_ARGS[@]}"
