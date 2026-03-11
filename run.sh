#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CONFIG_PATH="$ROOT_DIR/news-edits-pipeline/config.yaml"
CONFIG_PATH="$DEFAULT_CONFIG_PATH"
DB_DIR="$ROOT_DIR/article-versions"
EXTRA_ARGS=()
USER_SUPPLIED_DB=false

if [[ $# -gt 0 && "${1:-}" != "-"* && -f "${1:-}" ]]; then
  CONFIG_PATH="$1"
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --config" >&2
        exit 1
      fi
      CONFIG_PATH="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

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

for arg in "${EXTRA_ARGS[@]}"; do
  if [[ "$arg" == "--db" ]]; then
    USER_SUPPLIED_DB=true
    break
  fi
done

declare -a DB_ARGS=()
if [[ "$USER_SUPPLIED_DB" == true ]]; then
  echo "Info: Using user-supplied --db arguments from command line" >&2
else
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
fi

python "$ROOT_DIR/news-edits-pipeline/pipeline.py" \
  --config "$CONFIG_PATH" \
  "${DB_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
