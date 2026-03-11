#!/usr/bin/env bash
# run-matt.sh — fast launcher for --matt-features-only
#
# Two modes:
#   offline (default) — uses vllm.LLM directly for maximum batch throughput.
#                       Requires GPUs on this machine.  No server needed.
#   online  (--online) — hits a running vLLM server (http://localhost:8000).
#                        Use this when vLLM is already running elsewhere.
#
# Examples:
#   # Offline, all DBs, 4 GPUs, 4k context:
#   bash run-matt.sh --tensor-parallel-size 4 --max-model-len 4096
#
#   # Offline, single DB, cap 500 articles:
#   bash run-matt.sh --db article-versions/ap.db --max-articles-per-outlet 500
#
#   # Online mode (vLLM server already running):
#   bash run-matt.sh --online --article-workers 8 --max-in-flight-llm-requests 32

set -euo pipefail
export PYTHONUNBUFFERED=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="$ROOT_DIR/news-edits-pipeline/config.yaml"
DB_DIR="$ROOT_DIR/article-versions"
OUT_PATH="$ROOT_DIR/out"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MODE="offline"

# Offline-mode flags
TENSOR_PARALLEL_SIZE=""   # auto-detected below
MAX_MODEL_LEN="8192"
GPU_MEM_UTIL="0.95"

# Online-mode flags (passed to pipeline.py)
ARTICLE_WORKERS="8"
MAX_IN_FLIGHT="32"        # keep vLLM server queue saturated

MAX_ARTICLES_PER_OUTLET=""
EXTRA_DB_ARGS=()
USER_SUPPLIED_DB=false
EXTRA_ARGS=()

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --online)
      MODE="online"
      shift
      ;;
    --offline)
      MODE="offline"
      shift
      ;;
    --tensor-parallel-size)
      TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
    --max-model-len)
      MAX_MODEL_LEN="$2"; shift 2 ;;
    --gpu-memory-utilization)
      GPU_MEM_UTIL="$2"; shift 2 ;;
    --article-workers)
      ARTICLE_WORKERS="$2"; shift 2 ;;
    --max-in-flight-llm-requests)
      MAX_IN_FLIGHT="$2"; shift 2 ;;
    --max-articles-per-outlet)
      MAX_ARTICLES_PER_OUTLET="$2"; shift 2 ;;
    --config)
      CONFIG_PATH="$2"; shift 2 ;;
    --out-path)
      OUT_PATH="$2"; shift 2 ;;
    --db)
      EXTRA_DB_ARGS+=("--db" "$2")
      USER_SUPPLIED_DB=true
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

# ---------------------------------------------------------------------------
# Auto-detect GPUs for offline mode
# ---------------------------------------------------------------------------
if [[ "$MODE" == "offline" && -z "$TENSOR_PARALLEL_SIZE" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    # Count comma-separated entries in CUDA_VISIBLE_DEVICES.
    IFS=',' read -ra _GPUS <<< "$CUDA_VISIBLE_DEVICES"
    TENSOR_PARALLEL_SIZE="${#_GPUS[@]}"
  elif command -v nvidia-smi &>/dev/null; then
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
    TENSOR_PARALLEL_SIZE="${NUM_GPUS:-1}"
  else
    TENSOR_PARALLEL_SIZE="1"
    echo "Warning: nvidia-smi not found; defaulting to --tensor-parallel-size 1" >&2
  fi
  echo "Auto-detected GPUs: $TENSOR_PARALLEL_SIZE" >&2
fi

# ---------------------------------------------------------------------------
# Collect DB paths (same logic as run.sh)
# ---------------------------------------------------------------------------
declare -a DB_ARGS=()
if [[ "$USER_SUPPLIED_DB" == true ]]; then
  DB_ARGS=("${EXTRA_DB_ARGS[@]}")
else
  declare -a DB_NAMES=(
    "ap.db"
    "newssniffer-bbc.db"
    "newssniffer-guardian.db"
    "newssniffer-independent.db"
    "newssniffer-nytimes.db"
    "newssniffer-washpo.db"
    "reuters.db"
  )
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

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
cd "$ROOT_DIR/news-edits-pipeline"

if [[ "$MODE" == "offline" ]]; then
  echo "[run-matt] Mode: OFFLINE  tensor_parallel=$TENSOR_PARALLEL_SIZE  max_model_len=$MAX_MODEL_LEN  gpu_mem=$GPU_MEM_UTIL" >&2

  OFFLINE_ARGS=(
    --config  "$CONFIG_PATH"
    "${DB_ARGS[@]}"
    --out-path "$OUT_PATH"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
  )
  if [[ -n "$MAX_ARTICLES_PER_OUTLET" ]]; then
    OFFLINE_ARGS+=(--max-articles-per-outlet "$MAX_ARTICLES_PER_OUTLET")
  fi
  OFFLINE_ARGS+=("${EXTRA_ARGS[@]}")

  python "$ROOT_DIR/news-edits-pipeline/matt_offline_pipeline.py" "${OFFLINE_ARGS[@]}"

else
  echo "[run-matt] Mode: ONLINE  article_workers=$ARTICLE_WORKERS  max_in_flight=$MAX_IN_FLIGHT" >&2

  ONLINE_ARGS=(
    --config  "$CONFIG_PATH"
    "${DB_ARGS[@]}"
    --out-path "$OUT_PATH"
    --matt-features-only
    --article-workers     "$ARTICLE_WORKERS"
    --max-in-flight-llm-requests "$MAX_IN_FLIGHT"
  )
  if [[ -n "$MAX_ARTICLES_PER_OUTLET" ]]; then
    ONLINE_ARGS+=(--max-articles-per-outlet "$MAX_ARTICLES_PER_OUTLET")
  fi
  ONLINE_ARGS+=("${EXTRA_ARGS[@]}")

  python "$ROOT_DIR/news-edits-pipeline/pipeline.py" "${ONLINE_ARGS[@]}"
fi
