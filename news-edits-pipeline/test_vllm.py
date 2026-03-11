"""Smoke test for matt_offline_pipeline vllm setup.

Tests that vllm loads correctly after the safe (non-CUDA-poisoning) imports,
and that the deferred analysis imports work after LLM init.

Usage:
  cd news-edits-pipeline
  CUDA_VISIBLE_DEVICES=4 python test_vllm.py
"""

import sys
print(f"Python: {sys.version}")

# Safe imports (no torch/CUDA at import time)
from config import Config
from loader import iter_article_counts
from pipeline_utils import load_schema, prune_low_confidence
from prompt_utils import build_pair_prompt_payloads
print("Safe imports OK")

# vllm init
print("\nLoading vllm LLM...")
from vllm import LLM, SamplingParams
llm = LLM(model="facebook/opt-125m", enforce_eager=True)
print("LLM loaded OK")

# Deferred imports (spacy/torch — must come AFTER LLM init)
print("\nLoading deferred imports (analysis/article_processor)...")
from analysis import inter_update_timing
from article_processor import ArticleResult
from writer import write_article_result
print("Deferred imports OK")

# Quick inference check
print("\nRunning test inference...")
out = llm.generate(["Hello world"], SamplingParams(max_tokens=8, temperature=0.0))
print(f"Output: {out[0].outputs[0].text!r}")
print("\nAll good!")
