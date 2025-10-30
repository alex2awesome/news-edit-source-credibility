"""Classical (non-LLM) analyses for the news edits pipeline."""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

from canonicalize import fuzzy_match_source, normalize_source


@lru_cache(maxsize=1)
def _get_nlp(spacy_model: str):
    """Load spaCy model once."""
    if not spacy_model:
        raise ValueError("spaCy model name is required but was not provided.")
    try:
        return spacy.load(spacy_model)
    except OSError as exc:  # pragma: no cover - environment specific
        try:
            spacy.cli.download(spacy_model)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download spaCy model '{spacy_model}'. Please download it manually and try again."
            ) from exc
        return spacy.load(spacy_model)


_PARAGRAPH_SEP_RE = re.compile(r"(?:\n\s*\n|<p>|</p>)")
_TOKEN_SPLIT_RE = re.compile(r"\W+", re.UNICODE)


def _split_paragraphs(text: str) -> List[Dict]:
    paragraphs: List[Dict] = []
    start = 0
    idx = 0
    for match in _PARAGRAPH_SEP_RE.finditer(text):
        end = match.start()
        if end > start:
            segment = text[start:end]
            paragraphs.append({"start": start, "end": end, "text": segment, "index": idx})
            idx += 1
        start = match.end()
    if start < len(text):
        paragraphs.append({"start": start, "end": len(text), "text": text[start:], "index": idx})
    return paragraphs


def segment(text: str, spacy_model: str) -> Dict:
    """Return sentence, paragraph, token segmentation with char offsets."""
    doc = _get_nlp(spacy_model)(text)
    sentences = [
        {"start": sent.start_char, "end": sent.end_char, "text": sent.text, "index": i}
        for i, sent in enumerate(doc.sents)
    ]
    tokens = [
        {"start": token.idx, "end": token.idx + len(token.text), "text": token.text, "index": i}
        for i, token in enumerate(doc)
    ]
    paragraphs = _split_paragraphs(text)
    return {
        "sentences": sentences,
        "paragraphs": paragraphs,
        "tokens": tokens,
        "char_len": len(text),
    }


def extract_lede(title: str, text: str, paragraphs: List[Dict]) -> Dict:
    """Return the first paragraph as the lede along with offsets."""
    if paragraphs:
        para = paragraphs[0]
        return {"text": para["text"], "start": para["start"], "end": para["end"], "paragraph_index": para["index"]}
    # fallback to first ~80 words when paragraphs missing
    tokens = text.split()
    snippet = " ".join(tokens[:80])
    end = text.find(snippet) + len(snippet) if snippet else 0
    return {"text": snippet, "start": 0, "end": end, "paragraph_index": 0}


def compute_prominence_features(char_start: int, char_len: int, in_title: bool, in_lede: bool) -> Dict:
    """Compute prominence metrics for a mention."""
    lead_pct = 0.0
    if char_len > 0 and char_start >= 0:
        lead_pct = max(0.0, min(1.0, char_start / char_len))
    return {
        "lead_percentile": lead_pct,
        "is_in_title": bool(in_title),
        "is_in_lede": bool(in_lede),
    }


def ner_entities_spacy(text: str, spacy_model: str) -> Dict:
    """Extract entities using spaCy with canonical forms."""
    doc = _get_nlp(spacy_model)(text)
    entities = []
    for ent in doc.ents:
        entities.append(
            {
                "surface": ent.text,
                "label": ent.label_,
                "char_start": ent.start_char,
                "char_end": ent.end_char,
                "canonical": normalize_source(ent.text),
            }
        )
    return {"entities": entities}


def align_sentences(prev_text: str, curr_text: str, spacy_model: str) -> Dict:
    """Align sentences using TF-IDF cosine similarity and measure movement."""
    prev_doc = _get_nlp(spacy_model)(prev_text)
    curr_doc = _get_nlp(spacy_model)(curr_text)
    prev_sentences = [sent.text for sent in prev_doc.sents]
    curr_sentences = [sent.text for sent in curr_doc.sents]

    if not prev_sentences or not curr_sentences:
        return {"alignments": [], "avg_rank_change": 0.0, "movement_index": 0.0}

    vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(prev_sentences + curr_sentences)
    prev_mat = matrix[: len(prev_sentences)]
    curr_mat = matrix[len(prev_sentences) :]
    sim = prev_mat @ curr_mat.T

    alignments = []
    total_rank_change = 0.0
    total_direction = 0.0
    max_len = max(len(prev_sentences), len(curr_sentences)) or 1

    for i in range(len(prev_sentences)):
        if sim.shape[1] == 0:
            break
        row = sim.getrow(i).toarray()[0]
        best_idx = int(row.argmax())
        best_score = float(row[best_idx]) if row.size else 0.0
        alignments.append({"prev_index": i, "curr_index": best_idx, "score": best_score})
        total_rank_change += abs(i - best_idx)
        total_direction += (i - best_idx)

    if alignments:
        avg_rank_change = total_rank_change / len(alignments)
        movement_index = total_direction / (len(alignments) * max_len)
    else:
        avg_rank_change = 0.0
        movement_index = 0.0

    return {
        "alignments": alignments,
        "avg_rank_change": avg_rank_change,
        "movement_index": movement_index,
    }


def jaccard_title_body(title: str, body_first_paragraph: str) -> float:
    """Compute Jaccard overlap between title and first paragraph tokens."""
    title_tokens = {tok.lower() for tok in _TOKEN_SPLIT_RE.split(title) if tok}
    body_tokens = {tok.lower() for tok in _TOKEN_SPLIT_RE.split(body_first_paragraph) if tok}
    if not title_tokens or not body_tokens:
        return 0.0
    return len(title_tokens & body_tokens) / len(title_tokens | body_tokens)


def compute_diff_magnitude(prev_text: str, curr_text: str) -> Dict:
    """Return token-level diff metrics between consecutive versions."""
    prev_tokens = prev_text.split()
    curr_tokens = curr_text.split()
    prev_counter = Counter(prev_tokens)
    curr_counter = Counter(curr_tokens)

    added = 0
    removed = 0
    for token, count in curr_counter.items():
        delta = count - prev_counter.get(token, 0)
        if delta > 0:
            added += delta
    for token, count in prev_counter.items():
        delta = count - curr_counter.get(token, 0)
        if delta > 0:
            removed += delta

    percent_new = (added / max(len(curr_tokens), 1))
    return {
        "tokens_added": added,
        "tokens_deleted": removed,
        "percent_text_new": percent_new,
    }


def inter_update_timing(prev_ts: str, curr_ts: str) -> float:
    """Return minutes between two ISO-8601 timestamps."""
    def _parse(ts: str) -> datetime:
        if ts is None:
            return None
        ts = ts.strip()
        if not ts:
            return None
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(ts)
        except ValueError:
            return None

    prev_dt = _parse(prev_ts)
    curr_dt = _parse(curr_ts)
    if not prev_dt or not curr_dt:
        return 0.0
    delta = curr_dt - prev_dt
    return delta.total_seconds() / 60.0


def aggregate_sources_over_versions(source_mentions_by_version: List[Dict], char_lens_by_version: List[Dict]) -> Dict:
    """Aggregate source-level metrics across all versions of an article."""
    sources: Dict[str, Dict] = {}
    known_for_fuzzy: Dict[str, Dict[str, str]] = {}
    version_nums = sorted(meta["version_num"] for meta in char_lens_by_version)
    final_version = max(version_nums) if version_nums else 0

    for version_blob in source_mentions_by_version:
        version_num = version_blob.get("version_num")
        timestamp = version_blob.get("timestamp_utc")
        mentions = version_blob.get("mentions", [])
        for mention in mentions:
            canonical = mention.get("canonical") or mention.get("surface")
            source_type = mention.get("type", "unknown")
            norm = normalize_source(canonical)

            source_id = mention.get("source_id_within_article")
            match_key = source_id or fuzzy_match_source(canonical, known_for_fuzzy)
            if match_key is None:
                match_key = f"s{len(sources) + 1:03d}"
            known_for_fuzzy.setdefault(match_key, {"normalized": norm})
            if match_key not in sources:
                sources[match_key] = {
                    "source_id_within_article": match_key,
                    "source_canonical": canonical,
                    "source_type": source_type,
                    "normalized": norm,
                    "first_seen_version": version_num,
                    "first_seen_time": timestamp,
                    "last_seen_version": version_num,
                    "last_seen_time": timestamp,
                    "num_mentions_total": 0,
                    "num_versions_present": set(),
                    "total_attributed_words": 0,
                    "lead_appearance_count": 0,
                    "title_appearance_count": 0,
                    "prominence_values": [],
                    "presence_sequence": [],
                    "per_version_flags": {},
                    "doubted_any": False,
                }
            record = sources[match_key]
            record["source_canonical"] = canonical or record["source_canonical"]
            record["source_type"] = source_type or record["source_type"]
            record["last_seen_version"] = version_num
            record["last_seen_time"] = timestamp
            record["num_mentions_total"] += 1
            record["num_versions_present"].add(version_num)
            attributed_text = mention.get("attributed_text") or ""
            record["total_attributed_words"] += len(attributed_text.split())
            if mention.get("is_in_lede"):
                record["lead_appearance_count"] += 1
            if mention.get("is_in_title"):
                record["title_appearance_count"] += 1
            prom = mention.get("prominence", {}).get("lead_percentile")
            if prom is not None:
                record["prominence_values"].append(float(prom))
            record["doubted_any"] = record["doubted_any"] or mention.get("doubted", False)

            record["per_version_flags"].setdefault(version_num, {"in_lede": 0, "in_title": 0, "present": 0})
            flags = record["per_version_flags"][version_num]
            if mention.get("is_in_lede"):
                flags["in_lede"] = 1
            if mention.get("is_in_title"):
                flags["in_title"] = 1
            flags["present"] = 1

    for record in sources.values():
        presence = []
        for v in version_nums:
            flags = record["per_version_flags"].get(v)
            presence.append(1 if flags and flags.get("present") else 0)
        record["presence_sequence"] = presence
        consecutive = sum(1 for i in range(1, len(presence)) if presence[i] and presence[i - 1])
        possible = max(sum(presence) - 1, 0)
        record["voice_retention_index"] = consecutive / possible if possible else 0.0
        record["num_versions_present"] = len(record["num_versions_present"])
        record["mean_prominence"] = float(np.mean(record["prominence_values"])) if record["prominence_values"] else 0.0

        # deemphasized_any: appeared in lede/title in an earlier version and later only outside those zones
        deemphasized = False
        observed_lede_or_title = False
        for v in version_nums:
            flags = record["per_version_flags"].get(v, {"in_lede": 0, "in_title": 0, "present": 0})
            if flags["present"] and (flags["in_lede"] or flags["in_title"]):
                observed_lede_or_title = True
            if observed_lede_or_title and flags["present"] and not (flags["in_lede"] or flags["in_title"]):
                deemphasized = True
                break
        record["deemphasized_any"] = deemphasized

        final_flags = record["per_version_flags"].get(final_version, {"present": 0})
        record["disappeared_any"] = bool(sum(presence) > 0 and not final_flags.get("present"))

    sources_list = []
    for record in sources.values():
        sources_list.append(
            {
                key: value
                for key, value in record.items()
                if key not in {"normalized", "prominence_values", "per_version_flags"}
            }
        )

    article_vri = 0.0
    if sources_list:
        article_vri = float(np.mean([src["voice_retention_index"] for src in sources_list]))

    return {
        "sources": sources_list,
        "article_voice_retention_index": article_vri,
        "total_versions": len(char_lens_by_version),
    }


def diversity_indices(attributed_words_by_source: Dict[str, int]) -> Dict:
    """Compute Shannon entropy, Herfindahl index, and Gini coefficient."""
    counts = np.array([max(v, 0) for v in attributed_words_by_source.values()], dtype=float)
    total = counts.sum()
    if total == 0:
        return {"shannon": 0.0, "herfindahl": 0.0, "gini": 0.0}

    proportions = counts / total
    shannon = float(-np.sum([p * math.log(p) for p in proportions if p > 0]))
    herfindahl = float(np.sum(proportions ** 2))

    # Gini coefficient
    sorted_counts = np.sort(counts)
    index = np.arange(1, len(sorted_counts) + 1)
    gini = float((np.sum((2 * index - len(sorted_counts) - 1) * sorted_counts)) / (len(sorted_counts) * total))
    gini = abs(gini)

    return {"shannon": shannon, "herfindahl": herfindahl, "gini": gini}


def final_version_bias(metrics_first: Dict, metrics_final: Dict) -> Dict:
    """Compare first vs. final version sourcing metrics."""
    keys = {
        "institutional_share_words",
        "anonymous_source_share_words",
        "hedge_density_per_1k_tokens",
    }
    deltas = {}
    for key in keys:
        first_val = float(metrics_first.get(key, 0.0)) if metrics_first else 0.0
        final_val = float(metrics_final.get(key, 0.0)) if metrics_final else 0.0
        deltas[f"{key}_delta"] = final_val - first_val
    deltas["overstate_institutional_share"] = deltas.get("institutional_share_words_delta", 0.0)
    return deltas


__all__ = [
    "segment",
    "extract_lede",
    "compute_prominence_features",
    "ner_entities_spacy",
    "align_sentences",
    "jaccard_title_body",
    "compute_diff_magnitude",
    "inter_update_timing",
    "aggregate_sources_over_versions",
    "diversity_indices",
    "final_version_bias",
]
