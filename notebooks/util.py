"""Notebook utilities for side-by-side version diff rendering."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from html import escape

from IPython.display import HTML


def highlight_pair(old_text: str, new_text: str) -> tuple[str, str]:
    """Return HTML snippets for old/new texts with removed/added spans highlighted."""
    old_text = re.sub(r"</?p\s*/?>", "\n", old_text or "")
    new_text = re.sub(r"</?p\s*/?>", "\n", new_text or "")
    old_tokens = old_text.split()
    new_tokens = new_text.split()

    matcher = SequenceMatcher(None, old_tokens, new_tokens)
    old_parts: list[str] = []
    new_parts: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        old_chunk = escape(" ".join(old_tokens[i1:i2]).replace("$", r"\$"))
        new_chunk = escape(" ".join(new_tokens[j1:j2]).replace("$", r"\$"))
        if tag == "equal":
            if old_chunk:
                old_parts.append(old_chunk)
            if new_chunk:
                new_parts.append(new_chunk)
        elif tag == "delete":
            if old_chunk:
                old_parts.append(
                    f"<del class='diff-remove' style='background:#ff8f99;color:#4a0d13;text-decoration:line-through;'>"
                    f"{old_chunk}</del>"
                )
        elif tag == "insert":
            if new_chunk:
                new_parts.append(
                    f"<ins class='diff-add' style='background:#8dff9f;color:#053b1f;text-decoration:none;'>{new_chunk}</ins>"
                )
        elif tag == "replace":
            if old_chunk:
                old_parts.append(
                    f"<del class='diff-remove' style='background:#ff8f99;color:#4a0d13;text-decoration:line-through;'>"
                    f"{old_chunk}</del>"
                )
            if new_chunk:
                new_parts.append(
                    f"<ins class='diff-add' style='background:#8dff9f;color:#053b1f;text-decoration:none;'>{new_chunk}</ins>"
                )

    old_html = " ".join(old_parts).replace("\n", "<br>")
    new_html = " ".join(new_parts).replace("\n", "<br>")
    return old_html, new_html


def format_article_html(article_id, version_t1, version_t2, left_html: str, right_html: str) -> HTML:
    """Render side-by-side article versions with diff highlighting."""
    return HTML(
        f"""
        <style>
        .diff-columns {{ display: flex; gap: 1rem; font-family: 'Source Sans Pro', system-ui, sans-serif; }}
        .diff-panel {{ flex: 1; border: 1px solid #ddd; padding: 0.75rem; border-radius: 6px; background: #fafafa; }}
        .diff-panel h4 {{ margin-top: 0; font-size: 1rem; }}
        .diff-add {{ background: #8dff9f !important; color: #053b1f !important; text-decoration: none !important; }}
        .diff-remove {{ background: #ff8f99 !important; color: #4a0d13 !important; text-decoration: line-through !important; }}
        </style>
        <div class="diff-columns">
          <div class="diff-panel">
            <h4>Article {article_id} · v{int(version_t1)}</h4>
            <p>{left_html}</p>
          </div>
          <div class="diff-panel">
            <h4>v{int(version_t2)}</h4>
            <p>{right_html}</p>
          </div>
        </div>
        """
    )
