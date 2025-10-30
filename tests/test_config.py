from pathlib import Path

import sys
import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
PIPELINE_DIR = ROOT_DIR / "news-edits-pipeline"
sys.path.insert(0, str(PIPELINE_DIR))

from config import Config  # type: ignore  # noqa: E402


def test_config_requires_spacy_model(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model: test\n")

    with pytest.raises(ValueError, match="spacy_model"):
        Config.from_yaml(config_path)
