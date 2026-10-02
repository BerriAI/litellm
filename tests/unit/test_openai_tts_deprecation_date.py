import json
from functools import lru_cache
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]
MAIN_PATH = REPO_ROOT / "model_prices_and_context_window.json"
BACKUP_PATH = REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"

# OpenAI notified API customers that these legacy audio models shut down on 2027-01-06.
SHUTDOWN_DATE = "2027-01-06"
OPENAI_TTS_MODELS = (
    "tts-1",
    "tts-1-1106",
    "tts-1-hd",
    "tts-1-hd-1106",
    "gpt-4o-mini-tts",
    "gpt-4o-mini-tts-2025-03-20",
    "gpt-4o-mini-tts-2025-12-15",
)


@lru_cache(maxsize=2)
def _load(path: Path) -> dict[str, dict[str, object]]:
    with open(path) as f:
        return json.load(f)


@pytest.mark.parametrize("path", (MAIN_PATH, BACKUP_PATH), ids=("main", "backup"))
@pytest.mark.parametrize("model", OPENAI_TTS_MODELS)
def test_openai_tts_models_carry_shutdown_date(path: Path, model: str) -> None:
    entry = _load(path)[model]
    assert entry["litellm_provider"] == "openai"
    assert entry["deprecation_date"] == SHUTDOWN_DATE


@pytest.mark.parametrize("model", OPENAI_TTS_MODELS)
def test_openai_tts_models_backup_matches_main(model: str) -> None:
    assert _load(BACKUP_PATH)[model] == _load(MAIN_PATH)[model], (
        f"{model} differs between main and backup model cost maps"
    )
