import json
from functools import lru_cache
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]
MAIN_PATH = REPO_ROOT / "model_prices_and_context_window.json"
BACKUP_PATH = REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"

DOCUMENTED_MAX_INPUT_TOKENS = 272000
DOCUMENTED_MAX_OUTPUT_TOKENS = 128000

SMALL_MODEL_NAMES = (
    "gpt-5.4-mini",
    "gpt-5.4-mini-2026-03-17",
    "gpt-5.4-nano",
    "gpt-5.4-nano-2026-03-17",
)
SMALL_MODELS = tuple(f"{prefix}{name}" for prefix in ("", "azure/", "azure_ai/") for name in SMALL_MODEL_NAMES)

STANDARD_PRICING = {
    "gpt-5.4-mini": (7.5e-07, 4.5e-06, 7.5e-08),
    "gpt-5.4-nano": (2e-07, 1.25e-06, 2e-08),
}

LONG_CONTEXT_MODELS = ("gpt-5.4", "gpt-5.4-pro")


@lru_cache(maxsize=2)
def _load(path: Path) -> dict[str, dict[str, object]]:
    with open(path) as f:
        return json.load(f)


def _pricing_key(model: str) -> str:
    return "gpt-5.4-nano" if "nano" in model else "gpt-5.4-mini"


@pytest.mark.parametrize("model", SMALL_MODELS)
def test_gpt_5_4_small_models_backup_matches_main(model: str) -> None:
    assert _load(BACKUP_PATH).get(model) == _load(MAIN_PATH).get(model), (
        f"{model} differs between main and backup model cost maps"
    )
