import json
from pathlib import Path

import pytest

from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

REPO_ROOT = Path(__file__).parents[2]
MAIN_PATH = REPO_ROOT / "model_prices_and_context_window.json"
BACKUP_PATH = REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"

UNPREFIXED = "gemini-3.1-flash-lite-image"
GEMINI = "gemini/gemini-3.1-flash-lite-image"
VERTEX = "vertex_ai/gemini-3.1-flash-lite-image"
ALL_KEYS = (UNPREFIXED, GEMINI, VERTEX)


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.parametrize("model", ALL_KEYS)
def test_backup_matches_main(model: str):
    assert _load(BACKUP_PATH).get(model) == _load(MAIN_PATH).get(model)


def test_gemini_prefix_routes_to_gemini():
    routed_model, provider, _, _ = get_llm_provider(model=GEMINI)
    assert routed_model == UNPREFIXED
    assert provider == "gemini"


def test_vertex_prefix_routes_to_vertex():
    routed_model, provider, _, _ = get_llm_provider(model=VERTEX)
    assert routed_model == UNPREFIXED
    assert provider == "vertex_ai"
