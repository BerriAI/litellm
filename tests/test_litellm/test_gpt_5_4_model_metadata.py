import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]
MAIN_PATH = REPO_ROOT / "model_prices_and_context_window.json"
BACKUP_PATH = REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"

MINI_NANO_MODELS = (
    "gpt-5.4-mini",
    "gpt-5.4-mini-2026-03-17",
    "gpt-5.4-nano",
    "gpt-5.4-nano-2026-03-17",
    "azure/gpt-5.4-mini",
    "azure/gpt-5.4-mini-2026-03-17",
    "azure/gpt-5.4-nano",
    "azure/gpt-5.4-nano-2026-03-17",
    "azure_ai/gpt-5.4-mini",
    "azure_ai/gpt-5.4-mini-2026-03-17",
    "azure_ai/gpt-5.4-nano",
    "azure_ai/gpt-5.4-nano-2026-03-17",
)

LONG_CONTEXT_MODELS = (
    "gpt-5.4",
    "gpt-5.4-pro",
    "azure/gpt-5.4",
    "azure_ai/gpt-5.4",
)


def _load(path):
    with open(path) as f:
        return json.load(f)


@pytest.mark.parametrize("model", MINI_NANO_MODELS)
def test_gpt_5_4_mini_nano_max_input_tokens(model):
    """gpt-5.4-mini/nano share the 400k context window, so max input is 272k.

    Regression for azure/azure_ai variants that had leaked the base gpt-5.4
    1,050,000 window (and the raw 400,000 context window) into max_input_tokens.
    """
    info = _load(MAIN_PATH).get(model)
    assert info is not None, f"{model} missing from model_prices_and_context_window.json"
    assert info["max_input_tokens"] == 272000
    assert info["max_output_tokens"] == 128000
    assert info["max_tokens"] == 128000


@pytest.mark.parametrize("model", LONG_CONTEXT_MODELS)
def test_gpt_5_4_base_keeps_long_context_window(model):
    """Base gpt-5.4 and gpt-5.4-pro keep the full 1,050,000 token input window."""
    info = _load(MAIN_PATH).get(model)
    assert info is not None, f"{model} missing from model_prices_and_context_window.json"
    assert info["max_input_tokens"] == 1050000
    assert info["max_output_tokens"] == 128000


@pytest.mark.parametrize("model", MINI_NANO_MODELS + LONG_CONTEXT_MODELS)
def test_gpt_5_4_backup_matches_main(model):
    """The bundled cost map must stay in sync with the canonical file."""
    assert _load(BACKUP_PATH).get(model) == _load(MAIN_PATH).get(
        model
    ), f"{model} differs between main and backup model cost maps"
