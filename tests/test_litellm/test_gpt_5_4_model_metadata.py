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
def test_gpt_5_4_small_models_use_documented_token_limits(model: str) -> None:
    """gpt-5.4-mini/nano are 400K-window models: 272K in, 128K out, not gpt-5.4's 1.05M window."""
    info = _load(MAIN_PATH).get(model)
    assert info is not None, f"{model} not found in model_prices_and_context_window.json"

    assert info["max_input_tokens"] == DOCUMENTED_MAX_INPUT_TOKENS
    assert info["max_output_tokens"] == DOCUMENTED_MAX_OUTPUT_TOKENS
    assert info["max_tokens"] == DOCUMENTED_MAX_OUTPUT_TOKENS


@pytest.mark.parametrize("model", SMALL_MODELS)
def test_gpt_5_4_small_models_have_no_long_context_surcharge(model: str) -> None:
    """OpenAI prices prompts above 272K at 2x input / 1.5x output for the 1.05M-window models only."""
    info = _load(MAIN_PATH)[model]
    assert [key for key in info if "above_272k" in key] == []


@pytest.mark.parametrize("model", SMALL_MODELS)
def test_gpt_5_4_small_models_standard_pricing(model: str) -> None:
    info = _load(MAIN_PATH)[model]
    input_cost, output_cost, cache_read_cost = STANDARD_PRICING[_pricing_key(model)]

    assert info["input_cost_per_token"] == input_cost
    assert info["output_cost_per_token"] == output_cost
    assert info["cache_read_input_token_cost"] == cache_read_cost


@pytest.mark.parametrize("model", LONG_CONTEXT_MODELS)
def test_gpt_5_4_long_context_models_keep_surcharge(model: str) -> None:
    """The mini/nano correction must leave gpt-5.4 and gpt-5.4-pro tiered pricing intact."""
    info = _load(MAIN_PATH)[model]

    assert info["input_cost_per_token_above_272k_tokens"] == pytest.approx(info["input_cost_per_token"] * 2)
    assert info["output_cost_per_token_above_272k_tokens"] == pytest.approx(info["output_cost_per_token"] * 1.5)


@pytest.mark.parametrize("model", SMALL_MODELS)
def test_gpt_5_4_small_models_backup_matches_main(model: str) -> None:
    assert _load(BACKUP_PATH).get(model) == _load(MAIN_PATH).get(model), (
        f"{model} differs between main and backup model cost maps"
    )
