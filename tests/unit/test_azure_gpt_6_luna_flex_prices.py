"""Azure GPT-6 Luna flex prices: the Azure retail price meters "6-luna ShortCo/LongCo ... Flex Gl" are half of "... Std Gl"."""

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]
MODELS = ("azure/gpt-6-luna", "azure/gpt-6-luna-2026-09-22")
PRICE_KEYS = (
    "input_cost_per_token",
    "output_cost_per_token",
    "cache_read_input_token_cost",
    "cache_creation_input_token_cost",
)
FLEX_PRICES_PER_TOKEN = {
    "input_cost_per_token_flex": 5e-08,
    "output_cost_per_token_flex": 2.5e-07,
    "cache_read_input_token_cost_flex": 5e-09,
    "cache_creation_input_token_cost_flex": 6.25e-08,
    "input_cost_per_token_above_272k_tokens_flex": 1e-07,
    "output_cost_per_token_above_272k_tokens_flex": 3.75e-07,
    "cache_read_input_token_cost_above_272k_tokens_flex": 1e-08,
    "cache_creation_input_token_cost_above_272k_tokens_flex": 1.25e-07,
}


def _load(relative_path: str) -> dict:
    return json.loads((REPO_ROOT / relative_path).read_text())


@pytest.mark.parametrize("model", MODELS)
def test_flex_prices_match_the_azure_retail_meters(model):
    entry = _load("model_prices_and_context_window.json")[model]
    for key, price in FLEX_PRICES_PER_TOKEN.items():
        assert entry[key] == pytest.approx(price), key


@pytest.mark.parametrize("model", MODELS)
def test_flex_prices_are_half_of_the_standard_prices(model):
    entry = _load("model_prices_and_context_window.json")[model]
    for key in PRICE_KEYS:
        for suffix in ("", "_above_272k_tokens"):
            assert entry[f"{key}{suffix}_flex"] == pytest.approx(entry[f"{key}{suffix}"] / 2), f"{key}{suffix}"


@pytest.mark.parametrize("model", MODELS)
def test_backup_cost_map_matches_main(model):
    main = _load("model_prices_and_context_window.json")
    backup = _load("litellm/model_prices_and_context_window_backup.json")
    assert backup[model] == main[model]
