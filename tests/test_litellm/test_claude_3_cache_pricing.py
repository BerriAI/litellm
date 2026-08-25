"""
Unit tests for Claude 3 cache creation pricing above 1 hour in model prices map.

Fixes Issue #38056:
- claude-3-haiku-20240307 cache_creation_input_token_cost_above_1hr should be 5e-07 (2x input cost of 2.5e-07)
- claude-3-opus-20240229 cache_creation_input_token_cost_above_1hr should be 3e-05 (2x input cost of 1.5e-05)
"""

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]
MAIN_PATH = REPO_ROOT / "model_prices_and_context_window.json"
BACKUP_PATH = REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"


def _load_prices(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


@pytest.mark.parametrize("path", [MAIN_PATH, BACKUP_PATH], ids=["main", "backup"])
def test_claude_3_haiku_cache_creation_cost_above_1hr(path: Path):
    prices = _load_prices(path)
    model = "claude-3-haiku-20240307"
    assert model in prices, f"{model} not found in {path}"
    model_info = prices[model]

    input_cost = model_info["input_cost_per_token"]
    assert input_cost == 2.5e-07

    # Cache creation cost above 1hr is 2x standard input cost (5e-07)
    expected_cache_creation_above_1hr = input_cost * 2
    assert model_info["cache_creation_input_token_cost_above_1hr"] == expected_cache_creation_above_1hr
    assert model_info["cache_creation_input_token_cost_above_1hr"] == 5e-07


@pytest.mark.parametrize("path", [MAIN_PATH, BACKUP_PATH], ids=["main", "backup"])
def test_claude_3_opus_cache_creation_cost_above_1hr(path: Path):
    prices = _load_prices(path)
    model = "claude-3-opus-20240229"
    assert model in prices, f"{model} not found in {path}"
    model_info = prices[model]

    input_cost = model_info["input_cost_per_token"]
    assert input_cost == 1.5e-05

    # Cache creation cost above 1hr is 2x standard input cost (3e-05)
    expected_cache_creation_above_1hr = input_cost * 2
    assert model_info["cache_creation_input_token_cost_above_1hr"] == expected_cache_creation_above_1hr
    assert model_info["cache_creation_input_token_cost_above_1hr"] == 3e-05
