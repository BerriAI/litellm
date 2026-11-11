import json
from pathlib import Path

import pytest

# USD per token, from DeepInfra's published prices (https://api.deepinfra.com/models/<model>)
EXPECTED_PRICES = {
    "deepinfra/mistralai/Mistral-Nemo-Instruct-2407": (2.9e-08, 3e-08),
    "deepinfra/nvidia/NVIDIA-Nemotron-3.5-Lightning": (6e-08, 1.6e-07),
    "deepinfra/anthropic/claude-sonnet-5": (3e-06, 1.5e-05),
    "deepinfra/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B": (7.5e-08, 4e-07),
}

REPO_ROOT = Path(__file__).parents[2]
COST_MAPS = {
    "main": REPO_ROOT / "model_prices_and_context_window.json",
    "backup": REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json",
}


@pytest.mark.parametrize("cost_map", COST_MAPS, ids=list(COST_MAPS))
@pytest.mark.parametrize("model", EXPECTED_PRICES)
def test_deepinfra_prices_match_published_rates(cost_map, model):
    with open(COST_MAPS[cost_map]) as f:
        info = json.load(f)[model]

    expected_input, expected_output = EXPECTED_PRICES[model]
    assert info["litellm_provider"] == "deepinfra"
    assert info["input_cost_per_token"] == pytest.approx(expected_input)
    assert info["output_cost_per_token"] == pytest.approx(expected_output)
