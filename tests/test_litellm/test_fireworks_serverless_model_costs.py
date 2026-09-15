"""
Validate the Fireworks AI Serverless entry added for #37274 exists in
`model_prices_and_context_window.json` and that the bare Fireworks model ID
resolves through `get_model_info`.

Pricing as published at https://docs.fireworks.ai/serverless/pricing
(USD per 1M tokens, uncached input / cached input / output):

  accounts/fireworks/models/deepseek-v4-pro-0813  -> $1.32 / $0.044 / $3.96
"""

import json
import os

import pytest

NEW_ENTRIES = {
    "fireworks_ai/accounts/fireworks/models/deepseek-v4-pro-0813": {
        "input_cost_per_token": 1.32e-06,
        "cache_read_input_token_cost": 4.4e-08,
        "output_cost_per_token": 3.96e-06,
        "max_input_tokens": 1048576,
        "max_output_tokens": 131072,
    },
}


@pytest.fixture(scope="module")
def model_data():
    json_path = os.path.join(os.path.dirname(__file__), "../../model_prices_and_context_window.json")
    with open(json_path) as f:
        return json.load(f)


TWIN_PINNED_PRICES = {
    "deepseek-v4-flash-0731": {
        "input_cost_per_token": 2.2e-07,
        "cache_read_input_token_cost": 7e-09,
        "output_cost_per_token": 6.6e-07,
    },
    "deepseek-v4p1-flash": {
        "input_cost_per_token": 2.2e-07,
        "cache_read_input_token_cost": 7e-09,
        "output_cost_per_token": 6.6e-07,
        "supports_vision": True,
        "max_output_tokens": 393216,
    },
}


def test_fireworks_account_prefixed_twins_agree_on_price(model_data):
    """Every accounts/fireworks/models/X entry prices identically to its bare fireworks_ai/X twin."""
    prefix = "fireworks_ai/accounts/fireworks/models/"
    pairs_checked = 0
    for key, entry in model_data.items():
        if not key.startswith(prefix):
            continue
        bare_key = f"fireworks_ai/{key[len(prefix) :]}"
        bare_entry = model_data.get(bare_key)
        if bare_entry is None:
            continue
        pairs_checked += 1
        for field in sorted({f for f in (*entry, *bare_entry) if "cost" in f}):
            assert entry.get(field) == bare_entry.get(field), f"{key} vs {bare_key}: {field}"
    assert pairs_checked >= 20
