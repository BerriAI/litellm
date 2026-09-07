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

import litellm
from litellm.utils import get_model_info


@pytest.fixture(scope="module", autouse=True)
def _local_model_cost_map():
    """
    Point litellm at the bundled cost map for the duration of this module
    only. ``mp.undo()`` restores both the environment variable and
    ``litellm.model_cost`` so nothing leaks into later tests.
    """
    mp = pytest.MonkeyPatch()
    mp.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    mp.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    get_model_info.cache_clear()
    yield
    mp.undo()
    get_model_info.cache_clear()


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
    json_path = os.path.join(
        os.path.dirname(__file__), "../../model_prices_and_context_window.json"
    )
    with open(json_path) as f:
        return json.load(f)


def test_fireworks_serverless_entries_exist(model_data):
    """The new prefixed entry carries the pricing and metadata from #37274."""
    for key, expected in NEW_ENTRIES.items():
        assert key in model_data, f"{key} is missing from model_prices_and_context_window.json"
        entry = model_data[key]
        for field, value in expected.items():
            assert entry[field] == pytest.approx(value), f"{key}.{field}"
        assert entry["litellm_provider"] == "fireworks_ai"
        assert entry["mode"] == "chat"
        assert entry["supports_function_calling"] is True
        assert entry["supports_vision"] is False


def test_bare_fireworks_ids_resolve_through_prefixed_entries():
    """Bare IDs from #37274 resolve via the provider-prefix lookup path."""
    for bare_id, prefixed_key in [
        (
            "accounts/fireworks/models/deepseek-v4-pro-0813",
            "fireworks_ai/accounts/fireworks/models/deepseek-v4-pro-0813",
        ),
    ]:
        info = get_model_info(model=bare_id, custom_llm_provider="fireworks_ai")
        expected = NEW_ENTRIES[prefixed_key]
        assert info.get("key") == prefixed_key
        assert info["litellm_provider"] == "fireworks_ai"
        assert info["input_cost_per_token"] == pytest.approx(expected["input_cost_per_token"])
        assert info["cache_read_input_token_cost"] == pytest.approx(expected["cache_read_input_token_cost"])
        assert info["output_cost_per_token"] == pytest.approx(expected["output_cost_per_token"])
        assert info["max_input_tokens"] == expected["max_input_tokens"]
        assert info["max_output_tokens"] == expected["max_output_tokens"]


TWIN_PINNED_PRICES = {
    "deepseek-v4-flash-0731": {
        "input_cost_per_token": 2.2e-07,
        "cache_read_input_token_cost": 7e-09,
        "output_cost_per_token": 6.6e-07,
    },
}


def test_deepseek_v4_flash_0731_twins_pin_published_pricing(model_data):
    """Both 0731 entries carry the price published at docs.fireworks.ai/serverless/pricing."""
    for bare_suffix, expected in TWIN_PINNED_PRICES.items():
        for key in (
            f"fireworks_ai/{bare_suffix}",
            f"fireworks_ai/accounts/fireworks/models/{bare_suffix}",
        ):
            entry = model_data[key]
            for field, value in expected.items():
                assert entry[field] == pytest.approx(value), f"{key}.{field}"


def test_fireworks_account_prefixed_twins_agree_on_price(model_data):
    """Every accounts/fireworks/models/X entry prices identically to its bare fireworks_ai/X twin."""
    prefix = "fireworks_ai/accounts/fireworks/models/"
    pairs_checked = 0
    for key, entry in model_data.items():
        if not key.startswith(prefix):
            continue
        bare_key = f"fireworks_ai/{key[len(prefix):]}"
        bare_entry = model_data.get(bare_key)
        if bare_entry is None:
            continue
        pairs_checked += 1
        for field in sorted({f for f in (*entry, *bare_entry) if "cost" in f}):
            assert entry.get(field) == bare_entry.get(field), f"{key} vs {bare_key}: {field}"
    assert pairs_checked >= 20
