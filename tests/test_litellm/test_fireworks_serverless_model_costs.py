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

from litellm.utils import get_model_info


@pytest.fixture(scope="module")
def model_data():
    json_path = os.path.join(os.path.dirname(__file__), "../../model_prices_and_context_window.json")
    with open(json_path) as f:
        return json.load(f)


def test_bare_fireworks_ids_resolve_through_prefixed_entries():
    """Bare IDs from #37274 resolve via the provider-prefix lookup path."""
    for bare_id, prefixed_key in [
        (
            "accounts/fireworks/models/deepseek-v4-pro-0813",
            "fireworks_ai/accounts/fireworks/models/deepseek-v4-pro-0813",
        ),
    ]:
        info = get_model_info(model=bare_id, custom_llm_provider="fireworks_ai")
        assert info.get("key") == prefixed_key
        assert info["litellm_provider"] == "fireworks_ai"


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
