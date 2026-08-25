"""
Validate that the Fireworks AI Serverless entries added for #37274 exist in
`model_prices_and_context_window.json` and that the bare Fireworks model IDs
resolve through `get_model_info`.

Only two of the eight IDs requested in #37274 were genuinely missing: the
remaining six already resolve through the `fireworks_ai/` prefixed entries via
the provider-prefix fallback in `get_model_info`. This test pins the two new
entries and proves the bare IDs map onto them.

Pricing as published in #37274 (USD per 1M tokens, uncached input / cached
input / output):

  accounts/fireworks/models/deepseek-v4-pro-0813  -> $1.32 / $0.044 / $3.96
  accounts/fireworks/models/qwen3p8-2p4t-a95b     -> $2.00 / $0.25  / $6.00
"""

import json
import os

import pytest

# Force litellm to load the cost map from the bundled backup file instead of
# fetching it from GitHub, so the entries added in this PR are the ones tested
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

from litellm.utils import get_model_info  # noqa: E402

NEW_ENTRIES = {
    "fireworks_ai/accounts/fireworks/models/deepseek-v4-pro-0813": {
        "input_cost_per_token": 1.32e-06,
        "cache_read_input_token_cost": 4.4e-08,
        "output_cost_per_token": 3.96e-06,
        "max_input_tokens": 1048576,
        "max_output_tokens": 131072,
    },
    "fireworks_ai/accounts/fireworks/models/qwen3p8-2p4t-a95b": {
        "input_cost_per_token": 2e-06,
        "cache_read_input_token_cost": 2.5e-07,
        "output_cost_per_token": 6e-06,
        "max_input_tokens": 262144,
        "max_output_tokens": 32768,
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
    """The two new prefixed entries carry the pricing and metadata from #37274."""
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
        (
            "accounts/fireworks/models/qwen3p8-2p4t-a95b",
            "fireworks_ai/accounts/fireworks/models/qwen3p8-2p4t-a95b",
        ),
    ]:
        info = get_model_info(model=bare_id, custom_llm_provider="fireworks_ai")
        assert info["litellm_provider"] == "fireworks_ai"
        assert info["input_cost_per_token"] == pytest.approx(NEW_ENTRIES[prefixed_key]["input_cost_per_token"])
