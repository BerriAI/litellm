"""
Regression test for Fireworks Kimi K2.5 / K2.6 / K2.7 context and output limits.

Fireworks publishes a 262144-token context window for every Kimi K2.5, K2.6 and
K2.7 model, but caps generation well below that. A previous bulk edit had flattened
max_output_tokens/max_tokens to 262144 (equal to the context window), which let the
pre-call context-window check admit requests asking for a full 262144-token
completion that Fireworks then rejects. These assertions pin the corrected per-alias
limits so a future bulk edit can't silently flatten them again.
"""

import json
from importlib.resources import files

import pytest

CONTEXT_WINDOW = 262144
OUTPUT_LIMIT = 32768

KIMI_ALIASES = (
    "fireworks_ai/kimi-k2p5",
    "fireworks_ai/kimi-k2p6",
    "fireworks_ai/kimi-k2p6-fast",
    "fireworks_ai/kimi-k2p7-code",
    "fireworks_ai/kimi-k2p7-code-fast",
    "fireworks_ai/accounts/fireworks/models/kimi-k2p5",
    "fireworks_ai/accounts/fireworks/models/kimi-k2p6",
    "fireworks_ai/accounts/fireworks/models/kimi-k2p7-code",
    "fireworks_ai/accounts/fireworks/routers/kimi-k2p6-fast",
    "fireworks_ai/accounts/fireworks/routers/kimi-k2p7-code-fast",
)


@pytest.fixture(scope="module")
def use_local_model_cost_map():
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")

    import litellm
    from litellm.utils import _invalidate_model_cost_lowercase_map

    original_model_cost = litellm.model_cost
    litellm.model_cost = json.loads(
        files("litellm")
        .joinpath("model_prices_and_context_window_backup.json")
        .read_text(encoding="utf-8")
    )
    litellm.get_model_info.cache_clear()
    _invalidate_model_cost_lowercase_map()
    try:
        yield litellm
    finally:
        litellm.model_cost = original_model_cost
        litellm.get_model_info.cache_clear()
        _invalidate_model_cost_lowercase_map()
        monkeypatch.undo()


@pytest.mark.parametrize("alias", KIMI_ALIASES)
def test_fireworks_kimi_raw_cost_entry_limits(use_local_model_cost_map, alias):
    entry = use_local_model_cost_map.model_cost[alias]

    assert entry["litellm_provider"] == "fireworks_ai"
    assert entry["max_input_tokens"] == CONTEXT_WINDOW
    assert entry["max_output_tokens"] == OUTPUT_LIMIT
    assert entry["max_tokens"] == OUTPUT_LIMIT
    assert entry["max_output_tokens"] < entry["max_input_tokens"]


@pytest.mark.parametrize("alias", KIMI_ALIASES)
def test_fireworks_kimi_get_model_info_limits(use_local_model_cost_map, alias):
    model_info = use_local_model_cost_map.get_model_info(model=alias)

    assert model_info["max_input_tokens"] == CONTEXT_WINDOW
    assert model_info["max_output_tokens"] == OUTPUT_LIMIT
    assert model_info["max_tokens"] == OUTPUT_LIMIT
