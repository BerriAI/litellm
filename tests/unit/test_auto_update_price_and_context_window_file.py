"""Unit tests for the Friendli transform in
`.github/scripts/auto_update_price_and_context_window_file.py`."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / ".github"
    / "scripts"
    / "auto_update_price_and_context_window_file.py"
)


@pytest.fixture(scope="module")
def sync_module():
    spec = importlib.util.spec_from_file_location(
        "auto_update_price_and_context_window_file", SCRIPT_PATH
    )
    assert spec and spec.loader, f"Could not load spec for {SCRIPT_PATH}"
    module = importlib.util.module_from_spec(spec)
    sys.modules["auto_update_price_and_context_window_file"] = module
    spec.loader.exec_module(module)
    return module


def _reasoning_model(**overrides: object) -> dict:
    model = {
        "id": "zai-org/GLM-Test",
        "base_model": "zhipuai/glm-test",
        "context_length": 1048576,
        "max_completion_tokens": 131072,
        "pricing": {"input": "0.00000015", "output": "0.0000005", "input_cache_read": "0.00000003"},
        "reasoning": True,
        "reasoning_options": [{"type": "effort", "values": ["max", "high", "low"]}],
        "functionality": {
            "tool_call": True,
            "parallel_tool_call": True,
            "structured_output": True,
            "system_messages": True,
            "tool_choice": True,
        },
        "input_modalities": ["text", "image", "video"],
        "mode": "chat",
    }
    model.update(overrides)
    return model


def test_transform_emits_declared_effort_levels_in_canonical_order(sync_module):
    entry = sync_module.transform_friendli_data([_reasoning_model()], {})[
        "friendliai/zai-org/GLM-Test"
    ]
    assert entry["supports_reasoning"] is True
    assert entry["reasoning_effort_levels"] == ["low", "high", "max"]
    assert not any(k.endswith("_reasoning_effort") for k in entry)


def test_transform_reasoning_model_without_effort_options_declares_empty_levels(sync_module):
    model = _reasoning_model(reasoning_options=[{"type": "budget_tokens", "values": []}])
    entry = sync_module.transform_friendli_data([model], {})["friendliai/zai-org/GLM-Test"]
    assert entry["reasoning_effort_levels"] == []


def test_transform_non_reasoning_model_declares_no_levels(sync_module):
    model = _reasoning_model(reasoning=False, reasoning_options=[])
    entry = sync_module.transform_friendli_data([model], {})["friendliai/zai-org/GLM-Test"]
    assert entry["supports_reasoning"] is False
    assert "reasoning_effort_levels" not in entry


def test_transform_max_tokens_mirrors_output_cap_not_context(sync_module):
    entry = sync_module.transform_friendli_data([_reasoning_model()], {})[
        "friendliai/zai-org/GLM-Test"
    ]
    assert entry["max_input_tokens"] == 1048576
    assert entry["max_output_tokens"] == 131072
    assert entry["max_tokens"] == entry["max_output_tokens"]


def test_transform_prompt_caching_follows_cache_pricing(sync_module):
    cached = sync_module.transform_friendli_data([_reasoning_model()], {})[
        "friendliai/zai-org/GLM-Test"
    ]
    assert cached["supports_prompt_caching"] is True
    assert cached["cache_read_input_token_cost"] == 3e-08

    uncached_model = _reasoning_model(pricing={"input": "0.00000014", "output": "0.0000004"})
    uncached = sync_module.transform_friendli_data([uncached_model], {})[
        "friendliai/zai-org/GLM-Test"
    ]
    assert uncached["supports_prompt_caching"] is False
    assert "cache_read_input_token_cost" not in uncached


def test_transform_modalities_set_vision_image_and_video_flags(sync_module):
    entry = sync_module.transform_friendli_data([_reasoning_model()], {})[
        "friendliai/zai-org/GLM-Test"
    ]
    assert entry["supports_vision"] is True
    assert entry["supports_image_input"] is True
    assert entry["supports_video_input"] is True

    text_only = _reasoning_model(input_modalities=["text"])
    entry_text = sync_module.transform_friendli_data([text_only], {})[
        "friendliai/zai-org/GLM-Test"
    ]
    assert entry_text["supports_vision"] is False
    assert entry_text["supports_image_input"] is False
    assert entry_text["supports_video_input"] is False


def test_transform_skips_rows_without_valid_token_prices_so_priced_local_entries_survive(sync_module):
    local = {
        "friendliai/zai-org/GLM-Test": {
            "litellm_provider": "friendliai",
            "input_cost_per_token": 1.5e-07,
            "output_cost_per_token": 5e-07,
        }
    }
    unpriced_rows = [
        _reasoning_model(pricing={}),
        _reasoning_model(pricing=None),
        _reasoning_model(pricing={"input": "0.00000015"}),
        _reasoning_model(pricing={"output": "0.0000005"}),
        _reasoning_model(pricing={"input": "not-a-number", "output": "0.0000005"}),
        _reasoning_model(pricing={"input": "-0.00000015", "output": "0.0000005"}),
        _reasoning_model(pricing={"input": "inf", "output": "0.0000005"}),
        _reasoning_model(pricing={"input": "nan", "output": "0.0000005"}),
    ]
    remote = sync_module.transform_friendli_data(unpriced_rows, local)
    assert remote == {}
    sync_module.sync_local_data_with_remote(local, remote, replace_keys=frozenset(remote))
    assert local["friendliai/zai-org/GLM-Test"]["input_cost_per_token"] == 1.5e-07
    assert local["friendliai/zai-org/GLM-Test"]["output_cost_per_token"] == 5e-07


def test_transform_keeps_zero_priced_rows(sync_module):
    free_model = _reasoning_model(pricing={"input": "0", "output": "0"})
    entry = sync_module.transform_friendli_data([free_model], {})["friendliai/zai-org/GLM-Test"]
    assert entry["input_cost_per_token"] == 0.0
    assert entry["output_cost_per_token"] == 0.0


def test_transforms_survive_failed_fetch(sync_module):
    assert sync_module.transform_friendli_data(None, {}) == {}
    assert sync_module.transform_friendli_data([], {}) == {}
    assert sync_module.transform_openrouter_data(None) == {}
    assert sync_module.transform_vercel_ai_gateway_data(None) == {}


def test_vercel_transform_skips_rows_without_token_pricing_or_limits(sync_module):
    rows = [
        {
            "id": "wan-video",
            "pricing": {"video_duration_pricing": [{"resolution": "720p", "cost_per_second": "0.1"}]},
        },
        {
            "id": "qwen3-embedding",
            "context_window": 32768,
            "max_tokens": 32768,
            "pricing": {"input": "0.00000001"},
        },
        {
            "id": "no-limits-chat",
            "pricing": {"input": "0.000001", "output": "0.000002"},
        },
        {
            "id": "good-chat",
            "context_window": 128000,
            "max_tokens": 8192,
            "pricing": {"input": "0.000001", "output": "0.000002"},
        },
    ]
    transformed = sync_module.transform_vercel_ai_gateway_data(rows)
    assert list(transformed) == ["vercel_ai_gateway/good-chat"]
    assert transformed["vercel_ai_gateway/good-chat"]["input_cost_per_token"] == 1e-06
    assert transformed["vercel_ai_gateway/good-chat"]["output_cost_per_token"] == 2e-06


def test_sync_replaces_friendli_entries_so_dropped_cache_pricing_does_not_survive(sync_module):
    local = {
        "friendliai/zai-org/GLM-Test": {
            "litellm_provider": "friendliai",
            "cache_read_input_token_cost": 3e-08,
            "supports_prompt_caching": True,
        }
    }
    uncached_model = _reasoning_model(pricing={"input": "0.00000014", "output": "0.0000004"})
    remote = sync_module.transform_friendli_data([uncached_model], local)
    sync_module.sync_local_data_with_remote(local, remote, replace_keys=frozenset(remote))
    synced = local["friendliai/zai-org/GLM-Test"]
    assert "cache_read_input_token_cost" not in synced
    assert synced["supports_prompt_caching"] is False


def test_sync_still_merges_entries_outside_replace_keys(sync_module):
    local = {"openrouter/some-model": {"input_cost_per_token": 1e-06, "supports_vision": True}}
    remote = {"openrouter/some-model": {"input_cost_per_token": 2e-06}}
    sync_module.sync_local_data_with_remote(local, remote)
    assert local["openrouter/some-model"] == {"input_cost_per_token": 2e-06, "supports_vision": True}


def test_transform_inherits_allowlisted_keys_from_base_model_entry(sync_module):
    local = {
        "zhipuai/glm-test": {
            "supports_pdf_input": True,
            "supports_assistant_prefill": True,
            "input_cost_per_token": 9e-06,
        }
    }
    entry = sync_module.transform_friendli_data([_reasoning_model()], local)[
        "friendliai/zai-org/GLM-Test"
    ]
    assert entry["supports_pdf_input"] is True
    assert entry["supports_assistant_prefill"] is True
    assert entry["input_cost_per_token"] == 1.5e-07


# --- DotHoosh -----------------------------------------------------------------------------
# Trimmed from GET https://gateway.dothoosh.com/v1/models (fixture values, real shape).


def _dothoosh_row(**overrides: object) -> dict:
    row = {
        "id": "gpt-5",
        "context_length": 400000,
        "top_provider": {"context_length": 400000, "max_completion_tokens": 128000},
        "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]},
        "pricing": {
            "prompt": "0.00000125",
            "completion": "0.00001",
            "input_cache_read": "0.000000125",
            "input_cache_write": "0.0000015625",
            "input_cache_write_1h": "0.0000025",
            "request": "0",
        },
        "pricing_tiers": [
            {"above_input_tokens": 272000, "prompt": "0.0000025", "completion": "0.000015"},
            {"above_input_tokens": 128000, "prompt": "0.000002"},
        ],
        "supported_parameters": ["max_tokens", "reasoning", "tool_choice", "tools"],
        "protocols": [
            {"protocol": "openai-chat-completions"},
            {"protocol": "openai-responses"},
            {"protocol": "anthropic-messages"},
        ],
    }
    row.update(overrides)
    return row


def test_dothoosh_transform_maps_prices_limits_and_flags(sync_module):
    entry = sync_module.transform_dothoosh_data([_dothoosh_row()])["dothoosh/gpt-5"]
    assert entry["litellm_provider"] == "dothoosh"
    assert entry["mode"] == "chat"
    assert entry["input_cost_per_token"] == 1.25e-06
    assert entry["output_cost_per_token"] == 1e-05
    assert entry["cache_read_input_token_cost"] == 1.25e-07
    assert entry["cache_creation_input_token_cost"] == 1.5625e-06
    assert entry["cache_creation_input_token_cost_above_1hr"] == 2.5e-06
    assert entry["supports_prompt_caching"] is True
    assert entry["max_input_tokens"] == 400000
    assert entry["max_output_tokens"] == entry["max_tokens"] == 128000
    assert entry["supports_function_calling"] is True
    assert entry["supports_tool_choice"] is True
    assert entry["supports_reasoning"] is True
    assert entry["supports_vision"] is True
    assert entry["supported_endpoints"] == ["/v1/chat/completions", "/v1/responses"]
    assert entry["source"] == "https://www.dothoosh.com/gateway/v1/models"


def test_dothoosh_tiers_use_litellm_keys_and_carry_unchanged_prices_forward(sync_module):
    entry = sync_module.transform_dothoosh_data([_dothoosh_row()])["dothoosh/gpt-5"]
    # given out of order; 128k omits completion/cache, 272k omits cache
    assert entry["input_cost_per_token_above_128k_tokens"] == 2e-06
    assert entry["output_cost_per_token_above_128k_tokens"] == 1e-05
    assert entry["cache_read_input_token_cost_above_128k_tokens"] == 1.25e-07
    assert entry["input_cost_per_token_above_272k_tokens"] == 2.5e-06
    assert entry["output_cost_per_token_above_272k_tokens"] == 1.5e-05
    assert entry["cache_read_input_token_cost_above_272k_tokens"] == 1.25e-07


def test_dothoosh_tier_threshold_not_a_multiple_of_1000_uses_plain_token_key(sync_module):
    row = _dothoosh_row(pricing_tiers=[{"above_input_tokens": 100500, "prompt": "0.000003"}])
    entry = sync_module.transform_dothoosh_data([row])["dothoosh/gpt-5"]
    assert entry["input_cost_per_token_above_100500_tokens"] == 3e-06


def test_dothoosh_responses_only_model_and_no_cache_or_tools(sync_module):
    row = _dothoosh_row(
        id="resp-only",
        pricing={"prompt": "0.000001", "completion": "0.000002"},
        pricing_tiers=None,
        supported_parameters=["max_tokens"],
        architecture={"input_modalities": ["text"], "output_modalities": ["text"]},
        protocols=[{"protocol": "openai-responses"}],
    )
    entry = sync_module.transform_dothoosh_data([row])["dothoosh/resp-only"]
    assert entry["mode"] == "responses"
    assert entry["supported_endpoints"] == ["/v1/responses"]
    assert entry["supports_prompt_caching"] is False
    assert "cache_read_input_token_cost" not in entry
    assert entry["supports_function_calling"] is False
    assert entry["supports_reasoning"] is False
    assert entry["supports_vision"] is False
    assert not any("_above_" in key for key in entry)


def test_dothoosh_row_without_protocols_is_a_chat_model(sync_module):
    row = _dothoosh_row(protocols=None)
    entry = sync_module.transform_dothoosh_data([row])["dothoosh/gpt-5"]
    assert entry["mode"] == "chat"
    assert entry["supported_endpoints"] == ["/v1/chat/completions"]


def test_dothoosh_transform_skips_unpriced_or_unsized_rows(sync_module):
    rows = [
        _dothoosh_row(pricing=None),
        _dothoosh_row(pricing={"prompt": "0.000001"}),
        _dothoosh_row(pricing={"prompt": "free", "completion": "0.000001"}),
        _dothoosh_row(pricing={"prompt": "-1", "completion": "0.000001"}),
        _dothoosh_row(context_length=None, top_provider={}),
        _dothoosh_row(id=""),
    ]
    assert sync_module.transform_dothoosh_data(rows) == {}
    assert sync_module.transform_dothoosh_data(None) == {}
    assert sync_module.transform_dothoosh_data([]) == {}


def test_dothoosh_sync_replaces_entries_so_stale_fields_do_not_survive(sync_module):
    stale = {"dothoosh/gpt-5": {"cache_read_input_token_cost": 9.9e-05, "supports_vision": False, "legacy": 1}}
    remote = sync_module.transform_dothoosh_data(
        [_dothoosh_row(pricing={"prompt": "0.000001", "completion": "0.000002"}, pricing_tiers=None)]
    )
    sync_module.sync_local_data_with_remote(stale, remote, replace_keys=frozenset(remote))
    assert "cache_read_input_token_cost" not in stale["dothoosh/gpt-5"]
    assert "legacy" not in stale["dothoosh/gpt-5"]


def test_dothoosh_retired_models_are_dropped_only_for_a_full_list(sync_module):
    remote = sync_module.transform_dothoosh_data([_dothoosh_row(id=f"m{i}") for i in range(5)])
    local = {"dothoosh/m0": {}, "dothoosh/gone": {}, "openrouter/x": {}, "dothoosh-other/y": {}}
    assert sync_module.drop_retired_dothoosh_entries(local, remote) == ["dothoosh/gone"]
    assert set(local) == {"dothoosh/m0", "openrouter/x", "dothoosh-other/y"}

    short = sync_module.transform_dothoosh_data([_dothoosh_row(id="m0")])
    local = {"dothoosh/m0": {}, "dothoosh/gone": {}}
    assert sync_module.drop_retired_dothoosh_entries(local, short) == []
    assert set(local) == {"dothoosh/m0", "dothoosh/gone"}
    assert sync_module.drop_retired_dothoosh_entries(local, {}) == []
