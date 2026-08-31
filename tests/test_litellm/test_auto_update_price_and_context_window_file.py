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
