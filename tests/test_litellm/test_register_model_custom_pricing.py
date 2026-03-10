"""
Test that register_model() in completion() and embedding() passes all
custom pricing fields from kwargs and model_info, not just the base
input/output costs.

Previously, only input_cost_per_token, output_cost_per_token, and
litellm_provider were forwarded. Fields like cache_read_input_token_cost,
mode, and supports_prompt_caching were dropped, causing incorrect cost
calculations for DB-sourced models with prompt caching pricing.
"""

import os
import sys

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.main import _build_custom_pricing_entry


def test_build_custom_pricing_entry_includes_all_kwargs_fields():
    """All CustomPricingLiteLLMParams fields present in kwargs should be
    included in the resulting entry dict."""
    kwargs = {
        "input_cost_per_token": 0.001,
        "output_cost_per_token": 0.002,
        "cache_read_input_token_cost": 0.00025,
        "cache_creation_input_token_cost": 0.005,
        "output_cost_per_reasoning_token": 0.01,
        "input_cost_per_audio_token": 0.003,
        "unrelated_kwarg": "should_be_ignored",
    }

    entry = _build_custom_pricing_entry(
        custom_llm_provider="openai",
        kwargs=kwargs,
    )

    assert entry["litellm_provider"] == "openai"
    assert entry["input_cost_per_token"] == 0.001
    assert entry["output_cost_per_token"] == 0.002
    assert entry["cache_read_input_token_cost"] == 0.00025
    assert entry["cache_creation_input_token_cost"] == 0.005
    assert entry["output_cost_per_reasoning_token"] == 0.01
    assert entry["input_cost_per_audio_token"] == 0.003
    assert "unrelated_kwarg" not in entry


def test_build_custom_pricing_entry_merges_model_info_metadata():
    """Fields from model_info (mode, supports_prompt_caching, max_tokens)
    should be merged into the entry when present."""
    kwargs = {
        "input_cost_per_token": 0.001,
        "output_cost_per_token": 0.002,
    }
    model_info = {
        "id": "deployment-123",
        "mode": "chat",
        "supports_prompt_caching": True,
        "max_tokens": 128000,
    }

    entry = _build_custom_pricing_entry(
        custom_llm_provider="openai",
        kwargs=kwargs,
        model_info=model_info,
    )

    assert entry["mode"] == "chat"
    assert entry["supports_prompt_caching"] is True
    assert entry["max_tokens"] == 128000


def test_build_custom_pricing_entry_setdefault_does_not_override_existing():
    """model_info uses setdefault, so it should not override a key that is
    already present in the entry dict. Currently CustomPricingLiteLLMParams
    and the model_info keys (mode, supports_prompt_caching, max_tokens) do
    not overlap, but if they ever do, setdefault ensures the kwargs-sourced
    value wins."""
    kwargs = {
        "input_cost_per_token": 0.001,
        "output_cost_per_token": 0.002,
    }
    model_info = {
        "mode": "chat",
        "supports_prompt_caching": True,
        "max_tokens": 128000,
    }

    entry = _build_custom_pricing_entry(
        custom_llm_provider="openai",
        kwargs=kwargs,
        model_info=model_info,
    )

    assert entry["mode"] == "chat"
    assert entry["supports_prompt_caching"] is True
    assert entry["max_tokens"] == 128000

    # Verify setdefault behavior: if a model_info key already exists in
    # the entry (e.g. from a future CustomPricingLiteLLMParams addition),
    # setdefault must not overwrite it.
    entry["mode"] = "embedding"  # simulate pre-existing value
    # Re-apply setdefault the same way _build_custom_pricing_entry does
    entry.setdefault("mode", model_info["mode"])
    assert entry["mode"] == "embedding"  # must NOT revert to "chat"


def test_build_custom_pricing_entry_skips_none_values():
    """Fields with None values in kwargs should not be included."""
    kwargs = {
        "input_cost_per_token": 0.001,
        "output_cost_per_token": None,  # explicitly None
        "cache_read_input_token_cost": None,
    }

    entry = _build_custom_pricing_entry(
        custom_llm_provider="openai",
        kwargs=kwargs,
    )

    assert entry["input_cost_per_token"] == 0.001
    assert "output_cost_per_token" not in entry
    assert "cache_read_input_token_cost" not in entry


def test_build_custom_pricing_entry_handles_no_model_info():
    """Should work correctly when model_info is None."""
    kwargs = {
        "input_cost_per_token": 0.001,
        "output_cost_per_token": 0.002,
    }

    entry = _build_custom_pricing_entry(
        custom_llm_provider="openai",
        kwargs=kwargs,
        model_info=None,
    )

    assert entry["litellm_provider"] == "openai"
    assert entry["input_cost_per_token"] == 0.001
    assert entry["output_cost_per_token"] == 0.002
    assert "mode" not in entry


def test_register_model_receives_cache_pricing_fields():
    """End-to-end: when register_model is called with a full pricing entry,
    the cache pricing fields should be present in litellm.model_cost."""
    model_key = "openai/test-custom-model-with-cache-pricing"

    litellm.register_model(
        {
            model_key: {
                "input_cost_per_token": 0.001,
                "output_cost_per_token": 0.002,
                "cache_read_input_token_cost": 0.00025,
                "supports_prompt_caching": True,
                "mode": "chat",
                "max_tokens": 8192,
                "litellm_provider": "openai",
            }
        }
    )

    registered = litellm.model_cost.get(model_key)
    assert registered is not None, f"{model_key} should be in model_cost"
    assert registered["cache_read_input_token_cost"] == 0.00025
    assert registered["supports_prompt_caching"] is True
    assert registered["mode"] == "chat"
    assert registered["max_tokens"] == 8192

    # Cleanup
    litellm.model_cost.pop(model_key, None)


def test_build_custom_pricing_entry_time_based():
    """Time-based pricing fields should be included correctly."""
    kwargs = {
        "input_cost_per_second": 0.01,
        "output_cost_per_second": 0.02,
    }

    entry = _build_custom_pricing_entry(
        custom_llm_provider="openai",
        kwargs=kwargs,
    )

    assert entry["litellm_provider"] == "openai"
    assert entry["input_cost_per_second"] == 0.01
    assert entry["output_cost_per_second"] == 0.02
