"""
Tests for The Grid (thegrid) JSON-configured OpenAI-compatible provider.

Follows the pattern of other openai_like JSON providers (LibertAI, Darkbloom, PublicAI):
registry lookup, provider resolution, URL construction, and cost-map presence.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


class TestTheGridProvider:
    def test_thegrid_json_config_exists(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("thegrid")
        cfg = JSONProviderRegistry.get("thegrid")
        assert cfg is not None
        assert cfg.base_url == "https://api.thegrid.ai/v1"
        assert cfg.api_key_env == "THEGRID_API_KEY"
        assert "/v1/chat/completions" in cfg.supported_endpoints
        assert "/v1/responses" in cfg.supported_endpoints

    def test_thegrid_supports_responses_api(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.supports_responses_api("thegrid") is True

    def test_thegrid_provider_resolution(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, _api_key, api_base = get_llm_provider(
            model="thegrid/agent-max",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "agent-max"
        assert provider == "thegrid"
        assert api_base == "https://api.thegrid.ai/v1"

    def test_thegrid_dynamic_config_and_url(self):
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("thegrid")
        config = create_config_class(provider)()

        api_base, api_key = config._get_openai_compatible_provider_info(None, None)
        assert api_base == "https://api.thegrid.ai/v1"

        api_base, api_key = config._get_openai_compatible_provider_info(
            "https://custom.example.com/v1", "test-key"
        )
        assert api_base == "https://custom.example.com/v1"
        assert api_key == "test-key"

        url = config.get_complete_url(
            api_base="https://api.thegrid.ai/v1",
            api_key="test-key",
            model="thegrid/agent-max",
            optional_params={},
            litellm_params={},
            stream=False,
        )
        assert url == "https://api.thegrid.ai/v1/chat/completions"

    def test_thegrid_model_cost_map(self):
        root = Path(__file__).resolve().parents[4]
        cost_path = root / "model_prices_and_context_window.json"
        # Fallback when tests run from an installed package layout.
        if not cost_path.exists():
            cost_path = Path(os.getcwd()) / "model_prices_and_context_window.json"
        with open(cost_path) as f:
            model_cost = json.load(f)

        expected_keys = [
        "thegrid/agent-max",
        "thegrid/agent-prime",
        "thegrid/agent-standard",
        "thegrid/code-max",
        "thegrid/code-prime",
        "thegrid/code-standard",
        "thegrid/text-max",
        "thegrid/text-prime",
        "thegrid/text-standard",
        ]
        for key in expected_keys:
            assert key in model_cost, f"missing cost-map entry {key}"
            info = model_cost[key]
            assert info["litellm_provider"] == "thegrid"
            assert info["mode"] == "chat"
            assert info["max_input_tokens"] > 0
            assert info["max_output_tokens"] > 0
            # Uniform pricing: input == output == cache-read.
            assert info["input_cost_per_token"] == info["output_cost_per_token"]
            assert info["input_cost_per_token"] == info["cache_read_input_token_cost"]

        sample = model_cost["thegrid/agent-max"]
        assert sample["max_input_tokens"] == 922000
        assert sample["max_output_tokens"] == 128000
        assert sample["supports_function_calling"] is True
