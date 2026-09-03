"""
Tests for the AI Token King provider (JSON-configured, OpenAI-compatible gateway).
"""

import json
from pathlib import Path

import litellm

BASE_URL = "https://api.aitokenking.com.tw/api/v1"


class TestAITokenKingProviderConfig:
    """AI Token King provider configuration and resolution."""

    def test_aitokenking_in_provider_list(self):
        from litellm import LlmProviders

        assert hasattr(LlmProviders, "AITOKENKING")
        assert LlmProviders.AITOKENKING.value == "aitokenking"
        assert "aitokenking" in litellm.provider_list

    def test_aitokenking_json_config_exists(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("aitokenking")

        cfg = JSONProviderRegistry.get("aitokenking")
        assert cfg is not None
        assert cfg.base_url == BASE_URL
        assert cfg.api_key_env == "AITOKENKING_API_KEY"
        assert cfg.api_base_env == "AITOKENKING_API_BASE"
        assert cfg.param_mappings.get("max_completion_tokens") == "max_tokens"
        assert cfg.supported_endpoints == ["/v1/chat/completions"]

    def test_aitokenking_in_openai_compatible_providers(self):
        from litellm.constants import openai_compatible_endpoints, openai_compatible_providers

        assert "aitokenking" in openai_compatible_providers
        assert BASE_URL in openai_compatible_endpoints

    def test_aitokenking_provider_resolution(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="aitokenking/qwen3.7-max",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "qwen3.7-max"
        assert provider == "aitokenking"
        assert api_base == BASE_URL

    def test_aitokenking_api_base_override(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="aitokenking/qwen3.7-max",
            custom_llm_provider=None,
            api_base="https://custom.example.com/v1",
            api_key="sk-test",
        )

        assert provider == "aitokenking"
        assert api_base == "https://custom.example.com/v1"
        assert api_key == "sk-test"

    def test_aitokenking_url_autodetection(self):
        """Passing the gateway's api_base without a prefix resolves the provider."""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="qwen3.7-max",
            custom_llm_provider=None,
            api_base=BASE_URL,
            api_key=None,
        )
        assert provider == "aitokenking"
        assert api_base == BASE_URL

    def test_aitokenking_router_config(self):
        from litellm import Router

        router = Router(
            model_list=[
                {
                    "model_name": "atk-chat",
                    "litellm_params": {
                        "model": "aitokenking/qwen3.7-max",
                        "api_key": "test-key",
                    },
                }
            ]
        )

        assert len(router.model_list) == 1
        assert router.model_list[0]["model_name"] == "atk-chat"


class TestAITokenKingPricing:
    """Cost tracking is the point of shipping the price map with the provider."""

    @staticmethod
    def _load_price_map() -> dict:
        json_path = Path(__file__).parents[4] / "model_prices_and_context_window.json"
        with open(json_path) as f:
            return json.load(f)

    def test_price_map_entries_use_provider_prefix(self):
        model_cost = self._load_price_map()
        keys = [k for k in model_cost if k.startswith("aitokenking/")]
        assert len(keys) == 48
        for k in keys:
            entry = model_cost[k]
            assert entry["litellm_provider"] == "aitokenking"
            assert entry["mode"] == "chat"
            # A zero price would read as "free" rather than "unknown"; never ship one.
            assert entry["input_cost_per_token"] > 0
            assert entry["output_cost_per_token"] > 0

    def test_prefixed_keys_do_not_shadow_upstream_vendor_entries(self):
        """Some gateway model ids match vendor ids already in the map (e.g. gpt-5.5).

        The gateway resells at its own price, so the entry must live under the
        provider prefix only — a bare-id entry would overwrite the vendor's price
        for everyone.
        """
        model_cost = self._load_price_map()
        for bare in ("gpt-5.5", "claude-sonnet-5", "gemini-3.1-pro-preview"):
            assert f"aitokenking/{bare}" in model_cost
            assert model_cost[bare]["litellm_provider"] != "aitokenking"

    def test_completion_cost_resolves_for_prefixed_model(self, monkeypatch):
        from litellm.types.utils import ModelResponse

        model = "aitokenking/qwen3.7-max"
        entry = self._load_price_map()[model]
        monkeypatch.setitem(litellm.model_cost, model, entry)

        prompt_tokens, completion_tokens = 100_000, 100_000
        expected = entry["input_cost_per_token"] * prompt_tokens + entry["output_cost_per_token"] * completion_tokens

        response = ModelResponse(
            **{
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"},
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
        )

        cost = litellm.completion_cost(completion_response=response, model=model)
        assert abs(cost - expected) < 1e-9
        assert cost > 0
