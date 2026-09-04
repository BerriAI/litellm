"""
Tests for the Cognition provider identity.

Cognition serves an OpenAI-compatible /v1/chat/completions surface, but it must resolve to its
own `cognition` provider so OpenAI-specific pricing and provider-level reporting never apply to
its traffic.
"""

import json
from pathlib import Path

import pytest

import litellm


class TestCognitionProviderIdentity:
    def test_cognition_is_a_registered_provider(self):
        from litellm import LlmProviders

        assert LlmProviders.COGNITION.value == "cognition"
        assert "cognition" in litellm.provider_list

    def test_cognition_json_config(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        cognition = JSONProviderRegistry.get("cognition")
        assert cognition is not None
        assert cognition.base_url == "https://api.cognition.ai/v1"
        assert cognition.api_key_env == "COGNITION_API_KEY"
        assert cognition.api_base_env == "COGNITION_API_BASE"

    def test_cognition_in_openai_compatible_providers(self):
        from litellm.constants import openai_compatible_providers

        assert "cognition" in openai_compatible_providers

    def test_prefixed_model_resolves_to_cognition_not_openai(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, _, api_base = get_llm_provider(
            model="cognition/swe-1.7",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "swe-1.7"
        assert provider == "cognition"
        assert api_base == "https://api.cognition.ai/v1"

    def test_explicit_api_base_and_key_win(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        _, provider, api_key, api_base = get_llm_provider(
            model="cognition/swe-1.7",
            custom_llm_provider=None,
            api_base="https://cognition.internal.example/v1",
            api_key="sk-test",
        )

        assert provider == "cognition"
        assert api_base == "https://cognition.internal.example/v1"
        assert api_key == "sk-test"

    def test_api_base_autodetects_cognition(self, monkeypatch: pytest.MonkeyPatch):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        monkeypatch.setenv("COGNITION_API_KEY", "sk-cognition-env")

        _, provider, api_key, api_base = get_llm_provider(
            model="swe-1.7",
            custom_llm_provider=None,
            api_base="https://api.cognition.ai/v1",
            api_key=None,
        )

        assert provider == "cognition"
        assert api_base == "https://api.cognition.ai/v1"
        assert api_key == "sk-cognition-env"

    def test_autodetected_api_base_keeps_the_caller_api_key(self, monkeypatch: pytest.MonkeyPatch):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        monkeypatch.setenv("COGNITION_API_KEY", "sk-cognition-env")

        _, provider, api_key, _ = get_llm_provider(
            model="swe-1.7",
            custom_llm_provider=None,
            api_base="https://api.cognition.ai/v1",
            api_key="sk-cognition-caller",
        )

        assert provider == "cognition"
        assert api_key == "sk-cognition-caller"

    def test_env_api_key_is_read_from_cognition_variable(self, monkeypatch: pytest.MonkeyPatch):
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("COGNITION_API_KEY", "sk-cognition-env")

        provider = JSONProviderRegistry.get("cognition")
        assert provider is not None

        api_base, api_key = create_config_class(provider)()._get_openai_compatible_provider_info(None, None)
        assert api_base == "https://api.cognition.ai/v1"
        assert api_key == "sk-cognition-env"


class TestCognitionCostTracking:
    @pytest.mark.parametrize(
        "model, input_cost, output_cost, cache_read_cost",
        [
            ("cognition/swe-1.6", 5e-07, 2.5e-06, 2e-07),
            ("cognition/swe-1.7", 5e-07, 2.5e-06, 2e-07),
            ("cognition/swe-1.7-lightning", 2.5e-06, 1.25e-05, 1e-06),
        ],
    )
    def test_cost_map_entries(self, model: str, input_cost: float, output_cost: float, cache_read_cost: float):
        info = litellm.get_model_info(model=model)

        assert info["litellm_provider"] == "cognition"
        assert info["mode"] == "chat"
        assert info["input_cost_per_token"] == input_cost
        assert info["output_cost_per_token"] == output_cost
        assert info["cache_read_input_token_cost"] == cache_read_cost

    @pytest.mark.parametrize(
        "model, expected_prompt_cost, expected_completion_cost",
        [
            ("cognition/swe-1.7", 0.5, 2.5),
            ("cognition/swe-1.7-lightning", 2.5, 12.5),
        ],
    )
    def test_cost_differs_from_openai_pricing(
        self, model: str, expected_prompt_cost: float, expected_completion_cost: float
    ):
        """A cognition-prefixed model must never be priced off an OpenAI cost entry."""
        from litellm.cost_calculator import cost_per_token

        prompt_cost, completion_cost = cost_per_token(
            model=model,
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
            custom_llm_provider="cognition",
        )

        assert prompt_cost == pytest.approx(expected_prompt_cost)
        assert completion_cost == pytest.approx(expected_completion_cost)

    def test_lightning_is_five_times_the_standard_tier(self):
        standard = litellm.get_model_info(model="cognition/swe-1.7")
        lightning = litellm.get_model_info(model="cognition/swe-1.7-lightning")

        assert lightning["input_cost_per_token"] == pytest.approx(standard["input_cost_per_token"] * 5)
        assert lightning["output_cost_per_token"] == pytest.approx(standard["output_cost_per_token"] * 5)

    def test_supported_endpoints_matrix(self):
        matrix = json.loads((Path(litellm.__file__).parent / "provider_endpoints_support_backup.json").read_text())

        endpoints = matrix["providers"]["cognition"]["endpoints"]
        assert endpoints["chat_completions"] is True
        assert endpoints["messages"] is True
        assert endpoints["responses"] is True
        assert endpoints["embeddings"] is False


class TestCognitionRouting:
    @pytest.mark.asyncio
    async def test_router_spend_is_attributed_to_cognition_pricing(self):
        """Routed traffic is costed off the cognition entry, not an OpenAI one."""
        from litellm import Router

        router = Router(
            model_list=[
                {
                    "model_name": "swe",
                    "litellm_params": {"model": "cognition/swe-1.7", "api_key": "sk-test"},
                }
            ]
        )

        response = await router.acompletion(
            model="swe",
            messages=[{"role": "user", "content": "hi"}],
            mock_response="hello from swe",
        )

        usage = response.usage
        expected = usage.prompt_tokens * 5e-07 + usage.completion_tokens * 2.5e-06
        assert response._hidden_params["response_cost"] == pytest.approx(expected)

    @pytest.mark.asyncio
    async def test_router_spend_uses_the_lightning_entry_for_lightning(self):
        """The Lightning tier is its own model, costed off its own entry."""
        from litellm import Router

        router = Router(
            model_list=[
                {
                    "model_name": "swe-lightning",
                    "litellm_params": {"model": "cognition/swe-1.7-lightning", "api_key": "sk-test"},
                }
            ]
        )

        response = await router.acompletion(
            model="swe-lightning",
            messages=[{"role": "user", "content": "hi"}],
            mock_response="hello from swe lightning",
        )

        usage = response.usage
        expected = usage.prompt_tokens * 2.5e-06 + usage.completion_tokens * 1.25e-05
        assert response._hidden_params["response_cost"] == pytest.approx(expected)
