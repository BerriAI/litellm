"""
Tests for GitGot provider configuration and integration.
"""

import litellm


class TestGitGotProviderConfig:
    def test_gitgot_in_provider_list(self):
        from litellm import LlmProviders

        assert hasattr(LlmProviders, "GITGOT")
        assert LlmProviders.GITGOT.value == "gitgot"
        assert "gitgot" in litellm.provider_list

    def test_gitgot_json_config_exists(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("gitgot")

        gitgot = JSONProviderRegistry.get("gitgot")
        assert gitgot is not None
        assert gitgot.base_url == "https://inference.gitgot.ai/v1"
        assert gitgot.api_key_env == "GITGOT_API_KEY"
        assert gitgot.api_base_env == "GITGOT_API_BASE"

    def test_gitgot_in_openai_compatible_providers(self):
        from litellm.constants import openai_compatible_providers

        assert "gitgot" in openai_compatible_providers

    def test_gitgot_in_openai_compatible_endpoints(self):
        from litellm.constants import openai_compatible_endpoints

        assert "https://inference.gitgot.ai/v1" in openai_compatible_endpoints

    def test_gitgot_provider_resolution(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="gitgot/openai/gpt-oss-120b",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        # GitGot model ids carry a vendor prefix, so only the leading "gitgot/"
        # is stripped and the rest of the path is preserved.
        assert model == "openai/gpt-oss-120b"
        assert provider == "gitgot"
        assert api_base == "https://inference.gitgot.ai/v1"

    def test_gitgot_api_base_override(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="gitgot/openai/gpt-oss-120b",
            custom_llm_provider=None,
            api_base="https://custom.gitgot.ai/v1",
            api_key="gg-test",
        )

        assert provider == "gitgot"
        assert api_base == "https://custom.gitgot.ai/v1"
        assert api_key == "gg-test"

    def test_gitgot_model_prices_registered(self, local_model_cost_map):
        from litellm import model_cost

        entry = model_cost["gitgot/openai/gpt-oss-120b"]
        assert entry["litellm_provider"] == "gitgot"
        assert entry["mode"] == "chat"
        # Costs in this file are per token, not per million. A per-million
        # figure here would report the provider as a million times dearer than
        # it is in every client that reads it.
        assert 0 < entry["input_cost_per_token"] < 0.001
        assert 0 < entry["output_cost_per_token"] < 0.001
        assert entry["max_input_tokens"] > 0

    def test_gitgot_cost_calculation(self, local_model_cost_map):
        from litellm import completion_cost
        from litellm.types.utils import Choices, Message, ModelResponse, Usage

        response = ModelResponse(
            id="chatcmpl-test",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(content="hello", role="assistant"),
                )
            ],
            created=1234567890,
            model="gitgot/openai/gpt-oss-120b",
            object="chat.completion",
            usage=Usage(
                prompt_tokens=1_000_000,
                completion_tokens=1_000_000,
                total_tokens=2_000_000,
            ),
        )

        cost = completion_cost(completion_response=response)
        # A million tokens each way should cost dollars, not thousands.
        assert 0 < cost < 100

    def test_gitgot_router_config(self):
        from litellm import Router

        router = Router(
            model_list=[
                {
                    "model_name": "gitgot-chat",
                    "litellm_params": {
                        "model": "gitgot/openai/gpt-oss-120b",
                        "api_key": "test-key",
                    },
                }
            ]
        )

        assert len(router.model_list) == 1
        assert router.model_list[0]["model_name"] == "gitgot-chat"

    def test_gitgot_endpoint_support_declared(self):
        import json
        from pathlib import Path

        support = json.loads(
            (
                Path(litellm.__file__).parent.parent
                / "provider_endpoints_support.json"
            ).read_text()
        )["providers"]["gitgot"]

        assert support["endpoints"]["chat_completions"] is True
        assert support["endpoints"]["embeddings"] is True
