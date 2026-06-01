"""
Tests for Eden AI provider configuration and integration.

Eden AI (https://www.edenai.co/) is an OpenAI-compatible aggregator that
exposes 100+ underlying models from OpenAI, Anthropic, Google, Mistral and
others through a single chat-completions endpoint.
"""

import os

import pytest

import litellm


class TestEdenAIProviderConfig:
    """Test Eden AI provider configuration"""

    def test_edenai_in_provider_list(self):
        """Test that edenai is in the provider list"""
        from litellm import LlmProviders

        assert hasattr(LlmProviders, "EDENAI")
        assert LlmProviders.EDENAI.value == "edenai"
        assert "edenai" in litellm.provider_list

    def test_edenai_json_config_exists(self):
        """Test that edenai is configured in providers.json"""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("edenai")

        edenai = JSONProviderRegistry.get("edenai")
        assert edenai is not None
        assert edenai.base_url == "https://api.edenai.run/v3"
        assert edenai.api_key_env == "EDENAI_API_KEY"
        assert edenai.api_base_env == "EDENAI_API_BASE"
        assert edenai.param_mappings.get("max_completion_tokens") == "max_tokens"

    def test_edenai_provider_resolution(self):
        """Test that provider resolution finds edenai for prefixed model ids"""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="edenai/openai/gpt-4o-mini",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        # Eden AI's model ids are themselves prefixed (e.g. openai/gpt-4o-mini),
        # so the model returned is everything after the leading "edenai/".
        assert model == "openai/gpt-4o-mini"
        assert provider == "edenai"
        assert api_base == "https://api.edenai.run/v3"

    def test_edenai_api_base_env_override(self, monkeypatch):
        """Test that EDENAI_API_BASE overrides the default base URL"""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        monkeypatch.setenv("EDENAI_API_BASE", "https://custom-edenai.example.com/v3")
        _, provider, _, api_base = get_llm_provider(
            model="edenai/openai/gpt-4o-mini",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )
        assert provider == "edenai"
        assert api_base == "https://custom-edenai.example.com/v3"

    def test_edenai_router_config(self):
        """Test that edenai can be used in Router configuration"""
        from litellm import Router

        router = Router(
            model_list=[
                {
                    "model_name": "edenai-gpt-4o-mini",
                    "litellm_params": {
                        "model": "edenai/openai/gpt-4o-mini",
                        "api_key": "test-key",
                    },
                }
            ]
        )

        assert len(router.model_list) == 1
        assert router.model_list[0]["model_name"] == "edenai-gpt-4o-mini"


class TestEdenAICompletionMocked:
    """Mocked completion tests — no real API calls."""

    def test_edenai_completion_mocked(self):
        """Test that a mocked completion through the edenai/ prefix routes
        correctly and returns the stub content unchanged."""
        response = litellm.completion(
            model="edenai/openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "Say PING and nothing else"}],
            api_key="test-key",
            mock_response="PING",
        )

        assert response is not None
        assert hasattr(response, "choices")
        assert len(response.choices) > 0
        assert response.choices[0].message.content == "PING"


class TestEdenAILiveIntegration:
    """Live integration tests against the real Eden AI API.

    Skipped automatically when EDENAI_API_KEY is not set, so CI without the
    secret stays green. When the key is present (local dev, or a dedicated
    integration runner) these validate that each endpoint flagged `true` in
    provider_endpoints_support.json actually routes a request through to
    api.edenai.run/v3 end-to-end. Endpoints flagged `false` (`/responses`,
    `/embeddings`, `/image/generations`, `/audio/*`, `/moderations`,
    `/batches`, `/rerank`) intentionally have no live test here; their
    handlers ship in follow-up PRs and the tests come with them.
    """

    @pytest.mark.skipif(
        not os.environ.get("EDENAI_API_KEY"), reason="EDENAI_API_KEY not set"
    )
    def test_edenai_live_chat_completions(self):
        """`/v1/chat/completions` via litellm.completion() — the openai_like
        loader's primary surface."""
        response = litellm.completion(
            model="edenai/openai/gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": "Reply with 'PING' and nothing else.",
                }
            ],
            max_tokens=8,
        )

        assert response is not None
        assert response.choices[0].message.content
        assert response.usage is not None
        assert response.usage.total_tokens > 0

    @pytest.mark.skipif(
        not os.environ.get("EDENAI_API_KEY"), reason="EDENAI_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_edenai_live_anthropic_messages(self):
        """`/v1/messages` (Anthropic-format) via litellm.anthropic_messages().
        LiteLLM translates the Anthropic-shape request to OpenAI shape
        internally and routes through Eden AI's openai-compatible endpoint;
        the response is translated back to Anthropic shape before being
        returned to the caller."""
        response = await litellm.anthropic_messages(
            model="edenai/anthropic/claude-opus-4-6",
            messages=[
                {
                    "role": "user",
                    "content": "Reply with 'MSG' and nothing else.",
                }
            ],
            max_tokens=8,
        )

        assert response is not None
        # anthropic_messages returns a dict (Anthropic-shape response)
        assert response.get("type") == "message"
        assert response.get("role") == "assistant"
        content = response.get("content", [])
        assert content and content[0].get("type") == "text"
        assert content[0].get("text")
        usage = response.get("usage", {})
        assert usage.get("total_tokens", 0) > 0
