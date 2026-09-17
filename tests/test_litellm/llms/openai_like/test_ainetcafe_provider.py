"""
Tests for the ainetcafe (Kimi K3) provider: request routing, parameter passthrough,
cost tracking and the consistency of its metadata across the config surfaces.
"""

import json
from pathlib import Path

import pytest
import respx
from httpx import Response

import litellm

BASE_URL = "https://microquickjs.com/v1"
MODEL = "ainetcafe/Kimi-K3"


def _chat_response(prompt_tokens=100, cached_tokens=60, completion_tokens=20):
    return {
        "id": "chatcmpl-ainetcafe-test",
        "object": "chat.completion",
        "created": 1758000000,
        "model": "Kimi-K3",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "LiteLLM OK",
                    "reasoning_content": "The user wants an exact reply.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": cached_tokens},
        },
    }


@pytest.fixture(autouse=True)
def _local_cost_map(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")


class TestAinetcafeRequests:
    @respx.mock
    def test_completion_hits_default_base_with_bearer_key_and_passes_reasoning_params(self):
        route = respx.post(f"{BASE_URL}/chat/completions").mock(return_value=Response(200, json=_chat_response()))

        response = litellm.completion(
            model=MODEL,
            messages=[{"role": "user", "content": "Reply with exactly: LiteLLM OK"}],
            api_key="sk-test-ainetcafe",
            reasoning_effort="high",
            max_completion_tokens=256,
        )

        assert route.called
        request = route.calls.last.request
        assert request.headers["authorization"] == "Bearer sk-test-ainetcafe"
        body = json.loads(request.content)
        assert body["model"] == "Kimi-K3"
        assert body["reasoning_effort"] == "high"
        assert body["max_completion_tokens"] == 256
        assert response.choices[0].message.content == "LiteLLM OK"
        assert response.choices[0].message.reasoning_content == "The user wants an exact reply."

    @respx.mock
    def test_api_key_env_and_api_base_env_are_honoured(self, monkeypatch):
        monkeypatch.setenv("AINETCAFE_API_KEY", "sk-from-env")
        monkeypatch.setenv("AINETCAFE_API_BASE", "https://dedicated.example.com/v1")
        route = respx.post("https://dedicated.example.com/v1/chat/completions").mock(
            return_value=Response(200, json=_chat_response())
        )

        litellm.completion(model=MODEL, messages=[{"role": "user", "content": "hi"}])

        assert route.called
        assert route.calls.last.request.headers["authorization"] == "Bearer sk-from-env"

    def test_base_url_autodetects_provider(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, _, api_base = get_llm_provider(
            model="Kimi-K3", custom_llm_provider=None, api_base=BASE_URL, api_key=None
        )

        assert (model, provider, api_base) == ("Kimi-K3", "ainetcafe", BASE_URL)

    @respx.mock
    def test_completion_cost_uses_cost_map_and_cached_input_rate(self):
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=Response(200, json=_chat_response(prompt_tokens=100, cached_tokens=60, completion_tokens=20))
        )
        info = litellm.get_model_info(MODEL)

        response = litellm.completion(model=MODEL, messages=[{"role": "user", "content": "hi"}], api_key="sk-test")
        cost = litellm.completion_cost(completion_response=response)

        expected = (
            40 * info["input_cost_per_token"]
            + 60 * info["cache_read_input_token_cost"]
            + 20 * info["output_cost_per_token"]
        )
        assert cost == pytest.approx(expected)
        assert 0 < info["cache_read_input_token_cost"] < info["input_cost_per_token"] < info["output_cost_per_token"]


class TestAinetcafeMetadataConsistency:
    @staticmethod
    def _load(*parts):
        with open(Path(__file__).parents[4].joinpath(*parts)) as f:
            return json.load(f)

    def test_model_entry_matches_backup_and_declares_capabilities_the_provider_serves(self):
        root = self._load("model_prices_and_context_window.json")[MODEL]
        backup = self._load("litellm", "model_prices_and_context_window_backup.json")[MODEL]

        assert root == backup
        assert root["litellm_provider"] == "ainetcafe"
        assert root["max_tokens"] == root["max_output_tokens"] <= root["max_input_tokens"]
        # Source: https://ainetcafe.com/k3/#pricing and https://ainetcafe.com/k3/verifier.html, read 2026-09-17.
        # Kimi K3 on ainetcafe reasons, calls tools, returns JSON schema output and takes image input.
        assert all(
            root[flag]
            for flag in (
                "supports_reasoning",
                "supports_function_calling",
                "supports_tool_choice",
                "supports_response_schema",
                "supports_vision",
                "supports_prompt_caching",
            )
        )

    def test_provider_surfaces_agree_on_slug_base_url_and_env_keys(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("ainetcafe")
        assert provider is not None

        create_fields = [
            e
            for e in self._load("litellm", "proxy", "public_endpoints", "provider_create_fields.json")
            if e["litellm_provider"] == "ainetcafe"
        ]
        assert len(create_fields) == 1
        api_base_field = next(f for f in create_fields[0]["credential_fields"] if f["key"] == "api_base")
        assert api_base_field["placeholder"] == provider.base_url
        assert create_fields[0]["default_model_placeholder"] == MODEL

        support = self._load("provider_endpoints_support.json")["providers"]["ainetcafe"]
        support_backup = self._load("litellm", "provider_endpoints_support_backup.json")["providers"]["ainetcafe"]
        assert support == support_backup
        assert support["endpoints"]["chat_completions"] is ("/v1/chat/completions" in provider.supported_endpoints)

        from litellm.constants import openai_compatible_endpoints, openai_compatible_providers

        assert "ainetcafe" in openai_compatible_providers
        assert provider.base_url in openai_compatible_endpoints
