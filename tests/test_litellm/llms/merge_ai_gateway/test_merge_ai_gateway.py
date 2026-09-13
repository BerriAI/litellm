"""
Mock tests for merge_ai_gateway provider
"""

import httpx

import litellm
from litellm.llms.merge_ai_gateway.chat.transformation import (
    DEFAULT_API_BASE,
    MergeAIGatewayConfig,
)

MODELS_URL = f"{DEFAULT_API_BASE}/models"


def test_merge_ai_gateway_provider_routing():
    """merge_ai_gateway/<model> resolves provider and strips the prefix."""
    model, provider, _, _ = litellm.get_llm_provider(model="merge_ai_gateway/anthropic/claude-opus-4-6")
    assert provider == "merge_ai_gateway"
    assert model == "anthropic/claude-opus-4-6"


def test_merge_ai_gateway_default_api_base():
    """get_llm_provider fills in the default merge gateway base."""
    _, provider, _, api_base = litellm.get_llm_provider(model="merge_ai_gateway/x/y")
    assert provider == "merge_ai_gateway"
    assert api_base == DEFAULT_API_BASE


def test_merge_ai_gateway_in_provider_lists():
    assert "merge_ai_gateway" in litellm.provider_list
    assert "merge_ai_gateway" in litellm.models_by_provider


def test_merge_ai_gateway_models_endpoint(respx_mock):
    """get_models probes {api_base}/models and namespaces the catalog ids."""
    route = respx_mock.get(MODELS_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {"id": "anthropic/claude-opus-4-6"},
                    {"id": "openai/gpt-5"},
                ]
            },
        )
    )

    models = MergeAIGatewayConfig().get_models(api_key="sk-merge-test")

    assert models == [
        "merge_ai_gateway/anthropic/claude-opus-4-6",
        "merge_ai_gateway/openai/gpt-5",
    ]
    assert route.calls[0].request.headers["authorization"] == "Bearer sk-merge-test"


def test_merge_ai_gateway_get_valid_models_uses_live_catalog(respx_mock):
    """get_valid_models(check_provider_endpoint=True) hits the live catalog."""
    route = respx_mock.get(MODELS_URL).mock(
        return_value=httpx.Response(200, json={"data": [{"id": "anthropic/claude-opus-4-6"}]})
    )

    models = litellm.get_valid_models(
        check_provider_endpoint=True,
        custom_llm_provider="merge_ai_gateway",
        api_key="sk-merge-test",
    )

    assert models == ["merge_ai_gateway/anthropic/claude-opus-4-6"]
    assert len(route.calls) == 1
