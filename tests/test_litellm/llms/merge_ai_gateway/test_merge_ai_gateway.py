"""
Mock tests for merge_ai_gateway provider
"""

import os
from unittest.mock import MagicMock, patch

import litellm
from litellm.llms.merge_ai_gateway.chat.transformation import (
    DEFAULT_API_BASE,
    MergeAIGatewayConfig,
)


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


def test_merge_ai_gateway_models_endpoint():
    """get_models probes {api_base}/models and returns catalog ids."""
    config = MergeAIGatewayConfig()

    with patch("litellm.module_level_client.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"id": "anthropic/claude-opus-4-6"},
                {"id": "openai/gpt-5"},
            ]
        }
        mock_get.return_value = mock_response

        models = config.get_models(api_key="sk-merge-test")

    assert models == ["anthropic/claude-opus-4-6", "openai/gpt-5"]
    assert mock_get.call_args.kwargs["url"] == f"{DEFAULT_API_BASE}/models"
    assert mock_get.call_args.kwargs["headers"] == {"Authorization": "Bearer sk-merge-test"}


def test_merge_ai_gateway_get_valid_models_uses_live_catalog():
    """get_valid_models(check_provider_endpoint=True) hits the live catalog."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": [{"id": "anthropic/claude-opus-4-6"}]}

    with patch("litellm.module_level_client.get", return_value=mock_response) as mock_get:
        models = litellm.get_valid_models(
            check_provider_endpoint=True,
            custom_llm_provider="merge_ai_gateway",
            api_key="sk-merge-test",
        )

    assert models == ["anthropic/claude-opus-4-6"]
    assert mock_get.call_args.kwargs["url"] == f"{DEFAULT_API_BASE}/models"
