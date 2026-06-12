"""
Tests for DashScope (Alibaba Qwen) Responses API support.

Feature: https://github.com/BerriAI/litellm/issues/29780
DashScope exposes an OpenAI-compatible /responses endpoint. These tests pin the
two guarantees the issue asks for: dashscope/* responses calls route to
{api_base}/responses, and the upstream model id is passed through unchanged.
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

import litellm
from litellm.llms.dashscope.responses.transformation import (
    DashScopeResponsesAPIConfig,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

DASHSCOPE_RESPONSES_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/responses"


def _make_mock_responses_api_response() -> dict:
    return {
        "id": "resp-test123",
        "object": "response",
        "created_at": 1234567890,
        "model": "qwen-max",
        "output": [
            {
                "type": "message",
                "id": "msg-test123",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Hello from Qwen",
                        "annotations": [],
                    }
                ],
            }
        ],
        "status": "completed",
        "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
    }


def _make_mock_http_client(response_body: dict) -> MagicMock:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = response_body
    mock_response.text = json.dumps(response_body)
    mock_client.post.return_value = mock_response
    return mock_client


def _extract_posted_url_and_body(mock_client: MagicMock):
    call = mock_client.post.call_args
    url = call.kwargs.get("url") or (call.args[0] if call.args else None)
    raw_body = call.kwargs.get("data")
    if raw_body is None:
        raw_body = call.kwargs.get("json")
    body = json.loads(raw_body) if isinstance(raw_body, str) else raw_body
    return url, body


def test_dashscope_responses_routes_to_responses_endpoint_without_model_rewrite():
    """End-to-end: dashscope/* responses call hits {base}/responses and sends the
    upstream model id ('qwen-max') unchanged. This is the core of issue #29780."""
    mock_client = _make_mock_http_client(_make_mock_responses_api_response())

    with patch(
        "litellm.llms.custom_httpx.llm_http_handler._get_httpx_client",
        return_value=mock_client,
    ):
        response = litellm.responses(
            model="dashscope/qwen-max",
            input="Hello, how are you?",
            api_key="test-key",
        )

    url, body = _extract_posted_url_and_body(mock_client)
    assert url == DASHSCOPE_RESPONSES_URL
    assert body["model"] == "qwen-max"

    from litellm.types.llms.openai import ResponsesAPIResponse

    assert isinstance(response, ResponsesAPIResponse)
    assert response.output[0].content[0].text == "Hello from Qwen"  # type: ignore[union-attr]


def test_dashscope_provider_config_registration():
    """ProviderConfigManager must return the DashScope responses config so the call
    routes natively instead of falling back to chat-completions emulation."""
    config = ProviderConfigManager.get_provider_responses_api_config(
        model="dashscope/qwen-max",
        provider=LlmProviders.DASHSCOPE,
    )

    assert isinstance(config, DashScopeResponsesAPIConfig)
    assert config.custom_llm_provider == LlmProviders.DASHSCOPE


def test_dashscope_responses_api_url():
    config = DashScopeResponsesAPIConfig()

    assert config.get_complete_url(api_base=None, litellm_params={}) == (
        DASHSCOPE_RESPONSES_URL
    )
    assert (
        config.get_complete_url(
            api_base="https://dashscope.aliyuncs.com/compatible-mode/v1/",
            litellm_params={},
        )
        == DASHSCOPE_RESPONSES_URL
    )
    assert (
        config.get_complete_url(
            api_base="https://proxy.internal/api", litellm_params={}
        )
        == "https://proxy.internal/api/v1/responses"
    )


def test_dashscope_responses_api_url_uses_env_base(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_BASE", "https://intl.example/compatible-mode/v1")
    config = DashScopeResponsesAPIConfig()

    assert config.get_complete_url(api_base=None, litellm_params={}) == (
        "https://intl.example/compatible-mode/v1/responses"
    )


def test_dashscope_validate_environment_sets_bearer_from_param():
    config = DashScopeResponsesAPIConfig()

    headers = config.validate_environment(
        headers={},
        model="qwen-max",
        litellm_params=GenericLiteLLMParams(api_key="sk-dashscope"),
    )

    assert headers["Authorization"] == "Bearer sk-dashscope"


def test_dashscope_validate_environment_reads_env_key(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-from-env")
    config = DashScopeResponsesAPIConfig()

    headers = config.validate_environment(
        headers={}, model="qwen-max", litellm_params=GenericLiteLLMParams()
    )

    assert headers["Authorization"] == "Bearer sk-from-env"


def test_dashscope_validate_environment_requires_api_key(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    config = DashScopeResponsesAPIConfig()

    with pytest.raises(ValueError, match="DashScope API key not set"):
        config.validate_environment(
            headers={}, model="qwen-max", litellm_params=GenericLiteLLMParams()
        )
