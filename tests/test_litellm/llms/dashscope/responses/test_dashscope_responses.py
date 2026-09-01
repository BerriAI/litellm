"""
Tests for DashScope Responses API support (Issue #38474).
"""

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm.llms.dashscope.responses.transformation import (
    DEFAULT_DASHSCOPE_API_BASE,
    DashScopeResponsesAPIConfig,
)
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


def _make_mock_responses_api_response(content: str = "Hello from Qwen!") -> dict[str, Any]:
    return {
        "id": "resp-dashscope-123",
        "object": "response",
        "created_at": 1234567890,
        "model": "qwen3.8-flash",
        "output": [
            {
                "type": "message",
                "id": "msg-dashscope-123",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": content,
                        "annotations": [],
                    }
                ],
            }
        ],
        "status": "completed",
        "usage": {
            "input_tokens": 8,
            "output_tokens": 12,
            "total_tokens": 20,
        },
    }


def _make_mock_http_client(response_body: dict[str, Any]) -> MagicMock:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = response_body
    mock_response.text = json.dumps(response_body)
    mock_client.post.return_value = mock_response
    return mock_client


def test_dashscope_provider_config_registration() -> None:
    """Test that ProviderConfigManager returns DashScopeResponsesAPIConfig for dashscope."""
    config = ProviderConfigManager.get_provider_responses_api_config(
        model="dashscope/qwen3.8-flash",
        provider=LlmProviders.DASHSCOPE,
    )

    assert config is not None
    assert isinstance(config, DashScopeResponsesAPIConfig)
    assert config.custom_llm_provider == LlmProviders.DASHSCOPE


def test_dashscope_default_url() -> None:
    """Test default public DashScope /responses URL."""
    config = DashScopeResponsesAPIConfig()
    url = config.get_complete_url(api_base=None, litellm_params={})
    assert url == f"{DEFAULT_DASHSCOPE_API_BASE}/responses"
    assert url == "https://dashscope.aliyuncs.com/compatible-mode/v1/responses"


def test_dashscope_maas_workspace_responses_api_base_in_params() -> None:
    """
    Test MaaS workspace endpoint with dedicated responses_api_base in litellm_params.
    This validates the core fix for Issue #38474.
    """
    config = DashScopeResponsesAPIConfig()

    maas_chat_base = "https://ws-12345.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
    maas_responses_base = "https://ws-12345.ap-southeast-1.maas.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1"

    url = config.get_complete_url(
        api_base=maas_chat_base,
        litellm_params={"responses_api_base": maas_responses_base},
    )
    assert url == f"{maas_responses_base}/responses"


def test_dashscope_responses_api_base_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test DASHSCOPE_RESPONSES_API_BASE environment variable takes precedence."""
    config = DashScopeResponsesAPIConfig()

    monkeypatch.setenv(
        "DASHSCOPE_RESPONSES_API_BASE",
        "https://custom-ws.maas.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1",
    )
    monkeypatch.setenv(
        "DASHSCOPE_API_BASE",
        "https://custom-ws.maas.aliyuncs.com/compatible-mode/v1",
    )

    url = config.get_complete_url(api_base=None, litellm_params={})
    assert url == "https://custom-ws.maas.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1/responses"


def test_dashscope_api_base_without_v1() -> None:
    """Test api_base without /v1 suffix correctly appends /v1/responses."""
    config = DashScopeResponsesAPIConfig()
    url = config.get_complete_url(
        api_base="https://custom-dashscope.example.com",
        litellm_params={},
    )
    assert url == "https://custom-dashscope.example.com/v1/responses"


def test_dashscope_api_base_with_trailing_responses() -> None:
    """Test api_base already ending with /responses is not duplicated."""
    config = DashScopeResponsesAPIConfig()
    url = config.get_complete_url(
        api_base="https://custom-dashscope.example.com/compatible-mode/v1/responses",
        litellm_params={},
    )
    assert url == "https://custom-dashscope.example.com/compatible-mode/v1/responses"


def test_dashscope_validate_environment_success() -> None:
    """Test validate_environment sets Authorization header."""
    config = DashScopeResponsesAPIConfig()
    headers = config.validate_environment(
        headers={},
        model="qwen3.8-flash",
        litellm_params=GenericLiteLLMParams(api_key="sk-test-key-123"),
    )
    assert headers.get("Authorization") == "Bearer sk-test-key-123"


def test_dashscope_validate_environment_missing_key() -> None:
    """Test validate_environment raises ValueError when API key is missing."""
    config = DashScopeResponsesAPIConfig()
    with pytest.raises(ValueError, match="DashScope API key is required"):
        config.validate_environment(
            headers={},
            model="qwen3.8-flash",
            litellm_params=GenericLiteLLMParams(api_key=None),
        )


def test_dashscope_preserves_cache_control() -> None:
    """Test DashScopeResponsesAPIConfig preserves cache_control in tools/input."""
    config = DashScopeResponsesAPIConfig()
    sample_input = [
        {"role": "user", "content": [{"type": "input_text", "text": "Hi", "cache_control": {"type": "ephemeral"}}]}
    ]
    input_res, _ = config.remove_cache_control_flag_from_input_and_tools(
        model="qwen3.8-flash",
        input=sample_input,  # pyright: ignore[reportArgumentType] # test exercises supported cache-control input
        tools=None,
    )
    assert isinstance(input_res, list)
    assert input_res == sample_input
    first_item = input_res[0]
    assert isinstance(first_item, dict)
    content_list = first_item.get("content")
    assert isinstance(content_list, list)
    assert "cache_control" in content_list[0]


def test_dashscope_supports_native_websocket() -> None:
    """Test supports_native_websocket returns False."""
    config = DashScopeResponsesAPIConfig()
    assert config.supports_native_websocket() is False


def test_dashscope_responses_e2e_with_mock_client() -> None:
    """Test end-to-end litellm.responses call for dashscope."""
    mock_client = _make_mock_http_client(_make_mock_responses_api_response("Hello from Qwen!"))

    with patch(  # test-quality-ok: mock HTTP client for Responses API without live network
        "litellm.llms.custom_httpx.llm_http_handler._get_httpx_client",
        return_value=mock_client,
    ):
        response = litellm.responses(
            model="dashscope/qwen3.8-flash",
            input="Hello",
            api_key="sk-dashscope-test",
            responses_api_base="https://ws.maas.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1",
        )

    assert response is not None
    assert isinstance(response, ResponsesAPIResponse)
    assert len(response.output) > 0
    msg = response.output[0]
    assert getattr(msg, "role", None) == "assistant"
    content_list = getattr(msg, "content", None)
    assert content_list is not None
    assert content_list[0].text == "Hello from Qwen!"

    mock_client.post.assert_called_once()
    call_args = mock_client.post.call_args
    called_url = call_args[0][0] if call_args[0] else call_args[1].get("url")
    assert "https://ws.maas.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1/responses" in str(called_url)
