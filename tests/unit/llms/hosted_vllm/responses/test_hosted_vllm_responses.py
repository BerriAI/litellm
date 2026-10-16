"""
Tests for hosted_vllm responses API support.

Regression test for: https://github.com/BerriAI/litellm/issues
Bug: client.responses.create() raised TypeError: 'NoneType' object is not a mapping
when extra_body=None was passed through the responses→completion pipeline for
hosted_vllm (and any OpenAI-compatible provider using add_provider_specific_params_to_optional_params).
"""

import json
from unittest.mock import MagicMock, patch

import pytest


import litellm
from litellm.llms.hosted_vllm.responses.transformation import (
    HostedVLLMResponsesAPIConfig,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


def _make_mock_responses_api_response(content: str = "Hello! I'm doing well.") -> dict:
    return {
        "id": "resp-test123",
        "object": "response",
        "created_at": 1234567890,
        "model": "Qwen/Qwen3-8B",
        "output": [
            {
                "type": "message",
                "id": "msg-test123",
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
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
        },
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


def test_hosted_vllm_responses_create_with_string_input():
    """
    Test that hosted_vllm routes directly to the native /v1/responses endpoint
    when the Responses API config is registered, and correctly parses the response.
    """
    mock_client = _make_mock_http_client(_make_mock_responses_api_response("I'm doing well, thanks!"))

    with patch(
        "litellm.llms.custom_httpx.llm_http_handler.get_httpx_client",
        return_value=mock_client,
    ):
        response = litellm.responses(
            model="hosted_vllm/Qwen/Qwen3-8B",
            input="Hello, how are you?",
            api_base="https://test-vllm.example.com/v1",
            api_key="test-key",
        )

    from litellm.types.llms.openai import ResponsesAPIResponse

    assert response is not None
    assert isinstance(response, ResponsesAPIResponse)
    assert len(response.output) > 0
    output_message = response.output[0]
    assert output_message.role == "assistant"  # type: ignore[union-attr]
    assert len(output_message.content) > 0  # type: ignore[union-attr]
    assert "well" in output_message.content[0].text  # type: ignore[union-attr]


def test_hosted_vllm_responses_create_with_explicit_none_extra_body():
    """
    Directly verify the fix in add_provider_specific_params_to_optional_params:
    extra_body=None must not crash when building optional_params.
    """
    from litellm.utils import get_optional_params

    # This should not raise TypeError: 'NoneType' object is not a mapping
    optional_params = get_optional_params(
        model="Qwen/Qwen3-8B",
        custom_llm_provider="hosted_vllm",
        extra_body=None,
    )

    # extra_body=None should be normalized to an empty dict (or absent)
    assert optional_params.get("extra_body") is not None or "extra_body" not in optional_params


def test_hosted_vllm_provider_config_registration():
    """Test that ProviderConfigManager returns HostedVLLMResponsesAPIConfig for hosted_vllm."""
    config = ProviderConfigManager.get_provider_responses_api_config(
        model="hosted_vllm/Qwen/Qwen3-8B",
        provider=LlmProviders.HOSTED_VLLM,
    )

    assert config is not None
    assert isinstance(config, HostedVLLMResponsesAPIConfig)
    assert config.custom_llm_provider == LlmProviders.HOSTED_VLLM


def test_hosted_vllm_responses_api_url():
    """Test get_complete_url() constructs the correct URL."""
    config = HostedVLLMResponsesAPIConfig()

    # api_base without /v1
    url = config.get_complete_url(
        api_base="http://localhost:8000",
        litellm_params={},
    )
    assert url == "http://localhost:8000/v1/responses"

    # api_base with /v1
    url_with_v1 = config.get_complete_url(
        api_base="http://localhost:8000/v1",
        litellm_params={},
    )
    assert url_with_v1 == "http://localhost:8000/v1/responses"

    # api_base with trailing slash
    url_with_slash = config.get_complete_url(
        api_base="http://localhost:8000/v1/",
        litellm_params={},
    )
    assert url_with_slash == "http://localhost:8000/v1/responses"


def test_hosted_vllm_responses_api_url_requires_api_base():
    """Test get_complete_url() raises ValueError when api_base is not set."""
    config = HostedVLLMResponsesAPIConfig()

    with pytest.raises(ValueError, match="api_base not set"):
        config.get_complete_url(
            api_base=None,
            litellm_params={},
        )


def test_hosted_vllm_validate_environment_default_api_key():
    """Test validate_environment() defaults to 'fake-api-key' when no key is provided."""
    config = HostedVLLMResponsesAPIConfig()

    headers = config.validate_environment(
        headers={},
        model="Qwen/Qwen3-8B",
        litellm_params=GenericLiteLLMParams(),
    )

    assert headers.get("Authorization") == "Bearer fake-api-key"


def test_hosted_vllm_validate_environment_custom_api_key():
    """Test validate_environment() uses the provided api_key."""
    config = HostedVLLMResponsesAPIConfig()

    headers = config.validate_environment(
        headers={},
        model="Qwen/Qwen3-8B",
        litellm_params=GenericLiteLLMParams(api_key="my-custom-key"),
    )

    assert headers.get("Authorization") == "Bearer my-custom-key"


def _sent_responses_body(**call_kwargs) -> dict:
    mock_client = _make_mock_http_client(_make_mock_responses_api_response())
    with patch(
        "litellm.llms.custom_httpx.llm_http_handler.get_httpx_client",
        return_value=mock_client,
    ):
        litellm.responses(
            model="hosted_vllm/Qwen/Qwen3-8B",
            input="hi",
            api_base="https://test-vllm.example.com/v1",
            api_key="test-key",
            **call_kwargs,
        )
    kwargs = mock_client.post.call_args.kwargs
    return kwargs["json"] if "json" in kwargs else json.loads(kwargs["data"])


def test_responses_cache_salt_is_per_key():
    alice = _sent_responses_body(hosted_vllm_cache_salt=True, litellm_metadata={"user_api_key_hash": "hash-alice"})
    bob = _sent_responses_body(hosted_vllm_cache_salt=True, litellm_metadata={"user_api_key_hash": "hash-bob"})
    assert len(alice["cache_salt"]) == 64
    assert alice["cache_salt"] != bob["cache_salt"]
    assert "hosted_vllm_cache_salt" not in alice


def test_responses_cache_salt_mixes_caller_salt_from_extra_body():
    opted_in = {"hosted_vllm_cache_salt": True, "litellm_metadata": {"user_api_key_hash": "hash-alice"}}
    plain = _sent_responses_body(**opted_in)["cache_salt"]
    split = _sent_responses_body(**opted_in, extra_body={"cache_salt": "end-user-1"})["cache_salt"]
    forged = _sent_responses_body(
        hosted_vllm_cache_salt=True,
        litellm_metadata={"user_api_key_hash": "hash-bob"},
        extra_body={"cache_salt": plain},
    )["cache_salt"]
    assert split not in (plain, "end-user-1")
    assert forged != plain


@pytest.mark.parametrize(
    "call_kwargs, expected_salt",
    [
        pytest.param(
            {"hosted_vllm_cache_salt": True, "metadata": {"user_api_key_hash": "hash-alice"}},
            None,
            id="ignores-caller-responses-metadata",
        ),
        pytest.param({"litellm_metadata": {"user_api_key_hash": "hash-alice"}}, None, id="not-opted-in-sends-no-salt"),
        pytest.param(
            {"litellm_metadata": {"user_api_key_hash": "hash-alice"}, "extra_body": {"cache_salt": "caller"}},
            "caller",
            id="not-opted-in-keeps-caller-salt",
        ),
    ],
)
def test_responses_cache_salt_untouched_without_opt_in_or_caller_key(call_kwargs, expected_salt):
    assert _sent_responses_body(**call_kwargs).get("cache_salt") == expected_salt
