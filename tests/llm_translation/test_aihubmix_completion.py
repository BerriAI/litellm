"""
Unit tests for the AIHubMix provider integration.
AIHubMix is an OpenAI-compatible aggregation API gateway (https://aihubmix.com/v1).
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm import acompletion, completion
from litellm.llms.aihubmix.chat.transformation import AIHubMixChatConfig


# ---------------------------------------------------------------------------
# Transformation / config tests (no network)
# ---------------------------------------------------------------------------


def test_aihubmix_config_provider_name():
    config = AIHubMixChatConfig()
    assert config.custom_llm_provider == "aihubmix"


def test_aihubmix_config_default_api_base():
    config = AIHubMixChatConfig()
    api_base, _ = config._get_openai_compatible_provider_info(
        api_base=None, api_key=None
    )
    assert api_base == "https://aihubmix.com/v1"


def test_aihubmix_config_custom_api_base():
    config = AIHubMixChatConfig()
    api_base, _ = config._get_openai_compatible_provider_info(
        api_base="https://custom.aihubmix.com/v1", api_key=None
    )
    assert api_base == "https://custom.aihubmix.com/v1"


def test_aihubmix_config_api_key_passthrough():
    config = AIHubMixChatConfig()
    _, api_key = config._get_openai_compatible_provider_info(
        api_base=None, api_key="my-test-key"
    )
    assert api_key == "my-test-key"


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------


def test_get_llm_provider_aihubmix():
    model, provider, _, _ = litellm.get_llm_provider("aihubmix/gpt-5.4")
    assert provider == "aihubmix"
    assert model == "gpt-5.4"


def test_get_llm_provider_aihubmix_explicit():
    model, provider, _, _ = litellm.get_llm_provider(
        "gpt-5.4", custom_llm_provider="aihubmix"
    )
    assert provider == "aihubmix"
    assert model == "gpt-5.4"


# ---------------------------------------------------------------------------
# Mock completion tests (no real API calls)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stream", [True, False])
def test_aihubmix_mock_completion(stream):
    """Confirm the integration works end-to-end using mock_response."""
    response = completion(
        model="aihubmix/gpt-5.4",
        messages=[{"role": "user", "content": "Hello!"}],
        api_key="fake-key",
        mock_response="Hello from AIHubMix!",
        stream=stream,
    )
    if stream:
        for chunk in response:
            pass
    else:
        assert response is not None
        assert response.choices[0].message.content == "Hello from AIHubMix!"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_aihubmix_request_url_and_model(stream):
    """Verify that requests are sent to https://aihubmix.com/v1 with the correct model name."""
    messages = [{"role": "user", "content": "Hello!"}]
    mock_response_data = litellm.ModelResponse(
        choices=[
            litellm.Choices(
                message=litellm.Message(content="Hello!"),
                index=0,
                finish_reason="stop",
            )
        ]
    ).model_dump()

    with patch(
        "litellm.llms.custom_httpx.llm_http_handler.AsyncHTTPHandler.post"
    ) as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = json.dumps(mock_response_data)
        mock_resp.headers = {"Content-Type": "application/json"}
        mock_resp.json.return_value = mock_response_data
        mock_post.return_value = mock_resp

        await acompletion(
            model="aihubmix/gpt-5.4",
            messages=messages,
            api_key="fake-key",
            stream=stream,
        )

    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["url"] == "https://aihubmix.com/v1/chat/completions"
    request_body = json.loads(call_kwargs["data"])
    assert request_body["model"] == "gpt-5.4"
    assert request_body["messages"] == messages
