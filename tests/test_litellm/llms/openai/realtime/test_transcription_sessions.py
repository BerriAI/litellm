"""
Tests for the Realtime transcription_sessions surface used by gpt-realtime-whisper:
  - OpenAI / Azure URL construction (POST /v1/realtime/transcription_sessions)
  - RealtimeTranscriptionSessionRequest model-resolution + passthrough
  - BaseLLMHTTPHandler.async_realtime_transcription_session_handler targeting
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.azure.realtime.http_transformation import AzureRealtimeHTTPConfig
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.llms.openai.realtime.http_transformation import OpenAIRealtimeHTTPConfig
from litellm.types.realtime import RealtimeTranscriptionSessionRequest


def test_openai_transcription_session_url():
    cfg = OpenAIRealtimeHTTPConfig()
    assert (
        cfg.get_transcription_session_url(
            api_base="https://api.openai.com", model="gpt-realtime-whisper"
        )
        == "https://api.openai.com/v1/realtime/transcription_sessions"
    )


def test_openai_transcription_session_url_strips_trailing_v1():
    """A /v1 suffix must not be duplicated in the path."""
    cfg = OpenAIRealtimeHTTPConfig()
    assert (
        cfg.get_transcription_session_url(
            api_base="https://api.openai.com/v1", model="gpt-realtime-whisper"
        )
        == "https://api.openai.com/v1/realtime/transcription_sessions"
    )


def test_azure_transcription_session_url_uses_deployment_and_api_version():
    cfg = AzureRealtimeHTTPConfig()
    url = cfg.get_transcription_session_url(
        api_base="https://my.openai.azure.com",
        model="whisper-deploy",
        api_version="2025-04-01-preview",
    )
    assert (
        url
        == "https://my.openai.azure.com/openai/realtime/transcription_sessions?api-version=2025-04-01-preview"
    )


def test_request_resolves_model_returns_none_when_both_absent():
    req = RealtimeTranscriptionSessionRequest(input_audio_format="pcm16")
    assert req.resolved_model() is None

    req = RealtimeTranscriptionSessionRequest(
        model="openai/gpt-realtime-whisper",
        input_audio_transcription={"model": "gpt-realtime-whisper"},
    )
    assert req.resolved_model() == "openai/gpt-realtime-whisper"


def test_request_resolves_model_from_input_audio_transcription():
    req = RealtimeTranscriptionSessionRequest(
        input_audio_transcription={"model": "gpt-realtime-whisper", "language": "en"},
    )
    assert req.resolved_model() == "gpt-realtime-whisper"


def test_request_passthrough_excludes_routing_hint():
    """Unknown fields pass through; the litellm-only `model` hint is not forwarded."""
    req = RealtimeTranscriptionSessionRequest(
        model="openai/gpt-realtime-whisper",
        input_audio_format="pcm16",
        input_audio_transcription={"model": "gpt-realtime-whisper"},
        turn_detection=None,
    )
    forwarded = req.model_dump(exclude_none=True, exclude={"model"})
    assert "model" not in forwarded
    assert forwarded["input_audio_format"] == "pcm16"
    assert forwarded["input_audio_transcription"] == {"model": "gpt-realtime-whisper"}


@pytest.mark.asyncio
async def test_handler_posts_to_transcription_sessions_url():
    handler = BaseLLMHTTPHandler()

    mock_response = MagicMock(spec=httpx.Response)
    mock_client = MagicMock(spec=AsyncHTTPHandler)
    mock_client.post = AsyncMock(return_value=mock_response)

    logging_obj = MagicMock()
    logging_obj.pre_call = MagicMock()

    request_body = {"input_audio_transcription": {"model": "gpt-realtime-whisper"}}
    result = await handler.async_realtime_transcription_session_handler(
        api_base="https://api.openai.com",
        api_key="sk-test",
        request_data=request_body,
        logging_obj=logging_obj,
        timeout=10.0,
        provider_config=OpenAIRealtimeHTTPConfig(),
        model="gpt-realtime-whisper",
        client=mock_client,
    )

    assert result is mock_response
    _, kwargs = mock_client.post.call_args
    assert kwargs["url"] == "https://api.openai.com/v1/realtime/transcription_sessions"
    assert kwargs["json"] == request_body
    assert kwargs["headers"]["Authorization"] == "Bearer sk-test"


@pytest.mark.asyncio
async def test_client_secret_handler_still_targets_client_secrets_url():
    """Refactor regression: the client_secrets handler must keep its own URL."""
    handler = BaseLLMHTTPHandler()

    mock_response = MagicMock(spec=httpx.Response)
    mock_client = MagicMock(spec=AsyncHTTPHandler)
    mock_client.post = AsyncMock(return_value=mock_response)

    logging_obj = MagicMock()
    logging_obj.pre_call = MagicMock()

    await handler.async_realtime_client_secret_handler(
        api_base="https://api.openai.com",
        api_key="sk-test",
        request_data={"session": {"type": "realtime"}},
        logging_obj=logging_obj,
        timeout=10.0,
        provider_config=OpenAIRealtimeHTTPConfig(),
        model="gpt-4o-realtime-preview",
        client=mock_client,
    )

    _, kwargs = mock_client.post.call_args
    assert kwargs["url"] == "https://api.openai.com/v1/realtime/client_secrets"


@pytest.mark.asyncio
async def test_sdk_fn_routes_openai_transcription_session(monkeypatch):
    """
    litellm.acreate_realtime_transcription_session resolves the OpenAI provider
    from the transcription model and POSTs to the OpenAI transcription_sessions URL.
    """
    import litellm

    monkeypatch.setenv("OPENAI_API_KEY", "sk-unit-test")

    mock_response = MagicMock(spec=httpx.Response)
    mock_client = MagicMock(spec=AsyncHTTPHandler)
    mock_client.post = AsyncMock(return_value=mock_response)

    result = await litellm.acreate_realtime_transcription_session(
        model="openai/gpt-realtime-whisper",
        transcription_session={
            "input_audio_format": "pcm16",
            "input_audio_transcription": {"model": "gpt-realtime-whisper"},
        },
        client=mock_client,
    )

    assert result is mock_response
    _, kwargs = mock_client.post.call_args
    assert kwargs["url"].endswith("/v1/realtime/transcription_sessions")
    # The litellm-only routing hint must not be forwarded upstream.
    assert "model" not in kwargs["json"]
    assert kwargs["json"]["input_audio_transcription"] == {
        "model": "gpt-realtime-whisper"
    }


def test_append_query_params_skips_existing_keys():
    from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler

    url = "wss://example.com/v1/realtime?model=gpt-4o"
    result = BaseLLMHTTPHandler._append_query_params(
        url, {"model": "ignored", "intent": "transcription"}
    )
    assert "model=ignored" not in result
    assert "intent=transcription" in result


def test_append_query_params_no_params_returns_unchanged():
    from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler

    url = "wss://example.com/v1/realtime?model=gpt-4o"
    assert BaseLLMHTTPHandler._append_query_params(url, None) == url
    assert BaseLLMHTTPHandler._append_query_params(url, {}) == url


def test_append_query_params_encodes_special_chars():
    from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler

    url = "wss://example.com/v1/realtime"
    result = BaseLLMHTTPHandler._append_query_params(url, {"intent": "a&b=c"})
    assert "intent=a%26b%3Dc" in result
    assert "&b=c" not in result


def test_azure_construct_url_encodes_model_and_api_version():
    """model and api-version must be URL-encoded to prevent query-string injection."""
    from litellm.llms.azure.realtime.handler import AzureOpenAIRealtime

    h = AzureOpenAIRealtime()
    url = h._construct_url(
        "https://x.openai.azure.com",
        "deploy&evil=1",
        "2024-10-01-preview",
    )
    assert "evil=1" not in url.split("?", 1)[1]

    url_ga = h._construct_url(
        "https://x.openai.azure.com",
        "deploy&evil=1",
        None,
        realtime_protocol="GA",
    )
    assert "evil=1" not in url_ga.split("?", 1)[1]
