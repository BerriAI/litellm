import asyncio
import json
import logging
import time
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.code_interpreter_interception.handler import (
    CodeInterpreterInterceptionLogger,
    LITELLM_CODE_EXECUTION_TOOL_NAME,
)
from litellm.llms.base_llm.audio_transcription.transformation import (
    AudioTranscriptionRequestData,
    BaseAudioTranscriptionConfig,
)
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.custom_httpx.llm_http_handler import (
    BaseLLMHTTPHandler,
    _collect_ws_project_quota_callbacks,
    _google_genai_streaming_hidden_params,
    _has_pre_call_deployment_hook,
    _rust_responses_websocket_enabled,
)
from litellm.llms.azure.videos.transformation import AzureVideoConfig
from litellm.llms.openai.videos.transformation import OpenAIVideoConfig
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import TranscriptionResponse

_ACTIVE_KEY = "_code_interpreter_interception_active"
_SANDBOX_KEY = "_code_interpreter_interception_sandbox_key"


def test_prepare_fake_stream_request():
    # Initialize the BaseLLMHTTPHandler
    handler = BaseLLMHTTPHandler()

    # Test case 1: fake_stream is True
    stream = True
    data = {
        "stream": True,
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    fake_stream = True

    result_stream, result_data = handler._prepare_fake_stream_request(stream=stream, data=data, fake_stream=fake_stream)

    # Verify that stream is set to False
    assert result_stream is False
    # Verify that "stream" key is removed from data
    assert "stream" not in result_data
    # Verify other data remains unchanged
    assert result_data["model"] == "gpt-4"
    assert result_data["messages"] == [{"role": "user", "content": "Hello"}]

    # Test case 2: fake_stream is False
    stream = True
    data = {
        "stream": True,
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    fake_stream = False

    result_stream, result_data = handler._prepare_fake_stream_request(stream=stream, data=data, fake_stream=fake_stream)

    # Verify that stream remains True
    assert result_stream is True
    # Verify that data remains unchanged
    assert "stream" in result_data
    assert result_data["stream"] is True
    assert result_data["model"] == "gpt-4"
    assert result_data["messages"] == [{"role": "user", "content": "Hello"}]

    # Test case 3: data doesn't have stream key but fake_stream is True
    stream = True
    data = {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}
    fake_stream = True

    result_stream, result_data = handler._prepare_fake_stream_request(stream=stream, data=data, fake_stream=fake_stream)

    # Verify that stream is set to False
    assert result_stream is False
    # Verify that data remains unchanged (since there was no stream key to remove)
    assert "stream" not in result_data
    assert result_data["model"] == "gpt-4"
    assert result_data["messages"] == [{"role": "user", "content": "Hello"}]


def test_response_api_handler_streams_when_provider_transform_adds_stream():
    handler = BaseLLMHTTPHandler()
    config = Mock()
    config.validate_environment.return_value = {}
    config.get_complete_url.return_value = "https://chatgpt.example.com/responses"
    config.transform_responses_api_request.return_value = {
        "model": "gpt-5.3-codex",
        "input": "hi",
        "stream": True,
    }
    config.sign_request.return_value = ({}, None)
    client = HTTPHandler(client=httpx.Client())
    client.post = Mock(
        return_value=httpx.Response(
            200,
            request=httpx.Request("POST", "https://chatgpt.example.com/responses"),
        )
    )
    logging_obj = Mock()

    handler.response_api_handler(
        model="gpt-5.3-codex",
        input="hi",
        responses_api_provider_config=config,
        response_api_optional_request_params={},
        custom_llm_provider="chatgpt",
        litellm_params=GenericLiteLLMParams(),
        logging_obj=logging_obj,
        client=client,
    )

    assert client.post.call_args.kwargs["stream"] is True
    assert client.post.call_args.kwargs["json"]["stream"] is True


def test_response_api_handler_runs_agentic_hooks_in_sync_path(monkeypatch):
    handler = BaseLLMHTTPHandler()
    config = Mock()
    config.validate_environment.return_value = {}
    config.get_complete_url.return_value = "https://chatgpt.example.com/responses"
    config.transform_responses_api_request.return_value = {
        "model": "gpt-5",
        "input": "hi",
    }
    config.sign_request.return_value = ({}, None)
    initial_response = Mock()
    final_response = Mock()
    config.transform_response_api_response.return_value = initial_response

    client = HTTPHandler(client=httpx.Client())
    client.post = Mock(
        return_value=httpx.Response(
            200,
            request=httpx.Request("POST", "https://chatgpt.example.com/responses"),
        )
    )
    logging_obj = Mock()

    monkeypatch.setattr(handler, "_has_agentic_completion_hook", Mock(return_value=True))
    hook_mock = AsyncMock(return_value=final_response)
    monkeypatch.setattr(handler, "_call_agentic_completion_hooks", hook_mock)

    response = handler.response_api_handler(
        model="gpt-5",
        input="hi",
        responses_api_provider_config=config,
        response_api_optional_request_params={},
        custom_llm_provider="openai",
        litellm_params=GenericLiteLLMParams(),
        logging_obj=logging_obj,
        client=client,
    )

    assert response is final_response
    hook_mock.assert_awaited_once()
    assert hook_mock.call_args.kwargs["api_surface"] == "responses"
    assert hook_mock.call_args.kwargs["messages"] == [{"role": "user", "content": "hi"}]


def test_response_api_handler_runs_responses_pre_call_hook_before_transform():
    handler = BaseLLMHTTPHandler()
    config = Mock()
    config.validate_environment.return_value = {}
    config.get_complete_url.return_value = "https://api.openai.com/v1/responses"
    config.sign_request.return_value = ({}, None)
    initial_response = ResponsesAPIResponse(
        id="resp_1",
        created_at=0,
        output=[],
        status="completed",
        model="gpt-5",
    )
    config.transform_response_api_response.return_value = initial_response

    def transform_responses_api_request(**kwargs):
        return {
            "model": kwargs["model"],
            "input": kwargs["input"],
            **kwargs["response_api_optional_request_params"],
        }

    config.transform_responses_api_request.side_effect = transform_responses_api_request
    client = HTTPHandler(client=httpx.Client())
    client.post = Mock(
        return_value=httpx.Response(
            200,
            request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
        )
    )
    logging_obj = Mock()
    logging_obj.dynamic_success_callbacks = []

    old_callbacks = list(litellm.callbacks)
    litellm.callbacks = [CodeInterpreterInterceptionLogger()]
    try:
        response = handler.response_api_handler(
            model="gpt-5",
            input="use code",
            responses_api_provider_config=config,
            response_api_optional_request_params={
                "tools": [{"type": "code_interpreter", "container": {"type": "auto"}}]
            },
            custom_llm_provider="openai",
            litellm_params=GenericLiteLLMParams(api_key="sk-test"),
            logging_obj=logging_obj,
            client=client,
        )
    finally:
        litellm.callbacks = old_callbacks

    assert response is initial_response
    transform_kwargs = config.transform_responses_api_request.call_args.kwargs
    tools = transform_kwargs["response_api_optional_request_params"]["tools"]
    assert not any(tool.get("type") == "code_interpreter" for tool in tools)
    assert any(
        tool.get("type") == "function" and tool.get("name") == LITELLM_CODE_EXECUTION_TOOL_NAME for tool in tools
    )
    hook_litellm_params = transform_kwargs["litellm_params"]
    assert hook_litellm_params.get(_ACTIVE_KEY) is True
    assert hook_litellm_params.get(_SANDBOX_KEY)


@pytest.mark.asyncio
async def test_async_response_api_handler_streams_when_provider_transform_adds_stream():
    handler = BaseLLMHTTPHandler()
    config = Mock()
    config.validate_environment.return_value = {}
    config.get_complete_url.return_value = "https://chatgpt.example.com/responses"
    config.transform_responses_api_request.return_value = {
        "model": "gpt-5.3-codex",
        "input": "hi",
        "stream": True,
    }
    config.sign_request.return_value = ({}, None)
    client = AsyncHTTPHandler()
    client.post = AsyncMock(
        return_value=httpx.Response(
            200,
            request=httpx.Request("POST", "https://chatgpt.example.com/responses"),
        )
    )
    logging_obj = Mock()

    await handler.async_response_api_handler(
        model="gpt-5.3-codex",
        input="hi",
        responses_api_provider_config=config,
        response_api_optional_request_params={},
        custom_llm_provider="chatgpt",
        litellm_params=GenericLiteLLMParams(),
        logging_obj=logging_obj,
        client=client,
    )

    assert client.post.call_args.kwargs["stream"] is True
    assert client.post.call_args.kwargs["json"]["stream"] is True


def test_get_agentic_loop_settings_defaults_and_overrides():
    handler = BaseLLMHTTPHandler()

    depth, max_loops, fingerprints = handler._get_agentic_loop_settings(kwargs={})
    assert depth == 0
    assert max_loops == 3
    assert fingerprints == []

    depth, max_loops, fingerprints = handler._get_agentic_loop_settings(
        kwargs={
            "_agentic_loop_depth": 2,
            "max_agentic_loops": 7,
            "_agentic_loop_fingerprints": ["fp-1", "fp-2"],
        }
    )
    assert depth == 2
    assert max_loops == 7
    assert fingerprints == ["fp-1", "fp-2"]


def test_has_agentic_completion_hook_detection(monkeypatch):
    """The streaming path skips the agentic wrapper only when no callback
    overrides async_should_run_agentic_loop. Verify both directions."""
    from litellm.integrations.custom_logger import CustomLogger

    handler = BaseLLMHTTPHandler()
    logging_obj = Mock()
    logging_obj.dynamic_success_callbacks = []

    # No callbacks at all -> no agentic hook.
    monkeypatch.setattr(litellm, "callbacks", [])
    assert handler._has_agentic_completion_hook(logging_obj) is False

    # A plain CustomLogger that does NOT override the gate -> still no hook
    # (so the wrapper is safely skipped).
    class _PlainLogger(CustomLogger):
        pass

    monkeypatch.setattr(litellm, "callbacks", [_PlainLogger()])
    assert handler._has_agentic_completion_hook(logging_obj) is False

    # A logger that overrides the gate (directly) -> hook present.
    class _AgenticLogger(CustomLogger):
        async def async_should_run_agentic_loop(
            self, response, model, messages, tools, stream, custom_llm_provider, kwargs
        ):
            return True, {}

    monkeypatch.setattr(litellm, "callbacks", [_AgenticLogger()])
    assert handler._has_agentic_completion_hook(logging_obj) is True

    # Override inherited through an intermediate class is still detected
    # (function-identity check, not a leaf __dict__ check).
    class _DerivedAgenticLogger(_AgenticLogger):
        pass

    monkeypatch.setattr(litellm, "callbacks", [_DerivedAgenticLogger()])
    assert handler._has_agentic_completion_hook(logging_obj) is True

    # Hook supplied via logging_obj.dynamic_success_callbacks is detected too.
    monkeypatch.setattr(litellm, "callbacks", [])
    logging_obj.dynamic_success_callbacks = [_AgenticLogger()]
    assert handler._has_agentic_completion_hook(logging_obj) is True

    # String-named callback entry (e.g. "datadog") must be resolved to its
    # CustomLogger instance via get_custom_logger_compatible_class -- the same
    # way ProxyLogging._callback_capabilities handles them. Without that
    # resolution a string-registered agentic callback would be silently
    # skipped and the buffering wrapper would never fire.
    logging_obj.dynamic_success_callbacks = []
    agentic_via_string = _AgenticLogger()
    monkeypatch.setattr(litellm, "callbacks", ["fake_string_callback"])
    monkeypatch.setattr(
        "litellm.litellm_core_utils.litellm_logging.get_custom_logger_compatible_class",
        lambda name: agentic_via_string if name == "fake_string_callback" else None,
    )
    assert handler._has_agentic_completion_hook(logging_obj) is True

    # Unresolvable string (returns None) is skipped, no false positive.
    monkeypatch.setattr(litellm, "callbacks", ["unknown_callback"])
    monkeypatch.setattr(
        "litellm.litellm_core_utils.litellm_logging.get_custom_logger_compatible_class",
        lambda name: None,
    )
    assert handler._has_agentic_completion_hook(logging_obj) is False


def test_fingerprint_agentic_tools_is_deterministic():
    handler = BaseLLMHTTPHandler()
    tools_a = {"tool_calls": [{"id": "1", "input": {"q": "abc"}, "name": "web_search"}]}
    tools_b = {"tool_calls": [{"name": "web_search", "input": {"q": "abc"}, "id": "1"}]}

    assert handler._fingerprint_agentic_tools(tools_a) == handler._fingerprint_agentic_tools(tools_b)


@pytest.mark.asyncio
async def test_async_anthropic_messages_handler_extra_headers():
    """
    Test that async_anthropic_messages_handler correctly extracts and merges
    extra_headers from kwargs with proper priority.
    """
    handler = BaseLLMHTTPHandler()

    # Mock the config
    mock_config = Mock()
    mock_config.validate_anthropic_messages_environment = Mock(
        return_value=({"x-api-key": "test-key"}, "https://api.anthropic.com")
    )
    mock_config.transform_anthropic_messages_request = Mock(
        return_value={"model": "claude-3-opus-20240229", "messages": []}
    )

    # Mock the client
    mock_client = AsyncMock()
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello!"}],
        "model": "claude-3-opus-20240229",
        "stop_reason": "end_turn",
    }
    mock_client.post = AsyncMock(return_value=mock_response)

    # Mock logging object
    mock_logging_obj = Mock()
    mock_logging_obj.update_environment_variables = Mock()
    mock_logging_obj.model_call_details = {}
    mock_logging_obj.stream = False

    # Test case 1: Only extra_headers in kwargs
    kwargs = {
        "extra_headers": {
            "X-Custom-Header": "from-kwargs",
            "X-Auth-Token": "token123",
        }
    }

    with patch(
        "litellm.litellm_core_utils.get_provider_specific_headers.ProviderSpecificHeaderUtils.get_provider_specific_headers"
    ) as mock_provider_headers:
        mock_provider_headers.return_value = None

        # Capture what headers are passed to validate_anthropic_messages_environment
        captured_headers = {}

        def capture_validate(*args, **kwargs):
            captured_headers.update(kwargs.get("headers", {}))
            return ({"x-api-key": "test-key"}, "https://api.anthropic.com")

        mock_config.validate_anthropic_messages_environment = capture_validate

        try:
            await handler.async_anthropic_messages_handler(
                model="claude-3-opus-20240229",
                messages=[{"role": "user", "content": "Hello"}],
                anthropic_messages_provider_config=mock_config,
                anthropic_messages_optional_request_params={},
                custom_llm_provider="anthropic",
                litellm_params=GenericLiteLLMParams(),
                logging_obj=mock_logging_obj,
                client=mock_client,
                kwargs=kwargs,
            )
        except Exception:
            pass  # We're testing header extraction, not the full flow

        # Verify extra_headers were extracted and merged
        assert "X-Custom-Header" in captured_headers
        assert captured_headers["X-Custom-Header"] == "from-kwargs"
        assert "X-Auth-Token" in captured_headers
        assert captured_headers["X-Auth-Token"] == "token123"


@pytest.mark.asyncio
async def test_async_anthropic_messages_handler_streaming_forwards_provider_response_headers():
    """
    Regression test for LIT-3724 (issue 2): streaming /v1/messages responses
    dropped the upstream provider's HTTP response headers, so Bedrock's
    x-amzn-requestid / x-amzn-trace-id never reached clients even with
    `return_response_headers: true`. The returned stream object must carry
    them in `_hidden_params["additional_headers"]` (llm_provider-* prefixed),
    which the proxy merges into the client-facing response headers.
    """
    from collections.abc import AsyncIterator as ABCAsyncIterator

    from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
        AnthropicMessagesConfig,
    )

    handler = BaseLLMHTTPHandler()

    sse_body = (
        b'event: message_start\ndata: {"type": "message_start"}\n\n'
        b'event: message_stop\ndata: {"type": "message_stop"}\n\n'
    )
    upstream_response = httpx.Response(
        200,
        headers={
            "x-amzn-requestid": "amzn-req-123",
            "x-amzn-trace-id": "Root=1-abc-def",
        },
        content=sse_body,
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
    )
    mock_client = AsyncMock(spec=AsyncHTTPHandler)
    mock_client.post = AsyncMock(return_value=upstream_response)

    mock_logging_obj = Mock()
    mock_logging_obj.model_call_details = {}

    result = await handler.async_anthropic_messages_handler(
        model="claude-sonnet-4-20250514",
        messages=[{"role": "user", "content": "Hello"}],
        anthropic_messages_provider_config=AnthropicMessagesConfig(),
        anthropic_messages_optional_request_params={"max_tokens": 32},
        custom_llm_provider="anthropic",
        litellm_params=GenericLiteLLMParams(),
        logging_obj=mock_logging_obj,
        client=mock_client,
        api_key="sk-test",
        stream=True,
        kwargs={},
    )

    assert isinstance(result, ABCAsyncIterator)

    additional_headers = result._hidden_params["additional_headers"]
    assert additional_headers["llm_provider-x-amzn-requestid"] == "amzn-req-123"
    assert additional_headers["llm_provider-x-amzn-trace-id"] == "Root=1-abc-def"

    collected = b"".join([chunk async for chunk in result])
    assert b"message_start" in collected
    assert b"message_stop" in collected


@pytest.mark.asyncio
async def test_async_anthropic_messages_handler_agentic_streaming_forwards_provider_response_headers():
    """
    Companion to the test above for the agentic branch: when a callback
    overrides async_should_run_agentic_loop, the handler wraps
    AgenticAnthropicStreamingIterator in AnthropicMessagesStreamingResponse.
    That wrapping must still expose the provider headers and delegate
    iteration through the two-phase agentic iterator unchanged.
    """
    from collections.abc import AsyncIterator as ABCAsyncIterator

    from litellm.integrations.custom_logger import CustomLogger
    from litellm.llms.anthropic.experimental_pass_through.messages.agentic_streaming_iterator import (
        AgenticAnthropicStreamingIterator,
    )
    from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
        AnthropicMessagesConfig,
    )

    class NoOpAgenticCallback(CustomLogger):
        async def async_should_run_agentic_loop(
            self,
            response,
            model,
            messages,
            tools,
            stream,
            custom_llm_provider,
            kwargs,
        ):
            return False, {}

    handler = BaseLLMHTTPHandler()

    sse_body = (
        b'event: message_start\ndata: {"type": "message_start"}\n\n'
        b'event: message_stop\ndata: {"type": "message_stop"}\n\n'
    )
    upstream_response = httpx.Response(
        200,
        headers={"x-amzn-requestid": "amzn-req-456"},
        content=sse_body,
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
    )
    mock_client = AsyncMock(spec=AsyncHTTPHandler)
    mock_client.post = AsyncMock(return_value=upstream_response)

    mock_logging_obj = Mock()
    mock_logging_obj.model_call_details = {}
    mock_logging_obj.dynamic_success_callbacks = [NoOpAgenticCallback()]

    result = await handler.async_anthropic_messages_handler(
        model="claude-sonnet-4-20250514",
        messages=[{"role": "user", "content": "Hello"}],
        anthropic_messages_provider_config=AnthropicMessagesConfig(),
        anthropic_messages_optional_request_params={"max_tokens": 32},
        custom_llm_provider="anthropic",
        litellm_params=GenericLiteLLMParams(),
        logging_obj=mock_logging_obj,
        client=mock_client,
        api_key="sk-test",
        stream=True,
        kwargs={},
    )

    assert isinstance(result, ABCAsyncIterator)
    assert isinstance(result.completion_stream, AgenticAnthropicStreamingIterator)
    assert result._hidden_params["additional_headers"]["llm_provider-x-amzn-requestid"] == "amzn-req-456"

    collected = b"".join([chunk async for chunk in result])
    assert b"message_start" in collected
    assert b"message_stop" in collected


@pytest.mark.asyncio
async def test_anthropic_messages_streaming_response_aclose_closes_upstream_stream():
    """
    Regression test: the proxy's streaming cleanup calls aclose on the
    handler's return value (see _finalize_streaming_generator_cleanup's
    hasattr(response, "aclose") check). The wrapper must forward aclose to
    the upstream stream so provider connections are released on client
    disconnect instead of lingering until garbage collection.
    """
    from litellm.llms.anthropic.experimental_pass_through.messages.streaming_iterator import (
        AnthropicMessagesStreamingResponse,
    )

    class UpstreamTracker:
        def __init__(self):
            self.closed = False

    tracker = UpstreamTracker()

    async def upstream():
        try:
            yield b'data: {"type": "message_start"}\n\n'
            yield b'data: {"type": "message_stop"}\n\n'
        finally:
            tracker.closed = True

    stream = AnthropicMessagesStreamingResponse(
        completion_stream=upstream(),
        hidden_params={"additional_headers": {}},
    )

    first_chunk = await stream.__anext__()
    assert b"message_start" in first_chunk
    assert tracker.closed is False

    await stream.aclose()
    assert tracker.closed is True


@pytest.mark.asyncio
async def test_anthropic_messages_streaming_response_aclose_closes_agentic_upstream_stream():
    from litellm.llms.anthropic.experimental_pass_through.messages.agentic_streaming_iterator import (
        AgenticAnthropicStreamingIterator,
    )
    from litellm.llms.anthropic.experimental_pass_through.messages.streaming_iterator import (
        AnthropicMessagesStreamingResponse,
    )

    class UpstreamTracker:
        def __init__(self):
            self.closed = False

    tracker = UpstreamTracker()

    async def upstream():
        try:
            yield b'data: {"type": "message_start"}\n\n'
            yield b'data: {"type": "message_stop"}\n\n'
        finally:
            tracker.closed = True

    agentic_iterator = AgenticAnthropicStreamingIterator(
        completion_stream=upstream(),
        http_handler=Mock(),
        model="claude-sonnet-4-20250514",
        messages=[{"role": "user", "content": "Hello"}],
        anthropic_messages_provider_config=Mock(),
        anthropic_messages_optional_request_params={},
        logging_obj=Mock(),
        custom_llm_provider="anthropic",
        kwargs={},
    )
    stream = AnthropicMessagesStreamingResponse(
        completion_stream=agentic_iterator,
        hidden_params={"additional_headers": {}},
    )

    first_chunk = await stream.__anext__()
    assert b"message_start" in first_chunk
    assert tracker.closed is False

    await stream.aclose()
    assert tracker.closed is True


@pytest.mark.asyncio
async def test_async_anthropic_messages_handler_passes_litellm_metadata():
    """Ensure litellm_metadata from kwargs is forwarded via update_from_kwargs.

    Routes like /messages store model_info under kwargs['litellm_metadata'].
    The handler must forward this so that use_custom_pricing_for_model can
    detect custom pricing. Regression test for #23185.
    """
    handler = BaseLLMHTTPHandler()

    mock_config = Mock()
    mock_config.validate_anthropic_messages_environment = Mock(
        return_value=({"x-api-key": "test-key"}, "https://api.anthropic.com")
    )
    mock_config.transform_anthropic_messages_request = Mock(
        return_value={"model": "claude-sonnet-4-20250514", "messages": []}
    )

    mock_client = AsyncMock()
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello!"}],
        "model": "claude-sonnet-4-20250514",
        "stop_reason": "end_turn",
    }
    mock_client.post = AsyncMock(return_value=mock_response)

    mock_logging_obj = Mock()
    mock_logging_obj.update_from_kwargs = Mock()
    mock_logging_obj.model_call_details = {}
    mock_logging_obj.stream = False

    custom_model_info = {
        "id": "claude-sonnet-4-custom-pricing",
        "input_cost_per_token": 0.0003,
        "output_cost_per_token": 0.0015,
    }
    kwargs = {
        "litellm_metadata": {
            "model_info": custom_model_info,
            "deployment": "anthropic/claude-sonnet-4-20250514",
        },
    }

    try:
        await handler.async_anthropic_messages_handler(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "Hello"}],
            anthropic_messages_provider_config=mock_config,
            anthropic_messages_optional_request_params={},
            custom_llm_provider="anthropic",
            litellm_params=GenericLiteLLMParams(),
            logging_obj=mock_logging_obj,
            client=mock_client,
            kwargs=kwargs,
        )
    except Exception:
        pass

    mock_logging_obj.update_from_kwargs.assert_called_once()
    call_kwargs = mock_logging_obj.update_from_kwargs.call_args
    kwargs_arg = (
        call_kwargs.kwargs.get("kwargs", call_kwargs[1].get("kwargs", {}))
        if call_kwargs.kwargs
        else call_kwargs[1].get("kwargs", {})
    )

    assert "litellm_metadata" in kwargs_arg
    assert kwargs_arg["litellm_metadata"]["model_info"] == custom_model_info


@pytest.mark.asyncio
async def test_async_anthropic_messages_handler_forwards_router_model_info():
    """Ensure router deployment model_info is forwarded into litellm_params.

    The Router stamps kwargs['model_info'] on every deployment dispatch via
    _update_kwargs_with_deployment. Downstream cooldown / success callbacks
    (router.deployment_callback_on_failure, deployment_callback_on_success)
    look up the deployment id via kwargs['litellm_params']['model_info']['id'].
    If async_anthropic_messages_handler builds its own litellm_params dict
    without forwarding model_info, the id is missing and cooldown is silently
    skipped for /v1/messages requests under the Router.
    """
    handler = BaseLLMHTTPHandler()

    mock_config = Mock()
    mock_config.validate_anthropic_messages_environment = Mock(
        return_value=({"x-api-key": "test-key"}, "https://api.anthropic.com")
    )
    mock_config.transform_anthropic_messages_request = Mock(
        return_value={"model": "claude-sonnet-4-20250514", "messages": []}
    )

    mock_client = AsyncMock()
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello!"}],
        "model": "claude-sonnet-4-20250514",
        "stop_reason": "end_turn",
    }
    mock_client.post = AsyncMock(return_value=mock_response)

    mock_logging_obj = Mock()
    mock_logging_obj.update_from_kwargs = Mock()
    mock_logging_obj.model_call_details = {}
    mock_logging_obj.stream = False

    deployment_model_info = {
        "id": "deployment-123",
        "db_model": False,
    }

    try:
        await handler.async_anthropic_messages_handler(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "Hello"}],
            anthropic_messages_provider_config=mock_config,
            anthropic_messages_optional_request_params={},
            custom_llm_provider="anthropic",
            litellm_params=GenericLiteLLMParams(),
            logging_obj=mock_logging_obj,
            client=mock_client,
            kwargs={"model_info": deployment_model_info},
        )
    except Exception:
        pass

    mock_logging_obj.update_from_kwargs.assert_called_once()
    call_kwargs = mock_logging_obj.update_from_kwargs.call_args
    litellm_params_arg = (
        call_kwargs.kwargs.get("litellm_params", call_kwargs[1].get("litellm_params", {}))
        if call_kwargs.kwargs
        else call_kwargs[1].get("litellm_params", {})
    )

    assert litellm_params_arg.get("model_info") == deployment_model_info


@pytest.mark.asyncio
async def test_async_anthropic_messages_handler_header_priority():
    """
    Test that async_anthropic_messages_handler respects header priority:
    forwarded < extra_headers < provider_specific
    """
    handler = BaseLLMHTTPHandler()

    # Mock the config
    mock_config = Mock()
    mock_client = AsyncMock()
    mock_logging_obj = Mock()
    mock_logging_obj.update_environment_variables = Mock()
    mock_logging_obj.model_call_details = {}
    mock_logging_obj.stream = False

    # Test with all three header sources
    kwargs = {
        "headers": {"X-Priority": "forwarded", "X-Forwarded-Only": "keep"},
        "extra_headers": {"X-Priority": "extra", "X-Extra-Only": "also-keep"},
    }

    with patch(
        "litellm.litellm_core_utils.get_provider_specific_headers.ProviderSpecificHeaderUtils.get_provider_specific_headers"
    ) as mock_provider_headers:
        mock_provider_headers.return_value = {
            "X-Priority": "provider",
            "X-Provider-Only": "keep-this-too",
        }

        captured_headers = {}

        def capture_validate(*args, **kwargs):
            captured_headers.update(kwargs.get("headers", {}))
            return ({"x-api-key": "test-key"}, "https://api.anthropic.com")

        mock_config.validate_anthropic_messages_environment = capture_validate
        mock_config.transform_anthropic_messages_request = Mock(
            return_value={"model": "claude-3-opus-20240229", "messages": []}
        )

        try:
            await handler.async_anthropic_messages_handler(
                model="claude-3-opus-20240229",
                messages=[{"role": "user", "content": "Hello"}],
                anthropic_messages_provider_config=mock_config,
                anthropic_messages_optional_request_params={},
                custom_llm_provider="anthropic",
                litellm_params=GenericLiteLLMParams(),
                logging_obj=mock_logging_obj,
                client=mock_client,
                kwargs=kwargs,
            )
        except Exception:
            pass

        # Verify priority: provider_specific should win
        assert captured_headers["X-Priority"] == "provider"
        # Verify all unique headers from different sources are present
        assert captured_headers["X-Forwarded-Only"] == "keep"
        assert captured_headers["X-Extra-Only"] == "also-keep"
        assert captured_headers["X-Provider-Only"] == "keep-this-too"


@pytest.mark.asyncio
async def test_async_anthropic_messages_handler_drops_top_level_and_nested_params():
    """
    Regression for LIT-3988 / GitHub #25931: on the /v1/messages path,
    additional_drop_params must strip plain top-level keys (e.g. `thinking`,
    `context_management`) before the provider transform runs, not only nested
    dotted paths. Bedrock rejects these fields, so leaving them in produces a 400.
    """
    handler = BaseLLMHTTPHandler()

    mock_config = Mock()
    mock_config.validate_anthropic_messages_environment = Mock(
        return_value=({"x-api-key": "test-key"}, "https://api.anthropic.com")
    )

    captured = {}

    def capture_transform(*args, **kwargs):
        captured["optional_params"] = kwargs["anthropic_messages_optional_request_params"]
        return {"model": "claude-opus-4-7", "messages": []}

    mock_config.transform_anthropic_messages_request = capture_transform

    mock_client = AsyncMock()
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello!"}],
        "model": "claude-opus-4-7",
        "stop_reason": "end_turn",
    }
    mock_client.post = AsyncMock(return_value=mock_response)

    mock_logging_obj = Mock()
    mock_logging_obj.update_from_kwargs = Mock()
    mock_logging_obj.model_call_details = {}
    mock_logging_obj.stream = False

    optional_params = {
        "max_tokens": 1024,
        "thinking": {"type": "enabled", "budget_tokens": 2048},
        "context_management": {"edits": [{"type": "clear_thinking_20251015"}]},
        "metadata": {"user_id": "u1", "drop_me": "x"},
    }

    with patch(
        "litellm.litellm_core_utils.get_provider_specific_headers.ProviderSpecificHeaderUtils.get_provider_specific_headers"
    ) as mock_provider_headers:
        mock_provider_headers.return_value = None
        try:
            await handler.async_anthropic_messages_handler(
                model="claude-opus-4-7",
                messages=[{"role": "user", "content": "Hello"}],
                anthropic_messages_provider_config=mock_config,
                anthropic_messages_optional_request_params=optional_params,
                custom_llm_provider="bedrock",
                litellm_params=GenericLiteLLMParams(
                    additional_drop_params=[
                        "thinking",
                        "context_management",
                        "metadata.drop_me",
                    ]
                ),
                logging_obj=mock_logging_obj,
                client=mock_client,
            )
        except Exception:
            pass  # drop runs before the mocked sign_request; the capture is what we assert on

    transformed = captured["optional_params"]
    assert "thinking" not in transformed
    assert "context_management" not in transformed
    assert transformed["max_tokens"] == 1024
    assert transformed["metadata"] == {"user_id": "u1"}


def test_google_genai_streaming_hidden_params_model_info_and_router_fallback():
    logging_obj = Mock()
    logging_obj.get_router_model_id = Mock(return_value="router-model-id")

    from_model_info = _google_genai_streaming_hidden_params(
        api_base="https://generativelanguage.googleapis.com/v1beta",
        litellm_params=GenericLiteLLMParams(model_info={"id": "info-id"}),
        logging_obj=logging_obj,
        response_headers=httpx.Headers({"x-ratelimit-remaining": "10"}),
    )
    assert from_model_info["model_id"] == "info-id"
    assert from_model_info["api_base"] == "https://generativelanguage.googleapis.com/v1beta"
    assert isinstance(from_model_info["additional_headers"], dict)

    from_router = _google_genai_streaming_hidden_params(
        api_base="https://x",
        litellm_params=GenericLiteLLMParams(),
        logging_obj=logging_obj,
        response_headers=httpx.Headers({}),
    )
    assert from_router["model_id"] == "router-model-id"


def _build_delete_response_mock(captured: dict):
    """Returns a fake httpx delete that records its kwargs."""

    def _response() -> httpx.Response:
        return httpx.Response(
            status_code=200,
            headers={"content-type": "application/json"},
            content=b'{"id": "resp_x", "object": "response", "deleted": true}',
            request=httpx.Request(method="DELETE", url="https://test.openai.azure.com"),
        )

    async def fake_async_delete(*args, **kwargs):
        captured.update(kwargs)
        return _response()

    def fake_sync_delete(*args, **kwargs):
        captured.update(kwargs)
        return _response()

    return fake_async_delete, fake_sync_delete


def test_async_delete_responses_omits_body_for_azure():
    """Azure responses DELETE rejects requests with any body. Verify the handler
    does not pass `json=` to httpx when the transformer returns an empty dict."""
    captured: dict = {}
    fake_async_delete, _ = _build_delete_response_mock(captured)

    async def run():
        with patch.object(AsyncHTTPHandler, "delete", new=fake_async_delete):
            await litellm.adelete_responses(
                response_id="resp_xyz",
                custom_llm_provider="azure",
                api_base="https://test.openai.azure.com",
                api_key="test-key",
                api_version="2025-03-01-preview",
            )

    asyncio.run(run())

    assert "json" not in captured
    assert "data" not in captured
    assert captured["url"].endswith("/openai/responses/resp_xyz?api-version=2025-03-01-preview")


def test_sync_delete_responses_omits_body_for_azure():
    captured: dict = {}
    _, fake_sync_delete = _build_delete_response_mock(captured)

    with patch.object(HTTPHandler, "delete", new=fake_sync_delete):
        litellm.delete_responses(
            response_id="resp_xyz",
            custom_llm_provider="azure",
            api_base="https://test.openai.azure.com",
            api_key="test-key",
            api_version="2025-03-01-preview",
        )

    assert "json" not in captured
    assert "data" not in captured
    assert captured["url"].endswith("/openai/responses/resp_xyz?api-version=2025-03-01-preview")


def _content_type(headers: dict) -> str:
    for key, value in headers.items():
        if key.lower() == "content-type":
            return value
    return ""


def test_async_delete_responses_sets_json_content_type():
    """OpenAI rejects a responses DELETE with no Content-Type by treating it as
    application/octet-stream. The handler must declare application/json."""
    captured: dict = {}
    fake_async_delete, _ = _build_delete_response_mock(captured)

    async def run():
        with patch.object(AsyncHTTPHandler, "delete", new=fake_async_delete):
            await litellm.adelete_responses(
                response_id="resp_xyz",
                custom_llm_provider="openai",
                api_key="test-key",
            )

    asyncio.run(run())

    assert _content_type(captured["headers"]) == "application/json"


def test_sync_delete_responses_sets_json_content_type():
    captured: dict = {}
    _, fake_sync_delete = _build_delete_response_mock(captured)

    with patch.object(HTTPHandler, "delete", new=fake_sync_delete):
        litellm.delete_responses(
            response_id="resp_xyz",
            custom_llm_provider="openai",
            api_key="test-key",
        )

    assert _content_type(captured["headers"]) == "application/json"


# ---------------------------------------------------------------------------
# Parity tests: request-body is serialized once and reused for the wire.
# (_async_post_anthropic_messages_with_http_error_retry)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "litellm_params_kwargs, stream, global_timeout, expected",
    [
        ({"timeout": 12.0}, False, None, 12.0),
        ({"request_timeout": 30.0}, False, None, 30.0),
        ({}, False, 1500.0, 1500.0),
        ({"timeout": 5.0, "stream_timeout": 50.0}, True, None, 50.0),
        ({"timeout": 5.0, "stream_timeout": 50.0}, False, None, 5.0),
        ({"timeout": 5.0, "request_timeout": 30.0}, False, None, 5.0),
        ({}, False, None, None),
        ({}, True, None, None),
    ],
)
def test_resolve_anthropic_messages_timeout(
    monkeypatch, litellm_params_kwargs, stream, global_timeout, expected
):
    from litellm.constants import DEFAULT_REQUEST_TIMEOUT_SECONDS

    if global_timeout is None:
        monkeypatch.setattr(
            "litellm.request_timeout",
            float(DEFAULT_REQUEST_TIMEOUT_SECONDS),
            raising=False,
        )
        monkeypatch.setattr(
            "litellm.request_timeout_explicitly_set",
            False,
            raising=False,
        )
    else:
        monkeypatch.setattr("litellm.request_timeout", global_timeout, raising=False)
        monkeypatch.setattr(
            "litellm.request_timeout_explicitly_set", True, raising=False
        )

    resolved = BaseLLMHTTPHandler._resolve_anthropic_messages_timeout(
        litellm_params=GenericLiteLLMParams(**litellm_params_kwargs),
        stream=stream,
        custom_llm_provider="anthropic",
    )

    assert resolved == expected


@pytest.mark.asyncio
async def test_async_anthropic_messages_handler_forwards_request_timeout(monkeypatch):
    from litellm.constants import DEFAULT_REQUEST_TIMEOUT_SECONDS

    monkeypatch.setattr(litellm, "callbacks", [])
    monkeypatch.setattr(litellm, "request_timeout", float(DEFAULT_REQUEST_TIMEOUT_SECONDS))
    monkeypatch.setattr(litellm, "request_timeout_explicitly_set", False)
    handler = BaseLLMHTTPHandler()

    mock_config = Mock()
    mock_config.validate_anthropic_messages_environment = Mock(
        return_value=({"x-api-key": "k"}, "https://api.anthropic.com")
    )
    mock_config.should_filter_anthropic_beta_headers = Mock(return_value=False)
    mock_config.transform_anthropic_messages_request = Mock(
        return_value={"model": "claude", "messages": []}
    )
    mock_config.get_complete_url = Mock(return_value="https://api.anthropic.com/v1/messages")
    mock_config.sign_request = Mock(return_value=({"x-api-key": "k"}, None))
    mock_config.max_retry_on_anthropic_messages_http_error = 1
    expected_response = {"id": "msg_1", "content": []}
    mock_config.transform_anthropic_messages_response = Mock(return_value=expected_response)

    ok_response = Mock()
    ok_response.raise_for_status = Mock(return_value=None)
    mock_client = AsyncMock(spec=AsyncHTTPHandler)
    mock_client.post = AsyncMock(return_value=ok_response)

    logging_obj = Mock()
    logging_obj.model_call_details = {}
    logging_obj.dynamic_success_callbacks = []

    result = await handler.async_anthropic_messages_handler(
        model="claude",
        messages=[{"role": "user", "content": "hi"}],
        anthropic_messages_provider_config=mock_config,
        anthropic_messages_optional_request_params={},
        custom_llm_provider="anthropic",
        litellm_params=GenericLiteLLMParams(request_timeout=0.3),
        logging_obj=logging_obj,
        client=mock_client,
        kwargs={},
    )

    assert result is expected_response
    assert mock_client.post.await_args.kwargs["timeout"] == 0.3


@pytest.mark.asyncio
async def test_async_anthropic_messages_handler_forwards_stream_timeout(monkeypatch):
    from litellm.constants import DEFAULT_REQUEST_TIMEOUT_SECONDS

    monkeypatch.setattr(litellm, "callbacks", [])
    monkeypatch.setattr(litellm, "request_timeout", float(DEFAULT_REQUEST_TIMEOUT_SECONDS))
    monkeypatch.setattr(litellm, "request_timeout_explicitly_set", False)
    handler = BaseLLMHTTPHandler()

    mock_config = Mock()
    mock_config.validate_anthropic_messages_environment = Mock(
        return_value=({"x-api-key": "k"}, "https://api.anthropic.com")
    )
    mock_config.should_filter_anthropic_beta_headers = Mock(return_value=False)
    mock_config.transform_anthropic_messages_request = Mock(
        return_value={"model": "claude", "messages": []}
    )
    mock_config.get_complete_url = Mock(return_value="https://api.anthropic.com/v1/messages")
    mock_config.sign_request = Mock(return_value=({"x-api-key": "k"}, None))
    mock_config.max_retry_on_anthropic_messages_http_error = 1
    mock_config.get_async_streaming_response_iterator = Mock(return_value=Mock())

    ok_response = Mock()
    ok_response.raise_for_status = Mock(return_value=None)
    ok_response.headers = httpx.Headers({})
    mock_client = AsyncMock(spec=AsyncHTTPHandler)
    mock_client.post = AsyncMock(return_value=ok_response)

    logging_obj = Mock()
    logging_obj.model_call_details = {}
    logging_obj.dynamic_success_callbacks = []

    await handler.async_anthropic_messages_handler(
        model="claude",
        messages=[{"role": "user", "content": "hi"}],
        anthropic_messages_provider_config=mock_config,
        anthropic_messages_optional_request_params={},
        custom_llm_provider="anthropic",
        litellm_params=GenericLiteLLMParams(timeout=9.0, stream_timeout=0.7),
        logging_obj=logging_obj,
        client=mock_client,
        stream=True,
        kwargs={},
    )

    assert mock_client.post.await_args.kwargs["stream"] is True
    assert mock_client.post.await_args.kwargs["timeout"] == 0.7


@pytest.mark.asyncio
async def test_anthropic_post_uses_prebuilt_body_without_redumping():
    """When the caller passes a pre-serialized (unsigned) body, attempt 0 must
    send exactly those bytes -- no second json.dumps of request_body."""
    import json as _json

    handler = BaseLLMHTTPHandler()
    request_body = {"model": "claude", "messages": [{"role": "user", "content": "hi"}]}
    prebuilt = _json.dumps(request_body)

    ok_resp = Mock()
    ok_resp.raise_for_status = Mock(return_value=None)
    http_client = Mock()
    http_client.post = AsyncMock(return_value=ok_resp)

    provider_config = Mock()
    provider_config.max_retry_on_anthropic_messages_http_error = 2

    logging_obj = Mock()
    logging_obj.model_call_details = {}

    out = await handler._async_post_anthropic_messages_with_http_error_retry(
        async_httpx_client=http_client,
        request_url="http://x/v1/messages",
        headers={},
        signed_json_body=prebuilt,
        request_body=request_body,
        stream=False,
        logging_obj=logging_obj,
        provider_config=provider_config,
        litellm_params=GenericLiteLLMParams(),
        api_key="k",
        model="claude",
    )
    assert out is ok_resp
    http_client.post.assert_awaited_once()
    sent = http_client.post.await_args.kwargs["data"]
    # Byte-identical to the legacy wire serialization, and the SAME object the
    # caller already used for the pre-call log (no re-serialization).
    assert sent == prebuilt
    assert sent is prebuilt


@pytest.mark.asyncio
async def test_anthropic_post_falls_back_to_json_dumps_when_unsigned_none():
    """signed_json_body=None keeps the exact legacy behavior."""
    import json as _json

    handler = BaseLLMHTTPHandler()
    request_body = {"model": "claude", "messages": [{"role": "user", "content": "yo"}]}

    ok_resp = Mock()
    ok_resp.raise_for_status = Mock(return_value=None)
    http_client = Mock()
    http_client.post = AsyncMock(return_value=ok_resp)

    provider_config = Mock()
    provider_config.max_retry_on_anthropic_messages_http_error = 1
    logging_obj = Mock()
    logging_obj.model_call_details = {}

    await handler._async_post_anthropic_messages_with_http_error_retry(
        async_httpx_client=http_client,
        request_url="http://x/v1/messages",
        headers={},
        signed_json_body=None,
        request_body=request_body,
        stream=False,
        logging_obj=logging_obj,
        provider_config=provider_config,
        litellm_params=GenericLiteLLMParams(),
        api_key="k",
        model="claude",
    )
    sent = http_client.post.await_args.kwargs["data"]
    assert sent == _json.dumps(request_body)


@pytest.mark.asyncio
async def test_anthropic_post_retry_reserializes_mutated_body():
    """On a retryable HTTP error the body is mutated + re-signed; the prebuilt
    body must NOT be reused -- attempt 1 sends the freshly serialized body."""
    import json as _json

    handler = BaseLLMHTTPHandler()
    request_body = {"model": "claude", "messages": [{"role": "user", "content": "a"}]}
    prebuilt = _json.dumps(request_body)

    err_resp = Mock()
    http_error = httpx.HTTPStatusError("bad", request=Mock(), response=Mock(status_code=400))
    err_resp.raise_for_status = Mock(side_effect=http_error)
    ok_resp = Mock()
    ok_resp.raise_for_status = Mock(return_value=None)
    http_client = Mock()
    http_client.post = AsyncMock(side_effect=[err_resp, ok_resp])

    def _mutate(e, request_data):
        request_data["messages"][0]["content"] = "MUTATED"

    provider_config = Mock()
    provider_config.max_retry_on_anthropic_messages_http_error = 2
    provider_config.should_retry_anthropic_messages_on_http_error = Mock(return_value=True)
    provider_config.transform_anthropic_messages_request_on_http_error = _mutate
    # Re-sign returns no signed body (native anthropic path) -> must re-dump.
    provider_config.sign_request = Mock(return_value=({}, None))

    logging_obj = Mock()
    logging_obj.model_call_details = {}

    await handler._async_post_anthropic_messages_with_http_error_retry(
        async_httpx_client=http_client,
        request_url="http://x/v1/messages",
        headers={},
        signed_json_body=prebuilt,
        request_body=request_body,
        stream=False,
        logging_obj=logging_obj,
        provider_config=provider_config,
        litellm_params=GenericLiteLLMParams(),
        api_key="k",
        model="claude",
    )
    assert http_client.post.await_count == 2
    first_sent = http_client.post.await_args_list[0].kwargs["data"]
    second_sent = http_client.post.await_args_list[1].kwargs["data"]
    assert first_sent == prebuilt  # attempt 0 used prebuilt
    assert second_sent == _json.dumps(request_body)  # attempt 1 re-serialized
    assert "MUTATED" in second_sent  # ... the mutated body


def test_base_responses_config_sign_request_is_noop_by_default():
    """Default responses sign_request must be a no-op: unchanged headers, no signed body.

    Guards the 15 existing responses providers from accidental signing when the
    handler starts calling sign_request.
    """
    from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig

    cfg = OpenAIResponsesAPIConfig()
    headers = {"Authorization": "Bearer sk-existing"}
    out_headers, signed_body = cfg.sign_request(
        headers=headers,
        optional_params={},
        request_data={"input": "hi"},
        api_base="https://api.openai.com/v1/responses",
    )
    assert out_headers == {"Authorization": "Bearer sk-existing"}
    assert signed_body is None


def _make_responses_handler_call(signed_body):
    """Drive BaseLLMHTTPHandler.response_api_handler with a fully mocked provider
    config + sync client, returning the kwargs the client.post was called with.

    signed_body=None simulates a no-op (non-signing) provider; bytes simulates a
    signing provider (e.g. Bedrock Mantle).
    """
    from unittest.mock import MagicMock
    from litellm.llms.custom_httpx.http_handler import HTTPHandler
    from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
    from litellm.types.router import GenericLiteLLMParams

    provider_config = MagicMock()
    provider_config.validate_environment.return_value = {}
    provider_config.get_complete_url.return_value = "https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses"
    provider_config.transform_responses_api_request.return_value = {"input": "hi"}
    provider_config.should_fake_stream.return_value = False
    provider_config.sign_request.return_value = ({"X-Signed": "1"}, signed_body)

    mock_client = MagicMock(spec=HTTPHandler)
    mock_client.post.return_value = MagicMock()

    handler = BaseLLMHTTPHandler()
    handler.response_api_handler(
        model="openai.gpt-5.5",
        input="hi",
        responses_api_provider_config=provider_config,
        response_api_optional_request_params={},
        custom_llm_provider="bedrock_mantle",
        litellm_params=GenericLiteLLMParams(aws_region_name="us-east-2"),
        logging_obj=MagicMock(),
        client=mock_client,
        _is_async=False,
    )
    return mock_client.post.call_args.kwargs


def test_responses_handler_sends_json_when_not_signed():
    """No-op provider (signed_body is None) -> handler posts json=data, no data= bytes."""
    kwargs = _make_responses_handler_call(signed_body=None)
    assert kwargs.get("json") == {"input": "hi"}
    assert "data" not in kwargs


def test_responses_handler_sends_signed_bytes_when_signed():
    """Signing provider -> handler posts the exact signed bytes via data=, not json=."""
    kwargs = _make_responses_handler_call(signed_body=b'{"input": "hi"}')
    assert kwargs.get("data") == b'{"input": "hi"}'
    assert "json" not in kwargs
    assert kwargs["headers"] == {"X-Signed": "1"}


def test_responses_handler_signs_after_fake_stream_prep_strips_stream():
    """Fake-stream signing-order invariant: the bytes SIGNED must equal the bytes SENT.

    In the streaming + fake-stream path the handler first runs
    _prepare_fake_stream_request, which pops "stream" out of the body, and only
    then calls sign_request. If signing ran before that pop, the signed body
    would still carry "stream" while the body sent over the wire would not,
    producing a SigV4 payload-hash mismatch (401) for a real Mantle deployment.
    We snapshot request_data at sign time and assert "stream" is already gone.
    """
    from unittest.mock import MagicMock
    from litellm.llms.custom_httpx.http_handler import HTTPHandler
    from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
    from litellm.types.llms.openai import ResponsesAPIResponse
    from litellm.types.router import GenericLiteLLMParams

    provider_config = MagicMock()
    provider_config.validate_environment.return_value = {}
    provider_config.get_complete_url.return_value = "https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses"
    provider_config.transform_responses_api_request.return_value = {
        "input": "hi",
        "stream": True,
    }
    provider_config.should_fake_stream.return_value = True
    provider_config.transform_response_api_response.return_value = ResponsesAPIResponse(
        id="resp_1",
        created_at=0,
        output=[],
        status="completed",
        model="openai.gpt-5.5",
    )

    captured = {}

    def _capture_sign(**kwargs):
        captured["request_data"] = dict(kwargs["request_data"])
        return ({"X-Signed": "1"}, b'{"input": "hi"}')

    provider_config.sign_request.side_effect = _capture_sign

    mock_client = MagicMock(spec=HTTPHandler)
    mock_client.post.return_value = MagicMock()

    handler = BaseLLMHTTPHandler()
    handler.response_api_handler(
        model="openai.gpt-5.5",
        input="hi",
        responses_api_provider_config=provider_config,
        response_api_optional_request_params={"stream": True},
        custom_llm_provider="bedrock_mantle",
        litellm_params=GenericLiteLLMParams(aws_region_name="us-east-2"),
        logging_obj=MagicMock(),
        client=mock_client,
        _is_async=False,
        fake_stream=True,
    )

    assert "stream" not in captured["request_data"]
    assert "input" in captured["request_data"]

    post_kwargs = mock_client.post.call_args.kwargs
    assert post_kwargs.get("data") == b'{"input": "hi"}'
    assert "json" not in post_kwargs
    assert "stream" in post_kwargs


def _make_compact_handler_call(signed_body, is_async):
    """Drive (async_)compact_response_api_handler with a fully mocked provider config
    + client, returning the kwargs the client.post was called with.

    signed_body=None simulates a no-op (non-signing) provider; bytes simulates a
    signing provider (e.g. Bedrock Mantle SigV4 / bearer).
    """
    from unittest.mock import MagicMock
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
    from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
    from litellm.types.router import GenericLiteLLMParams

    compact_url = "https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses/compact"
    provider_config = MagicMock()
    provider_config.validate_environment.return_value = {}
    provider_config.get_complete_url.return_value = "https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses"
    provider_config.transform_compact_response_api_request.return_value = (
        compact_url,
        {"model": "openai.gpt-5.5", "input": "hi"},
    )
    provider_config.sign_request.return_value = ({"X-Signed": "1"}, signed_body)
    provider_config.transform_compact_response_api_response.return_value = "ok"

    spec = AsyncHTTPHandler if is_async else HTTPHandler
    mock_client = MagicMock(spec=spec)
    if is_async:
        mock_client.post = AsyncMock(return_value=MagicMock())
    else:
        mock_client.post.return_value = MagicMock()

    handler = BaseLLMHTTPHandler()
    result = handler.compact_response_api_handler(
        model="openai.gpt-5.5",
        input="hi",
        responses_api_provider_config=provider_config,
        response_api_optional_request_params={},
        custom_llm_provider="bedrock_mantle",
        litellm_params=GenericLiteLLMParams(aws_region_name="us-east-2"),
        logging_obj=MagicMock(),
        client=mock_client,
        _is_async=is_async,
    )
    if is_async:
        asyncio.run(result)
    return provider_config, mock_client.post.call_args.kwargs


def test_compact_handler_sends_json_when_not_signed():
    """No-op provider on compact (signed_body is None) -> posts json=data, no data= bytes."""
    provider_config, kwargs = _make_compact_handler_call(signed_body=None, is_async=False)
    provider_config.sign_request.assert_called_once()
    assert kwargs.get("json") == {"model": "openai.gpt-5.5", "input": "hi"}
    assert "data" not in kwargs


def test_compact_handler_sends_signed_bytes_when_signed():
    """Signing provider on compact -> posts the signed bytes via data=, not json=.

    Regression for the adversarial-review finding that /responses/compact bypassed
    the SigV4 signing hook, so IAM-only Mantle callers sent unsigned bodies.
    """
    provider_config, kwargs = _make_compact_handler_call(
        signed_body=b'{"model": "openai.gpt-5.5", "input": "hi"}', is_async=False
    )
    assert kwargs.get("data") == b'{"model": "openai.gpt-5.5", "input": "hi"}'
    assert "json" not in kwargs
    assert kwargs["headers"] == {"X-Signed": "1"}
    # signing must use the compact endpoint as api_base, not the create URL
    assert provider_config.sign_request.call_args.kwargs["api_base"].endswith("/openai/v1/responses/compact")


def test_async_compact_handler_sends_signed_bytes_when_signed():
    """Async compact must sign identically to sync (same omission in the async twin)."""
    provider_config, kwargs = _make_compact_handler_call(
        signed_body=b'{"model": "openai.gpt-5.5", "input": "hi"}', is_async=True
    )
    assert kwargs.get("data") == b'{"model": "openai.gpt-5.5", "input": "hi"}'
    assert "json" not in kwargs
    assert kwargs["headers"] == {"X-Signed": "1"}


def test_async_compact_handler_sends_json_when_not_signed():
    """Async no-op provider on compact -> posts json=data, no data= bytes."""
    _provider_config, kwargs = _make_compact_handler_call(signed_body=None, is_async=True)
    assert kwargs.get("json") == {"model": "openai.gpt-5.5", "input": "hi"}
    assert "data" not in kwargs


@pytest.mark.asyncio
async def test_async_anthropic_messages_handler_passes_api_key_to_agentic_hooks():
    """
    Regression: async_anthropic_messages_handler must inject api_key into the
    kwargs dict forwarded to _call_agentic_completion_hooks.

    Without this, follow-up calls made by agentic hooks (e.g. websearch
    interception's second LLM call after executing searches) have no api_key
    and fail with "x-api-key header is required".
    """
    handler = BaseLLMHTTPHandler()

    mock_config = Mock()
    mock_config.validate_anthropic_messages_environment = Mock(
        return_value=({"x-api-key": "sk-test"}, "https://api.anthropic.com")
    )
    mock_config.transform_anthropic_messages_request = Mock(
        return_value={"model": "claude-haiku", "messages": [], "max_tokens": 16}
    )
    mock_config.sign_request = Mock(return_value=({}, None))

    fake_raw_response = {"id": "msg_1", "type": "message", "role": "assistant", "content": [], "stop_reason": "end_turn"}
    mock_config.transform_anthropic_messages_response = Mock(return_value=fake_raw_response)

    mock_logging_obj = Mock()
    mock_logging_obj.update_environment_variables = Mock()
    mock_logging_obj.model_call_details = {}
    mock_logging_obj.stream = False
    mock_logging_obj.dynamic_success_callbacks = None

    captured_kwargs: dict = {}
    sentinel_response = object()

    async def fake_agentic_hooks(**call_kwargs):
        captured_kwargs.update(call_kwargs)
        return sentinel_response

    mock_httpx_response = Mock()
    mock_httpx_response.status_code = 200

    with (
        patch.object(handler, "_async_post_anthropic_messages_with_http_error_retry", new=AsyncMock(return_value=mock_httpx_response)),
        patch.object(handler, "_call_agentic_completion_hooks", side_effect=fake_agentic_hooks),
        patch("litellm.llms.custom_httpx.llm_http_handler.get_async_httpx_client"),
        patch("litellm.litellm_core_utils.get_provider_specific_headers.ProviderSpecificHeaderUtils.get_provider_specific_headers", return_value=None),
    ):
        result = await handler.async_anthropic_messages_handler(
            model="claude-haiku",
            messages=[{"role": "user", "content": "hi"}],
            anthropic_messages_provider_config=mock_config,
            anthropic_messages_optional_request_params={"stream": False},
            custom_llm_provider="anthropic",
            litellm_params=GenericLiteLLMParams(api_key="sk-real-anthropic-key"),
            logging_obj=mock_logging_obj,
            api_key="sk-real-anthropic-key",
            stream=False,
        )

    assert result is sentinel_response
    assert "kwargs" in captured_kwargs, "_call_agentic_completion_hooks not called"
    forwarded = captured_kwargs["kwargs"]
    assert forwarded.get("api_key") == "sk-real-anthropic-key", (
        "api_key must be injected into kwargs passed to _call_agentic_completion_hooks "
        "so follow-up calls in agentic hooks (e.g. websearch) can authenticate"
    )


class _FakeWSExceptions:
    class WebSocketException(Exception):
        pass

    class InvalidStatusCode(WebSocketException):
        def __init__(self) -> None:
            super().__init__("HTTP 403")

    # websockets>=15 raises InvalidStatus (not InvalidStatusCode) for a rejected
    # client handshake; both must be treated as deterministic.
    class InvalidStatus(WebSocketException):
        def __init__(self) -> None:
            super().__init__("HTTP 401")


class _FakeWebsocketsModule:
    """Stand-in for the ``websockets`` module so the realtime backend-open retry
    can be exercised without a real network handshake (dependency injection,
    no monkeypatching)."""

    def __init__(self, outcomes):
        # outcomes: list where each item is either an Exception to raise or a
        # sentinel object to return as the "connected" websocket.
        self._outcomes = list(outcomes)
        self.exceptions = _FakeWSExceptions
        self.attempts = 0
        self.open_timeouts: list = []

    async def connect(self, *args, **kwargs):
        self.attempts += 1
        self.open_timeouts.append(kwargs.get("open_timeout"))
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.mark.asyncio
async def test_realtime_backend_open_retries_then_succeeds():
    """A hung/slow open handshake is retried; a later fresh attempt connects.

    Regression for intermittent ``1011 timed out during opening handshake``:
    the proxy used to surface a single slow upstream handshake to the caller as
    a fatal 1011 with no retry.
    """
    sentinel = object()
    fake = _FakeWebsocketsModule([TimeoutError("timed out during opening handshake"), sentinel])

    result = await BaseLLMHTTPHandler._open_realtime_backend_ws(
        fake, "wss://backend.example/live", {"Authorization": "Bearer x"}, None
    )

    assert result is sentinel
    assert fake.attempts == 2
    # Each attempt must be bounded by a finite open_timeout (not the default/None).
    assert all(t is not None and t > 0 for t in fake.open_timeouts)


@pytest.mark.asyncio
async def test_realtime_backend_open_raises_after_max_attempts():
    """When every attempt times out, the final error propagates (so the caller
    still closes the client socket) rather than looping forever."""
    fake = _FakeWebsocketsModule([TimeoutError("hang")] * 2)

    with pytest.raises(TimeoutError):
        await BaseLLMHTTPHandler._open_realtime_backend_ws(fake, "wss://backend.example/live", {}, None, max_attempts=2)

    assert fake.attempts == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rejection",
    [_FakeWSExceptions.InvalidStatusCode, _FakeWSExceptions.InvalidStatus],
)
async def test_realtime_backend_open_does_not_retry_auth_failure(rejection):
    """A deterministic handshake-status rejection (auth/4xx) must not be retried;
    retrying cannot help and the upstream status must surface, not a 1011. Both
    the websockets<15 (InvalidStatusCode) and >=15 (InvalidStatus) shapes apply."""
    fake = _FakeWebsocketsModule([rejection()])

    with pytest.raises(_FakeWSExceptions.WebSocketException):
        await BaseLLMHTTPHandler._open_realtime_backend_ws(fake, "wss://backend.example/live", {}, None)

    assert fake.attempts == 1


class _FakeClientWebSocket:
    def __init__(self, send_error=None):
        self.events = []
        self._send_error = send_error

    async def send_text(self, payload):
        if self._send_error is not None:
            raise self._send_error
        self.events.append(("send_text", payload))

    async def close(self, code=None, reason=None):
        self.events.append(("close", (code, reason)))


async def _run_async_realtime_with_backend_failure(client_ws):
    import websockets.exceptions  # noqa: F401  # binds the submodule so async_realtime's except clause resolves, as in the proxy process

    handler = BaseLLMHTTPHandler()
    provider_config = Mock()
    provider_config.get_complete_url.return_value = "wss://backend.example/live"
    provider_config.validate_environment.return_value = {}

    with patch.object(
        handler,
        "_open_realtime_backend_ws",
        AsyncMock(side_effect=Exception("vertex token refresh exploded")),
    ):
        await handler.async_realtime(
            model="gemini-live-2.5-flash",
            websocket=client_ws,
            logging_obj=Mock(),
            provider_config=provider_config,
            headers={},
        )


@pytest.mark.asyncio
async def test_async_realtime_generic_failure_sends_error_event_then_reasoned_close():
    """Regression for the realtime accept-then-silence hang: a generic backend
    failure used to close the client socket without any error event, so callers
    only saw a bare 1011. The client must receive an OpenAI-style error event
    before the reasoned close."""
    client_ws = _FakeClientWebSocket()

    await _run_async_realtime_with_backend_failure(client_ws)

    assert [name for name, _ in client_ws.events] == ["send_text", "close"]

    error_event = json.loads(client_ws.events[0][1])
    assert error_event["type"] == "error"
    assert error_event["error"]["type"] == "server_error"
    assert "vertex token refresh exploded" in error_event["error"]["message"]

    assert client_ws.events[1][1] == (1011, "Internal server error: vertex token refresh exploded")


@pytest.mark.asyncio
async def test_async_realtime_error_event_send_failure_still_closes():
    """A client socket that already dropped must not turn the loud-failure path
    into a new exception: the error-event send may fail, but the reasoned close
    must still be attempted."""
    client_ws = _FakeClientWebSocket(send_error=RuntimeError("client already disconnected"))

    await _run_async_realtime_with_backend_failure(client_ws)

    assert client_ws.events == [("close", (1011, "Internal server error: vertex token refresh exploded"))]


class _JSONBodyAudioTranscriptionConfig(BaseAudioTranscriptionConfig):
    def get_supported_openai_params(self, model):
        return []

    def map_openai_params(self, non_default_params, optional_params, model, drop_params):
        return optional_params

    def validate_environment(
        self,
        headers,
        model,
        messages,
        optional_params,
        litellm_params,
        api_key=None,
        api_base=None,
    ):
        return {**headers, "Authorization": "Bearer test-token"}

    def get_complete_url(self, api_base, api_key, model, optional_params, litellm_params, stream=None):
        return "https://transcription.example/recognize"

    def transform_audio_transcription_request(self, model, audio_file, optional_params, litellm_params):
        return AudioTranscriptionRequestData(data={"config": {"model": model}, "content": "YXVkaW8="})

    def transform_audio_transcription_response(self, raw_response):
        return TranscriptionResponse(text=raw_response.json()["text"])

    def get_error_class(self, error_message, status_code, headers):
        return BaseLLMException(message=error_message, status_code=status_code, headers=headers)


def _json_transcription_call_kwargs(provider_config):
    return {
        "model": "test-model",
        "audio_file": b"raw-audio",
        "optional_params": {},
        "litellm_params": {},
        "model_response": TranscriptionResponse(),
        "timeout": 10.0,
        "max_retries": 0,
        "logging_obj": Mock(),
        "api_key": None,
        "api_base": None,
        "custom_llm_provider": "custom",
        "headers": {},
        "provider_config": provider_config,
    }


def _capture_json_transcription_request(captured):
    def respond(request):
        captured["content_type"] = request.headers.get("content-type")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"text": "transcribed"})

    return respond


def test_audio_transcriptions_sends_dict_data_as_json_body():
    """Regression: dict request data was passed to httpx's data= param, which
    form-encodes it and silently ignores json=; JSON-body providers (e.g.
    Google Speech-to-Text) need an application/json body."""
    captured = {}
    client = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(_capture_json_transcription_request(captured))))

    response = BaseLLMHTTPHandler().audio_transcriptions(
        client=client,
        atranscription=False,
        **_json_transcription_call_kwargs(_JSONBodyAudioTranscriptionConfig()),
    )

    assert captured["content_type"] == "application/json"
    assert captured["body"] == {"config": {"model": "test-model"}, "content": "YXVkaW8="}
    assert response.text == "transcribed"


@pytest.mark.asyncio
async def test_async_audio_transcriptions_sends_dict_data_as_json_body():
    captured = {}
    client = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(_capture_json_transcription_request(captured)))

    response = await BaseLLMHTTPHandler().async_audio_transcriptions(
        client=client,
        **_json_transcription_call_kwargs(_JSONBodyAudioTranscriptionConfig()),
    )

    assert captured["content_type"] == "application/json"
    assert captured["body"] == {"config": {"model": "test-model"}, "content": "YXVkaW8="}
    assert response.text == "transcribed"


@pytest.mark.asyncio
async def test_async_retrieve_file_content_raises_on_http_error():
    """
    LIT-4008 regression: a provider error response (e.g. Anthropic's 400
    "File id must have `file_` prefix") must raise instead of being wrapped
    as file content, which downstream batch cost tracking would parse as an
    empty results file and bill $0.
    """
    from litellm.llms.anthropic.common_utils import AnthropicError
    from litellm.llms.anthropic.files.transformation import AnthropicFilesConfig

    handler = BaseLLMHTTPHandler()
    client = Mock(spec=AsyncHTTPHandler)
    client.get = AsyncMock(
        return_value=httpx.Response(
            status_code=400,
            content=b'{"type":"error","error":{"type":"invalid_request_error","message":"File id must have `file_` prefix."}}',
        )
    )

    with pytest.raises(AnthropicError) as exc_info:
        await handler.async_retrieve_file_content(
            file_content_request={"file_id": "msgbatch_123"},
            provider_config=AnthropicFilesConfig(),
            litellm_params={"api_key": "sk-test"},
            headers={},
            logging_obj=Mock(),
            client=client,
        )

    assert exc_info.value.status_code == 400
    assert "file_" in str(exc_info.value)


def test_sync_retrieve_file_content_raises_on_http_error():
    from litellm.llms.anthropic.common_utils import AnthropicError
    from litellm.llms.anthropic.files.transformation import AnthropicFilesConfig

    handler = BaseLLMHTTPHandler()
    client = Mock(spec=HTTPHandler)
    client.get = Mock(
        return_value=httpx.Response(
            status_code=404,
            content=b'{"type":"error","error":{"type":"not_found_error","message":"not found"}}',
        )
    )

    with pytest.raises(AnthropicError) as exc_info:
        handler.retrieve_file_content(
            file_content_request={"file_id": "file-abc"},
            provider_config=AnthropicFilesConfig(),
            litellm_params={"api_key": "sk-test"},
            headers={},
            logging_obj=Mock(),
            client=client,
        )

    assert exc_info.value.status_code == 404


_UPSTREAM_NOT_FOUND_BODY = {
    "error": {
        "message": "Response with id 'resp_abc' not found.",
        "type": "invalid_request_error",
        "param": None,
        "code": None,
    }
}


def _async_handler_returning(status_code: int, body: dict) -> AsyncHTTPHandler:
    handler = AsyncHTTPHandler()
    handler.client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(status_code, json=body))
    )
    return handler


def _sync_handler_returning(status_code: int, body: dict) -> HTTPHandler:
    handler = HTTPHandler()
    handler.client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(status_code, json=body)))
    return handler


@pytest.mark.asyncio
async def test_aget_responses_surfaces_upstream_error_status_instead_of_500():
    client = _async_handler_returning(404, _UPSTREAM_NOT_FOUND_BODY)

    with pytest.raises(litellm.NotFoundError) as excinfo:
        await litellm.aget_responses(
            response_id="resp_abc",
            custom_llm_provider="azure",
            api_base="https://test.openai.azure.com",
            api_key="test-key",
            api_version="2025-03-01-preview",
            client=client,
        )

    assert excinfo.value.status_code == 404
    assert "Response with id 'resp_abc' not found." in excinfo.value.message


def test_get_responses_surfaces_upstream_error_status_instead_of_500():
    client = _sync_handler_returning(404, _UPSTREAM_NOT_FOUND_BODY)

    with pytest.raises(litellm.NotFoundError) as excinfo:
        litellm.get_responses(
            response_id="resp_abc",
            custom_llm_provider="azure",
            api_base="https://test.openai.azure.com",
            api_key="test-key",
            api_version="2025-03-01-preview",
            client=client,
        )

    assert excinfo.value.status_code == 404
    assert "Response with id 'resp_abc' not found." in excinfo.value.message


def test_list_input_items_surfaces_upstream_error_status():
    client = _sync_handler_returning(404, _UPSTREAM_NOT_FOUND_BODY)

    with pytest.raises(litellm.NotFoundError) as excinfo:
        litellm.list_input_items(
            response_id="resp_abc",
            custom_llm_provider="azure",
            api_base="https://test.openai.azure.com",
            api_key="test-key",
            api_version="2025-03-01-preview",
            client=client,
        )

    assert excinfo.value.status_code == 404


@pytest.mark.asyncio
async def test_alist_input_items_surfaces_upstream_error_status():
    client = _async_handler_returning(404, _UPSTREAM_NOT_FOUND_BODY)

    with pytest.raises(litellm.NotFoundError) as excinfo:
        await litellm.alist_input_items(
            response_id="resp_abc",
            custom_llm_provider="azure",
            api_base="https://test.openai.azure.com",
            api_key="test-key",
            api_version="2025-03-01-preview",
            client=client,
        )

    assert excinfo.value.status_code == 404


@pytest.mark.asyncio
async def test_anthropic_invalid_thinking_signature_retry_resigns_bedrock_request(monkeypatch):
    """Regression: after Bedrock rejects a replayed thinking block (400 invalid signature),
    the strip-and-retry re-sign must not inherit attempt 1's SigV4 Authorization/X-Amz-Date;
    reusing them over the new stripped body makes AWS return 403 SignatureDoesNotMatch."""
    from litellm.llms.bedrock.messages.invoke_transformations.anthropic_claude3_transformation import (
        AmazonAnthropicClaudeMessagesConfig,
    )

    for env_var in ("AWS_BEARER_TOKEN_BEDROCK", "AWS_SESSION_TOKEN", "AWS_PROFILE"):
        monkeypatch.delenv(env_var, raising=False)

    handler = BaseLLMHTTPHandler()
    provider_config = AmazonAnthropicClaudeMessagesConfig()
    litellm_params = GenericLiteLLMParams(
        aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
        aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        aws_region_name="us-east-1",
    )
    request_url = "https://bedrock-runtime.us-east-1.amazonaws.com/model/test-model/invoke"
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "x", "signature": ""},
                    {"type": "text", "text": "ok"},
                ],
            },
            {"role": "user", "content": "continue"},
        ],
    }
    first_attempt_headers, signed_json_body = provider_config.sign_request(
        headers={"Content-Type": "application/json"},
        optional_params=dict(litellm_params),
        request_data=request_body,
        api_base=request_url,
        api_key=None,
        stream=False,
        fake_stream=False,
        model="test-model",
    )

    posts: list = []
    invalid_signature_response = httpx.Response(
        400,
        text='{"message": "messages.1.content.0: Invalid `signature` in `thinking` block"}',
        request=httpx.Request("POST", request_url),
    )
    ok_response = httpx.Response(200, json={"id": "msg_1"}, request=httpx.Request("POST", request_url))

    class FakeAsyncClient:
        async def post(
            self, url, headers, data, stream=False, logging_obj=None, timeout=None
        ):
            posts.append({"headers": dict(headers), "data": data})
            return invalid_signature_response if len(posts) == 1 else ok_response

    logging_obj = Mock()
    logging_obj.model_call_details = {}

    response = await handler._async_post_anthropic_messages_with_http_error_retry(
        async_httpx_client=FakeAsyncClient(),
        request_url=request_url,
        headers=dict(first_attempt_headers),
        signed_json_body=signed_json_body,
        request_body=request_body,
        stream=False,
        logging_obj=logging_obj,
        provider_config=provider_config,
        litellm_params=litellm_params,
        api_key=None,
        model="test-model",
    )

    assert response.status_code == 200
    assert len(posts) == 2
    retry_payload = json.loads(posts[1]["data"])
    retry_blocks = [
        block
        for message in retry_payload["messages"]
        if isinstance(message.get("content"), list)
        for block in message["content"]
    ]
    assert retry_blocks and all(block["type"] != "thinking" for block in retry_blocks)
    retry_authorization = posts[1]["headers"]["Authorization"]
    assert retry_authorization.startswith("AWS4-HMAC-SHA256")
    assert retry_authorization != first_attempt_headers["Authorization"]


class TestServerFulfilledToolsInRequest:
    """_server_fulfilled_tools_in_request gates the buffered (non-leaking) streaming
    mode for server-fulfilled tools like headroom_retrieve."""

    @staticmethod
    def _logging_obj_with(callbacks):
        logging_obj = Mock()
        logging_obj.dynamic_success_callbacks = callbacks
        return logging_obj

    def test_should_hold_back_when_callback_owns_tool_in_request(self):
        from litellm.integrations.custom_logger import CustomLogger

        class RetrievalCallback(CustomLogger):
            server_fulfilled_tool_names = frozenset({"headroom_retrieve"})

        tools = [
            {"name": "Bash", "input_schema": {"type": "object"}},
            {"name": "headroom_retrieve", "input_schema": {"type": "object"}},
        ]
        assert BaseLLMHTTPHandler._server_fulfilled_tools_in_request(
            logging_obj=self._logging_obj_with([RetrievalCallback()]), tools=tools
        ) == frozenset({"headroom_retrieve"})

    def test_should_stream_live_when_tool_absent_from_request(self):
        from litellm.integrations.custom_logger import CustomLogger

        class RetrievalCallback(CustomLogger):
            server_fulfilled_tool_names = frozenset({"headroom_retrieve"})

        tools = [{"name": "Bash", "input_schema": {"type": "object"}}]
        assert (
            BaseLLMHTTPHandler._server_fulfilled_tools_in_request(
                logging_obj=self._logging_obj_with([RetrievalCallback()]), tools=tools
            )
            == frozenset()
        )

    def test_should_stream_live_when_no_callback_declares_tool_names(self):
        from litellm.integrations.custom_logger import CustomLogger

        tools = [{"name": "headroom_retrieve", "input_schema": {"type": "object"}}]
        assert (
            BaseLLMHTTPHandler._server_fulfilled_tools_in_request(
                logging_obj=self._logging_obj_with([CustomLogger()]), tools=tools
            )
            == frozenset()
        )

    def test_should_stream_live_without_tools(self):
        assert (
            BaseLLMHTTPHandler._server_fulfilled_tools_in_request(logging_obj=self._logging_obj_with([]), tools=None)
            == frozenset()
        )

    def test_interception_callbacks_declare_their_retrieval_tools(self):
        from litellm.integrations.compression_interception.handler import (
            LITELLM_CONTENT_RETRIEVE_TOOL_NAME,
            CompressionInterceptionLogger,
        )
        from litellm.proxy.guardrails.guardrail_hooks.headroom.headroom import (
            HEADROOM_RETRIEVE_TOOL_NAME,
            HeadroomGuardrail,
        )

        assert HeadroomGuardrail.server_fulfilled_tool_names == frozenset({HEADROOM_RETRIEVE_TOOL_NAME})
        assert CompressionInterceptionLogger.server_fulfilled_tool_names == frozenset(
            {LITELLM_CONTENT_RETRIEVE_TOOL_NAME}
        )


def _make_stub_direct_vector_store_config(response):
    from litellm.llms.base_llm.vector_store.transformation import (
        BaseDirectVectorStoreConfig,
    )

    class StubDirectVectorStoreConfig(BaseDirectVectorStoreConfig):
        def __init__(self):
            super().__init__()
            self.sync_calls = []
            self.async_calls = []

        def execute_search_vector_store_request(self, **kwargs):
            self.sync_calls.append(kwargs)
            return response

        async def aexecute_search_vector_store_request(self, **kwargs):
            self.async_calls.append(kwargs)
            return response

    return StubDirectVectorStoreConfig()


def test_vector_store_search_handler_direct_config_sync_skips_http():
    handler = BaseLLMHTTPHandler()
    stub_response = {"object": "vector_store.search_results.page", "search_query": "q", "data": []}
    config = _make_stub_direct_vector_store_config(stub_response)
    logging_obj = Mock()

    with patch("litellm.llms.custom_httpx.llm_http_handler._get_httpx_client") as mock_get_client:
        result = handler.vector_store_search_handler(
            vector_store_id="vs_direct",
            query="q",
            vector_store_search_optional_params={"max_num_results": 4},
            vector_store_provider_config=config,
            custom_llm_provider="valkey",
            litellm_params=GenericLiteLLMParams(valkey_host="localhost"),
            logging_obj=logging_obj,
            timeout=12.5,
            _is_async=False,
        )

    assert result is stub_response
    mock_get_client.assert_not_called()
    assert len(config.sync_calls) == 1
    call = config.sync_calls[0]
    assert call["vector_store_id"] == "vs_direct"
    assert call["query"] == "q"
    assert call["timeout"] == 12.5
    assert call["vector_store_search_optional_params"] == {"max_num_results": 4}
    assert isinstance(call["litellm_params"], dict)
    assert call["litellm_params"]["valkey_host"] == "localhost"
    pre_call_args = logging_obj.pre_call.call_args.kwargs["additional_args"]
    assert pre_call_args["query"] == "q"
    assert pre_call_args["vector_store_id"] == "vs_direct"


@pytest.mark.asyncio
async def test_vector_store_search_handler_direct_config_async_skips_http():
    handler = BaseLLMHTTPHandler()
    stub_response = {"object": "vector_store.search_results.page", "search_query": "q", "data": []}
    config = _make_stub_direct_vector_store_config(stub_response)
    logging_obj = Mock()

    with patch("litellm.llms.custom_httpx.llm_http_handler.get_async_httpx_client") as mock_get_client:
        result = await handler.vector_store_search_handler(
            vector_store_id="vs_direct",
            query=["q1", "q2"],
            vector_store_search_optional_params={},
            vector_store_provider_config=config,
            custom_llm_provider="valkey",
            litellm_params=GenericLiteLLMParams(valkey_host="localhost"),
            logging_obj=logging_obj,
            timeout=7.0,
            _is_async=True,
        )

    assert result is stub_response
    mock_get_client.assert_not_called()
    assert len(config.async_calls) == 1
    assert config.async_calls[0]["query"] == ["q1", "q2"]
    assert config.async_calls[0]["litellm_params"]["valkey_host"] == "localhost"
    assert config.async_calls[0]["timeout"] == 7.0
    pre_call_args = logging_obj.pre_call.call_args.kwargs["additional_args"]
    assert pre_call_args["query"] == ["q1", "q2"]
    assert pre_call_args["vector_store_id"] == "vs_direct"


def _direct_vector_store_debug_logging_obj():
    from litellm.litellm_core_utils.litellm_logging import Logging as LitellmLogging

    logging_obj = LitellmLogging(
        model="valkey",
        messages=[{"role": "user", "content": "q"}],
        stream=False,
        call_type="vector_store_search",
        start_time=time.time(),
        litellm_call_id="vs-debug-call-id",
        function_id="vs-debug-function-id",
        log_raw_request_response=True,
    )
    logging_obj.update_environment_variables(
        model="valkey",
        optional_params={"vector_store_id": "vs_direct", "query": "q"},
        litellm_params={
            "litellm_call_id": "vs-debug-call-id",
            "vector_store_id": "vs_direct",
            "litellm_request_debug": True,
            "metadata": {"user_api_key_alias": "vs-test-key"},
            "valkey_host": "valkey.internal",
            "valkey_password": "sup3r-s3cret-valkey-pw",
            "litellm_embedding_config": {"api_key": "sk-embedding-s3cret"},
        },
    )
    return logging_obj


@pytest.mark.parametrize("is_async", [False, True])
def test_direct_vector_store_search_debug_log_omits_stored_credentials(caplog, is_async):
    """Regression: an empty api_base made pre_call dump the whole model_call_details, so every
    search shipped the stored valkey_password / embedding api_key into the raw_request metadata."""
    handler = BaseLLMHTTPHandler()
    stub_response = {"object": "vector_store.search_results.page", "search_query": "q", "data": []}
    config = _make_stub_direct_vector_store_config(stub_response)
    logging_obj = _direct_vector_store_debug_logging_obj()

    with caplog.at_level(logging.DEBUG, logger=verbose_logger.name):
        result = handler.vector_store_search_handler(
            vector_store_id="vs_direct",
            query="q",
            vector_store_search_optional_params={"max_num_results": 4},
            vector_store_provider_config=config,
            custom_llm_provider="valkey",
            litellm_params=GenericLiteLLMParams(
                valkey_host="valkey.internal",
                valkey_password="sup3r-s3cret-valkey-pw",
            ),
            logging_obj=logging_obj,
            _is_async=is_async,
        )
        if is_async:
            result = asyncio.run(result)

    assert result is stub_response
    raw_request = logging_obj.model_call_details["litellm_params"]["metadata"]["raw_request"]
    assert "sup3r-s3cret-valkey-pw" not in raw_request
    assert "sk-embedding-s3cret" not in raw_request
    assert "valkey://vs_direct" in raw_request
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "sup3r-s3cret-valkey-pw" not in logged
    assert "sk-embedding-s3cret" not in logged


@pytest.mark.asyncio
async def test_async_anthropic_messages_handler_carries_deployment_vertex_location_for_pricing(monkeypatch):
    """
    The proxy pre-creates the logging object before the router picks a deployment, so the
    native /v1/messages path must copy the deployment's vertex_location into the logging
    params it updates; otherwise cost resolution falls back to the environment and every
    call on this surface prices with the regional uplift (#34393).
    """
    import contextlib
    from datetime import datetime

    from litellm.litellm_core_utils.litellm_logging import (
        Logging,
        _resolve_vertex_location_for_cost,
    )

    monkeypatch.setenv("VERTEXAI_LOCATION", "us-east5")
    monkeypatch.setattr(litellm, "vertex_location", None)

    handler = BaseLLMHTTPHandler()

    async def logging_obj_after_handler(generic_params):
        logging_obj = Logging(
            model="vertex_ai/claude-haiku-4-5@20251001",
            messages=[{"role": "user", "content": "hi"}],
            stream=False,
            call_type="anthropic_messages",
            start_time=datetime.now(),
            litellm_call_id="vertex-messages-location",
            function_id="f",
        )
        logging_obj.update_environment_variables(
            model="vertex_ai/claude-haiku-4-5@20251001",
            user="",
            optional_params={},
            litellm_params={"api_base": ""},
            custom_llm_provider="vertex_ai",
        )
        mock_config = Mock()
        mock_config.validate_anthropic_messages_environment = Mock(
            return_value=({"authorization": "Bearer t"}, "https://us-east5-aiplatform.googleapis.com")
        )
        mock_config.transform_anthropic_messages_request = Mock(
            return_value={"model": "claude-haiku-4-5@20251001", "messages": []}
        )
        with contextlib.suppress(Exception):
            await handler.async_anthropic_messages_handler(
                model="claude-haiku-4-5@20251001",
                messages=[{"role": "user", "content": "hi"}],
                anthropic_messages_provider_config=mock_config,
                anthropic_messages_optional_request_params={"max_tokens": 10},
                custom_llm_provider="vertex_ai",
                litellm_params=generic_params,
                logging_obj=logging_obj,
                client=AsyncMock(),
                kwargs={},
            )
        return logging_obj

    global_deployment = await logging_obj_after_handler(GenericLiteLLMParams(vertex_location="global"))
    assert global_deployment.litellm_params["vertex_location"] == "global"
    assert (
        _resolve_vertex_location_for_cost(
            custom_llm_provider="vertex_ai",
            litellm_params=global_deployment.litellm_params,
            optional_params=global_deployment.optional_params,
            model="claude-haiku-4-5@20251001",
        )
        == "global"
    )

    unconfigured_deployment = await logging_obj_after_handler(GenericLiteLLMParams())
    assert "vertex_location" not in unconfigured_deployment.litellm_params


_GENERIC_STREAM_SSE = (
    b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,'
    b'"model":"test-model","choices":[{"index":0,"delta":{"content":"hi"},'
    b'"finish_reason":null}]}\n\n'
    b"data: [DONE]\n\n"
)


def _generic_stream_upstream_response() -> httpx.Response:
    return httpx.Response(
        200,
        headers={
            "x-request-id": "generic-req-123",
            "x-ratelimit-remaining-requests": "42",
        },
        content=_GENERIC_STREAM_SSE,
        request=httpx.Request("POST", "https://fake-vllm.test/v1/chat/completions"),
    )


def test_generic_http_handler_sync_streaming_forwards_provider_response_headers():
    """
    Regression test for the generic BaseLLMHTTPHandler streaming path used by
    ~30 providers (deepseek, groq, hosted_vllm, databricks, openrouter, ...).

    The sync `completion()` streaming branch builds the CustomStreamWrapper from
    `make_sync_call`, which returns the upstream response headers alongside the
    stream. Those headers must reach the caller as `llm_provider-*` entries in
    `_hidden_params["additional_headers"]`, which is what the proxy merges into
    the client-facing response headers.
    """
    mock_client = Mock(spec=HTTPHandler)
    mock_client.post = Mock(return_value=_generic_stream_upstream_response())

    response = litellm.completion(
        model="hosted_vllm/test-model",
        messages=[{"role": "user", "content": "Hello"}],
        api_base="https://fake-vllm.test/v1",
        api_key="sk-test",
        stream=True,
        client=mock_client,
    )

    additional_headers = response._hidden_params["additional_headers"]
    assert additional_headers["llm_provider-x-request-id"] == "generic-req-123"
    assert additional_headers["llm_provider-x-ratelimit-remaining-requests"] == "42"

    assert "".join([chunk.choices[0].delta.content or "" for chunk in response]) == "hi"


@pytest.mark.asyncio
async def test_generic_http_handler_async_streaming_forwards_provider_response_headers():
    """
    Companion to the sync test above for `acompletion_stream_function`, which
    builds its CustomStreamWrapper from `make_async_call_stream_helper`.
    """
    mock_client = AsyncMock(spec=AsyncHTTPHandler)
    mock_client.post = AsyncMock(return_value=_generic_stream_upstream_response())

    response = await litellm.acompletion(
        model="hosted_vllm/test-model",
        messages=[{"role": "user", "content": "Hello"}],
        api_base="https://fake-vllm.test/v1",
        api_key="sk-test",
        stream=True,
        client=mock_client,
    )

    additional_headers = response._hidden_params["additional_headers"]
    assert additional_headers["llm_provider-x-request-id"] == "generic-req-123"
    assert additional_headers["llm_provider-x-ratelimit-remaining-requests"] == "42"

    collected = [chunk async for chunk in response]
    assert "".join([chunk.choices[0].delta.content or "" for chunk in collected]) == "hi"


@pytest.mark.parametrize(
    "custom_llm_provider, litellm_params, expected",
    [
        ("openai", GenericLiteLLMParams(rust=True), True),
        ("openai", GenericLiteLLMParams(), False),
        ("openai", GenericLiteLLMParams(rust=False), False),
        ("azure", GenericLiteLLMParams(rust=True), False),
        ("hosted_vllm", GenericLiteLLMParams(rust=True), False),
        (None, GenericLiteLLMParams(rust=True), False),
    ],
)
def test_the_rust_responses_websocket_needs_both_openai_and_the_rust_flag(
    custom_llm_provider, litellm_params, expected
):
    assert _rust_responses_websocket_enabled(custom_llm_provider, litellm_params) is expected


def test_a_plain_callback_does_not_advertise_a_pre_call_deployment_hook(monkeypatch):
    from litellm.integrations.custom_logger import CustomLogger

    class _PlainLogger(CustomLogger):
        pass

    logging_obj = Mock()
    logging_obj.dynamic_success_callbacks = []

    monkeypatch.setattr(litellm, "callbacks", [])
    assert _has_pre_call_deployment_hook(logging_obj) is False

    monkeypatch.setattr(litellm, "callbacks", [_PlainLogger()])
    assert _has_pre_call_deployment_hook(logging_obj) is False


def test_a_callback_that_overrides_the_deployment_hook_is_detected(monkeypatch):
    from litellm.integrations.custom_logger import CustomLogger

    class _DeploymentHookLogger(CustomLogger):
        async def async_pre_call_deployment_hook(self, kwargs, call_type):
            return None

    class _InheritsTheHook(_DeploymentHookLogger):
        pass

    logging_obj = Mock()
    logging_obj.dynamic_success_callbacks = []

    monkeypatch.setattr(litellm, "callbacks", [_DeploymentHookLogger()])
    assert _has_pre_call_deployment_hook(logging_obj) is True

    monkeypatch.setattr(litellm, "callbacks", [_InheritsTheHook()])
    assert _has_pre_call_deployment_hook(logging_obj) is True

    monkeypatch.setattr(litellm, "callbacks", [])
    logging_obj.dynamic_success_callbacks = [_DeploymentHookLogger()]
    assert _has_pre_call_deployment_hook(logging_obj) is True


def test_only_callbacks_that_can_charge_a_frame_are_collected_for_ws_quota(monkeypatch):
    from litellm.integrations.custom_logger import CustomLogger

    class _PlainLogger(CustomLogger):
        pass

    class _QuotaLogger(CustomLogger):
        async def enforce_project_io_token_quota_for_frame(self, *args, **kwargs):
            return None

    class _NotCallableAttribute:
        enforce_project_io_token_quota_for_frame = "not a method"

    plain, quota, decoy = _PlainLogger(), _QuotaLogger(), _NotCallableAttribute()

    monkeypatch.setattr(litellm, "callbacks", [plain, decoy])
    assert _collect_ws_project_quota_callbacks() == ()

    monkeypatch.setattr(litellm, "callbacks", [plain, quota, decoy])
    assert _collect_ws_project_quota_callbacks() == (quota,)


@pytest.mark.asyncio
async def test_async_rerank_records_llm_api_duration():
    """arerank must feed the httpx timing into the logging obj, so the proxy can emit
    x-litellm-overhead-duration-ms / x-litellm-timing-* on /rerank."""

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "rerank-1",
                "results": [{"index": 0, "relevance_score": 0.9}],
                "meta": {"api_version": {"version": "2"}, "billed_units": {"search_units": 1}},
            },
        )

    client = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))

    response = await litellm.arerank(
        model="cohere/rerank-v3.5",
        query="what is the capital of france",
        documents=["paris", "berlin"],
        top_n=1,
        api_key="fake-key",
        client=client,
    )

    assert response._hidden_params["litellm_overhead_time_ms"] is not None
    assert response._hidden_params["_response_ms"] >= response._hidden_params["litellm_overhead_time_ms"]


class _JSONBodyVideoConfig(OpenAIVideoConfig):
    def use_multipart_form_data(self) -> bool:
        return False


def _video_create_call_kwargs(config, **optional_params):
    return {
        "model": "sora-2",
        "prompt": "a cat surfing",
        "video_generation_provider_config": config,
        "video_generation_optional_request_params": {"seconds": "4", **optional_params},
        "custom_llm_provider": "openai",
        "litellm_params": GenericLiteLLMParams(api_key="sk-test", api_base="https://video.example/v1"),
        "logging_obj": Mock(),
        "timeout": 10.0,
    }


def _capture_video_create_request(captured):
    def respond(request):
        captured["content_type"] = request.headers.get("content-type")
        captured["body"] = request.content
        return httpx.Response(
            200,
            json={"id": "video_123", "object": "video", "status": "queued", "created_at": 1712697600, "model": "sora-2"},
        )

    return respond


def _multipart_text_fields(content_type: str, body: bytes) -> dict:
    boundary = content_type.split("boundary=")[1].encode()
    return {
        part.split(b'name="')[1].split(b'"')[0].decode(): part.partition(b"\r\n\r\n")[2].rstrip(b"\r\n-").decode()
        for part in body.split(b"--" + boundary)
        if b'name="' in part and b"filename=" not in part
    }


def test_video_generation_without_file_sends_multipart_form_data():
    """Regression for #36493: the OpenAI SDK always sends /videos requests as
    multipart/form-data, so OpenAI-compatible backends (SGLang Diffusion,
    vLLM-Omni) reject the JSON body LiteLLM used to send when no
    input_reference file was attached."""
    captured = {}
    client = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(_capture_video_create_request(captured))))

    result = BaseLLMHTTPHandler().video_generation_handler(client=client, **_video_create_call_kwargs(OpenAIVideoConfig()))

    assert captured["content_type"].startswith("multipart/form-data")
    assert _multipart_text_fields(captured["content_type"], captured["body"]) == {
        "model": "sora-2",
        "prompt": "a cat surfing",
        "seconds": "4",
    }
    assert result.status == "queued"


@pytest.mark.asyncio
async def test_async_video_generation_without_file_sends_multipart_form_data():
    captured = {}
    client = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(_capture_video_create_request(captured)))

    result = await BaseLLMHTTPHandler().async_video_generation_handler(
        client=client, **_video_create_call_kwargs(OpenAIVideoConfig())
    )

    assert captured["content_type"].startswith("multipart/form-data")
    assert _multipart_text_fields(captured["content_type"], captured["body"]) == {
        "model": "sora-2",
        "prompt": "a cat surfing",
        "seconds": "4",
    }
    assert result.status == "queued"


def test_azure_video_generation_without_file_sends_multipart_form_data():
    """AzureVideoConfig subclasses OpenAIVideoConfig, so it inherits the
    file-less multipart behavior. Azure's /openai/v1/videos surface is
    OpenAI-SDK-compatible (the SDK sends multipart there too), so this is
    intentional; lock it so the inherited flip can't silently regress to JSON."""
    assert AzureVideoConfig().use_multipart_form_data() is True

    captured = {}
    client = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(_capture_video_create_request(captured))))

    result = BaseLLMHTTPHandler().video_generation_handler(client=client, **_video_create_call_kwargs(AzureVideoConfig()))

    assert captured["content_type"].startswith("multipart/form-data")
    assert _multipart_text_fields(captured["content_type"], captured["body"]) == {
        "model": "sora-2",
        "prompt": "a cat surfing",
        "seconds": "4",
    }
    assert result.status == "queued"


def test_video_generation_json_provider_keeps_json_body():
    captured = {}
    client = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(_capture_video_create_request(captured))))

    result = BaseLLMHTTPHandler().video_generation_handler(client=client, **_video_create_call_kwargs(_JSONBodyVideoConfig()))

    assert captured["content_type"] == "application/json"
    assert json.loads(captured["body"]) == {"model": "sora-2", "prompt": "a cat surfing", "seconds": "4"}
    assert result.status == "queued"


def test_video_generation_with_input_reference_keeps_file_multipart():
    captured = {}
    client = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(_capture_video_create_request(captured))))

    result = BaseLLMHTTPHandler().video_generation_handler(
        client=client,
        **_video_create_call_kwargs(OpenAIVideoConfig(), input_reference=b"\x89PNG\r\n\x1a\nfakepng"),
    )

    assert captured["content_type"].startswith("multipart/form-data")
    assert b'name="input_reference"' in captured["body"]
    assert b'filename="input_reference.png"' in captured["body"]
    assert _multipart_text_fields(captured["content_type"], captured["body"]) == {
        "model": "sora-2",
        "prompt": "a cat surfing",
        "seconds": "4",
    }
    assert result.status == "queued"


AZURE_AI_BASE = "https://myfoundry.services.ai.azure.com"
AZURE_AI_CHAT_COMPLETIONS_URL = f"{AZURE_AI_BASE}/models/chat/completions"

def _a_tool_with_an_unsupported_field() -> dict:
    return {
        "type": "function",
        "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}},
        "strict": True,
    }

A_COMPLETION = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1,
    "model": "grok-3",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "sent"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}

TOOL_LEVEL_REJECTION = "Extra inputs are not permitted: tools[0].strict"
UNRELATED_REJECTION = "Extra inputs are not permitted: temperature"
A_REJECTION_THE_PROVIDER_CANNOT_FIX = "The model is not available in this region"


class _RecordedAzureAI:
    def __init__(self, responses: list[httpx.Response]) -> None:
        self._responses = responses
        self.bodies: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        return self._responses[min(len(self.bodies) - 1, len(self._responses) - 1)]


@pytest.fixture
def httpx_transport(monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)


def _rejection(message: str) -> httpx.Response:
    return httpx.Response(422, json={"error": {"message": message}})


def _call_azure_ai(recorder: _RecordedAzureAI, **overrides):
    import respx

    with respx.mock(assert_all_called=True) as router:
        router.post(AZURE_AI_CHAT_COMPLETIONS_URL).mock(side_effect=recorder)
        return litellm.completion(
            model="azure_ai/grok-3",
            messages=[{"role": "user", "content": "hi"}],
            tools=[_a_tool_with_an_unsupported_field()],
            api_base=AZURE_AI_BASE,
            api_key="fake-key",
            **overrides,
        )


def test_a_tool_field_the_provider_rejects_is_dropped_and_the_call_retried():
    recorder = _RecordedAzureAI(
        [_rejection(TOOL_LEVEL_REJECTION), httpx.Response(200, json=A_COMPLETION)]
    )

    response = _call_azure_ai(recorder)

    assert len(recorder.bodies) == 2
    assert recorder.bodies[0]["tools"][0]["strict"] is True
    assert "strict" not in recorder.bodies[1]["tools"][0]
    assert response.choices[0].message.content == "sent"


def test_the_retry_changes_only_the_field_the_provider_named():
    recorder = _RecordedAzureAI(
        [_rejection(TOOL_LEVEL_REJECTION), httpx.Response(200, json=A_COMPLETION)]
    )

    _call_azure_ai(recorder)

    first, second = recorder.bodies
    assert second["messages"] == first["messages"]
    assert second["model"] == first["model"]
    assert second["tools"][0]["function"] == first["tools"][0]["function"]


def test_a_provider_that_keeps_rejecting_is_not_retried_forever():
    recorder = _RecordedAzureAI([_rejection(TOOL_LEVEL_REJECTION)])

    with pytest.raises(litellm.BadRequestError) as raised:
        _call_azure_ai(recorder)

    assert len(recorder.bodies) == 2
    assert raised.value.status_code == 422


def test_a_rejection_the_provider_cannot_fix_is_not_retried_at_all():
    recorder = _RecordedAzureAI([_rejection(A_REJECTION_THE_PROVIDER_CANNOT_FIX)])

    with pytest.raises(litellm.BadRequestError):
        _call_azure_ai(recorder)

    assert len(recorder.bodies) == 1


def test_an_extra_input_outside_a_tool_is_not_retried_unless_dropping_params_was_asked_for():
    recorder = _RecordedAzureAI([_rejection(UNRELATED_REJECTION)])

    with pytest.raises(litellm.BadRequestError):
        _call_azure_ai(recorder)

    assert len(recorder.bodies) == 1


def test_an_extra_input_outside_a_tool_is_retried_when_dropping_params_was_asked_for():
    recorder = _RecordedAzureAI(
        [_rejection(UNRELATED_REJECTION), httpx.Response(200, json=A_COMPLETION)]
    )

    response = _call_azure_ai(recorder, drop_params=True)

    assert len(recorder.bodies) == 2
    assert response.choices[0].message.content == "sent"


@pytest.mark.asyncio
async def test_a_tool_field_the_provider_rejects_is_dropped_and_retried_on_the_async_path(
    httpx_transport,
):
    import respx

    recorder = _RecordedAzureAI(
        [_rejection(TOOL_LEVEL_REJECTION), httpx.Response(200, json=A_COMPLETION)]
    )

    with respx.mock(assert_all_called=True) as router:
        router.post(AZURE_AI_CHAT_COMPLETIONS_URL).mock(side_effect=recorder)
        response = await litellm.acompletion(
            model="azure_ai/grok-3",
            messages=[{"role": "user", "content": "hi"}],
            tools=[_a_tool_with_an_unsupported_field()],
            api_base=AZURE_AI_BASE,
            api_key="fake-key",
        )

    assert len(recorder.bodies) == 2
    assert recorder.bodies[0]["tools"][0]["strict"] is True
    assert "strict" not in recorder.bodies[1]["tools"][0]
    assert response.choices[0].message.content == "sent"


@pytest.mark.asyncio
async def test_a_provider_that_keeps_rejecting_is_not_retried_forever_on_the_async_path(
    httpx_transport,
):
    import respx

    recorder = _RecordedAzureAI([_rejection(TOOL_LEVEL_REJECTION)])

    with respx.mock(assert_all_called=True) as router:
        router.post(AZURE_AI_CHAT_COMPLETIONS_URL).mock(side_effect=recorder)
        with pytest.raises(litellm.BadRequestError):
            await litellm.acompletion(
                model="azure_ai/grok-3",
                messages=[{"role": "user", "content": "hi"}],
                tools=[_a_tool_with_an_unsupported_field()],
                api_base=AZURE_AI_BASE,
                api_key="fake-key",
            )

    assert len(recorder.bodies) == 2
