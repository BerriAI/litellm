#!/usr/bin/env python3
"""
Test to verify the Google GenAI generate_content handler functionality
"""

import asyncio
import datetime
import json
import logging
import threading
from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Final, NotRequired, TypedDict

import httpx
import pytest
import respx

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger

from litellm.google_genai.adapters.handler import GenerateContentToCompletionHandler
from litellm.google_genai.adapters.transformation import GoogleGenAIAdapter, GoogleGenAIStreamWrapper
from litellm.types.utils import Delta, ModelResponseStream, StandardLoggingPayload, StreamingChoices, Usage
from tests._wait_helpers import await_until, wait_until


def test_stream_response_when_stream_requested_sync():
    """
    Test that when a stream response is returned and streaming was requested,
    the sync handler correctly transforms it to generate_content streaming format.
    """
    # Mock a stream response
    mock_stream = MagicMock()
    mock_stream.__iter__ = MagicMock(return_value=iter([]))

    # Mock the GoogleGenAIAdapter's translate_completion_output_params_streaming method
    with patch.object(
        GoogleGenAIAdapter,
        "translate_completion_output_params_streaming",
        return_value=mock_stream,
    ) as mock_translate:
        with patch("litellm.completion", return_value=mock_stream):
            # Call the handler with stream=True
            result = GenerateContentToCompletionHandler.generate_content_handler(
                model="gemini-pro",
                contents=[{"role": "user", "parts": [{"text": "Hello"}]}],
                litellm_params={},  # Empty dict for params
                stream=True,
            )

            # Verify that translate_completion_output_params_streaming was called
            mock_translate.assert_called_once_with(mock_stream)
            # Verify the result is the transformed stream
            assert result == mock_stream


@pytest.mark.asyncio
async def test_stream_response_when_stream_requested_async():
    """
    Test that when a stream response is returned and streaming was requested,
    the async handler correctly transforms it to generate_content streaming format.
    """
    # Mock a stream response
    mock_stream = MagicMock()
    mock_stream.__aiter__ = AsyncMock(return_value=iter([]))  # Return an empty async iterator

    # Mock the GoogleGenAIAdapter's translate_completion_output_params_streaming method
    with patch.object(
        GoogleGenAIAdapter,
        "translate_completion_output_params_streaming",
        return_value=mock_stream,
    ) as mock_translate:
        with patch("litellm.acompletion", return_value=mock_stream):
            # Call the handler with stream=True
            result = await GenerateContentToCompletionHandler.async_generate_content_handler(
                model="gemini-pro",
                contents=[{"role": "user", "parts": [{"text": "Hello"}]}],
                litellm_params={},  # Empty dict for params
                stream=True,
            )

            # Verify that translate_completion_output_params_streaming was called
            mock_translate.assert_called_once_with(mock_stream)
            # Verify the result is the transformed stream
            assert result == mock_stream


def test_stream_transformation_error_sync():
    """
    Test that when a stream transformation fails, the sync handler raises a ValueError.
    """
    # Mock a stream response
    mock_stream = MagicMock()
    mock_stream.__iter__ = MagicMock(return_value=iter([]))

    # Mock the GoogleGenAIAdapter's translate_completion_output_params_streaming method to return None
    with patch.object(
        GoogleGenAIAdapter,
        "translate_completion_output_params_streaming",
        return_value=None,
    ):
        # Patch litellm.completion directly to prevent real API calls
        with patch("litellm.completion", return_value=mock_stream):
            # Call the handler with stream=True and expect a ValueError
            with pytest.raises(ValueError, match="Failed to transform streaming response"):
                GenerateContentToCompletionHandler.generate_content_handler(
                    model="gemini-pro",
                    contents=[{"role": "user", "parts": [{"text": "Hello"}]}],
                    litellm_params={},  # Empty dict for params
                    stream=True,
                )


@pytest.mark.asyncio
async def test_stream_transformation_error_async():
    """
    Test that when a stream transformation fails, the async handler raises a ValueError.
    """
    # Mock a stream response
    mock_stream = MagicMock()
    mock_stream.__aiter__ = AsyncMock(return_value=mock_stream)

    # Mock the GoogleGenAIAdapter's translate_completion_output_params_streaming method to return None
    with patch.object(
        GoogleGenAIAdapter,
        "translate_completion_output_params_streaming",
        return_value=None,
    ):
        # Mock litellm.acompletion at the module level where it's imported
        # We need to patch it in the handler module, not in litellm itself
        with patch("litellm.google_genai.adapters.handler.litellm") as mock_litellm:
            # Use AsyncMock for async function
            mock_litellm.acompletion = AsyncMock(return_value=mock_stream)
            # Call the handler with stream=True and expect a ValueError
            with pytest.raises(ValueError, match="Failed to transform streaming response"):
                await GenerateContentToCompletionHandler.async_generate_content_handler(
                    model="gemini-pro",
                    contents=[{"role": "user", "parts": [{"text": "Hello"}]}],
                    litellm_params={},  # Empty dict for params
                    stream=True,
                )


def test_citation_metadata_transformation():
    """
    Test that citationMetadata.citationSources is properly transformed to citationMetadata.citations
    to avoid Pydantic validation errors.
    """
    from unittest.mock import MagicMock

    import httpx

    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.llms.gemini.google_genai.transformation import GoogleGenAIConfig

    # Create a mock response with citationMetadata.citationSources (the problematic format)
    mock_response_data = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "This is a video analysis response with citation metadata."}],
                    "role": "model",
                },
                "finishReason": "STOP",
                "index": 0,
                "safetyRatings": [],
                "citationMetadata": {
                    "citationSources": [
                        {
                            "startIndex": 5848,
                            "endIndex": 5900,
                            "uri": "https://example.com/video-source",
                            "license": "MIT",
                            "title": "Video Analysis Source",
                            "publicationDate": "2024-01-15",
                        },
                        {
                            "startIndex": 6200,
                            "endIndex": 6250,
                            "uri": "https://another-source.com/reference",
                            "license": "CC-BY",
                            "title": "Another Reference",
                            "publicationDate": "2024-02-01",
                        },
                    ]
                },
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 150,
            "candidatesTokenCount": 200,
            "totalTokenCount": 350,
        },
        "responseId": "test-response-123",
    }

    # Create mock httpx response
    mock_httpx_response = MagicMock(spec=httpx.Response)
    mock_httpx_response.json.return_value = mock_response_data
    mock_httpx_response.status_code = 200
    mock_httpx_response.headers = {}

    # Create logging object
    logging_obj = LiteLLMLoggingObj(
        model="gemini-2.5-flash",
        messages=[],
        stream=False,
        call_type="generate_content",
        start_time=1234567890,
        litellm_call_id="test-call-123",
        function_id="test-function-123",
    )

    # Create GoogleGenAI config
    config = GoogleGenAIConfig()

    # Test the transformation
    try:
        result = config.transform_generate_content_response(
            model="gemini-2.5-flash",
            raw_response=mock_httpx_response,
            logging_obj=logging_obj,
        )

        # Verify the transformation worked
        assert result is not None

        # Check that citationSources was transformed to citations
        if hasattr(result, "candidates") and result.candidates:
            candidate = result.candidates[0]
            if hasattr(candidate, "citationMetadata") and candidate.citationMetadata:
                # The citationMetadata should now have 'citations' instead of 'citationSources'
                citation_metadata = candidate.citationMetadata

                # Check that citations field exists
                assert hasattr(citation_metadata, "citations"), "citations field should exist after transformation"

                # Verify the citations data is preserved
                if hasattr(citation_metadata, "citations") and citation_metadata.citations:
                    assert len(citation_metadata.citations) == 2, "Should have 2 citations"
                    assert citation_metadata.citations[0]["uri"] == "https://example.com/video-source"
                    assert citation_metadata.citations[1]["uri"] == "https://another-source.com/reference"

        print("✅ Citation metadata transformation test passed!")

    except Exception as e:
        pytest.fail(f"Citation metadata transformation failed: {e}")


@pytest.mark.asyncio
async def test_generate_content_adapter_preserves_proxy_server_request():
    """
    Ensure GenerateContentToCompletionHandler forwards proxy_server_request
    to the downstream completion call so proxy spend logging captures the request body.
    """
    from litellm.types.router import GenericLiteLLMParams
    from litellm.types.utils import Choices, Message, ModelResponse

    handler = GenerateContentToCompletionHandler()

    dummy_proxy_request: dict[str, object] = {
        "url": "http://localhost:4000/v1beta/models/gemini-2.0-flash:generateContent",
        "method": "POST",
        "headers": {"content-type": "application/json"},
        "body": {"contents": [{"role": "user", "parts": [{"text": "Hello, world!"}]}]},
    }

    gemini_data: list[dict[str, object]] = [{"role": "user", "parts": [{"text": "Hello, world!"}]}]

    mock_response = ModelResponse(choices=[Choices(message=Message(content="Hi!", role="assistant"))])

    with patch(
        "litellm.google_genai.adapters.handler.litellm.acompletion",
        new_callable=AsyncMock,
    ) as mock_acompletion:
        mock_acompletion.return_value = mock_response

        await handler.async_generate_content_handler(
            model="gemini-2.0-flash",
            contents=gemini_data,
            litellm_params=GenericLiteLLMParams(),
            proxy_server_request=dummy_proxy_request,
            metadata={"source": "unit_test"},
        )

        assert mock_acompletion.called, "Inner acompletion was not called"
        called_kwargs = mock_acompletion.call_args.kwargs

        assert "proxy_server_request" in called_kwargs, "proxy_server_request was dropped from completion_kwargs"
        assert called_kwargs["proxy_server_request"] == dummy_proxy_request


_CHAT_SSE = (
    'data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"m",'
    '"choices":[{"index":0,"delta":{"role":"assistant","content":"144"},"finish_reason":null}]}\n\n'
    'data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"m",'
    '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
    "data: [DONE]\n\n"
)
_CHAT_BODY: dict[str, object] = {
    "id": "c",
    "object": "chat.completion",
    "created": 1,
    "model": "m",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "144"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
}


_CHAT_SSE_WITH_USAGE = _CHAT_SSE.replace(
    "data: [DONE]",
    'data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[],'
    '"usage":{"prompt_tokens":340,"completion_tokens":1,"total_tokens":341}}\n\n'
    "data: [DONE]",
)

_ANTHROPIC_SSE: Final = "".join(
    f"event: {event}\ndata: {json.dumps(data)}\n\n"
    for event, data in (
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-haiku-4-5",
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 20, "output_tokens": 1},
                },
            },
        ),
        (
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "144"}},
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}},
        ),
        ("message_stop", {"type": "message_stop"}),
    )
)


class _CallbackParams(TypedDict):
    litellm_session_id: NotRequired[str | None]


class _CallbackKwargs(TypedDict):
    call_type: str
    litellm_call_id: str
    litellm_params: _CallbackParams
    standard_logging_object: StandardLoggingPayload


class _LoggedEvents(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[tuple[str, str, str, str | None]] = []
        self.response_costs: list[float] = []
        self.session_ids: list[str | None] = []
        self.messages: list[object] = []

    def _record(self, outcome: str, kwargs: _CallbackKwargs) -> None:
        payload: Final = kwargs["standard_logging_object"]
        self.events.append((outcome, kwargs["call_type"], kwargs["litellm_call_id"], payload["model_group"]))
        self.response_costs.append(payload["response_cost"])
        self.session_ids.append(kwargs["litellm_params"].get("litellm_session_id"))
        self.messages.append(payload["messages"])

    def log_success_event(
        self, kwargs: _CallbackKwargs, response_obj: object, start_time: datetime.datetime, end_time: datetime.datetime
    ) -> None:
        self._record("success", kwargs)

    async def async_log_success_event(
        self, kwargs: _CallbackKwargs, response_obj: object, start_time: datetime.datetime, end_time: datetime.datetime
    ) -> None:
        self._record("success", kwargs)

    def log_failure_event(
        self, kwargs: _CallbackKwargs, response_obj: object, start_time: datetime.datetime, end_time: datetime.datetime
    ) -> None:
        self._record("failure", kwargs)

    async def async_log_failure_event(
        self, kwargs: _CallbackKwargs, response_obj: object, start_time: datetime.datetime, end_time: datetime.datetime
    ) -> None:
        self._record("failure", kwargs)


class _CacheHitFlags(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.cache_hits: list[object] = []
        self.call_returned: Final = threading.Event()

    def logging_hook(self, kwargs: dict[str, object], result: object, call_type: str) -> tuple[dict[str, object], object]:
        self.call_returned.wait(timeout=1.0)
        return kwargs, result

    def log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: datetime.datetime, end_time: datetime.datetime
    ) -> None:
        self.cache_hits.append(kwargs.get("cache_hit"))


class _AsyncCacheHitFlags(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.cache_hits: list[object] = []

    async def async_logging_hook(
        self, kwargs: dict[str, object], result: object, call_type: str
    ) -> tuple[dict[str, object], object]:
        await asyncio.sleep(0.3)
        return kwargs, result

    async def async_log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: datetime.datetime, end_time: datetime.datetime
    ) -> None:
        self.cache_hits.append(kwargs.get("cache_hit"))


class _SuccessPayloads(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.model_groups: list[str | None] = []

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.model_groups.append(kwargs["standard_logging_object"]["model_group"])


class _SuccessRows(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[tuple[bool | None, str | None, str]] = []

    async def async_log_success_event(
        self, kwargs: _CallbackKwargs, response_obj: object, start_time: datetime.datetime, end_time: datetime.datetime
    ) -> None:
        payload: Final = kwargs["standard_logging_object"]
        self.rows.append((payload["cache_hit"], payload["model_group"], payload["model"]))


@pytest.fixture
def _httpx_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)


@pytest.fixture
def _litellm_errors(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    verbose_logger.addHandler(caplog.handler)
    yield caplog
    verbose_logger.removeHandler(caplog.handler)


_ADAPTER_CALL_ID: Final = "generate-content-call"
_SUCCESS_SSE: Final = _CHAT_SSE.replace(
    "data: [DONE]",
    'data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[],'
    '"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}\n\n'
    "data: [DONE]",
)
_INPUT_COST_PER_TOKEN: Final = 0.001
_OUTPUT_COST_PER_TOKEN: Final = 0.002


def _adapter_router() -> litellm.Router:
    return litellm.Router(
        model_list=[
            {
                "model_name": "grok-group",
                "litellm_params": {
                    "model": "hosted_vllm/grok",
                    "api_base": "http://upstream.test/v1",
                    "api_key": "k",
                    "input_cost_per_token": _INPUT_COST_PER_TOKEN,
                    "output_cost_per_token": _OUTPUT_COST_PER_TOKEN,
                },
            }
        ]
    )


def _upstream_reply(stream: bool) -> httpx.Response:
    if stream:
        return httpx.Response(200, text=_SUCCESS_SSE, headers={"content-type": "text/event-stream"})
    return httpx.Response(200, json=_CHAT_BODY)


def _litellm_error_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.levelno >= logging.ERROR]


async def _drain(stream: object) -> None:
    async for _ in stream:
        pass


@pytest.mark.asyncio
async def test_generate_content_adapter_keeps_provider_error_status(_httpx_transport: None):
    from litellm.types.router import GenericLiteLLMParams

    with respx.mock() as upstream:
        upstream.post("http://upstream.test/v1/chat/completions").mock(
            return_value=httpx.Response(400, json={"error": {"message": "max_tokens must be positive"}})
        )
        with pytest.raises(litellm.BadRequestError) as exc_info:
            await GenerateContentToCompletionHandler.async_generate_content_handler(
                model="hosted_vllm/grok",
                contents=[{"role": "user", "parts": [{"text": "Hello"}]}],
                litellm_params=GenericLiteLLMParams(api_base="http://upstream.test/v1", api_key="k"),
                config={"maxOutputTokens": -5},
            )

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_generate_content_adapter_logs_router_model_group(
    _httpx_transport: None, monkeypatch: pytest.MonkeyPatch
):
    from litellm.types.router import GenericLiteLLMParams

    success_payloads = _SuccessPayloads()
    monkeypatch.setattr(litellm, "callbacks", [success_payloads])

    with respx.mock() as upstream:
        upstream.post("http://upstream.test/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=_CHAT_BODY)
        )
        await GenerateContentToCompletionHandler.async_generate_content_handler(
            model="hosted_vllm/grok",
            contents=[{"role": "user", "parts": [{"text": "Hello"}]}],
            litellm_params=GenericLiteLLMParams(api_base="http://upstream.test/v1", api_key="k"),
            litellm_metadata={"model_group": "grok-group"},
        )
    await asyncio.sleep(0.5)

    assert "grok-group" in success_payloads.model_groups


@pytest.mark.parametrize(
    ("stream", "requested_stream_options", "expected_stream_options"),
    [
        (True, None, {"include_usage": True}),
        (True, {"include_usage": True}, {"include_usage": True}),
        (True, {"include_usage": False}, None),
        (False, None, None),
    ],
)
@pytest.mark.asyncio
async def test_generate_content_adapter_asks_openai_compatible_upstream_for_stream_usage(
    _httpx_transport: None,
    stream: bool,
    requested_stream_options: dict[str, bool] | None,
    expected_stream_options: dict[str, bool] | None,
):
    from litellm.types.router import GenericLiteLLMParams

    with respx.mock() as upstream:
        route = upstream.post("http://upstream.test/v1/chat/completions").mock(
            return_value=(
                httpx.Response(200, text=_CHAT_SSE, headers={"content-type": "text/event-stream"})
                if stream
                else httpx.Response(200, json=_CHAT_BODY)
            )
        )
        response = await GenerateContentToCompletionHandler.async_generate_content_handler(
            model="hosted_vllm/grok",
            contents=[{"role": "user", "parts": [{"text": "Hello"}]}],
            litellm_params=GenericLiteLLMParams(api_base="http://upstream.test/v1", api_key="k"),
            stream=stream,
            **({} if requested_stream_options is None else {"stream_options": requested_stream_options}),
        )
        if stream:
            await _drain(response)

    assert json.loads(route.calls[0].request.content).get("stream_options") == expected_stream_options


@pytest.mark.parametrize(
    ("model", "api_base", "upstream_url", "upstream_sse", "prompt_tokens"),
    [
        (
            "hosted_vllm/grok",
            "http://upstream.test/v1",
            "http://upstream.test/v1/chat/completions",
            _CHAT_SSE_WITH_USAGE,
            340,
        ),
        ("anthropic/claude-haiku-4-5", "http://upstream.test", "http://upstream.test/v1/messages", _ANTHROPIC_SSE, 20),
    ],
    ids=["openai_compatible", "anthropic"],
)
@pytest.mark.asyncio
async def test_generate_content_adapter_stream_ends_with_upstream_usage(
    _httpx_transport: None, model: str, api_base: str, upstream_url: str, upstream_sse: str, prompt_tokens: int
):
    payloads: Final = await _adapter_stream_payloads(model, api_base, upstream_url, upstream_sse)

    assert payloads[-1]["candidates"][0]["finishReason"] == "STOP"
    assert payloads[-1]["usageMetadata"]["promptTokenCount"] == prompt_tokens


@pytest.mark.parametrize(
    ("model", "api_base", "upstream_url", "upstream_sse"),
    [
        ("hosted_vllm/grok", "http://upstream.test/v1", "http://upstream.test/v1/chat/completions", _CHAT_SSE),
        ("anthropic/claude-haiku-4-5", "http://upstream.test", "http://upstream.test/v1/messages", _ANTHROPIC_SSE),
    ],
    ids=["openai_compatible", "anthropic"],
)
@pytest.mark.asyncio
async def test_generate_content_adapter_stream_without_usage_still_ends_with_stop(
    _httpx_transport: None, model: str, api_base: str, upstream_url: str, upstream_sse: str
):
    payloads: Final = await _adapter_stream_payloads(
        model, api_base, upstream_url, upstream_sse, stream_options={"include_usage": False}
    )

    assert [payload["candidates"][0].get("finishReason") for payload in payloads][-1] == "STOP"


async def _adapter_stream_payloads(
    model: str, api_base: str, upstream_url: str, upstream_sse: str, **kwargs: object
):
    from litellm.types.router import GenericLiteLLMParams

    with respx.mock() as upstream:
        upstream.post(upstream_url).mock(
            return_value=httpx.Response(200, text=upstream_sse, headers={"content-type": "text/event-stream"})
        )
        response = await GenerateContentToCompletionHandler.async_generate_content_handler(
            model=model,
            contents=[{"role": "user", "parts": [{"text": "Hello"}]}],
            litellm_params=GenericLiteLLMParams(api_base=api_base, api_key="k"),
            stream=True,
            **kwargs,
        )
        events: Final = [chunk async for chunk in response]

    return [
        json.loads(line.removeprefix("data: "))
        for line in b"".join(events).decode().splitlines()
        if line.startswith("data: {")
    ]


@pytest.mark.parametrize(("entrypoint", "stream"), [("agenerate_content", False), ("agenerate_content_stream", True)])
@pytest.mark.asyncio
async def test_generate_content_adapter_success_is_logged_once_on_the_request(
    _httpx_transport: None,
    _litellm_errors: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
    stream: bool,
):
    logged: Final = _LoggedEvents()
    monkeypatch.setattr(litellm, "callbacks", [logged])
    per_request_calls: list[tuple[str, str]] = []

    async def per_request_callback(
        kwargs: _CallbackKwargs, completion_response: object, start_time: datetime.datetime, end_time: datetime.datetime
    ) -> None:
        per_request_calls.append((kwargs["call_type"], kwargs["litellm_call_id"]))

    with respx.mock() as upstream:
        upstream.post("http://upstream.test/v1/chat/completions").mock(return_value=_upstream_reply(stream))
        response: Final = await getattr(_adapter_router(), entrypoint)(
            model="grok-group",
            contents=[{"role": "user", "parts": [{"text": "Hello"}]}],
            litellm_call_id=_ADAPTER_CALL_ID,
            success_callback=[per_request_callback],
        )
        if stream:
            await _drain(response)
    await await_until(
        lambda: bool(logged.events) and bool(per_request_calls), message="the request's success callbacks ran"
    )

    assert logged.events == [("success", entrypoint, _ADAPTER_CALL_ID, "grok-group")]
    assert logged.response_costs == [pytest.approx(3 * _INPUT_COST_PER_TOKEN + 1 * _OUTPUT_COST_PER_TOKEN)]
    assert per_request_calls == [(entrypoint, _ADAPTER_CALL_ID)]
    assert _litellm_error_messages(_litellm_errors) == []


def test_sync_generate_content_adapter_success_is_logged_once_on_the_request(
    _httpx_transport: None, _litellm_errors: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
):
    logged: Final = _LoggedEvents()
    monkeypatch.setattr(litellm, "callbacks", [logged])
    per_request_calls: list[tuple[str, str]] = []

    def per_request_callback(
        kwargs: _CallbackKwargs, completion_response: object, start_time: datetime.datetime, end_time: datetime.datetime
    ) -> None:
        per_request_calls.append((kwargs["call_type"], kwargs["litellm_call_id"]))

    with respx.mock() as upstream:
        upstream.post("http://upstream.test/v1/chat/completions").mock(return_value=_upstream_reply(stream=False))
        _adapter_router().generate_content(
            model="grok-group",
            contents=[{"role": "user", "parts": [{"text": "Hello"}]}],
            litellm_call_id=_ADAPTER_CALL_ID,
            success_callback=[per_request_callback],
        )
    wait_until(lambda: bool(logged.events) and bool(per_request_calls), message="the request's success callbacks ran")

    assert logged.events == [("success", "generate_content", _ADAPTER_CALL_ID, "grok-group")]
    assert logged.response_costs == [pytest.approx(3 * _INPUT_COST_PER_TOKEN + 1 * _OUTPUT_COST_PER_TOKEN)]
    assert per_request_calls == [("generate_content", _ADAPTER_CALL_ID)]
    assert _litellm_error_messages(_litellm_errors) == []


def test_sync_generate_content_adapter_cache_hit_reaches_callbacks_as_a_cache_hit(
    _httpx_transport: None, monkeypatch: pytest.MonkeyPatch
):
    logged: Final = _CacheHitFlags()
    monkeypatch.setattr(litellm, "callbacks", [logged])
    monkeypatch.setattr(litellm, "cache", litellm.Cache(type="local"))
    router: Final = _adapter_router()
    prompts: Final = ("first", "second", "third")

    def call(prompt: str, logged_count: int) -> None:
        logged.call_returned.clear()
        router.generate_content(
            model="grok-group", contents=[{"role": "user", "parts": [{"text": prompt}]}], caching=True
        )
        logged.call_returned.set()
        wait_until(lambda: len(logged.cache_hits) == logged_count, message="the call's success callback ran")

    with respx.mock() as upstream:
        route: Final = upstream.post("http://upstream.test/v1/chat/completions").mock(
            return_value=_upstream_reply(stream=False)
        )
        for index, prompt in enumerate(prompts):
            call(prompt, 2 * index + 1)
            call(prompt, 2 * index + 2)

    assert route.call_count == len(prompts)
    assert [hit is True for hit in logged.cache_hits] == [False, True] * len(prompts)


@pytest.mark.asyncio
async def test_async_generate_content_adapter_cache_hit_reaches_callbacks_as_a_cache_hit(
    _httpx_transport: None, monkeypatch: pytest.MonkeyPatch, _litellm_errors: pytest.LogCaptureFixture
):
    logged: Final = _AsyncCacheHitFlags()
    monkeypatch.setattr(litellm, "callbacks", [logged])
    monkeypatch.setattr(litellm, "cache", litellm.Cache(type="local"))
    router: Final = _adapter_router()
    prompts: Final = ("first", "second", "third")

    async def call(prompt: str, logged_count: int) -> None:
        await router.agenerate_content(
            model="grok-group", contents=[{"role": "user", "parts": [{"text": prompt}]}], caching=True
        )
        await await_until(lambda: len(logged.cache_hits) == logged_count, message="the call's success callback ran")

    with respx.mock() as upstream:
        route: Final = upstream.post("http://upstream.test/v1/chat/completions").mock(
            return_value=_upstream_reply(stream=False)
        )
        for index, prompt in enumerate(prompts):
            await call(prompt, 2 * index + 1)
            await call(prompt, 2 * index + 2)

    assert route.call_count == len(prompts)
    assert [hit is True for hit in logged.cache_hits] == [False, True] * len(prompts)
    assert _litellm_error_messages(_litellm_errors) == []


@pytest.mark.asyncio
async def test_generate_content_adapter_cache_hit_on_a_proxy_logging_object_logs_the_routed_model(
    _httpx_transport: None, monkeypatch: pytest.MonkeyPatch
):
    from litellm.litellm_core_utils.rules import Rules
    from litellm.utils import function_setup

    logged: Final = _SuccessRows()
    monkeypatch.setattr(litellm, "callbacks", [logged])
    monkeypatch.setattr(litellm, "cache", litellm.Cache(type="local"))
    router: Final = _adapter_router()
    contents: Final = [{"role": "user", "parts": [{"text": "Hello"}]}]

    async def call(call_id: str, logged_count: int) -> None:
        logging_obj, _ = function_setup(
            "agenerate_content",
            Rules(),
            datetime.datetime.now(),
            model="grok-group",
            contents=contents,
            litellm_call_id=call_id,
        )
        await router.agenerate_content(
            model="grok-group",
            contents=contents,
            caching=True,
            litellm_call_id=call_id,
            litellm_logging_obj=logging_obj,
        )
        await await_until(lambda: len(logged.rows) == logged_count, message="the call's success callback ran")

    with respx.mock() as upstream:
        upstream.post("http://upstream.test/v1/chat/completions").mock(return_value=_upstream_reply(stream=False))
        await call("miss", 1)
        await call("hit", 2)

    miss, hit = logged.rows
    assert (miss[0] is True, hit[0] is True) == (False, True)
    assert miss[1] == "grok-group"
    assert hit[1:] == miss[1:]


@pytest.mark.parametrize(("entrypoint", "stream"), [("agenerate_content", False), ("agenerate_content_stream", True)])
@pytest.mark.asyncio
async def test_generate_content_adapter_logs_the_whole_conversation(
    _httpx_transport: None, monkeypatch: pytest.MonkeyPatch, entrypoint: str, stream: bool
):
    logged: Final = _LoggedEvents()
    monkeypatch.setattr(litellm, "callbacks", [logged])

    with respx.mock() as upstream:
        upstream.post("http://upstream.test/v1/chat/completions").mock(return_value=_upstream_reply(stream))
        response: Final = await getattr(_adapter_router(), entrypoint)(
            model="grok-group",
            contents=[
                {"role": "user", "parts": [{"text": "Say hi."}]},
                {"role": "model", "parts": [{"text": "hi"}]},
                {"role": "user", "parts": [{"text": "Now say bye."}]},
            ],
            systemInstruction={"parts": [{"text": "Answer in three words."}]},
        )
        if stream:
            await _drain(response)
    await await_until(lambda: bool(logged.messages), message="the request's success callback ran")

    assert logged.messages == [
        [
            {"role": "system", "content": "Answer in three words."},
            {"role": "user", "content": "Say hi."},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "Now say bye."},
        ]
    ]


@pytest.mark.asyncio
async def test_generate_content_adapter_keeps_the_request_no_log_and_session_id(
    _httpx_transport: None, monkeypatch: pytest.MonkeyPatch
):
    logged: Final = _LoggedEvents()
    monkeypatch.setattr(litellm, "callbacks", [logged])

    with respx.mock() as upstream:
        upstream.post("http://upstream.test/v1/chat/completions").mock(return_value=_upstream_reply(stream=False))
        for call_id, request_settings in (
            ("unlogged-call", {"no-log": True}),
            (_ADAPTER_CALL_ID, {"litellm_session_id": "adapter-session"}),
        ):
            await litellm.agenerate_content(
                model="hosted_vllm/grok",
                contents=[{"role": "user", "parts": [{"text": "Hello"}]}],
                api_base="http://upstream.test/v1",
                api_key="k",
                litellm_call_id=call_id,
                **request_settings,
            )
    await await_until(lambda: bool(logged.events), message="the logged request reached the callback")

    assert [call_id for _, _, call_id, _ in logged.events] == [_ADAPTER_CALL_ID]
    assert logged.session_ids == ["adapter-session"]


@pytest.mark.parametrize("entrypoint", ["agenerate_content", "agenerate_content_stream"])
@pytest.mark.asyncio
async def test_generate_content_adapter_pre_call_rules_reject_the_request_before_the_upstream_call(
    _httpx_transport: None, monkeypatch: pytest.MonkeyPatch, entrypoint: str
):
    logged: Final = _LoggedEvents()
    monkeypatch.setattr(litellm, "callbacks", [logged])
    monkeypatch.setattr(litellm, "pre_call_rules", [lambda text: "forbidden" not in text])
    generate: Final = getattr(_adapter_router(), entrypoint)

    with respx.mock(assert_all_called=False) as upstream:
        route: Final = upstream.post("http://upstream.test/v1/chat/completions").mock(
            return_value=_upstream_reply(stream=entrypoint == "agenerate_content_stream")
        )
        with pytest.raises(litellm.APIResponseValidationError):
            await generate(model="grok-group", contents=[{"role": "user", "parts": [{"text": "a forbidden question"}]}])
        allowed: Final = await generate(
            model="grok-group",
            contents=[{"role": "user", "parts": [{"text": "hello"}]}],
            litellm_call_id=_ADAPTER_CALL_ID,
        )
        if entrypoint == "agenerate_content_stream":
            await _drain(allowed)
    await await_until(lambda: bool(logged.events), message="the allowed request was logged")

    assert route.call_count == 1
    assert [(outcome, call_id) for outcome, _, call_id, _ in logged.events] == [("success", _ADAPTER_CALL_ID)]


def test_sync_generate_content_adapter_pre_call_rule_rejection_logs_no_failure(
    _httpx_transport: None, monkeypatch: pytest.MonkeyPatch
):
    logged: Final = _LoggedEvents()
    monkeypatch.setattr(litellm, "callbacks", [logged])
    monkeypatch.setattr(litellm, "pre_call_rules", [lambda text: "forbidden" not in text])
    router: Final = _adapter_router()

    with respx.mock(assert_all_called=False) as upstream:
        route: Final = upstream.post("http://upstream.test/v1/chat/completions").mock(
            return_value=_upstream_reply(stream=False)
        )
        with pytest.raises(litellm.APIResponseValidationError):
            router.generate_content(
                model="grok-group", contents=[{"role": "user", "parts": [{"text": "a forbidden question"}]}]
            )
        router.generate_content(
            model="grok-group",
            contents=[{"role": "user", "parts": [{"text": "hello"}]}],
            litellm_call_id=_ADAPTER_CALL_ID,
        )
    wait_until(lambda: bool(logged.events), message="the allowed request was logged")

    assert route.call_count == 1
    assert [(outcome, call_id) for outcome, _, call_id, _ in logged.events] == [("success", _ADAPTER_CALL_ID)]


@pytest.mark.asyncio
async def test_generate_content_adapter_pre_call_rules_skip_an_empty_conversation_like_chat(
    _httpx_transport: None, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(litellm, "pre_call_rules", [lambda text: text != ""])

    with respx.mock() as upstream:
        route: Final = upstream.post("http://upstream.test/v1/chat/completions").mock(
            return_value=_upstream_reply(stream=False)
        )
        await _adapter_router().agenerate_content(model="grok-group", contents=[])

    assert route.call_count == 1


@pytest.mark.asyncio
async def test_generate_content_adapter_pre_call_rule_rejections_leave_the_deployments_available(
    _httpx_transport: None, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(litellm, "pre_call_rules", [lambda text: "forbidden" not in text])
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "grok-group",
                "litellm_params": {"model": "hosted_vllm/grok", "api_base": "http://upstream.test/v1", "api_key": "k"},
                "model_info": {"id": deployment_id},
            }
            for deployment_id in ("grok-1", "grok-2")
        ],
        allowed_fails=1,
    )

    with respx.mock(assert_all_called=False) as upstream:
        route: Final = upstream.post("http://upstream.test/v1/chat/completions").mock(
            return_value=_upstream_reply(stream=False)
        )
        for _ in range(3):
            with pytest.raises(litellm.APIResponseValidationError):
                await router.agenerate_content(
                    model="grok-group",
                    contents=[{"role": "user", "parts": [{"text": "a forbidden question"}]}],
                )
        await router.agenerate_content(model="grok-group", contents=[{"role": "user", "parts": [{"text": "hello"}]}])

    assert route.call_count == 1


@pytest.mark.parametrize("entrypoint", ["agenerate_content", "agenerate_content_stream"])
@pytest.mark.asyncio
async def test_generate_content_adapter_failure_is_logged_once(
    _httpx_transport: None, monkeypatch: pytest.MonkeyPatch, entrypoint: str
):
    logged: Final = _LoggedEvents()
    monkeypatch.setattr(litellm, "callbacks", [logged])

    with respx.mock() as upstream:
        upstream.post("http://upstream.test/v1/chat/completions").mock(
            return_value=httpx.Response(400, json={"error": {"message": "max_tokens must be positive"}})
        )
        with pytest.raises(litellm.BadRequestError):
            await getattr(_adapter_router(), entrypoint)(
                model="grok-group",
                contents=[{"role": "user", "parts": [{"text": "Hello"}]}],
                litellm_call_id=_ADAPTER_CALL_ID,
            )
    await await_until(lambda: bool(logged.events), message="the request's failure was logged")

    assert logged.events == [("failure", entrypoint, _ADAPTER_CALL_ID, "grok-group")]


def test_sync_generate_content_adapter_failure_is_logged_once(_httpx_transport: None, monkeypatch: pytest.MonkeyPatch):
    logged: Final = _LoggedEvents()
    monkeypatch.setattr(litellm, "callbacks", [logged])

    with respx.mock() as upstream:
        upstream.post("http://upstream.test/v1/chat/completions").mock(
            return_value=httpx.Response(400, json={"error": {"message": "max_tokens must be positive"}})
        )
        with pytest.raises(litellm.BadRequestError):
            _adapter_router().generate_content(
                model="grok-group",
                contents=[{"role": "user", "parts": [{"text": "Hello"}]}],
                litellm_call_id=_ADAPTER_CALL_ID,
            )
    wait_until(lambda: bool(logged.events), message="the request's failure was logged")

    assert logged.events == [("failure", "generate_content", _ADAPTER_CALL_ID, "grok-group")]


def _stream_chunk(content: str | None, finish_reason: str | None, usage: Usage | None = None) -> ModelResponseStream:
    return ModelResponseStream(
        choices=[StreamingChoices(index=0, delta=Delta(content=content), finish_reason=finish_reason)],
        **({} if usage is None else {"usage": usage}),
    )


_FINISH_WITHOUT_USAGE: Final = (_stream_chunk("144", None), _stream_chunk(None, "stop"))
_FINISH_WITH_USAGE: Final = (
    _stream_chunk("144", None),
    _stream_chunk(None, "stop", Usage(prompt_tokens=20, completion_tokens=1, total_tokens=21)),
)


@pytest.mark.parametrize(
    ("chunks", "prompt_tokens"), [(_FINISH_WITHOUT_USAGE, 0), (_FINISH_WITH_USAGE, 20)], ids=["no_usage", "usage"]
)
def test_stream_wrapper_iteration_ends_with_one_stop_event(chunks: tuple[ModelResponseStream, ...], prompt_tokens: int):
    events: Final = list(GoogleGenAIStreamWrapper(completion_stream=iter(chunks)))

    assert [event["candidates"][0].get("finishReason") for event in events] == [None, "STOP"]
    assert events[-1]["usageMetadata"]["promptTokenCount"] == prompt_tokens


@pytest.mark.parametrize(
    ("chunks", "prompt_tokens"), [(_FINISH_WITHOUT_USAGE, 0), (_FINISH_WITH_USAGE, 20)], ids=["no_usage", "usage"]
)
@pytest.mark.asyncio
async def test_stream_wrapper_async_iteration_ends_with_one_stop_event(
    chunks: tuple[ModelResponseStream, ...], prompt_tokens: int
):
    async def completion_stream():
        for chunk in chunks:
            yield chunk

    events: Final = [event async for event in GoogleGenAIStreamWrapper(completion_stream=completion_stream())]

    assert [event["candidates"][0].get("finishReason") for event in events] == [None, "STOP"]
    assert events[-1]["usageMetadata"]["promptTokenCount"] == prompt_tokens
