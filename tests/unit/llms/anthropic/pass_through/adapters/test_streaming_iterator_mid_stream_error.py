"""
Regression tests for the ``/v1/messages`` async adapter dropping the socket on a
mid-stream provider error.

When a non-Anthropic model (e.g. Bedrock Converse) is served through
``/v1/messages``, the proxy hands Starlette the async SSE iterator directly. If
the upstream provider stream raises while being pulled (Bedrock raises
``BedrockError`` when a ConverseStream ends without a terminal ``messageStop``
event, common on cross-region inference profiles), the exception escaped the
request handler's try/except and tore down the connection. Clients like Claude
Code then showed a bare "Connection closed mid-response".

The async SSE wrapper must instead surface the failure as a well-formed
Anthropic ``error`` event so the stream stays valid and the client can retry.
"""

import asyncio
import json
import os
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Final, Iterator, List, Optional
from unittest.mock import MagicMock

import httpx
import pytest
from pydantic import TypeAdapter

import litellm

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.exceptions import MidStreamFallbackError
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.llms.anthropic.pass_through.adapters.streaming_iterator import (
    AnthropicStreamWrapper,
    _mid_stream_error_sse_event,
)
from litellm.llms.bedrock.common_utils import BedrockError
from litellm.types.utils import Delta, ModelResponseStream, StandardLoggingPayload, StreamingChoices, Usage


def _make_chunk(delta: Delta, finish_reason: Optional[str] = None) -> MagicMock:
    chunk = MagicMock()
    chunk.choices = [
        StreamingChoices(finish_reason=finish_reason, index=0, delta=delta, logprobs=None)
    ]
    chunk.usage = None
    chunk._hidden_params = {}
    return chunk


class _AsyncStreamThenRaise:
    """Yields the given chunks, then raises ``exc`` (mimics a provider stream
    that terminates mid-response)."""

    def __init__(self, items: Sequence[object], exc: BaseException) -> None:
        self._it = iter(items)
        self._exc = exc

    def __aiter__(self) -> "_AsyncStreamThenRaise":
        return self

    async def __anext__(self) -> object:
        try:
            return next(self._it)
        except StopIteration:
            raise self._exc


def _parse_sse(raw: bytes) -> tuple[str, dict]:
    text = raw.decode()
    event_line, data_line = text.strip().split("\n", 1)
    return event_line.removeprefix("event: "), json.loads(data_line.removeprefix("data: "))


async def _drain_sse(wrapper: AnthropicStreamWrapper) -> List[bytes]:
    return [event async for event in wrapper.async_anthropic_sse_wrapper()]


@pytest.mark.asyncio
async def test_mid_stream_bedrock_error_becomes_anthropic_error_event():
    """A ``BedrockError`` raised after partial content must be surfaced as a
    terminal Anthropic ``error`` event, not propagated (which drops the socket
    and yields "Connection closed mid-response")."""
    chunks = [_make_chunk(Delta(content="Creating a file"))]
    bedrock_err = BedrockError(
        status_code=500,
        message="Bedrock ConverseStream ended without a terminal 'messageStop' event",
    )
    wrapper = AnthropicStreamWrapper(
        completion_stream=_AsyncStreamThenRaise(chunks, bedrock_err),
        model="bedrock-converse-sonnet-4-6",
    )

    events = await _drain_sse(wrapper)

    parsed = [_parse_sse(e) for e in events]
    event_types = [name for name, _ in parsed]
    assert "message_start" in event_types
    assert event_types[-1] == "error"
    _, error_payload = parsed[-1]
    assert error_payload["type"] == "error"
    assert error_payload["error"]["type"] == "api_error"
    assert "messageStop" in error_payload["error"]["message"]


@pytest.mark.asyncio
async def test_mid_stream_error_does_not_raise_out_of_wrapper():
    """The async wrapper must fully drain without letting the upstream exception
    escape — escaping is exactly what tore down the connection before the fix."""
    wrapper = AnthropicStreamWrapper(
        completion_stream=_AsyncStreamThenRaise([], BedrockError(status_code=500, message="boom")),
        model="claude-x",
    )
    events = await _drain_sse(wrapper)
    assert _parse_sse(events[-1])[0] == "error"


@pytest.mark.parametrize(
    "status_code, expected_type",
    [(500, "api_error"), (529, "overloaded_error"), (429, "rate_limit_error")],
)
def test_error_event_maps_status_code_to_anthropic_type(status_code, expected_type):
    raw = _mid_stream_error_sse_event(BedrockError(status_code=status_code, message="upstream failed"))
    name, payload = _parse_sse(raw)
    assert name == "error"
    assert payload["error"]["type"] == expected_type
    assert payload["error"]["message"] == "upstream failed"


def test_error_event_defaults_to_500_when_status_missing():
    raw = _mid_stream_error_sse_event(ValueError("no status here"))
    _, payload = _parse_sse(raw)
    assert payload["error"]["type"] == "api_error"
    assert payload["error"]["message"] == "no status here"


def test_error_event_preserves_midstream_fallback_error():
    exc = MidStreamFallbackError(
        message="BedrockException - internalServerException",
        model="bedrock-converse-sonnet-4-6",
        llm_provider="bedrock",
        original_exception=BedrockError(status_code=500, message="internalServerException"),
    )
    name, payload = _parse_sse(_mid_stream_error_sse_event(exc))
    assert name == "error"
    assert payload["error"]["type"] == "api_error"
    assert "internalServerException" in payload["error"]["message"]


def test_error_event_keeps_status_of_mapped_litellm_exception():
    exc = litellm.ContextWindowExceededError(
        message="prompt is too long: your prompt exceeds the model's context window",
        model="openai.gpt-5.6-luna",
        llm_provider="bedrock_mantle",
    )
    name, payload = _parse_sse(_mid_stream_error_sse_event(exc))
    assert name == "error"
    assert payload["error"]["type"] == "invalid_request_error"
    assert "prompt is too long" in payload["error"]["message"]


@pytest.mark.parametrize(
    "exc",
    [
        litellm.BadRequestError(
            message="temperature must be in the range [0.0, 2.0]", model="m", llm_provider="gemini"
        ),
        litellm.AuthenticationError(message="API key not valid", llm_provider="gemini", model="m"),
        litellm.NotFoundError(message="model is not found", model="m", llm_provider="gemini"),
    ],
)
def test_error_event_reports_other_provider_4xx_as_retriable_500(exc):
    name, payload = _parse_sse(_mid_stream_error_sse_event(exc))
    assert name == "error"
    assert payload["error"]["type"] == "api_error"


class _AsyncFailureRecorder(CustomLogger):
    def __init__(self):
        super().__init__()
        self.exceptions: list[BaseException] = []
        self.usages: tuple[tuple[int, int], ...] = ()

    async def async_log_failure_event(
        self,
        kwargs: Mapping[str, object],
        response_obj: object,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        self.exceptions.append(kwargs["exception"])
        payload: Final = TypeAdapter(StandardLoggingPayload).validate_python(kwargs["standard_logging_object"])
        self.usages = (*self.usages, (payload["prompt_tokens"], payload["completion_tokens"]))


class _BoundaryAsyncFailureRecorder(_AsyncFailureRecorder):
    def __init__(self) -> None:
        super().__init__()
        self.started = 0
        self.completed = 0

    async def async_log_failure_event(
        self,
        kwargs: Mapping[str, object],
        response_obj: object,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        self.started += 1
        await asyncio.sleep(0.05)
        await super().async_log_failure_event(kwargs, response_obj, start_time, end_time)
        self.completed += 1


def test_anthropic_adapter_does_not_repeat_custom_stream_failure_dispatch() -> None:
    async_recorder: Final = _BoundaryAsyncFailureRecorder()
    sync_recorder: Final = _SyncFailureRecorder()
    logging_obj: Final = _make_logging_obj("wrapped-custom-stream", async_recorder, sync_recorder)
    chunk: Final = ModelResponseStream(
        id="chatcmpl-dropped",
        object="chat.completion.chunk",
        created=1,
        model="gpt-5.6",
        choices=[StreamingChoices(index=0, delta=Delta(content="partial"), finish_reason=None)],
        usage=Usage(prompt_tokens=30, completion_tokens=1, total_tokens=31),
    )
    inner_stream: Final = CustomStreamWrapper(
        completion_stream=_AsyncStreamThenRaise([chunk], httpx.ReadError("stream dropped")),
        model="gpt-5.6",
        logging_obj=logging_obj,
        custom_llm_provider="openai",
    )
    wrapper: Final = AnthropicStreamWrapper(
        completion_stream=inner_stream,
        model="openai/gpt-5.6",
        litellm_logging_obj=logging_obj,
    )

    events: Final = asyncio.run(_drain_sse(wrapper))

    assert async_recorder.started == 1
    assert async_recorder.completed == 1
    assert sync_recorder.called.wait(timeout=5), "sync failure callback never ran"
    assert _parse_sse(events[-1])[0] == "error"
    assert len(async_recorder.exceptions) == 1
    assert len(sync_recorder.exceptions) == 1
    assert async_recorder.usages == ((30, 1),)
    assert sync_recorder.exceptions[0] == async_recorder.exceptions[0]


class _SyncFailureRecorder:
    def __init__(self):
        self.exceptions: list[BaseException] = []
        self.called = threading.Event()

    def __call__(self, kwargs, completion_response, start_time, end_time):
        self.exceptions.append(kwargs["exception"])
        self.called.set()


def _make_logging_obj(
    test_name: str,
    async_recorder: _AsyncFailureRecorder,
    sync_recorder: _SyncFailureRecorder,
) -> LiteLLMLoggingObj:
    return LiteLLMLoggingObj(
        model="bedrock-converse-sonnet-4-6",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        call_type="anthropic_messages",
        start_time=datetime.now(),
        litellm_call_id=test_name,
        function_id=test_name,
        dynamic_failure_callbacks=[sync_recorder],
        dynamic_async_failure_callbacks=[async_recorder],
    )


async def _proxy_boundary_hook(exc: Exception) -> None:
    return None


async def _wait_for_sync_failure(sync_recorder: _SyncFailureRecorder) -> None:
    for _ in range(500):
        if sync_recorder.called.is_set():
            return
        await asyncio.sleep(0.01)


def _bedrock_drop() -> BedrockError:
    return BedrockError(status_code=500, message="ConverseStream ended without messageStop")


def _chat_wrapper_envelope() -> MidStreamFallbackError:
    provider_error = _bedrock_drop()
    return MidStreamFallbackError(
        message=str(provider_error),
        model="bedrock-converse-sonnet-4-6",
        llm_provider="bedrock",
        original_exception=provider_error,
        is_pre_first_chunk=False,
    )


_RAISED_ERRORS = pytest.mark.parametrize(
    "raised",
    [_bedrock_drop, _chat_wrapper_envelope],
    ids=["provider_error", "chat_wrapper_envelope"],
)


def _failing_wrapper(
    logging_obj: LiteLLMLoggingObj | None,
    raised: Callable[[], Exception] = _bedrock_drop,
) -> AnthropicStreamWrapper:
    return AnthropicStreamWrapper(
        completion_stream=_AsyncStreamThenRaise([_make_chunk(Delta(content="partial"))], raised()),
        model="bedrock-converse-sonnet-4-6",
        litellm_logging_obj=logging_obj,
    )


@_RAISED_ERRORS
@pytest.mark.asyncio
async def test_mid_stream_error_reraises_the_provider_error_for_proxy_managed_stream(raised):
    async_recorder = _AsyncFailureRecorder()
    sync_recorder = _SyncFailureRecorder()
    logging_obj = _make_logging_obj("proxy-managed", async_recorder, sync_recorder)
    logging_obj.on_detached_stream_failure = _proxy_boundary_hook

    with pytest.raises(BedrockError) as raised_info:
        await _drain_sse(_failing_wrapper(logging_obj, raised))

    assert str(raised_info.value) == "ConverseStream ended without messageStop"
    await asyncio.sleep(0.1)
    assert async_recorder.exceptions == []
    assert sync_recorder.exceptions == []


@_RAISED_ERRORS
@pytest.mark.asyncio
async def test_mid_stream_error_dispatches_the_provider_error_to_failure_handlers_for_standalone_stream(raised):
    async_recorder = _AsyncFailureRecorder()
    sync_recorder = _SyncFailureRecorder()
    wrapper = _failing_wrapper(_make_logging_obj("standalone", async_recorder, sync_recorder), raised)

    events = await _drain_sse(wrapper)
    await _wait_for_sync_failure(sync_recorder)

    assert [str(exc) for exc in async_recorder.exceptions] == ["ConverseStream ended without messageStop"]
    assert [str(exc) for exc in sync_recorder.exceptions] == ["ConverseStream ended without messageStop"]
    assert _parse_sse(events[-1])[0] == "error"


class _BrokenFailureDispatchLogging(LiteLLMLoggingObj):
    async def dispatch_failure_handlers(self, *args, **kwargs):
        raise RuntimeError("failure sink is down")


@pytest.mark.asyncio
async def test_mid_stream_error_frame_survives_a_raising_failure_dispatch():
    logging_obj = _BrokenFailureDispatchLogging(
        model="bedrock-converse-sonnet-4-6",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        call_type="anthropic_messages",
        start_time=datetime.now(),
        litellm_call_id="broken-dispatch",
        function_id="broken-dispatch",
    )

    events = await _drain_sse(_failing_wrapper(logging_obj))

    assert _parse_sse(events[-1])[0] == "error"


@pytest.mark.asyncio
async def test_mid_stream_error_emits_error_event_without_logging_obj():
    events = await _drain_sse(_failing_wrapper(None))

    assert _parse_sse(events[-1])[0] == "error"
