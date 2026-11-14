"""
Scripted streams for the streaming characterization tests.

``ScriptedHost`` outcomes may be callables, so each builder here returns a callable that the host
calls with the request kwargs and that answers with a fresh stream:

- ``chat_stream``: a ``CustomStreamWrapper`` over chat chunks, optionally raising ``error`` after
  ``fail_after`` chunks, and ending with a ``finish_reason`` chunk unless ``finish`` is False.
- ``anthropic_stream``: an async iterator of raw ``/v1/messages`` SSE byte frames, optionally raising.
- ``responses_stream``: a ``BaseResponsesAPIStreamingIterator`` over Responses API events,
  optionally raising.

A stream built with ``logged=True`` logs through the ``ScriptedHost`` that served it when it ends
(``HostLogging``), as litellm's stream logging does, rather than when the call returns it.
"""

import datetime
import json
import sys
import types
from collections.abc import AsyncIterator, Callable, Coroutine, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Final

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
from litellm.types.utils import ModelResponseStream

if TYPE_CHECKING:
    from tests.unit.test_router.core.scripted_host import Call, ScriptedHost

MESSAGES: Final = [{"role": "user", "content": "hi"}]


def install_orjson_stub_if_missing() -> None:
    """
    The /v1/messages stream wrapper imports proxy modules that import ``orjson`` and ``soundfile`` at load
    time. Where either is not installed (some dev containers), a stand-in is registered so the wrapper can
    load (json-backed for ``orjson``, empty for ``soundfile``, which only audio passthrough logging uses);
    where they are installed (CI), nothing changes.
    """
    try:
        import orjson  # noqa: F401  # pyright: ignore[reportUnusedImport]  # only probing for the module
    except ImportError:
        stub: Final = types.ModuleType("orjson")
        stub.loads = json.loads  # pyright: ignore[reportAttributeAccessIssue]  # module attribute
        stub.dumps = lambda value, **_options: json.dumps(value).encode()  # pyright: ignore[reportAttributeAccessIssue]  # module attribute
        stub.JSONDecodeError = json.JSONDecodeError  # pyright: ignore[reportAttributeAccessIssue]  # module attribute
        sys.modules["orjson"] = stub
    try:
        import soundfile  # noqa: F401  # pyright: ignore[reportUnusedImport, reportMissingImports]  # only probing for the module
    except (ImportError, OSError):
        sys.modules["soundfile"] = types.ModuleType("soundfile")


def _logging_obj(model: str, call_type: str = "acompletion", logged: bool = False) -> LiteLLMLogging:
    logging_obj: Final = (HostLogging if logged else LiteLLMLogging)(
        model=model,
        messages=MESSAGES,
        stream=True,
        call_type=call_type,
        start_time=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
        litellm_call_id=f"stream-{model}",
        function_id=f"stream-{model}",
    )
    logging_obj.update_environment_variables(model=model, user=None, optional_params={}, litellm_params={})
    return logging_obj


def request_logging(model: str = "a") -> LiteLLMLogging:
    """The logging object a request shares across its attempts (``litellm_logging_obj``)."""
    return _logging_obj(model)


class HostLogging(LiteLLMLogging):
    """A stream's logging object that runs the serving ``ScriptedHost``'s callbacks where litellm's
    stream logging runs litellm's: once with the assembled response when the stream ends, and once
    per failure the stream raises (``dispatch_attempt_failure_handlers``). The host binds itself and
    the call when it returns the stream (``ScriptedHost._logs_at_stream_end``)."""

    def bind_host(self, host: "ScriptedHost", call: "Call") -> None:
        self._scripted: tuple[ScriptedHost, Call] | None = (host, call)  # pyright: ignore[reportUninitializedInstanceVariable]  # set when the host binds
        self._logged_success = False

    def _bound(self) -> "tuple[ScriptedHost, Call] | None":
        return getattr(self, "_scripted", None)

    async def dispatch_success_handlers(  # pyright: ignore[reportIncompatibleMethodOverride]  # the arguments the stream passes
        self, result: object = None, *_args: object, **_kwargs: object
    ) -> None:
        bound: Final = self._bound()
        if bound is None or self._logged_success:
            return
        self._logged_success = True
        await bound[0].log_async(bound[1], None, result)

    def dispatch_attempt_failure_handlers(
        self, exception: Exception, traceback_exception: str
    ) -> Coroutine[None, None, None]:
        bound: Final = self._bound()

        async def dispatch() -> None:
            if bound is not None:
                await bound[0].log_async(bound[1], exception, None)

        return dispatch()

    def success_handler(self, result: object = None, *_args: object, **_kwargs: object) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]  # see dispatch_success_handlers
        bound: Final = self._bound()
        if bound is None or self._logged_success or not getattr(result, "usage", None):
            return
        self._logged_success = True
        bound[0].log_sync(bound[1], None, result)

    def failure_handler(self, exception: Exception, *_args: object, **_kwargs: object) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]  # see dispatch_success_handlers
        bound: Final = self._bound()
        if bound is not None:
            bound[0].log_sync(bound[1], exception, None)


class _EndLogged:
    """What a hand-written scripted stream logs once when it ends: its failure, or a response carrying
    ``total_tokens`` (the usage the provider's own stream logging would assemble)."""

    def __init__(self, model: str, call_type: str, logged: bool, total_tokens: int) -> None:
        self.logging_obj: Final[LiteLLMLogging | None] = _logging_obj(model, call_type, True) if logged else None
        self._total_tokens: Final = total_tokens
        self._logged = False

    async def log_end(self, error: BaseException | None) -> None:
        if not isinstance(self.logging_obj, HostLogging) or self._logged:
            return
        self._logged = True
        if error is not None:
            # The provider's own error, as a stream logs it before wrapping it for the router.
            original: Final = getattr(error, "original_exception", None)
            logged: Final = original if isinstance(original, Exception) else error
            await self.logging_obj.dispatch_attempt_failure_handlers(
                logged if isinstance(logged, Exception) else Exception(logged), ""
            )
            return
        usage: Final = types.SimpleNamespace(total_tokens=self._total_tokens)
        await self.logging_obj.dispatch_success_handlers(types.SimpleNamespace(usage=usage))


def text_chunk(text: str) -> ModelResponseStream:
    return ModelResponseStream(choices=[{"index": 0, "delta": {"role": "assistant", "content": text}}])


def role_chunk() -> ModelResponseStream:
    return ModelResponseStream(choices=[{"index": 0, "delta": {"role": "assistant", "content": ""}}])


def finish_chunk(reason: str = "stop", usage: Mapping[str, int] | None = None) -> ModelResponseStream:
    chunk: Final = ModelResponseStream(choices=[{"index": 0, "delta": {}, "finish_reason": reason}])
    if usage is not None:
        setattr(chunk, "usage", dict(usage))
    return chunk


def chat_stream(
    *texts: str,
    error: BaseException | None = None,
    fail_after: int | None = None,
    finish: bool = True,
    leading_role_chunk: bool = False,
    model: str = "d1",
    hidden_params: Mapping[str, object] | None = None,
    sync: bool = False,
    usage: Mapping[str, int] | None = None,
    logged: bool = False,
) -> Callable[..., CustomStreamWrapper]:
    """
    A stream of ``texts`` (one chunk each). With ``error``, raises it after ``fail_after`` chunks
    (default: after every text chunk, before the finish chunk). ``leading_role_chunk`` sends a
    role-only, empty-content chunk first, which counts as the first chunk but carries no content.
    ``usage`` rides on the finish chunk; ``logged`` logs through the host when the stream ends.
    """

    def answer(**request: object) -> CustomStreamWrapper:  # kwargs-ok: the host passes the request kwargs
        produced: list[ModelResponseStream] = []
        if leading_role_chunk:
            produced.append(role_chunk())
        produced.extend(text_chunk(text) for text in texts)
        cut_at: Final = len(produced) if fail_after is None else fail_after

        def items() -> Iterator[ModelResponseStream]:
            for index, chunk in enumerate(produced):
                if error is not None and index == cut_at:
                    raise error
                yield chunk
            if error is not None:
                raise error
            if finish:
                yield finish_chunk(usage=usage)

        async def achunks() -> AsyncIterator[ModelResponseStream]:
            for chunk in items():
                yield chunk

        wrapper: Final = CustomStreamWrapper(
            completion_stream=items() if sync else achunks(),
            model=model,
            custom_llm_provider="openai",
            logging_obj=_logging_obj(model, "completion" if sync else "acompletion", logged),
        )
        if hidden_params is not None:
            wrapper._hidden_params.update(hidden_params)  # pyright: ignore[reportPrivateUsage]  # test seam
        return wrapper

    return answer


async def collect_texts(stream: object) -> str:
    """The text content of every chunk of an async chat stream, joined."""
    texts: Final[list[str]] = []
    async for chunk in stream:  # pyright: ignore[reportGeneralTypeIssues]  # an async stream
        choices = getattr(chunk, "choices", None)
        if choices:
            content = choices[0].delta.content
            if content:
                texts.append(content)
    return "".join(texts)


def collect_texts_sync(stream: object) -> str:
    texts: Final[list[str]] = []
    for chunk in stream:  # pyright: ignore[reportGeneralTypeIssues]  # a sync stream
        choices = getattr(chunk, "choices", None)
        if choices:
            content = choices[0].delta.content
            if content:
                texts.append(content)
    return "".join(texts)


# --- /v1/messages ------------------------------------------------------------------------------


def sse(event: str, data: Mapping[str, object]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


MESSAGE_START: Final = sse(
    "message_start",
    {"type": "message_start", "message": {"id": "msg_1", "type": "message", "role": "assistant", "content": []}},
)
BLOCK_START: Final = sse(
    "content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}
)
PING: Final = sse("ping", {"type": "ping"})
MESSAGE_STOP: Final = sse("message_stop", {"type": "message_stop"})


def text_delta(text: str) -> bytes:
    return sse(
        "content_block_delta",
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
    )


def error_frame(error_type: str = "overloaded_error", message: str = "Overloaded") -> bytes:
    return sse("error", {"type": "error", "error": {"type": error_type, "message": message}})


def refusal_delta(stop_details: Mapping[str, object] | None = None) -> bytes:
    delta: Final[dict[str, object]] = {"stop_reason": "refusal"}
    if stop_details is not None:
        delta["stop_details"] = dict(stop_details)
    return sse("message_delta", {"type": "message_delta", "delta": delta, "usage": {"output_tokens": 1}})


class ByteStream:
    """An async iterator of SSE frames that records whether it was closed. ``logged`` logs through the host
    when it ends, with ``total_tokens`` (``_EndLogged``)."""

    def __init__(
        self,
        frames: Sequence[bytes],
        error: BaseException | None,
        hidden_params: Mapping[str, object] | None,
        logged: bool = False,
        total_tokens: int = 0,
    ) -> None:
        self._frames: Final = list(frames)
        self._error: Final = error
        self._index = 0
        self.closed = False
        self._end: Final = _EndLogged("d1", "anthropic_messages", logged, total_tokens)
        if logged:
            self.logging_obj = self._end.logging_obj
        if hidden_params is not None:
            self._hidden_params = dict(hidden_params)

    def __aiter__(self) -> "ByteStream":
        return self

    async def __anext__(self) -> bytes:
        if self._index < len(self._frames):
            self._index += 1
            return self._frames[self._index - 1]
        await self._end.log_end(self._error)
        if self._error is not None:
            raise self._error
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.closed = True


def anthropic_stream(
    *frames: bytes,
    error: BaseException | None = None,
    hidden_params: Mapping[str, object] | None = None,
    opened: list[ByteStream] | None = None,
    logged: bool = False,
    total_tokens: int = 0,
) -> Callable[..., ByteStream]:
    def answer(**_request: object) -> ByteStream:  # kwargs-ok: the host passes the request kwargs
        stream: Final = ByteStream(frames, error, hidden_params, logged, total_tokens)
        if opened is not None:
            opened.append(stream)
        return stream

    return answer


async def collect_bytes(stream: object) -> list[bytes]:
    return [frame async for frame in stream]  # pyright: ignore[reportGeneralTypeIssues]  # an async stream


# --- Responses API -----------------------------------------------------------------------------


class Event:
    """A Responses API stream event: only ``type`` (and optionally ``response``) is read."""

    def __init__(self, event_type: str, response: object = None, delta: str | None = None) -> None:
        self.type = event_type
        self.response = response
        self.delta = delta
        self._hidden_params: dict[str, object] = {}

    def __repr__(self) -> str:
        return f"Event({self.type!r})"


class BackgroundResponse:
    background = True
    id = "resp_bg"


class ScriptedResponsesStream(BaseResponsesAPIStreamingIterator):
    """A Responses stream over ``events`` that bypasses the provider plumbing of the base class."""

    def __init__(  # pyright: ignore[reportMissingSuperCall]  # the base needs an httpx response
        self,
        events: Sequence[object],
        error: BaseException | None,
        model: str,
        custom_llm_provider: str,
        hidden_params: Mapping[str, object] | None,
        started_tool_call: bool = False,
        completed_response: object = None,
        logged: bool = False,
        total_tokens: int = 0,
    ) -> None:
        self._end: Final = _EndLogged(model, "aresponses", logged, total_tokens)
        self._events: Final = list(events)
        self._started_tool_call = started_tool_call
        self._error: Final = error
        self._index = 0
        self._hidden_params = dict(hidden_params or {})
        self.model = model
        self.custom_llm_provider = custom_llm_provider
        self.logging_obj = self._end.logging_obj or _logging_obj(model, "aresponses")
        self.litellm_metadata = None
        self.responses_api_provider_config = None
        self.finished = False
        self.completed_response = completed_response
        self.response = None
        self.start_time = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
        self.request_data = {}
        self.call_type = None
        self.closed = False

    def __aiter__(self) -> "ScriptedResponsesStream":
        return self

    async def __anext__(self) -> object:
        if self._index < len(self._events):
            self._index += 1
            return self._events[self._index - 1]
        await self._end.log_end(self._error)
        if self._error is not None:
            raise self._error
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.closed = True


def responses_stream(
    *events: object,
    error: BaseException | None = None,
    model: str = "d1",
    custom_llm_provider: str = "anthropic",
    hidden_params: Mapping[str, object] | None = None,
    opened: list[ScriptedResponsesStream] | None = None,
    started_tool_call: bool = False,
    completed_response: object = None,
    logged: bool = False,
    total_tokens: int = 0,
) -> Callable[..., ScriptedResponsesStream]:
    """``completed_response`` is the terminal event the stream latched before failing, if any. ``logged``
    logs through the host when the stream ends, with ``total_tokens``."""

    def answer(**_request: object) -> ScriptedResponsesStream:  # kwargs-ok: the host passes the request kwargs
        stream: Final = ScriptedResponsesStream(
            events,
            error,
            model,
            custom_llm_provider,
            hidden_params,
            started_tool_call,
            completed_response,
            logged,
            total_tokens,
        )
        if opened is not None:
            opened.append(stream)
        return stream

    return answer


async def drain_into(stream: object, seen: list[object]) -> None:
    """Append every item of an async stream to ``seen``, letting an error the stream raises propagate."""
    async for item in stream:  # pyright: ignore[reportGeneralTypeIssues]  # an async stream
        seen.append(item)


async def collect_event_types(stream: object) -> list[str]:
    return [getattr(event, "type", repr(event)) async for event in stream]  # pyright: ignore[reportGeneralTypeIssues]  # an async stream


__all__ = [
    "BLOCK_START",
    "MESSAGES",
    "MESSAGE_START",
    "MESSAGE_STOP",
    "PING",
    "BackgroundResponse",
    "ByteStream",
    "Event",
    "HostLogging",
    "ScriptedResponsesStream",
    "anthropic_stream",
    "chat_stream",
    "collect_bytes",
    "collect_event_types",
    "collect_texts",
    "collect_texts_sync",
    "drain_into",
    "error_frame",
    "finish_chunk",
    "install_orjson_stub_if_missing",
    "refusal_delta",
    "responses_stream",
    "role_chunk",
    "sse",
    "text_chunk",
    "text_delta",
]
