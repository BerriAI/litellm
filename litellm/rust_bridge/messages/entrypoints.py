from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from dataclasses import dataclass
from typing import Final, Protocol, cast  # noqa: TID251  # validates dynamically loaded native callables

from litellm.rust_bridge import execution
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.public_call import NativeCall
from litellm.types.llms.anthropic_messages.anthropic_response import AnthropicMessagesResponse


class MessagesStreamHead(dict[str, object]):
    agentic: Callable[[execution.AsyncStream], object] | None = None
    deliver: Callable[[object], Awaitable[object]] | None = None
    python_types: bool = False
    cached: object | None = None
    cache_writer: Callable[[object], object] | None = None


@dataclass(frozen=True, slots=True)
class _Undelivered:
    stream: object
    deliver: Callable[[object], Awaitable[object]]


def _public_stream(stream: execution.AsyncStream) -> object:
    from litellm.llms.anthropic.messages.native.streaming_iterator import (
        AnthropicMessagesStreamHiddenParams,
        AnthropicMessagesStreamingResponse,
    )

    return AnthropicMessagesStreamingResponse(
        completion_stream=cast(  # cast-ok: the native stream yields the provider's SSE bytes
            AsyncIterator[bytes], stream
        ),
        hidden_params=cast(  # cast-ok: the stream's stamped hidden params, shared as Python's handler shares them
            AnthropicMessagesStreamHiddenParams,
            stream._hidden_params,  # pyright: ignore[reportPrivateUsage]  # the native stream's own hidden params
        ),
    )


def _opened(native: execution.Execution, head: object) -> object:
    cached: Final = head.cached if isinstance(head, MessagesStreamHead) else None
    if cached is not None:
        native.close()
        return cached
    stream: Final = execution.AsyncStream(native, head)
    agentic: Final = head.agentic if isinstance(head, MessagesStreamHead) else None
    if agentic is not None:
        return agentic(stream)
    return _public_stream(stream) if isinstance(head, MessagesStreamHead) and head.python_types else stream


def _open_stream(native: execution.Execution, head: object) -> object:
    opened: Final = _opened(native, head)
    deliver: Final = head.deliver if isinstance(head, MessagesStreamHead) else None
    if deliver is not None:
        return _Undelivered(opened, deliver)
    writer: Final = head.cache_writer if isinstance(head, MessagesStreamHead) and head.cached is None else None
    return opened if writer is None else writer(opened)


async def _delivered(opened: Awaitable[object]) -> object:
    result: Final = await opened
    return await result.deliver(result.stream) if isinstance(result, _Undelivered) else result


class NativeMessages(Protocol):
    def __call__(
        self,
        request: NativeCall,
    ) -> AnthropicMessagesResponse | Iterator[bytes]: ...


class NativeAmessages(Protocol):
    def __call__(
        self,
        request: NativeCall,
    ) -> Awaitable[AnthropicMessagesResponse | AsyncIterator[bytes]]: ...


def _messages_binding(value: object) -> NativeMessages | None:
    if not callable(value):
        return None
    native: Final = execution.synchronous(value)
    return cast("NativeMessages", native)  # cast-ok: the execution adapter returns the route's result


def _amessages_binding(value: object) -> NativeAmessages | None:
    if not callable(value):
        return None
    native: Final = execution.asynchronous(value, _open_stream)

    def amessages(request: NativeCall) -> Awaitable[object]:
        return _delivered(native(request))

    return cast("NativeAmessages", amessages)  # cast-ok: the execution adapter returns the route's result


NATIVE_MESSAGES: Final = NativeBinding("messages", validate=_messages_binding)
NATIVE_AMESSAGES: Final = NativeBinding("amessages", validate=_amessages_binding)
