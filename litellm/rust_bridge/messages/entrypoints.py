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


@dataclass(frozen=True, slots=True)
class _Undelivered:
    stream: object
    deliver: Callable[[object], Awaitable[object]]


def _open_stream(native: execution.Execution, head: object) -> object:
    stream: Final = execution.AsyncStream(native, head)
    agentic: Final = head.agentic if isinstance(head, MessagesStreamHead) else None
    opened: Final = stream if agentic is None else agentic(stream)
    deliver: Final = head.deliver if isinstance(head, MessagesStreamHead) else None
    return opened if deliver is None else _Undelivered(opened, deliver)


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
