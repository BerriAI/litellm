from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Iterator
from typing import Final, Protocol, cast  # noqa: TID251  # validates dynamically loaded native callables

from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.public_call import NativeCall
from litellm.types.llms.anthropic_messages.anthropic_response import AnthropicMessagesResponse


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
    return cast("NativeMessages", value)  # cast-ok: callable validated at the native binding boundary


def _amessages_binding(value: object) -> NativeAmessages | None:
    if not callable(value):
        return None
    return cast("NativeAmessages", value)  # cast-ok: callable validated at the native binding boundary


NATIVE_MESSAGES: Final = NativeBinding("messages", validate=_messages_binding)
NATIVE_AMESSAGES: Final = NativeBinding("amessages", validate=_amessages_binding)
