from __future__ import annotations

from collections.abc import Awaitable, Mapping, Sequence
from typing import Protocol

from .callbacks import SessionCallbackHandle
from .request import (
    NativeChatCompletionsRequest,
    NativeFunction,
    NativeMessagesRequest,
    NativeOCRRequest,
    NativeRequestContext,
    NativeResponsesWebSocketRequest,
    NativeTranscriptionRequest,
)

RustChatCompletions = NativeFunction[NativeChatCompletionsRequest, Mapping[str, object]]
RustAchatCompletions = NativeFunction[NativeChatCompletionsRequest, Awaitable[Mapping[str, object]]]
RustMessages = NativeFunction[NativeMessagesRequest, dict[str, object]]
RustAmessages = NativeFunction[NativeMessagesRequest, Awaitable[dict[str, object]]]
RustOcr = NativeFunction[NativeOCRRequest, dict[str, object]]
RustAocr = NativeFunction[NativeOCRRequest, Awaitable[dict[str, object]]]
RustTranscription = NativeFunction[NativeTranscriptionRequest, dict[str, object]]
RustAtranscription = NativeFunction[NativeTranscriptionRequest, Awaitable[dict[str, object]]]


class RustChatCompletionsDecline(Protocol):
    def __call__(
        self,
        model: str,
        messages: Sequence[object],
        optional_params: Mapping[str, object] | None,
        custom_llm_provider: str | None,
        *,
        context: NativeRequestContext,
        stream: bool,
        has_custom_client: bool = False,
        has_agentic_hook: bool = False,
    ) -> str | None: ...


class RustResponsesWebSocket(Protocol):
    async def send_text(self, text: str) -> None: ...

    async def recv_text(self) -> str | None: ...

    async def close(self) -> None: ...


class RustResponsesWebSocketConnection(Protocol):
    @classmethod
    async def connect(
        cls,
        request: NativeResponsesWebSocketRequest,
        *,
        context: NativeRequestContext,
        callback_adapter: SessionCallbackHandle | None = None,
        auth_provider: object | None = None,
    ) -> RustResponsesWebSocket: ...


class RustRouteDecline(Protocol):
    def __call__(
        self,
        model: str,
        custom_llm_provider: str,
        *,
        stream: bool = False,
        has_agentic_hook: bool = False,
        has_custom_client: bool = False,
        request_format: str | None = None,
    ) -> str | None: ...


class NativeModule(Protocol):
    @property
    def chat_completions(self) -> RustChatCompletions: ...

    @property
    def achat_completions(self) -> RustAchatCompletions: ...

    @property
    def chat_completions_decline(self) -> RustChatCompletionsDecline: ...

    @property
    def ResponsesWebSocketConnection(self) -> type[RustResponsesWebSocketConnection]: ...

    @property
    def RustBridgeDeclined(self) -> type[BaseException]: ...

    @property
    def RustUpstreamError(self) -> type[BaseException]: ...

    @property
    def messages(self) -> RustMessages: ...

    @property
    def amessages(self) -> RustAmessages: ...

    @property
    def ocr(self) -> RustOcr: ...

    @property
    def aocr(self) -> RustAocr: ...

    @property
    def transcription(self) -> RustTranscription: ...

    @property
    def atranscription(self) -> RustAtranscription: ...

    @property
    def ocr_decline(self) -> RustRouteDecline: ...

    @property
    def messages_decline(self) -> RustRouteDecline: ...

    @property
    def transcription_decline(self) -> RustRouteDecline: ...

    @property
    def responses_websocket_decline(self) -> RustRouteDecline: ...
