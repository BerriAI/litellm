from __future__ import annotations

from collections.abc import Awaitable, Mapping, Sequence
from typing import Protocol


class RustChatCompletions(Protocol):
    def __call__(
        self,
        model: str,
        messages: Sequence[object],
        optional_params: Mapping[str, object] | None,
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str | None,
        extra_headers: Mapping[str, object] | None,
        timeout_seconds: float | None,
    ) -> Mapping[str, object]: ...


class RustAchatCompletions(Protocol):
    def __call__(
        self,
        model: str,
        messages: Sequence[object],
        optional_params: Mapping[str, object] | None,
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str | None,
        extra_headers: Mapping[str, object] | None,
        timeout_seconds: float | None,
    ) -> Awaitable[Mapping[str, object]]: ...


class RustChatCompletionsDecline(Protocol):
    def __call__(
        self,
        model: str,
        messages: Sequence[object],
        optional_params: Mapping[str, object] | None,
        custom_llm_provider: str | None,
    ) -> str | None: ...


class RustResponsesWebSocket(Protocol):
    async def send_text(self, text: str) -> None: ...

    async def recv_text(self) -> str | None: ...

    async def close(self) -> None: ...


class RustResponsesWebSocketConnection(Protocol):
    @classmethod
    async def connect(
        cls,
        url: str,
        headers: dict[str, str],
        timeout_seconds: float | None,
    ) -> RustResponsesWebSocket: ...


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


class RustMessages(Protocol):
    def __call__(
        self,
        model: str,
        body: dict[str, object],
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str | None,
        extra_headers: dict[str, object] | None,
        timeout_seconds: float | None,
    ) -> dict[str, object]: ...


class RustAmessages(Protocol):
    def __call__(
        self,
        model: str,
        body: dict[str, object],
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str | None,
        extra_headers: dict[str, object] | None,
        timeout_seconds: float | None,
    ) -> Awaitable[dict[str, object]]: ...


class RustOcr(Protocol):
    def __call__(
        self,
        model: str,
        document: dict[str, object],
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str | None,
        extra_headers: dict[str, object] | None,
        optional_params: dict[str, object],
        timeout_seconds: float | None,
    ) -> dict[str, object]: ...


class RustAocr(Protocol):
    def __call__(
        self,
        model: str,
        document: dict[str, object],
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str | None,
        extra_headers: dict[str, object] | None,
        optional_params: dict[str, object],
        timeout_seconds: float | None,
    ) -> Awaitable[dict[str, object]]: ...


class RustTranscription(Protocol):
    def __call__(
        self,
        model: str,
        audio: dict[str, object],
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str | None,
        extra_headers: dict[str, object] | None,
        optional_params: dict[str, object],
        timeout_seconds: float | None,
    ) -> dict[str, object]: ...


class RustAtranscription(Protocol):
    def __call__(
        self,
        model: str,
        audio: dict[str, object],
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str | None,
        extra_headers: dict[str, object] | None,
        optional_params: dict[str, object],
        timeout_seconds: float | None,
    ) -> Awaitable[dict[str, object]]: ...
