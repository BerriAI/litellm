from __future__ import annotations

from typing import Protocol

from litellm.types.guardrails import PresidioPerRequestConfig


class ResponsesClientWebSocket(Protocol):
    """Client-facing websocket surface used by the Responses API websocket handlers."""

    async def send_text(self, data: str) -> None: ...

    async def receive_text(self) -> str: ...


class ResponsesBackendWebSocket(Protocol):
    """Upstream provider websocket surface used when proxying a native Responses API socket."""

    async def recv(self, decode: bool = ...) -> str | bytes: ...

    async def send(self, message: str) -> None: ...

    async def close(self) -> None: ...


class PresidioGuardrailCallback(Protocol):
    """
    Duck-typed PII guardrail surface consumed by the Responses API websocket handlers.

    Declared structurally so the SDK does not import from the proxy guardrail package.
    """

    def get_presidio_settings_from_request_data(self, data: dict[str, object]) -> PresidioPerRequestConfig | None: ...

    async def check_pii(
        self,
        text: str,
        output_parse_pii: bool,
        presidio_config: PresidioPerRequestConfig | None,
        request_data: dict[str, object],
    ) -> str: ...
