from collections.abc import Callable
from typing import Final

from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

_LLM_PROVIDER_HEADER_PREFIX: Final = b"llm_provider-"


class LLMProviderHeadersMiddleware:
    """Drops the provider response headers forwarded as `llm_provider-<name>` when `include()` is false."""

    def __init__(self, app: ASGIApp, include: Callable[[], bool]) -> None:
        self.app = app
        self.include = include

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_without_provider_headers(message: Message) -> None:
            if message["type"] != "http.response.start" or self.include():
                await send(message)
                return
            kept: Final = tuple(
                (name, value)
                for name, value in Headers(scope=message).raw
                if not name.startswith(_LLM_PROVIDER_HEADER_PREFIX)
            )
            await send({**message, "headers": kept})

        await self.app(scope, receive, send_without_provider_headers)
