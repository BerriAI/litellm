"""Apply the proxy's header opt-outs to all HTTP responses, including streams."""

from collections.abc import Callable
from typing import Final

from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class LiteLLMResponseHeadersMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        include_litellm_headers: Callable[[], bool],
        include_llm_provider_headers: Callable[[], bool],
    ) -> None:
        self.app = app
        self.include_litellm_headers = include_litellm_headers
        self.include_llm_provider_headers = include_llm_provider_headers

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_response(message: Message) -> None:
            if message["type"] != "http.response.start":
                await send(message)
                return
            dropped: Final = tuple(
                prefix
                for prefix, include in (
                    (b"x-litellm-", self.include_litellm_headers()),
                    (b"llm_provider-", self.include_llm_provider_headers()),
                )
                if not include
            )
            if not dropped:
                await send(message)
                return
            response: Final[Message] = {
                **message,
                "headers": [
                    (name, value) for name, value in Headers(scope=message).raw if not name.lower().startswith(dropped)
                ],
            }
            await send(response)

        await self.app(scope, receive, send_response)
