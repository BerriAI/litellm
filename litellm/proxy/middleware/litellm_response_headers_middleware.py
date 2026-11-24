"""Apply the proxy's opt-out to all HTTP response headers, including streams."""

from collections.abc import Callable
from typing import Final

from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class LiteLLMResponseHeadersMiddleware:
    def __init__(self, app: ASGIApp, include_litellm_headers: Callable[[], bool]) -> None:
        self.app = app
        self.include_litellm_headers = include_litellm_headers

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_response(message: Message) -> None:
            if message["type"] != "http.response.start" or self.include_litellm_headers():
                await send(message)
                return
            response: Final[Message] = {
                **message,
                "headers": [
                    (name, value)
                    for name, value in Headers(scope=message).raw
                    if not name.lower().startswith(b"x-litellm-")
                ],
            }
            await send(response)

        await self.app(scope, receive, send_response)
