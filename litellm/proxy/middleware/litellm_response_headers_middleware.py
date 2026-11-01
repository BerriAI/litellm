"""Apply the proxy's opt-out to all HTTP response headers, including streams."""

from starlette.types import ASGIApp, Message, Receive, Scope, Send

import litellm


class LiteLLMResponseHeadersMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_response(message: Message) -> None:
            if message["type"] == "http.response.start" and litellm.include_litellm_headers is False:
                message = {
                    **message,
                    "headers": [
                        (name, value)
                        for name, value in message.get("headers", [])
                        if not name.lower().startswith(b"x-litellm-")
                    ],
                }
            await send(message)

        await self.app(scope, receive, send_response)
