from typing import Final

from starlette.requests import Request
from starlette.types import ASGIApp, Receive, Scope, Send

from litellm.proxy.common_utils.active_request import active_request


class ActiveRequestMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        token: Final = active_request.set(Request(scope, receive))
        try:
            await self.app(scope, receive, send)
        finally:
            active_request.reset(token)
