from typing import Final

from starlette.types import ASGIApp, Receive, Scope, Send

from litellm.caching.redis_batch import request_redis_batch_scope

_REQUEST_SCOPES: Final = frozenset({"http", "websocket"})


class RedisRequestBatchMiddleware:
    """Opens the request's Redis batch scope so auth, admission and routing reads issued anywhere in the
    request (dependencies, the endpoint, tasks it spawns) share one pipeline per Redis backend."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in _REQUEST_SCOPES:
            await self.app(scope, receive, send)
            return
        with request_redis_batch_scope() as batches:
            try:
                await self.app(scope, receive, send)
            finally:
                await batches.flush_all()
