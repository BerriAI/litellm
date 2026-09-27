from typing import Final

from starlette.types import ASGIApp, Receive, Scope, Send

from litellm._logging import verbose_proxy_logger
from litellm.caching.redis_request_plan import redis_request_plan_scope

_SCOPES_WITH_A_REQUEST_PLAN: Final = frozenset({"http", "websocket"})


class RedisRequestPlanMiddleware:
    """Gives each request a RedisRequestPlan so pre-call Redis work (auth prefetch,
    spend counters, routing reads) shares one pipeline per RedisCache.

    Declarations left unexecuted when the response has been sent are flushed here so
    write-backs still reach Redis instead of being silently dropped.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in _SCOPES_WITH_A_REQUEST_PLAN:
            await self.app(scope, receive, send)
            return
        with redis_request_plan_scope() as plan:
            try:
                await self.app(scope, receive, send)
            finally:
                try:
                    await plan.flush()
                except Exception as e:
                    verbose_proxy_logger.debug("redis request plan flush failed: %s", e)
