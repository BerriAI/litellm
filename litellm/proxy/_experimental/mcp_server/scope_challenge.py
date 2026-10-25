"""Keep upstream scope challenges ahead of Streamable HTTP response headers."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Final

from pydantic import TypeAdapter, ValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import Message, Send

from litellm.proxy._experimental.mcp_server.exceptions import MCPUpstreamAuthError

if TYPE_CHECKING:
    from mcp.server.context import CallNext, HandlerResult, ServerRequestContext

SCOPE_RESPONSE_KEY: Final = "litellm.mcp.scope_response"
_REQUEST: Final = TypeAdapter[dict[str, object]](dict[str, object])


def is_tool_call_request(body: bytes) -> bool:
    try:
        return _REQUEST.validate_json(body).get("method") == "tools/call"
    except ValidationError:
        return False


class OAuthScopeResponse:
    def __init__(self, send: Send, base_url: str, request_path: str | None) -> None:
        self._send = send
        self._base_url = base_url
        self._request_path = request_path
        self._ready = asyncio.Event()
        self._lock = asyncio.Lock()
        self._start: Message | None = None
        self._challenge: MCPUpstreamAuthError | None = None
        self._refused = False

    @property
    def pending(self) -> bool:
        return not self._ready.is_set()

    def allow(self) -> None:
        self._ready.set()

    def deny(self, error: MCPUpstreamAuthError) -> None:
        if self.pending and error.status_code == 403 and error.required_scope is not None:
            self._challenge = error
            self._ready.set()

    async def __call__(self, message: Message) -> None:
        if message["type"] == "http.response.start":
            self._start = message
            if message["status"] != 200:
                self.allow()
            return
        await self._ready.wait()
        async with self._lock:
            if self._refused:
                return
            if self._start is not None:
                start: Final = self._start
                self._start = None
                if self._challenge is not None:
                    error: Final = self._challenge.to_http_exception(self._base_url, self._request_path)
                    response: Final = JSONResponse(
                        status_code=error.status_code, content={"detail": error.detail}, headers=error.headers
                    )
                    self._refused = True
                    await self._send(
                        {"type": "http.response.start", "status": response.status_code, "headers": response.raw_headers}
                    )
                    await self._send({"type": "http.response.body", "body": bytes(response.body)})
                    return
                await self._send(start)
            await self._send(message)


def scope_response_from_request(request: object) -> OAuthScopeResponse | None:
    if not isinstance(request, Request):
        return None
    response: Final = request.scope.get(SCOPE_RESPONSE_KEY)
    return response if isinstance(response, OAuthScopeResponse) else None


async def finish_scope_response(context: ServerRequestContext[object, object], call_next: CallNext) -> HandlerResult:
    response: Final = scope_response_from_request(context.request)
    try:
        return await call_next(context)
    finally:
        if response is not None:
            response.allow()
