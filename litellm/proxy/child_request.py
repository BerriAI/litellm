"""
Child requests through the proxy's own routes

A feature that needs a second model call during a request (a compaction summary, an image description) sends it
back through the proxy with the caller's credentials, so the child goes through the same auth, budget, rate limit,
guardrail and logging path as any other request
"""

import asyncio
from collections.abc import Awaitable, Mapping
from contextlib import AbstractContextManager
from contextvars import Context, ContextVar
from dataclasses import dataclass
from typing import Final, Protocol, TypeVar

from fastapi import Request
from pydantic import TypeAdapter
from starlette.types import ASGIApp

from litellm.litellm_core_utils.initialize_dynamic_callback_params import inherit_message_logging_privacy
from litellm.llms.custom_httpx.asgi_handler import get_async_asgi_client
from litellm.proxy.litellm_pre_call_utils import UNTRUSTED_REQUEST_HEADER_CONTROL_FIELDS

_ResultT: Final = TypeVar("_ResultT")
_ASGI_APP: Final = TypeAdapter[ASGIApp](ASGIApp)
_ROOT_PATH: Final = TypeAdapter(str)
_JSON_OBJECT: Final = TypeAdapter(Mapping[str, object])
_REMOVED_HEADERS: Final = frozenset(
    (
        b"content-length",
        b"x-litellm-call-id",
        b"x-litellm-num-retries",
        b"x-litellm-timeout",
        b"x-litellm-stream-timeout",
    )
)


@dataclass(frozen=True, slots=True)
class ChildResponse:
    status_code: int
    body: bytes


async def send_child_request(
    request: Request,
    path: str,
    payload: Mapping[str, object],
    *,
    logging_disabled: bool,
    scope: AbstractContextManager[None],
) -> ChildResponse:
    """POST `payload` to `path` on the proxy app that served `request`, with the parent's headers. The child runs in
    a fresh context, so the parent's request state does not leak into it. `scope` is entered inside that context, to
    mark the child for the feature that sent it. When the parent turned message logging off, the child inherits that
    and the parent's logging control headers are dropped"""

    async def dispatch() -> ChildResponse:
        asgi_scope: Final = _JSON_OBJECT.validate_python(request.scope)
        root_path: Final = _ROOT_PATH.validate_python(asgi_scope.get("root_path", ""))
        url: Final = str(request.url.replace(path=root_path.rstrip("/") + path, query="", fragment=""))
        headers: Final = tuple(
            (name, value)
            for name, value in request.headers.raw
            if name.lower() not in _REMOVED_HEADERS
            and not (logging_disabled and name.decode("latin-1").lower() in UNTRUSTED_REQUEST_HEADER_CONTROL_FIELDS)
        )
        with scope, inherit_message_logging_privacy(logging_disabled):
            with get_async_asgi_client(
                app=_ASGI_APP.validate_python(asgi_scope["app"]),
                root_path=root_path,
                client=request.client,
            ) as client:
                async with client.stream(
                    "POST", url, headers=headers, json=_JSON_OBJECT.validate_python(payload)
                ) as response:
                    return ChildResponse(status_code=response.status_code, body=await response.aread())

    task: Final = Context().run(asyncio.create_task, dispatch())
    return await task


class ChatChildRequest(Protocol):
    async def __call__(
        self, payload: Mapping[str, object], *, logging_disabled: bool, scope: AbstractContextManager[None]
    ) -> ChildResponse: ...


chat_child_request: Final[ContextVar[ChatChildRequest | None]] = ContextVar("chat_child_request", default=None)


async def with_proxy_chat_child_request(call: Awaitable[_ResultT], request: Request) -> _ResultT:
    """Run `call` with `chat_child_request` bound to a sender of /v1/chat/completions children for `request`"""

    async def execute(
        payload: Mapping[str, object], *, logging_disabled: bool, scope: AbstractContextManager[None]
    ) -> ChildResponse:
        return await send_child_request(
            request, "/v1/chat/completions", payload, logging_disabled=logging_disabled, scope=scope
        )

    token: Final = chat_child_request.set(execute)
    try:
        return await call
    finally:
        chat_child_request.reset(token)
