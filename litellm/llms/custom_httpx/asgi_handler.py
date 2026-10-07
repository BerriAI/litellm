from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Final

import httpx
from starlette.types import ASGIApp, Receive, Scope, Send

from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # shared cache retains its legacy parameter mapping
)
from litellm.types.llms.custom_http import httpxSpecialProvider


@dataclass(frozen=True, slots=True)
class _ASGITarget:
    app: ASGIApp
    root_path: str
    client: tuple[str, int] | None


_target: Final[ContextVar[_ASGITarget]] = ContextVar("httpx_asgi_target")


async def _dispatch(scope: Scope, receive: Receive, send: Send) -> None:
    target: Final = _target.get()
    await target.app({**scope, "root_path": target.root_path, "client": target.client}, receive, send)


_TRANSPORT: Final = httpx.ASGITransport(app=_dispatch, raise_app_exceptions=False)


@contextmanager
def get_async_asgi_client(
    app: ASGIApp, root_path: str = "", client: tuple[str, int] | None = None
) -> Generator[httpx.AsyncClient]:
    handler: Final = get_async_httpx_client(
        llm_provider=httpxSpecialProvider.ASGI,
        params={"transport": _TRANSPORT, "timeout": httpx.Timeout(None), "follow_redirects": False},
    )
    token: Final = _target.set(_ASGITarget(app, root_path, client))
    try:
        yield handler.client
    finally:
        _target.reset(token)
