import json
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from types import MappingProxyType
from typing import Final

import pytest
from fastapi import FastAPI, HTTPException, Request
from pydantic import TypeAdapter

from litellm.proxy.child_request import ChildResponse, chat_child_request, with_proxy_chat_child_request
from litellm.proxy.hooks.parallel_request_limiter_v3 import get_or_create_request_stash, get_request_stash

_HEADERS: Final = (
    (b"authorization", b"Bearer sk-child-fixture"),
    (b"content-length", b"99999"),
    (b"x-litellm-call-id", b"parent"),
    (b"litellm-disable-message-redaction", b"true"),
    (b"x-litellm-timeout", b"600"),
)
_marked: Final[ContextVar[bool]] = ContextVar("child_request_test_marker", default=False)


@contextmanager
def _mark() -> Generator[None]:
    token: Final = _marked.set(True)
    try:
        yield
    finally:
        _marked.reset(token)


def _request(app: FastAPI) -> Request:
    return Request(
        TypeAdapter(dict[str, object]).validate_python(
            MappingProxyType(
                {
                    "type": "http",
                    "app": app,
                    "scheme": "https",
                    "server": ("proxy.test", 443),
                    "path": "/gateway/v1/chat/completions",
                    "root_path": "/gateway",
                    "query_string": b"parent=1",
                    "client": ("192.0.2.1", 4321),
                    "headers": _HEADERS,
                }
            )
        )
    )


async def _send_child(logging_disabled: bool) -> ChildResponse:
    send: Final = chat_child_request.get()
    assert send is not None
    return await send({"model": "vision-describer"}, logging_disabled=logging_disabled, scope=_mark())


@pytest.mark.asyncio
@pytest.mark.parametrize("logging_disabled", [False, True])
async def test_chat_child_request_runs_as_the_caller_in_a_fresh_marked_context(logging_disabled: bool):
    app: Final = FastAPI()
    stash: Final = get_or_create_request_stash()
    seen: Final[list[Mapping[str, object]]] = []  # mutable-ok: records what the child endpoint saw

    @app.post("/v1/chat/completions")
    async def endpoint(request: Request) -> Mapping[str, object]:
        seen.append(
            {
                "stash": get_request_stash(),
                "sender": chat_child_request.get(),
                "marked": _marked.get(),
                "authorization": request.headers.get("authorization"),
                "call_id": request.headers.get("x-litellm-call-id"),
                "timeout": request.headers.get("x-litellm-timeout"),
                "redaction": request.headers.get("litellm-disable-message-redaction"),
                "root_path": request.scope["root_path"],
                "query": str(request.query_params),
                "body": await request.json(),
            }
        )
        return {"choices": [{"message": {"content": "a cat"}}]}

    response: Final = await with_proxy_chat_child_request(_send_child(logging_disabled), _request(app))

    assert (response.status_code, json.loads(response.body)) == (200, {"choices": [{"message": {"content": "a cat"}}]})
    assert seen == [
        {
            "stash": None,
            "sender": None,
            "marked": True,
            "authorization": "Bearer sk-child-fixture",
            "call_id": None,
            "timeout": None,
            "redaction": None if logging_disabled else "true",
            "root_path": "/gateway",
            "query": "",
            "body": {"model": "vision-describer"},
        }
    ]
    assert (get_request_stash(), chat_child_request.get(), _marked.get()) == (stash, None, False)


@pytest.mark.asyncio
async def test_chat_child_request_returns_the_child_status_without_raising():
    app: Final = FastAPI()

    @app.post("/v1/chat/completions")
    async def endpoint() -> None:
        raise HTTPException(status_code=429, detail="budget exceeded")

    response: Final = await with_proxy_chat_child_request(_send_child(False), _request(app))

    assert (response.status_code, json.loads(response.body)) == (429, {"detail": "budget exceeded"})
