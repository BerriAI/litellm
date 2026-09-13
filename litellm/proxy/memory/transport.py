import asyncio
import json
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager, suppress
from contextvars import ContextVar
from io import BytesIO
from typing import Final
from uuid import uuid4

import anyio
from fastapi import Request
from pydantic import BaseModel, ConfigDict, TypeAdapter
from starlette.types import ASGIApp, Message, Scope

from litellm.proxy.hooks.parallel_request_limiter_v3 import wait_for_request_parallel_release

_IN_GATEWAY_ROUND: Final[ContextVar[bool]] = ContextVar("litellm_gateway_memory_round", default=False)
_HEADERS: Final = TypeAdapter(tuple[tuple[bytes, bytes], ...])
_BYTES: Final = TypeAdapter(bytes)
_OBJECT: Final = TypeAdapter(dict[str, object])
_ROUND_HEADERS: Final = frozenset(("idempotency-key", "x-request-id", "x-litellm-call-id"))


def _round_body(body: Mapping[str, object]) -> bytes:
    return json.dumps(
        {  # mutable-ok: Native provider JSON containers.
            **body,
            **{  # mutable-ok: Native provider JSON containers.
                field: {  # mutable-ok: Native provider JSON containers.
                    key: value
                    for key, value in _OBJECT.validate_python(body[field]).items()
                    if key.lower() not in _ROUND_HEADERS
                }
                for field in ("headers", "extra_headers")
                if isinstance(body.get(field), dict)
            },
            "litellm_call_id": str(uuid4()),
        }
    ).encode()


class RoundStart(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: int
    headers: tuple[tuple[bytes, bytes], ...] = ()


def in_gateway_round() -> bool:
    return _IN_GATEWAY_ROUND.get()


class GatewayRound:
    def __init__(self, app: ASGIApp, request: Request, body: Mapping[str, object]) -> None:
        self.app = app
        self.request = request
        self.body = _round_body(body)
        self.writer, self.reader = anyio.create_memory_object_stream[bytes](8)
        self.started: asyncio.Future[RoundStart] = asyncio.get_running_loop().create_future()
        self.disconnected = asyncio.Event()
        self.body_received = False
        self.task: asyncio.Task[None] | None = None

    async def receive(self) -> Message:
        if not self.body_received:
            self.body_received = True
            return {  # mutable-ok: Native ASGI or JSON payload.
                "type": "http.request",
                "body": self.body,
                "more_body": False,
            }
        await self.disconnected.wait()
        return {  # mutable-ok: Native ASGI or JSON payload.
            "type": "http.disconnect"
        }

    async def send(self, message: Message) -> None:
        if message["type"] == "http.response.start":
            if not self.started.done():
                self.started.set_result(RoundStart.model_validate(message))
            return
        if message["type"] == "http.response.body":
            await self.writer.send(_BYTES.validate_python(message.get("body", b"")))

    async def run(self) -> None:
        token: Final = _IN_GATEWAY_ROUND.set(True)
        headers: Final = tuple(
            (name, value)
            for name, value in _HEADERS.validate_python(self.request.scope["headers"])
            if name.lower()
            not in (
                b"content-length",
                b"content-type",
                b"accept-encoding",
                b"idempotency-key",
                b"x-request-id",
                b"x-litellm-call-id",
            )
        )
        scope: Final[Scope] = {
            **{  # mutable-ok: Native ASGI or JSON payload.
                key: self.request.scope[key]
                for key in (
                    "type",
                    "asgi",
                    "http_version",
                    "method",
                    "scheme",
                    "path",
                    "raw_path",
                    "query_string",
                    "root_path",
                    "server",
                    "client",
                )
                if key in self.request.scope
            },
            "headers": [  # mutable-ok: Native ASGI or JSON payload.
                *headers,
                (b"content-type", b"application/json"),
                (b"content-length", str(len(self.body)).encode()),
            ],
            "state": {},
        }
        try:
            async with self.writer:
                await self.app(scope, self.receive, self.send)
                await wait_for_request_parallel_release()
        except BaseException as exc:
            if not self.started.done():
                self.started.set_exception(exc)
            raise
        finally:
            _IN_GATEWAY_ROUND.reset(token)

    async def chunks(self) -> AsyncGenerator[bytes, None]:
        async with self.reader:
            async for body in self.reader:
                if body:
                    yield body
        if self.task is not None:
            await self.task

    async def read(self, limit: int = 16 * 1024 * 1024) -> bytes:
        with BytesIO() as buffer:
            async for chunk in self.chunks():
                if buffer.tell() + len(chunk) > limit:
                    raise ValueError("The gateway response exceeded its retained-output limit")
                buffer.write(chunk)
            return buffer.getvalue()

    async def close(self) -> None:
        self.disconnected.set()
        if self.task is not None:
            if not self.task.done():
                self.task.cancel()
            with suppress(asyncio.CancelledError):
                await self.task
        await self.reader.aclose()


@asynccontextmanager
async def gateway_round(
    app: ASGIApp, request: Request, body: Mapping[str, object]
) -> AsyncGenerator[GatewayRound, None]:
    call: Final = GatewayRound(app, request, body)
    call.task = asyncio.create_task(call.run())
    try:
        await call.started
        yield call
    finally:
        await call.close()
