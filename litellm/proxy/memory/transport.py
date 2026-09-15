import asyncio
import json
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager, suppress
from contextvars import ContextVar
from io import BytesIO
from typing import Final, TypeAlias
from uuid import uuid4

import anyio
from fastapi import HTTPException, Request
from pydantic import BaseModel, ConfigDict, TypeAdapter
from starlette.responses import Response, StreamingResponse

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.parallel_request_limiter_v3 import wait_for_request_parallel_release

_GATEWAY_ROUND: Final[ContextVar[int | None]] = ContextVar("litellm_gateway_memory_round", default=None)
_ROUND_ACCOUNTING: Final[ContextVar[tuple[str, asyncio.Future[None]] | None]] = ContextVar(
    "litellm_memory_round_accounting", default=None
)
_OBJECT: Final = TypeAdapter(dict[str, object])
_ROUND_HEADERS: Final = frozenset(("idempotency-key", "x-request-id", "x-litellm-call-id"))
RoundExecutor: TypeAlias = Callable[
    [Request, dict[str, object], UserAPIKeyAuth], Awaitable[Response]
]  # mutable-ok: The processor mutates its fresh request copy.


def begin_gateway_accounting(call_id: str) -> None:
    _ROUND_ACCOUNTING.set((call_id, asyncio.get_running_loop().create_future()))


def gateway_accounting(call_id: str | None = None) -> asyncio.Future[None] | None:
    accounting: Final = _ROUND_ACCOUNTING.get()
    if accounting is None or (call_id is not None and accounting[0] != call_id):
        return None
    return accounting[1]


def in_gateway_round() -> bool:
    return _GATEWAY_ROUND.get() is not None


def is_memory_continuation_round() -> bool:
    return (_GATEWAY_ROUND.get() or 0) > 0


def _round_body(body: Mapping[str, object]) -> bytes:
    return json.dumps(
        {  # mutable-ok: Starlette and the gateway processor consume native request containers.
            **body,
            **{  # mutable-ok: Starlette and the gateway processor consume native request containers.
                field: {  # mutable-ok: Starlette and the gateway processor consume native request containers.
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


class GatewayRound:
    def __init__(
        self, execute: RoundExecutor, request: Request, body: Mapping[str, object], auth: UserAPIKeyAuth, index: int
    ) -> None:
        self.execute = execute
        self.request = request
        self.body = _round_body(body)
        self.auth = auth
        self.index = index
        self.writer, self.reader = anyio.create_memory_object_stream[bytes](8)
        self.started: asyncio.Future[RoundStart] = asyncio.get_running_loop().create_future()
        self.task: asyncio.Task[None] | None = None

    async def run(self) -> None:
        from litellm.proxy.hooks.parallel_request_limiter_v3 import reset_request_stash

        token: Final = _GATEWAY_ROUND.set(self.index)
        reset_request_stash()
        headers: Final = tuple(
            (name, value)
            for name, value in self.request.headers.raw
            if name.decode("latin-1").lower() not in _ROUND_HEADERS
            and name.lower() not in (b"content-length", b"content-type")
        )
        inner: Final = Request(
            {  # mutable-ok: Starlette and the gateway processor consume native request containers.
                **{
                    key: value for key, value in self.request.scope.items() if key != "parsed_body"
                },  # mutable-ok: Starlette and the gateway processor consume native request containers.
                "state": {},  # mutable-ok: Starlette and the gateway processor consume native request containers.
                "headers": [  # mutable-ok: Starlette and the gateway processor consume native request containers.
                    *headers,
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(self.body)).encode()),
                ],
            },
            receive=self.request.receive,
        )
        inner._body = self.body
        try:
            async with self.writer:
                response: Final = await self.execute(inner, _OBJECT.validate_json(self.body), self.auth)
                self.started.set_result(RoundStart(status=response.status_code, headers=tuple(response.raw_headers)))
                if isinstance(response, StreamingResponse):
                    try:
                        async for chunk in response.body_iterator:
                            await self.writer.send(chunk.encode() if isinstance(chunk, str) else bytes(chunk))
                    finally:
                        close: Final = getattr(response.body_iterator, "aclose", None)
                        if close is not None:
                            await close()
                else:
                    await self.writer.send(bytes(response.body))
                if response.background is not None:
                    await response.background()
                accounting: Final = gateway_accounting()
                if accounting is not None:
                    try:
                        await asyncio.wait_for(asyncio.shield(accounting), timeout=15)
                    except TimeoutError as exc:
                        raise HTTPException(
                            status_code=503, detail="Memory could not confirm model spend; retry later"
                        ) from exc
                await wait_for_request_parallel_release()
        except BaseException as exc:
            if not self.started.done():
                self.started.set_exception(exc)
            raise
        finally:
            _GATEWAY_ROUND.reset(token)

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
        if self.task is not None:
            if not self.task.done():
                self.task.cancel()
            with suppress(asyncio.CancelledError):
                await self.task
        await self.reader.aclose()


@asynccontextmanager
async def gateway_round(
    execute: RoundExecutor, request: Request, body: Mapping[str, object], auth: UserAPIKeyAuth, index: int = 0
) -> AsyncGenerator[GatewayRound, None]:
    call: Final = GatewayRound(execute, request, body, auth, index)
    call.task = asyncio.create_task(call.run())
    try:
        await call.started
        yield call
    finally:
        await call.close()
