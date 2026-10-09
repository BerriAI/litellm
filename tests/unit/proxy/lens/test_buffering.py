import asyncio
import gc
from collections.abc import AsyncIterator, Mapping
from typing import Annotated, Final

import httpx
import pytest
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Response
from starlette.requests import ClientDisconnect, Request
from starlette.types import Message

from litellm.proxy._types import LitellmUserRoles
from litellm.proxy.common_utils.http_parsing_utils import read_request_body
from litellm.proxy.lens.adapter import Connection, Identity, forward, router
from litellm.proxy.lens.buffering import FORWARD_BUFFER_BUDGET, BodyReservation, BufferBudget, LensRoute
from litellm.tracing.remote import MAX_RESPONSE_BYTES, LensConnection

IDENTITY: Final = Identity(
    user_role=LitellmUserRoles.PROXY_ADMIN,
    user_id="admin",
    team_id=None,
    org_id=None,
    token=None,
    models=(),
    log_team_ids=(),
)
CONNECTION: Final = Connection(LensConnection("http://lens.test", "fixture-service-secret-32-characters"), "x" * 32)


def request(body: bytes) -> Request:
    async def receive() -> Message:
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(
        {"type": "http", "method": "POST", "path": "/lens/datasets", "headers": [], "query_string": b""},
        receive,
    )


async def send_response(response: Response) -> None:
    messages: list[Message] = []

    async def receive() -> Message:
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        messages.append(message)

    await response({"type": "http"}, receive, send)
    assert messages[-1]["body"] == b"result"


async def fetch(client: httpx.AsyncClient, budget: BufferBudget, body: bytes = b"data") -> Response:
    return await forward(request(body), IDENTITY, "datasets", CONNECTION, client, 1000, budget=budget)


@pytest.mark.asyncio
async def test_body_budget_is_shared_and_held_until_the_downstream_send_finishes() -> None:
    budget: Final = BufferBudget(10)
    sending: Final = asyncio.Event()
    finish: Final = asyncio.Event()

    async def receive() -> Message:
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        if message["type"] == "http.response.body":
            sending.set()
            await finish.wait()

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b"result"))
    ) as client:
        first: Final = await fetch(client, budget)
        task: Final = asyncio.create_task(first({"type": "http"}, receive, send))
        await asyncio.wait_for(sending.wait(), timeout=1)
        try:
            with pytest.raises(HTTPException) as full:
                await fetch(client, budget)
            assert full.value.status_code == 503
            assert full.value.headers == {"Retry-After": "1"}
        finally:
            finish.set()
            await task
        await send_response(await fetch(client, budget))


@pytest.mark.asyncio
@pytest.mark.parametrize("body,payload", [(b"oversized request", b""), (b"data", b"oversized response")])
async def test_capacity_failure_releases_partial_request_and_response_reservations(body: bytes, payload: bytes) -> None:
    budget: Final = BufferBudget(10)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=payload))
    ) as client:
        with pytest.raises(HTTPException) as full:
            await fetch(client, budget, body)
        assert full.value.status_code == 503
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b"result"))
    ) as client:
        await send_response(await fetch(client, budget))


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_upstream_failure_or_cancellation_releases_request_bytes(cancel: bool) -> None:
    budget: Final = BufferBudget(10)

    async def unavailable(incoming: httpx.Request) -> httpx.Response:
        assert incoming.content == b"data"
        if cancel:
            raise asyncio.CancelledError
        raise httpx.ConnectError("Lens is unavailable")

    async with httpx.AsyncClient(transport=httpx.MockTransport(unavailable)) as client:
        with pytest.raises(asyncio.CancelledError if cancel else HTTPException):
            await fetch(client, budget)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b"result"))
    ) as client:
        await send_response(await fetch(client, budget))


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_downstream_failure_or_cancellation_releases_response_bytes(cancel: bool) -> None:
    budget: Final = BufferBudget(10)

    async def receive() -> Message:
        return {"type": "http.disconnect"}

    async def broken_send(message: Message) -> None:
        if cancel:
            raise asyncio.CancelledError
        raise OSError("Browser disconnected")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b"result"))
    ) as client:
        response: Final = await fetch(client, budget)
        with pytest.raises(asyncio.CancelledError if cancel else OSError):
            await response({"type": "http"}, receive, broken_send)
        await send_response(await fetch(client, budget))


@pytest.mark.asyncio
async def test_discarded_unsent_response_releases_the_body_reservation() -> None:
    budget: Final = BufferBudget(10)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b"result"))
    ) as client:
        response = await fetch(client, budget)
        assert response.body == b"result"
        del response
        gc.collect()
        await send_response(await fetch(client, budget))


@pytest.mark.asyncio
async def test_capacity_rejection_does_not_consume_an_unbounded_incoming_stream() -> None:
    budget: Final = BufferBudget(4)
    consumed: list[bytes] = []
    chunks: Final = iter((b"data", b"next", b"must-not-be-read"))

    async def receive() -> Mapping[str, object]:
        chunk: Final = next(chunks)
        consumed.append(chunk)
        return {"type": "http.request", "body": chunk, "more_body": True}

    incoming: Final = Request(
        {"type": "http", "method": "POST", "path": "/lens/datasets", "headers": [], "query_string": b""},
        receive,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200))) as client:
        with pytest.raises(HTTPException) as full:
            await forward(incoming, IDENTITY, "datasets", CONNECTION, client, 1000, budget=budget)
        assert full.value.status_code == 503
    assert consumed == [b"data", b"next"]


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/lens/datasets", "/lens/service"])
async def test_router_rejects_capacity_before_authentication_reads_the_upload(path: str) -> None:

    app: Final = FastAPI()
    app.include_router(router)
    occupied: Final = BodyReservation(FORWARD_BUFFER_BUDGET)
    occupied.reserve(FORWARD_BUFFER_BUDGET.capacity - 4)
    consumed: list[bytes] = []

    async def chunks() -> AsyncIterator[bytes]:
        for chunk in (b"part", b"next", b"must-not-be-read"):
            consumed.append(chunk)
            yield chunk

    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway.test") as client:
            response: Final = await client.request(
                "GET" if path.endswith("service") else "POST", path, content=chunks()
            )
        assert response.status_code == 503
        assert response.headers["retry-after"] == "1"
        assert consumed == [b"part", b"next"]
        occupied.reserve(4)
    finally:
        occupied.release()


@pytest.mark.asyncio
async def test_router_rejects_oversized_chunked_upload_before_authentication() -> None:

    app: Final = FastAPI()
    app.include_router(router)

    async def chunks() -> AsyncIterator[bytes]:
        yield b" " * MAX_RESPONSE_BYTES
        yield b"x"
        pytest.fail("Upload continued after the per-request limit")

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway.test") as client:
        response: Final = await client.post("/lens/datasets", content=chunks())
    assert response.status_code == 413
    assert response.json() == {"detail": "Lens request is too large"}
    remaining: Final = BodyReservation(FORWARD_BUFFER_BUDGET)
    remaining.reserve(FORWARD_BUFFER_BUDGET.capacity)
    remaining.release()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_admission_releases_upload_on_disconnect_or_cancellation(cancel: bool) -> None:

    app: Final = FastAPI()
    app.include_router(router)

    async def chunks() -> AsyncIterator[bytes]:
        yield b"partial upload"
        if cancel:
            raise asyncio.CancelledError
        raise ClientDisconnect

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway.test") as client:
        with pytest.raises(asyncio.CancelledError if cancel else ClientDisconnect):
            await client.post("/lens/datasets", content=chunks())
    remaining: Final = BodyReservation(FORWARD_BUFFER_BUDGET)
    remaining.reserve(FORWARD_BUFFER_BUDGET.capacity)
    remaining.release()


@pytest.mark.asyncio
@pytest.mark.parametrize("reject_auth", [False, True])
async def test_admitted_bytes_survive_auth_parsing_and_remain_reserved_until_sent(reject_auth: bool) -> None:

    app: Final = FastAPI()
    router: Final = APIRouter(route_class=LensRoute)
    sending: Final = asyncio.Event()
    finish: Final = asyncio.Event()
    occupied: Final = BodyReservation(FORWARD_BUFFER_BUDGET)
    occupied.reserve(FORWARD_BUFFER_BUDGET.capacity - 10)
    observed: list[Message] = []

    async def auth_reader(request: Request) -> None:
        assert await read_request_body(request) == {}
        if reject_auth:
            raise HTTPException(401, "Invalid key")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b"result"))
    ) as upstream:

        @router.post("/lens/datasets")
        async def endpoint(request: Request, auth: Annotated[None, Depends(auth_reader)]) -> Response:
            return await forward(request, IDENTITY, "datasets", CONNECTION, upstream, 1000)

        @app.post("/ordinary")
        async def ordinary(request: Request) -> Response:
            return Response(await request.body())

        app.include_router(router)

        async def send(message: Message) -> None:
            observed.append(message)
            if message["type"] == "http.response.body":
                sending.set()
                await finish.wait()

        incoming: Final = request(b" {} ")
        task: Final = asyncio.create_task(app(incoming.scope, incoming.receive, send))
        try:
            await asyncio.wait_for(sending.wait(), 2)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://gateway.test"
            ) as client:
                blocked: Final = await client.post("/lens/datasets", content=b" {} " * 2)
                unrelated: Final = await client.post("/ordinary", content=b"ordinary body")
            assert blocked.status_code == 503
            assert unrelated.content == b"ordinary body"
            assert observed[0]["status"] == (401 if reject_auth else 200)
            assert observed[-1]["body"] == (b'{"detail":"Invalid key"}' if reject_auth else b"result")
        finally:
            finish.set()
            await task
            occupied.release()
    remaining: Final = BodyReservation(FORWARD_BUFFER_BUDGET)
    remaining.reserve(FORWARD_BUFFER_BUDGET.capacity)
    remaining.release()
