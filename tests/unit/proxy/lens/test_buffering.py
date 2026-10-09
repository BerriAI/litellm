import asyncio
import gc
from collections.abc import Mapping
from typing import Final

import httpx
import pytest
from fastapi import HTTPException, Response
from starlette.requests import Request
from starlette.types import Message

from litellm.proxy._types import LitellmUserRoles
from litellm.proxy.lens.adapter import Connection, Identity, forward
from litellm.proxy.lens.buffering import BufferBudget
from litellm.tracing.remote import LensConnection

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
