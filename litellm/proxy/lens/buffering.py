"""Account for forwarded Lens bodies until the downstream response finishes."""

from io import BytesIO
from threading import RLock
from typing import Final
from weakref import finalize

from fastapi import HTTPException, Request, Response
from fastapi.routing import APIRoute
from starlette.types import Message, Receive, Scope, Send

from litellm.tracing.remote import MAX_RESPONSE_BYTES


class BufferBudget:
    def __init__(self, capacity: int) -> None:
        self.capacity: Final = capacity
        self._used = 0
        self._lock: Final = RLock()

    def reserve(self, size: int) -> None:
        with self._lock:
            if self._used + size > self.capacity:
                raise HTTPException(
                    503, "Lens forwarding capacity is busy; retry shortly", headers={"Retry-After": "1"}
                )
            self._used += size

    def release(self, size: int) -> None:
        with self._lock:
            self._used -= size


class BodyReservation:
    def __init__(self, budget: BufferBudget) -> None:
        self._budget: Final = budget
        self._size = 0

    def reserve(self, size: int) -> None:
        self._budget.reserve(size)
        self._size += size

    def release(self) -> None:
        self._budget.release(self._size)
        self._size = 0


class BufferedResponse(Response):
    def __init__(self, body: bytes, status_code: int, headers: dict[str, str], reservation: BodyReservation) -> None:
        super().__init__(body, status_code=status_code, headers=headers)
        # A response discarded before ASGI sends it must release its reservation too.
        self._release: Final = finalize(self, reservation.release)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._release()


# Per gateway process; this accounts for the adapter's request and response buffers.
# Keep a full 64 MiB request plus a full 64 MiB response admissible.
FORWARD_BUFFER_BUDGET: Final = BufferBudget(128 * 1024 * 1024)


async def request_body(
    request: Request, limit: int = MAX_RESPONSE_BYTES, reservation: BodyReservation | None = None
) -> bytes:
    with BytesIO() as body:
        async for chunk in request.stream():
            if body.tell() + len(chunk) > limit:
                raise HTTPException(413, "Lens request is too large")
            if reservation is not None:
                reservation.reserve(len(chunk))
            body.write(chunk)
        return body.getvalue()


class AdmittedBody:
    def __init__(self, body: bytes, receive: Receive) -> None:
        self.body: Final = body
        self._receive: Final = receive
        self._sent = False

    async def __call__(self) -> Message:
        if self._sent:
            return await self._receive()
        self._sent = True
        return {"type": "http.request", "body": self.body, "more_body": False}


class LensRoute(APIRoute):
    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        reservation: Final = BodyReservation(FORWARD_BUFFER_BUDGET)
        try:
            body: Final = await request_body(Request(scope, receive), reservation=reservation)
            await super().handle(scope, AdmittedBody(body, receive), send)
        finally:
            reservation.release()
