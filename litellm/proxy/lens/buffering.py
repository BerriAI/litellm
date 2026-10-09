"""Account for forwarded Lens bodies until the downstream response finishes."""

from threading import RLock
from typing import Final
from weakref import finalize

from fastapi import HTTPException, Response
from starlette.types import Receive, Scope, Send


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
