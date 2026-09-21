from collections.abc import Awaitable, Callable, Mapping
from typing import Final

from starlette.types import ASGIApp, Receive, Scope, Send


class BudgetReservationReleaseMiddleware:
    """Releases the budget reservation auth made for a request once no callback owns it.

    Auth stamps the reservation on ``request.state``; a handler that builds a logging
    object binds it to the cost callbacks, which settle it on success or failure. When
    the response has been sent and the reservation is still unbound, nothing else ever
    would, so it is released here instead of pinning the spend counter until its TTL.
    """

    def __init__(self, app: ASGIApp, release: Callable[[Mapping[str, object]], Awaitable[None]]) -> None:
        self.app = app
        self.release = release

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        try:
            await self.app(scope, receive, send)
        finally:
            state: Final = scope.get("state")
            budget_reservation: Final = state.get("budget_reservation") if isinstance(state, Mapping) else None
            if isinstance(budget_reservation, Mapping):
                await self.release(budget_reservation)
