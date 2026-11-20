import asyncio
from collections.abc import Callable
from contextvars import ContextVar
from typing import Final

from starlette.routing import get_route_path
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from litellm._logging import verbose_proxy_logger
from litellm.types.integrations.prometheus import ProxyRequestOutcome

_CHAT_PATHS: Final = frozenset({"/chat/completions", "/v1/chat/completions"})


class _RequestOutcome:
    def __init__(self) -> None:
        self.status = 0
        self.completed = False
        self.failed = False
        self.cancelled = False

    def result(self) -> ProxyRequestOutcome:
        if self.cancelled or self.status == 499:
            return "cancelled"
        if self.failed or self.status >= 400 or not self.completed:
            return "error"
        return "success"


_state: Final[ContextVar[_RequestOutcome | None]] = ContextVar("prometheus_request_outcome", default=None)


def mark_request_stream_failed() -> None:
    state: Final = _state.get()
    if state is not None:
        state.failed = True


def mark_request_stream_cancelled() -> None:
    state: Final = _state.get()
    if state is not None and not state.completed:
        state.cancelled = True


class PrometheusRequestOutcomeMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        recorder: Callable[[], Callable[[ProxyRequestOutcome], None] | None],
    ) -> None:
        self.app: Final = app
        self.recorder: Final = recorder

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return
        record: Final = self.recorder() if get_route_path(scope) in _CHAT_PATHS else None
        if record is None:
            await self.app(scope, receive, send)
            return

        state: Final = _RequestOutcome()

        async def receive_message() -> Message:
            message: Final = await receive()
            if message["type"] == "http.disconnect" and not state.completed:
                state.cancelled = True
            return message

        async def send_message(message: Message) -> None:
            try:
                await send(message)
            except OSError:
                if not state.completed:
                    state.cancelled = True
                raise
            if message["type"] == "http.response.start":
                state.status = message["status"]
            elif message["type"] == "http.response.body" and not message.get("more_body", False):
                state.completed = True

        token: Final = _state.set(state)
        try:
            await self.app(scope, receive_message, send_message)
        except asyncio.CancelledError:
            if not state.completed:
                state.cancelled = True
            raise
        except Exception:
            if not state.completed:
                state.failed = True
            raise
        finally:
            _state.reset(token)
            try:
                record(state.result())
            except Exception:  # noqa: BLE001  # an exporter failure cannot change the response
                verbose_proxy_logger.debug("Could not record Prometheus request outcome", exc_info=True)
