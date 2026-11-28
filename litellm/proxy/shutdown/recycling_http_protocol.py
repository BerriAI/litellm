import asyncio
from typing import TYPE_CHECKING, Final

from pydantic import TypeAdapter
from uvicorn._types import ASGI3Application, ASGIReceiveCallable, ASGISendCallable, ASGISendEvent, Scope
from uvicorn.protocols.http.flow_control import CLOSE_HEADER

if TYPE_CHECKING:
    from uvicorn.protocols.http.h11_impl import H11Protocol as AutoHTTPProtocol
else:
    from uvicorn.protocols.http.auto import AutoHTTPProtocol

_ASGI_APPLICATION: Final = TypeAdapter[ASGI3Application](ASGI3Application)


class RecyclingHTTPProtocol(AutoHTTPProtocol):
    _shutdown_deadline: float | None = None
    _drain_idle_timer: asyncio.TimerHandle | None = None

    def connection_made(self, transport: asyncio.Transport) -> None:
        super().connection_made(transport)
        self._idle_graceful_timeout = self.config.timeout_graceful_shutdown
        self._application = _ASGI_APPLICATION.validate_python(self.app)
        self.app = self._draining_app

    async def _draining_app(self, scope: Scope, receive: ASGIReceiveCallable, send: ASGISendCallable) -> None:
        async def send_with_close(message: ASGISendEvent) -> None:
            if self._shutdown_deadline is not None and message["type"] == "http.response.start":
                headers: Final = tuple(message.get("headers", ()))
                if CLOSE_HEADER not in headers:
                    await send({**message, "headers": (*headers, CLOSE_HEADER)})
                    return
            await send(message)

        await self._application(scope, receive, send_with_close)

    def shutdown(self) -> None:
        if self._shutdown_deadline is not None:
            return
        now: Final = self.loop.time()
        timeout: Final = self._idle_graceful_timeout
        self._shutdown_deadline = now + (timeout if timeout is not None else self.timeout_keep_alive)
        if not self.cycle or self.cycle.response_complete:
            idle_timer: Final = self.timeout_keep_alive_task
            self._keep_idle_deadline(idle_timer.when() if idle_timer is not None else now + self.timeout_keep_alive)

    def on_response_complete(self) -> None:
        super().on_response_complete()
        idle_timer: Final = self.timeout_keep_alive_task
        if idle_timer is not None:
            self._keep_idle_deadline(idle_timer.when())

    def _keep_idle_deadline(self, idle_deadline: float) -> None:
        deadline: Final = self._shutdown_deadline
        if deadline is None:
            return
        if self._drain_idle_timer is not None:
            self._drain_idle_timer.cancel()
        self._drain_idle_timer = self.loop.call_at(min(deadline, idle_deadline), self._close_if_idle)
        self._unset_keepalive_if_required()

    def _close_if_idle(self) -> None:
        self._drain_idle_timer = None
        if not self.cycle or self.cycle.response_complete:
            self.transport.close()

    def connection_lost(self, exc: Exception | None) -> None:
        if self._drain_idle_timer is not None:
            self._drain_idle_timer.cancel()
        super().connection_lost(exc)
