import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from typing import Final, Protocol

from pydantic import BaseModel
from websockets.exceptions import ConnectionClosedOK

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.realtime_streaming import REALTIME_SESSION_SUCCESS_LOGGED_KEY
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.spend_tracking.budget_reservation import (
    invalidate_budget_reservation_counters,
    release_or_invalidate_budget_reservation,
)


class ObserverSocket(Protocol):
    def __aiter__(self) -> AsyncIterator[str | bytes]: ...

    async def close(self) -> None: ...


class UsageSink(Protocol):
    def store_message(self, message: str) -> None: ...

    async def log_messages(self, *, wait_for_dispatch: bool = False) -> None: ...


class _ObserverEvent(BaseModel):
    type: str


class CallSupervisor:
    def __init__(
        self,
        upstream: ObserverSocket,
        stream: UsageSink,
        logging_obj: Logging,
        auth: UserAPIKeyAuth,
        close_call: Callable[[], Awaitable[None]],
        *,
        ready_timeout: float = 20,
        lifetime: float = 3600,
        drain_timeout: float = 5,
        termination_timeout: float = 60,
        terminal_usage_required: bool = True,
    ) -> None:
        self._upstream = upstream
        self._stream = stream
        self._logging = logging_obj
        self._auth = auth
        self._close_call = close_call
        self._ready_timeout = ready_timeout
        self._lifetime = lifetime
        self._drain_timeout = drain_timeout
        self._termination_timeout = termination_timeout
        self._terminal_usage_required = terminal_usage_required
        self._ready = asyncio.Event()
        self._stop = asyncio.Event()
        self._started = False
        self._terminal = False
        self._close_confirmed = False
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._task is not None:
            raise RuntimeError("Call observer already started")
        self._task = asyncio.create_task(self._run())
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=self._ready_timeout)
            if not self._started or self._task.done():
                raise RuntimeError("Call observer ended before session became available")
        except BaseException:
            await self.close()
            raise

    async def close(self) -> None:
        self._stop.set()
        await self.wait()

    async def wait(self) -> None:
        if self._task is not None:
            await asyncio.shield(self._task)

    async def _read(self) -> None:
        try:
            await self._read_events()
        except ConnectionClosedOK:
            return

    async def _read_events(self) -> None:
        event: _ObserverEvent
        async for message in self._upstream:
            self._stream.store_message(message.decode("utf-8") if isinstance(message, bytes) else message)
            event = _ObserverEvent.model_validate_json(message)
            if event.type in ("session.started", "session.created"):
                self._started = True
                self._ready.set()
            if event.type == "session.closed":
                self._terminal = True
                return

    def _usage_complete(self) -> bool:
        return self._terminal or (not self._terminal_usage_required and self._close_confirmed)

    async def _run(self) -> None:
        reader: Final = asyncio.create_task(self._read())
        stopped: Final = asyncio.create_task(self._stop.wait())
        try:
            await asyncio.wait((reader, stopped), timeout=self._lifetime, return_when=asyncio.FIRST_COMPLETED)
        finally:
            try:
                if not self._terminal:
                    try:
                        await asyncio.wait_for(self._close_call(), timeout=self._termination_timeout)
                        self._close_confirmed = True
                    except Exception:  # noqa: BLE001  # provider exceptions can contain credentials
                        verbose_proxy_logger.error("Realtime observer could not terminate upstream call")
                    await self._drain(reader)
            finally:
                stopped.cancel()
                reader.cancel()
                await asyncio.gather(reader, stopped, return_exceptions=True)
                with suppress(Exception):
                    await self._upstream.close()
                if not self._usage_complete():
                    self._logging.model_call_details["realtime_usage_incomplete"] = True
                    verbose_proxy_logger.error(
                        "Realtime observer ended without terminal usage; recorded usage is partial"
                    )
                try:
                    try:
                        await self._stream.log_messages(wait_for_dispatch=True)
                    finally:
                        if self._started and not self._usage_complete():
                            await invalidate_budget_reservation_counters(
                                budget_reservation=self._auth.budget_reservation
                            )
                        elif not self._logging.model_call_details.get(REALTIME_SESSION_SUCCESS_LOGGED_KEY):
                            await release_or_invalidate_budget_reservation(
                                budget_reservation=self._auth.budget_reservation
                            )
                finally:
                    self._ready.set()

    async def _drain(self, reader: asyncio.Task[None]) -> None:
        try:
            await asyncio.wait_for(asyncio.shield(reader), timeout=self._drain_timeout)
        except asyncio.TimeoutError:
            if not self._usage_complete():
                verbose_proxy_logger.error("Realtime observer timed out draining terminal usage")
        except Exception:  # noqa: BLE001  # cleanup must settle the socket even when reading or closing fails
            verbose_proxy_logger.error("Realtime observer could not drain terminal usage")
            return


class CallSupervisors:
    def __init__(self) -> None:
        self._tasks: tuple[asyncio.Task[None], ...] = ()
        self._calls: tuple[CallSupervisor, ...] = ()

    async def start(self, supervisor: CallSupervisor) -> None:
        self._calls = (*self._calls, supervisor)
        try:
            await supervisor.start()
        except BaseException:
            self._calls = tuple(call for call in self._calls if call is not supervisor)
            raise
        task: Final = asyncio.create_task(self._watch(supervisor))
        self._tasks = (*self._tasks, task)

    async def _watch(self, supervisor: CallSupervisor) -> None:
        try:
            try:
                await supervisor.wait()
            except Exception:  # noqa: BLE001  # task must be consumed without exposing provider exception payloads
                verbose_proxy_logger.error("Realtime observer accounting failed")
        finally:
            self._calls = tuple(call for call in self._calls if call is not supervisor)
            self._tasks = tuple(task for task in self._tasks if task is not asyncio.current_task())

    async def shutdown(self) -> None:
        await asyncio.gather(*(call.close() for call in self._calls), return_exceptions=True)
        await asyncio.gather(*self._tasks, return_exceptions=True)


CALL_SUPERVISORS: Final = CallSupervisors()
