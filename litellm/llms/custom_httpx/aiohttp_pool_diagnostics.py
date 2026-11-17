import asyncio
from types import SimpleNamespace
from typing import Final

from aiohttp import (
    ClientSession,
    TraceConfig,
    TraceConnectionQueuedEndParams,
    TraceConnectionQueuedStartParams,
    TraceRequestExceptionParams,
    TraceRequestRedirectParams,
    TraceRequestStartParams,
)
from yarl import URL

from litellm._logging import verbose_logger

_POOL_WAIT_WARNING_SECONDS: Final = 1.0


class _PoolWaitContext(SimpleNamespace):
    def __init__(self, trace_request_ctx: object = None) -> None:
        super().__init__()
        self.host: str | None = None
        self.port: int | None = None
        self.started_at: float | None = None
        self.waited: float = 0.0
        self.warning: asyncio.TimerHandle | None = None
        self.warned: bool = False

    def set_destination(self, params: TraceRequestStartParams) -> None:
        self.host = params.url.host
        self.port = params.url.port

    def redirect(self, params: TraceRequestRedirectParams) -> None:
        location: Final = params.response.headers.get("Location") or params.response.headers.get("URI")
        if location is None:
            return
        try:
            destination: Final = params.url.join(URL(location))
            host: Final = destination.host
            port: Final = destination.port
        except ValueError:
            return
        self.host = host
        self.port = port

    def start(self, session: ClientSession) -> None:
        loop: Final = asyncio.get_running_loop()
        self.started_at = loop.time()
        if not self.warned:
            self.warning = loop.call_later(max(0.0, _POOL_WAIT_WARNING_SECONDS - self.waited), self.warn, session)

    def elapsed(self) -> float:
        return self.waited + (
            asyncio.get_running_loop().time() - self.started_at if self.started_at is not None else 0.0
        )

    def warn(self, session: ClientSession) -> None:
        if self.warned or self.started_at is None:
            return
        self.warned = True
        connector: Final = session.connector
        if connector is None:
            return
        verbose_logger.warning(
            "aiohttp connection pool saturated: host=%s port=%s waited=%.3fs for a connection slot "
            "(acquired_total=%s, limit=%s, limit_per_host=%s). "
            "Wait occurred before sending the request upstream.",
            self.host,
            self.port,
            self.elapsed(),
            len(connector._acquired),  # pyright: ignore[reportPrivateUsage]  # aiohttp exposes no public acquired count
            connector.limit,
            connector.limit_per_host,
        )

    def finish(self, session: ClientSession, failed: bool = False) -> None:
        if self.warning is not None:
            self.warning.cancel()
            self.warning = None
        if self.started_at is None:
            return
        if failed or self.elapsed() >= _POOL_WAIT_WARNING_SECONDS:
            self.warn(session)
        self.waited = self.elapsed()
        self.started_at = None


async def _on_request_start(session: ClientSession, context: SimpleNamespace, params: TraceRequestStartParams) -> None:
    if isinstance(context, _PoolWaitContext):
        context.set_destination(params)


async def _on_queued_start(
    session: ClientSession, context: SimpleNamespace, params: TraceConnectionQueuedStartParams
) -> None:
    if isinstance(context, _PoolWaitContext):
        context.start(session)


async def _on_queued_end(
    session: ClientSession, context: SimpleNamespace, params: TraceConnectionQueuedEndParams
) -> None:
    if isinstance(context, _PoolWaitContext):
        context.finish(session)


async def _on_request_exception(
    session: ClientSession, context: SimpleNamespace, params: TraceRequestExceptionParams
) -> None:
    if isinstance(context, _PoolWaitContext):
        context.finish(session, failed=True)


async def _on_request_redirect(
    session: ClientSession, context: SimpleNamespace, params: TraceRequestRedirectParams
) -> None:
    if isinstance(context, _PoolWaitContext):
        context.redirect(params)


def build_pool_wait_trace_config() -> TraceConfig:
    trace: Final = TraceConfig(trace_config_ctx_factory=_PoolWaitContext)
    trace.on_request_start.append(_on_request_start)
    trace.on_connection_queued_start.append(_on_queued_start)
    trace.on_connection_queued_end.append(_on_queued_end)
    trace.on_request_exception.append(_on_request_exception)
    trace.on_request_redirect.append(_on_request_redirect)
    return trace
