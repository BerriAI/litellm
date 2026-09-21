"""Collector sidecar: consume spend events from the pod's inference workers and run the cost pipeline.

Runs the proxy startup lifespan (config, Prisma, Redis transaction buffer, scheduled spend flushes)
without serving HTTP, then listens on ``LITELLM_COLLECTOR_ADDRESS`` for newline-delimited spend
events. Each event goes through the unchanged ``_ProxyDBLogger._PROXY_track_cost_callback``, so
spend logs, spend counters, budget reservation reconciliation and cache updates happen exactly as
they would in-process, just in this container. Events are handled in order per producer connection
(one per uvicorn worker); a slow pipeline fills the socket buffer and the producer's bounded queue,
which is the backpressure that triggers its fallback or drop policy. ``SIGTERM`` stops accepting
connections, half-closes every producer connection so the producers switch to their unavailable
policy, finishes the events already sent, then runs the proxy shutdown (which flushes the buffered
spend transactions).

``DATABASE_URL`` is assembled from the same ``DATABASE_*`` inputs as the proxy container, and when
``LITELLM_PGBOUNCER_ENABLED`` is set it points at the PgBouncer that container already runs on the
pod's loopback, so the sidecar must see the same env as the proxy. Under ``IAM_TOKEN_DB_AUTH`` or
``AZURE_POSTGRESQL_AUTH`` that PgBouncer only accepts the token the proxy container minted, so the
sidecar goes to Postgres directly and mints its own. Works from any image that has ``litellm``
installed:

    python -m litellm.proxy.collector [--address unix:///path.sock]
"""

import asyncio
import logging
import os
import signal
import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Final

from litellm._logging import verbose_logger, verbose_proxy_logger, verbose_router_logger
from litellm.proxy.db.db_url_settings import DatabaseURLSettings
from litellm.proxy.db.pgbouncer import (
    PgBouncerError,
    PgBouncerSettings,
    export_pooled_database_url,
    pooled_database_url,
)
from litellm.proxy.spend_tracking.spend_event_producer import (
    COLLECTOR_JOB_ROLE,
    AddressError,
    CollectorAddress,
    CollectorSettings,
    TcpAddress,
    UnixAddress,
    parse_collector_address,
)

MAX_EVENT_BYTES: Final = 64 * 1024 * 1024


class SpendEventConsumer:
    """Accepts producer connections and runs ``handler`` on every line each one sends, in order."""

    def __init__(self, handler: Callable[[bytes], Awaitable[None]]) -> None:
        self._handler = handler
        self._open_connections: set[asyncio.StreamWriter] = set()  # mutable-ok: live producer connections
        self._idle = asyncio.Event()
        self._idle.set()
        self._received = 0
        self._handled = 0
        self._failed = 0

    @property
    def received(self) -> int:
        return self._received

    @property
    def handled(self) -> int:
        return self._handled

    @property
    def failed(self) -> int:
        return self._failed

    async def serve(self, address: CollectorAddress) -> asyncio.Server:
        match address:
            case UnixAddress(path=path):
                socket_path: Final = Path(path)
                socket_path.parent.mkdir(parents=True, exist_ok=True)
                socket_path.unlink(missing_ok=True)
                return await asyncio.start_unix_server(self._on_connection, path=path, limit=MAX_EVENT_BYTES)
            case TcpAddress(host=host, port=port):
                return await asyncio.start_server(self._on_connection, host=host, port=port, limit=MAX_EVENT_BYTES)

    async def _on_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._open_connections.add(writer)
        self._idle.clear()
        try:
            while line := await reader.readline():
                if not line.endswith(b"\n"):
                    verbose_proxy_logger.error("collector: discarding truncated spend event (%d bytes)", len(line))
                    break
                self._received += 1
                await self._handle(line)
        except (ConnectionError, asyncio.IncompleteReadError, asyncio.LimitOverrunError) as error:
            verbose_proxy_logger.warning("collector: producer connection ended abnormally: %s", error)
        finally:
            writer.close()
            self._open_connections.discard(writer)
            if not self._open_connections:
                self._idle.set()

    async def _handle(self, line: bytes) -> None:
        try:
            await self._handler(line)
            self._handled += 1
        except Exception:  # noqa: BLE001  # the cost pipeline raises anything; one bad event must not stop the sidecar
            self._failed += 1
            verbose_proxy_logger.exception("collector: spend event failed")

    async def drain(self, timeout: float) -> int:
        """Half-close every producer connection, then keep reading until each producer hangs up or ``timeout``.

        Returns how many producer connections were still open when the timeout hit.
        """
        for writer in tuple(self._open_connections):
            if writer.is_closing() or not writer.can_write_eof():
                continue
            try:
                writer.write_eof()
            except (OSError, RuntimeError) as error:
                verbose_proxy_logger.debug("collector: producer already gone before half-close: %s", error)
        try:
            await asyncio.wait_for(self._idle.wait(), timeout)
        except TimeoutError:
            pass
        return len(self._open_connections)


def _install_stop_signals(loop: asyncio.AbstractEventLoop, stop: asyncio.Event) -> None:
    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, stop.set)


async def run_collector(address: CollectorAddress, drain_timeout: float) -> None:
    from fastapi import FastAPI

    from litellm.proxy.hooks.proxy_track_cost_callback import run_spend_event
    from litellm.proxy.proxy_server import proxy_startup_event

    stop: Final = asyncio.Event()
    _install_stop_signals(asyncio.get_running_loop(), stop)
    consumer: Final = SpendEventConsumer(handler=run_spend_event)
    async with proxy_startup_event(FastAPI()):
        server: Final = await consumer.serve(address)
        verbose_proxy_logger.info("collector: listening on %s", address)
        await stop.wait()
        server.close()
        still_open: Final = await consumer.drain(drain_timeout)
        verbose_proxy_logger.info(
            "collector: stopping. received=%d handled=%d failed=%d connections_cut=%d",
            consumer.received,
            consumer.handled,
            consumer.failed,
            still_open,
        )


def address_argument(argv: Sequence[str], default: str) -> str | AddressError:
    match tuple(argv):
        case ():
            return default
        case ("--address", value):
            return value
        case _:
            return AddressError(f"usage: python -m litellm.proxy.collector [--address ADDRESS], got {tuple(argv)}")


def apply_log_level(litellm_log: str | None) -> None:
    """Mirror the proxy's ``LITELLM_LOG`` handling: the sidecar has no CLI flags to turn logging on."""
    level: Final = logging.getLevelNamesMapping().get((litellm_log or "").upper())
    if level is None:
        return
    for logger in (verbose_logger, verbose_router_logger, verbose_proxy_logger):
        logger.setLevel(level)


def pod_pgbouncer_database_url(
    pgbouncer: PgBouncerSettings, environ: Mapping[str, str], *, token_auth: bool
) -> str | PgBouncerError | None:
    """The proxy container's PgBouncer URL for ``environ["DATABASE_URL"]``, or None to connect to Postgres directly.

    Direct is the answer when PgBouncer is off, and also under token auth: that PgBouncer's auth file
    only holds the token its own container minted, which this container cannot present.
    """
    if not pgbouncer.enabled or token_auth:
        return None
    upstream_url: Final = environ.get("DATABASE_URL")
    if upstream_url is None:
        return PgBouncerError("LITELLM_PGBOUNCER_ENABLED is set but no DATABASE_URL could be assembled")
    return pooled_database_url(upstream_url, pgbouncer)


def main(argv: Sequence[str]) -> None:
    os.environ.setdefault("LITELLM_JOB_ROLE", COLLECTOR_JOB_ROLE)
    apply_log_level(os.environ.get("LITELLM_LOG"))
    database: Final = DatabaseURLSettings.from_env()
    database.apply_to_env()
    pooled: Final = pod_pgbouncer_database_url(
        PgBouncerSettings(),
        os.environ,
        token_auth=database.iam_token_db_auth or database.azure_postgresql_auth,
    )
    if isinstance(pooled, PgBouncerError):
        sys.exit(f"LiteLLM collector: cannot use the pod's pgbouncer: {pooled.reason}")
    if pooled is not None:
        export_pooled_database_url(pooled)
    settings: Final = CollectorSettings()
    raw_address: Final = address_argument(argv, default=settings.address)
    address: Final = raw_address if isinstance(raw_address, AddressError) else parse_collector_address(raw_address)
    if isinstance(address, AddressError):
        sys.exit(f"LiteLLM collector: {address.reason}")
    asyncio.run(run_collector(address, drain_timeout=settings.drain_timeout_seconds))


if __name__ == "__main__":
    main(sys.argv[1:])
