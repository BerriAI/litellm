"""Fire-and-forget push of serialized spend events from an inference worker to the pod-local sidecar.

``LITELLM_COLLECTOR_ENABLED=true`` turns the push on in the gateway; the sidecar process sets
``LITELLM_JOB_ROLE=collector`` and always runs the pipeline in-process. Events queue in a bounded
in-memory buffer that a single writer task flushes over a unix socket or loopback TCP connection.
When the sidecar is unreachable, the buffer is full, or the connection breaks mid-write, each affected
event follows ``LITELLM_COLLECTOR_ON_UNAVAILABLE``: ``fallback`` runs the existing cost pipeline in
the worker, ``drop`` counts it and moves on. Transitions are logged with the counters, so a sidecar
outage is visible without scraping anything.

Delivery is at-most-once: a sidecar crash loses the events the kernel already took from its socket.
A sidecar that stops gracefully half-closes each connection first (EOF towards the producer) and
keeps reading until the producer hangs up, so the producer switches to the unavailable policy without
losing the events in flight. A write that fails part-way follows the unavailable policy without double
counting: ``drain()`` only fails while part of the line is still buffered in this process, so the
sidecar can at most have read a truncated line, which it discards. When the gateway itself stops with
the writer stuck mid-send, only an event whose bytes are still in the producer's write buffer follows
the unavailable policy; the connection is aborted first so the sidecar discards the truncated line
instead of also counting it. Events from one uvicorn worker are handled in the order it produced them;
events from different workers interleave, exactly like the in-process callbacks do today.
"""

import asyncio
import ipaddress
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias
from urllib.parse import urlsplit

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from litellm._logging import verbose_proxy_logger

COLLECTOR_ENV_PREFIX: Final = "LITELLM_COLLECTOR_"
COLLECTOR_JOB_ROLE: Final = "collector"
DEFAULT_COLLECTOR_ADDRESS: Final = "unix:///var/run/litellm/collector.sock"
RECONNECT_BACKOFF_SECONDS: Final = 1.0
DROP_LOG_EVERY: Final = 1000

UnavailablePolicy: TypeAlias = Literal["fallback", "drop"]
PublishOutcome: TypeAlias = Literal["queued", "fallback", "dropped"]


class CollectorSettings(BaseSettings):
    """``LITELLM_COLLECTOR_*`` env vars, shared by the gateway producer and the sidecar consumer."""

    model_config = SettingsConfigDict(
        env_prefix=COLLECTOR_ENV_PREFIX, case_sensitive=False, extra="ignore", frozen=True, populate_by_name=True
    )

    enabled: bool = False
    address: str = DEFAULT_COLLECTOR_ADDRESS
    buffer_size: int = Field(default=1000, ge=1)
    on_unavailable: UnavailablePolicy = "fallback"
    drain_timeout_seconds: float = Field(default=10.0, gt=0)
    connect_timeout_seconds: float = Field(default=1.0, gt=0)
    job_role: str | None = Field(default=None, validation_alias=AliasChoices("LITELLM_JOB_ROLE"))

    @property
    def produces(self) -> bool:
        return self.enabled and self.job_role != COLLECTOR_JOB_ROLE


@dataclass(frozen=True, slots=True)
class UnixAddress:
    path: str


@dataclass(frozen=True, slots=True)
class TcpAddress:
    host: str
    port: int


@dataclass(frozen=True, slots=True)
class AddressError:
    reason: str


CollectorAddress: TypeAlias = UnixAddress | TcpAddress


def _is_loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host == "localhost"


def parse_collector_address(address: str) -> CollectorAddress | AddressError:
    """``unix:///path/to.sock`` or ``tcp://127.0.0.1:port``; the socket carries unauthenticated spend events."""
    parsed: Final = urlsplit(address)
    if parsed.scheme == "unix" and parsed.path:
        return UnixAddress(path=parsed.path)
    if parsed.scheme == "tcp" and parsed.hostname and parsed.port is not None:
        if not _is_loopback(parsed.hostname):
            return AddressError(reason=f"tcp collector address must be a loopback host, got {address!r}")
        return TcpAddress(host=parsed.hostname, port=parsed.port)
    return AddressError(reason=f"expected unix:///path or tcp://127.0.0.1:port, got {address!r}")


async def open_collector_connection(
    address: CollectorAddress, timeout: float
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    match address:
        case UnixAddress(path=path):
            return await asyncio.wait_for(asyncio.open_unix_connection(path), timeout)
        case TcpAddress(host=host, port=port):
            return await asyncio.wait_for(asyncio.open_connection(host, port), timeout)


def build_spend_event_producer(
    settings: CollectorSettings, fallback: Callable[[bytes], Awaitable[None]]
) -> "SpendEventProducer | None":
    """The gateway producer for these settings, or ``None`` when the pipeline stays in-process."""
    if not settings.produces:
        return None
    address: Final = parse_collector_address(settings.address)
    if isinstance(address, AddressError):
        verbose_proxy_logger.error("collector: %s; running the spend pipeline in-process", address.reason)
        return None
    verbose_proxy_logger.info(
        "collector: offloading spend tracking to %s (buffer=%d, on_unavailable=%s)",
        settings.address,
        settings.buffer_size,
        settings.on_unavailable,
    )
    return SpendEventProducer(
        address=address,
        on_unavailable=settings.on_unavailable,
        buffer_size=settings.buffer_size,
        connect_timeout=settings.connect_timeout_seconds,
        fallback=fallback,
    )


@dataclass(frozen=True, slots=True)
class _Connection:
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter

    @property
    def alive(self) -> bool:
        return not self.writer.is_closing() and not self.reader.at_eof()


@dataclass(frozen=True, slots=True)
class SpendEventProducerStats:
    queued: int
    sent: int
    fallback: int
    dropped: int
    connected: bool


class SpendEventProducer:
    """Bounded buffer plus one writer task per process; see the module docstring for the contract."""

    def __init__(
        self,
        address: CollectorAddress,
        on_unavailable: UnavailablePolicy,
        buffer_size: int,
        connect_timeout: float,
        fallback: Callable[[bytes], Awaitable[None]],
        clock: Callable[[], float] = time.monotonic,
        open_connection: Callable[
            [CollectorAddress, float], Awaitable[tuple[asyncio.StreamReader, asyncio.StreamWriter]]
        ] = open_collector_connection,
    ) -> None:
        self._address = address
        self._on_unavailable = on_unavailable
        self._buffer_size = buffer_size
        self._connect_timeout = connect_timeout
        self._fallback = fallback
        self._clock = clock
        self._open_connection = open_connection
        self._queue: asyncio.Queue[bytes] | None = None
        self._writer_task: asyncio.Task[None] | None = None
        self._connection: _Connection | None = None
        self._in_flight: bytes | None = None
        self._closing = False
        self._next_connect_at = 0.0
        self._queued = 0
        self._sent = 0
        self._fallback_count = 0
        self._dropped = 0

    def stats(self) -> SpendEventProducerStats:
        return SpendEventProducerStats(
            queued=self._queued,
            sent=self._sent,
            fallback=self._fallback_count,
            dropped=self._dropped,
            connected=self._connection is not None,
        )

    async def publish(self, line: bytes) -> PublishOutcome:
        """Hand one serialized event to the writer task, or apply the unavailable policy right away."""
        if self._closing or self._clock() < self._next_connect_at:
            return await self._unavailable(line, "sidecar unreachable")
        queue: Final = self._ensure_writer()
        try:
            queue.put_nowait(line)
        except asyncio.QueueFull:
            return await self._unavailable(line, "buffer full")
        self._queued += 1
        return "queued"

    async def close(self, drain_timeout: float) -> None:
        """Flush the buffer for up to ``drain_timeout`` seconds, then apply the unavailable policy to the rest."""
        self._closing = True
        queue: Final = self._queue
        task: Final = self._writer_task
        if queue is None or task is None:
            return
        try:
            await asyncio.wait_for(queue.join(), drain_timeout)
        except asyncio.TimeoutError:
            verbose_proxy_logger.warning(
                "collector: %s events still buffered after %.1fs drain timeout", queue.qsize(), drain_timeout
            )
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        unsent: Final = self._take_unsent()
        await self._disconnect()
        if unsent is not None:
            await self._unavailable(unsent, "shutdown")
        while not queue.empty():
            await self._unavailable(queue.get_nowait(), "shutdown")

    def _take_unsent(self) -> bytes | None:
        """The in-flight event if any of its bytes never left this process, aborting the half-written connection."""
        in_flight: Final = self._in_flight
        self._in_flight = None
        connection: Final = self._connection
        if in_flight is None:
            return None
        if connection is None:
            return in_flight
        if connection.writer.transport.get_write_buffer_size() == 0:
            return None
        connection.writer.transport.abort()
        return in_flight

    def _ensure_writer(self) -> asyncio.Queue[bytes]:
        if self._queue is None:
            self._queue = asyncio.Queue(maxsize=self._buffer_size)
        if self._writer_task is None or self._writer_task.done():
            self._writer_task = asyncio.get_running_loop().create_task(self._run_writer(self._queue))
        return self._queue

    async def _run_writer(self, queue: asyncio.Queue[bytes]) -> None:
        while True:
            line = await queue.get()
            try:
                await self._send(line)
            finally:
                queue.task_done()

    async def _send(self, line: bytes) -> None:
        self._in_flight = line
        connection: Final = await self._connect()
        if connection is None:
            self._in_flight = None
            await self._unavailable(line, "sidecar unreachable")
            return
        try:
            connection.writer.write(line)
            await connection.writer.drain()
        except (ConnectionError, OSError, RuntimeError) as error:  # uvloop: RuntimeError on a closed transport
            self._in_flight = None
            await self._disconnect()
            self._next_connect_at = self._clock() + RECONNECT_BACKOFF_SECONDS
            await self._unavailable(line, f"write failed: {error}")
            return
        self._in_flight = None
        self._sent += 1

    async def _connect(self) -> _Connection | None:
        if self._connection is not None and self._connection.alive:
            return self._connection
        await self._disconnect()
        if self._clock() < self._next_connect_at:
            return None
        try:
            reader, writer = await self._open_connection(self._address, self._connect_timeout)
        except (ConnectionError, OSError, asyncio.TimeoutError) as error:
            self._next_connect_at = self._clock() + RECONNECT_BACKOFF_SECONDS
            verbose_proxy_logger.warning(
                "collector: cannot reach %s (%s); applying %s policy for %.0fs. stats=%s",
                self._address,
                error,
                self._on_unavailable,
                RECONNECT_BACKOFF_SECONDS,
                self.stats(),
            )
            return None
        self._connection = _Connection(reader=reader, writer=writer)
        verbose_proxy_logger.info("collector: connected to %s. stats=%s", self._address, self.stats())
        return self._connection

    async def _disconnect(self) -> None:
        connection: Final = self._connection
        self._connection = None
        if connection is None:
            return
        connection.writer.close()
        try:
            await connection.writer.wait_closed()
        except (ConnectionError, OSError):
            pass

    async def _unavailable(self, line: bytes, reason: str) -> PublishOutcome:
        if self._on_unavailable == "fallback":
            self._fallback_count += 1
            fallback: Final = asyncio.ensure_future(self._run_fallback(line, reason))
            try:
                await asyncio.shield(fallback)
            except asyncio.CancelledError:
                await fallback
                raise
            return "fallback"
        self._dropped += 1
        if self._dropped % DROP_LOG_EVERY == 1:
            verbose_proxy_logger.warning("collector: dropping spend event (%s). stats=%s", reason, self.stats())
        return "dropped"

    async def _run_fallback(self, line: bytes, reason: str) -> None:
        try:
            await self._fallback(line)
        except Exception:  # noqa: BLE001  # one failing event must not kill the writer task
            verbose_proxy_logger.exception("collector: in-process fallback failed (%s)", reason)
