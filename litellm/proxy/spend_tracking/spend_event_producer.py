"""Fire-and-forget push of serialized spend events from an inference worker to the pod-local sidecar.

``LITELLM_SPEND_WORKER_ENABLED=true`` turns the push on in the gateway; the sidecar process sets
``LITELLM_JOB_ROLE=spend_worker`` and always runs the pipeline in-process. Events queue in a bounded
in-memory buffer that a single writer task flushes over a unix socket or loopback TCP connection.
When the sidecar is unreachable, the buffer is full, or the connection breaks mid-write, each affected
event follows ``LITELLM_SPEND_WORKER_ON_UNAVAILABLE``: ``fallback`` runs the existing cost pipeline in
the worker, ``drop`` counts it and moves on. Transitions are logged with the counters, so a sidecar
outage is visible without scraping anything.

Delivery is at-most-once: a sidecar crash loses the events already handed to its socket. Events from
one uvicorn worker are handled in the order it produced them; events from different workers interleave,
exactly like the in-process callbacks do today.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias
from urllib.parse import urlsplit

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from litellm._logging import verbose_proxy_logger

SPEND_WORKER_ENV_PREFIX: Final = "LITELLM_SPEND_WORKER_"
SPEND_WORKER_JOB_ROLE: Final = "spend_worker"
DEFAULT_SPEND_WORKER_ADDRESS: Final = "unix:///var/run/litellm/spend-worker.sock"
RECONNECT_BACKOFF_SECONDS: Final = 1.0
DROP_LOG_EVERY: Final = 1000

UnavailablePolicy: TypeAlias = Literal["fallback", "drop"]
PublishOutcome: TypeAlias = Literal["queued", "fallback", "dropped"]


class SpendWorkerSettings(BaseSettings):
    """``LITELLM_SPEND_WORKER_*`` env vars, shared by the gateway producer and the sidecar consumer."""

    model_config = SettingsConfigDict(
        env_prefix=SPEND_WORKER_ENV_PREFIX, case_sensitive=False, extra="ignore", frozen=True, populate_by_name=True
    )

    enabled: bool = False
    address: str = DEFAULT_SPEND_WORKER_ADDRESS
    buffer_size: int = Field(default=1000, ge=1)
    on_unavailable: UnavailablePolicy = "fallback"
    drain_timeout_seconds: float = Field(default=10.0, gt=0)
    connect_timeout_seconds: float = Field(default=1.0, gt=0)
    job_role: str | None = Field(default=None, validation_alias=AliasChoices("LITELLM_JOB_ROLE"))

    @property
    def produces(self) -> bool:
        return self.enabled and self.job_role != SPEND_WORKER_JOB_ROLE


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


SpendWorkerAddress: TypeAlias = UnixAddress | TcpAddress


def parse_spend_worker_address(address: str) -> SpendWorkerAddress | AddressError:
    """``unix:///path/to.sock`` or ``tcp://127.0.0.1:port``."""
    parsed: Final = urlsplit(address)
    if parsed.scheme == "unix" and parsed.path:
        return UnixAddress(path=parsed.path)
    if parsed.scheme == "tcp" and parsed.hostname and parsed.port is not None:
        return TcpAddress(host=parsed.hostname, port=parsed.port)
    return AddressError(reason=f"expected unix:///path or tcp://host:port, got {address!r}")


async def open_spend_worker_connection(
    address: SpendWorkerAddress, timeout: float
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    match address:
        case UnixAddress(path=path):
            return await asyncio.wait_for(asyncio.open_unix_connection(path), timeout)
        case TcpAddress(host=host, port=port):
            return await asyncio.wait_for(asyncio.open_connection(host, port), timeout)


def build_spend_event_producer(
    settings: SpendWorkerSettings, fallback: Callable[[bytes], Awaitable[None]]
) -> "SpendEventProducer | None":
    """The gateway producer for these settings, or ``None`` when the pipeline stays in-process."""
    if not settings.produces:
        return None
    address: Final = parse_spend_worker_address(settings.address)
    if isinstance(address, AddressError):
        verbose_proxy_logger.error("spend worker: %s; running the spend pipeline in-process", address.reason)
        return None
    verbose_proxy_logger.info(
        "spend worker: offloading spend tracking to %s (buffer=%d, on_unavailable=%s)",
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
        address: SpendWorkerAddress,
        on_unavailable: UnavailablePolicy,
        buffer_size: int,
        connect_timeout: float,
        fallback: Callable[[bytes], Awaitable[None]],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._address = address
        self._on_unavailable = on_unavailable
        self._buffer_size = buffer_size
        self._connect_timeout = connect_timeout
        self._fallback = fallback
        self._clock = clock
        self._queue: asyncio.Queue[bytes] | None = None
        self._writer_task: asyncio.Task[None] | None = None
        self._writer: asyncio.StreamWriter | None = None
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
            connected=self._writer is not None,
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
                "spend worker: %s events still buffered after %.1fs drain timeout", queue.qsize(), drain_timeout
            )
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        while not queue.empty():
            await self._unavailable(queue.get_nowait(), "shutdown")
        await self._disconnect()

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
        writer: Final = await self._connect()
        if writer is None:
            await self._unavailable(line, "sidecar unreachable")
            return
        try:
            writer.write(line)
            await writer.drain()
        except (ConnectionError, OSError) as error:
            await self._disconnect()
            self._next_connect_at = self._clock() + RECONNECT_BACKOFF_SECONDS
            await self._unavailable(line, f"write failed: {error}")
            return
        self._sent += 1

    async def _connect(self) -> asyncio.StreamWriter | None:
        if self._writer is not None:
            return self._writer
        if self._clock() < self._next_connect_at:
            return None
        try:
            _, writer = await open_spend_worker_connection(self._address, self._connect_timeout)
        except (ConnectionError, OSError, asyncio.TimeoutError) as error:
            self._next_connect_at = self._clock() + RECONNECT_BACKOFF_SECONDS
            verbose_proxy_logger.warning(
                "spend worker: cannot reach %s (%s); applying %s policy for %.0fs. stats=%s",
                self._address,
                error,
                self._on_unavailable,
                RECONNECT_BACKOFF_SECONDS,
                self.stats(),
            )
            return None
        self._writer = writer
        verbose_proxy_logger.info("spend worker: connected to %s. stats=%s", self._address, self.stats())
        return writer

    async def _disconnect(self) -> None:
        writer: Final = self._writer
        self._writer = None
        if writer is None:
            return
        writer.close()
        try:
            await writer.wait_closed()
        except (ConnectionError, OSError):
            pass

    async def _unavailable(self, line: bytes, reason: str) -> PublishOutcome:
        if self._on_unavailable == "fallback":
            self._fallback_count += 1
            try:
                await self._fallback(line)
            except Exception:
                verbose_proxy_logger.exception("spend worker: in-process fallback failed (%s)", reason)
            return "fallback"
        self._dropped += 1
        if self._dropped % DROP_LOG_EVERY == 1:
            verbose_proxy_logger.warning("spend worker: dropping spend event (%s). stats=%s", reason, self.stats())
        return "dropped"
