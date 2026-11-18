import asyncio
from collections.abc import AsyncGenerator, Callable, Mapping
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Final

import httpx
from fastapi import HTTPException, Request
from pydantic import ConfigDict, TypeAdapter

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.repositories.accounting_delivery_repository import LedgerDatabase
from litellm.repositories.lens_export_repository import LensExportRepository
from litellm.tracing import TraceReceiver
from litellm.tracing.exporter import LensExporter
from litellm.tracing.ledger_export import (
    UNREACHABLE_CAPABILITIES,
    CostRunner,
    ExportSettings,
    Probe,
    Supported,
    Unreachable,
    Unsupported,
    UploadRunner,
    exporter_owner,
    probe_lens,
    run_loop,
)
from litellm.tracing.remote import LensConnection, RemoteTraceStore
from litellm.tracing.storage import LensTraceStorage
from litellm.tracing.types import LedgerCapabilities

_RECEIVER_ADAPTER: Final[TypeAdapter[TraceReceiver | None]] = TypeAdapter(
    TraceReceiver | None, config=ConfigDict(arbitrary_types_allowed=True)
)
_UNAVAILABLE_DETAIL: Final = "Agent tracing is not enabled. Configure the Lens service and LITELLM_LENS_URL."


def require_receiver(tracing: TraceReceiver | None) -> TraceReceiver:
    if tracing is None:
        raise HTTPException(status_code=501, detail=_UNAVAILABLE_DETAIL)
    return tracing


async def provide_receiver(request: Request) -> TraceReceiver | None:
    return _RECEIVER_ADAPTER.validate_python(getattr(request.state, "tracing_receiver", None))


async def provide_storage(request: Request) -> LensTraceStorage | None:
    tracing: Final = await provide_receiver(request)
    return tracing.storage if tracing is not None else None


@dataclass(frozen=True, slots=True)
class LedgerExportOptions:
    enabled: bool
    database: LedgerDatabase
    publish: Callable[[LedgerCapabilities | None], None]
    settings: ExportSettings


@asynccontextmanager
async def manage_tracing(
    enabled: bool,
    receiver_factory: Callable[[], TraceReceiver] | None = None,
    settings: Mapping[str, object] | None = None,
    client_factory: Callable[[LensConnection], httpx.AsyncClient] = LensConnection.lifespan_client,
    ledger: LedgerExportOptions | None = None,
) -> AsyncGenerator[TraceReceiver | None, None]:
    if not enabled:
        yield None
        return
    try:
        connection: Final = LensConnection.from_env()
    except ValueError as error:
        verbose_proxy_logger.error("Agent tracing store lens failed to start, tracing is disabled: %s", error)
        yield None
        return
    async with client_factory(connection) as client:
        tracing: Final = (
            receiver_factory()
            if receiver_factory
            else TraceReceiver(storage=LensTraceStorage(RemoteTraceStore(client)))
        )
        if ledger is None:
            async with _export_requests(LensExporter(client)):
                yield tracing
            return
        async with _ledger_export(client, ledger):
            yield tracing


@asynccontextmanager
async def _export_requests(spend_logger: LensExporter) -> AsyncGenerator[None, None]:
    spend_logger.start()
    manager: Final = litellm.logging_callback_manager
    manager.add_litellm_callback(spend_logger)
    manager.add_litellm_success_callback(spend_logger)
    manager.add_litellm_failure_callback(spend_logger)
    manager.add_litellm_async_success_callback(spend_logger)
    manager.add_litellm_async_failure_callback(spend_logger)
    verbose_proxy_logger.info("Agent tracing enabled (store=lens)")
    try:
        yield None
    finally:
        manager.remove_callback_from_all_lists(spend_logger)
        await spend_logger.aclose()


class _LegacyProducer:
    """The in-memory Lens callback, registered only while the ledger path is not the producer."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client: Final = client
        self._exporter: LensExporter | None = None

    def start(self) -> None:
        if self._exporter is not None:
            return
        self._exporter = LensExporter(self._client)
        self._exporter.start()
        manager: Final = litellm.logging_callback_manager
        manager.add_litellm_callback(self._exporter)
        manager.add_litellm_success_callback(self._exporter)
        manager.add_litellm_failure_callback(self._exporter)
        manager.add_litellm_async_success_callback(self._exporter)
        manager.add_litellm_async_failure_callback(self._exporter)
        verbose_proxy_logger.info("Agent tracing enabled (store=lens, producer=callback)")

    async def stop(self) -> None:
        exporter: Final = self._exporter
        if exporter is None:
            return
        self._exporter = None
        litellm.logging_callback_manager.remove_callback_from_all_lists(exporter)
        await exporter.aclose()


class _LedgerProducer:
    """Chooses the single Lens producer from the latest probe.

    Only a Lens that answers without ledger version 1 hands production back to the callback. An outage
    keeps the ledger producing with the last known contract, because its outbox is durable.
    """

    def __init__(self, client: httpx.AsyncClient, ledger: LedgerExportOptions) -> None:
        self._ledger: Final = ledger
        self._legacy: Final = _LegacyProducer(client)
        self._known: LedgerCapabilities = UNREACHABLE_CAPABILITIES
        self.probe: Probe = Unreachable("NotProbed")

    def draining(self) -> bool:
        return not isinstance(self.probe, Unsupported)

    async def apply(self, probe: Probe) -> None:
        self.probe = probe
        if isinstance(probe, Supported):
            self._known = probe.capabilities
        if self._ledger.enabled and self.draining():
            self._ledger.publish(self._known)
            await self._legacy.stop()
            return
        self._legacy.start()
        self._ledger.publish(None)

    async def close(self) -> None:
        self._ledger.publish(None)
        await self._legacy.stop()


async def _reprobe(client: httpx.AsyncClient, producer: _LedgerProducer, stop: asyncio.Event, seconds: float) -> None:
    while not stop.is_set():
        probe = await probe_lens(client)  # rebind-ok: one probe per cycle
        if stop.is_set():
            return
        await producer.apply(probe)
        with suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=seconds)


async def _join(tasks: tuple[asyncio.Task[None], ...], seconds: float) -> None:
    _, pending = await asyncio.wait(tasks, timeout=seconds)
    for task in pending:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


@asynccontextmanager
async def _ledger_export(client: httpx.AsyncClient, ledger: LedgerExportOptions) -> AsyncGenerator[None, None]:
    producer: Final = _LedgerProducer(client, ledger)
    await producer.apply(Unreachable("NotProbed"))
    settings: Final = ledger.settings
    costs: Final = LensExportRepository(ledger.database, exporter_owner("cost"))
    uploads: Final = LensExportRepository(ledger.database, exporter_owner("upload"))
    stop: Final = asyncio.Event()
    tasks: Final = (
        asyncio.create_task(
            run_loop(
                CostRunner(costs, client, settings).step,
                costs.release,
                stop=stop,
                active=producer.draining,
                idle_seconds=settings.cost_idle_seconds,
            )
        ),
        asyncio.create_task(
            run_loop(
                UploadRunner(uploads, client, settings).step,
                uploads.release,
                stop=stop,
                active=producer.draining,
                idle_seconds=settings.upload_idle_seconds,
            )
        ),
        asyncio.create_task(_reprobe(client, producer, stop, settings.probe_seconds)),
    )
    try:
        yield None
    finally:
        await producer.close()
        stop.set()
        await _join(tasks, settings.stop_seconds)
