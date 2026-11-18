import asyncio
from collections.abc import AsyncGenerator, Callable, Mapping
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass
from typing import Final

import httpx
from fastapi import HTTPException, Request
from pydantic import ConfigDict, TypeAdapter

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.tracing import TraceReceiver
from litellm.repositories.accounting_delivery_repository import LedgerDatabase
from litellm.repositories.lens_export_repository import LensExportRepository
from litellm.tracing.exporter import LensExporter
from litellm.tracing.ledger_export import ExportSettings, LensLedgerExporter, exporter_owner, ledger_capabilities
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
        capabilities: Final = await ledger_capabilities(client) if ledger is not None else None
        ledger_active: Final = ledger is not None and ledger.enabled and capabilities is not None
        async with (
            _drain_ledger(client, ledger, capabilities if ledger_active else None)
            if ledger is not None and capabilities is not None
            else nullcontext()
        ):
            async with nullcontext() if ledger_active else _export_requests(LensExporter(client)):
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


@asynccontextmanager
async def _drain_ledger(
    client: httpx.AsyncClient, ledger: LedgerExportOptions, active: LedgerCapabilities | None
) -> AsyncGenerator[None, None]:
    exporter: Final = LensLedgerExporter(
        LensExportRepository(ledger.database, exporter_owner()), client, ledger.settings
    )
    stop: Final = asyncio.Event()
    task: Final = asyncio.create_task(exporter.run(stop))
    ledger.publish(active)
    verbose_proxy_logger.info(
        "Lens ledger export %s", "enabled" if active is not None else "draining committed records only"
    )
    try:
        yield None
    finally:
        ledger.publish(None)
        stop.set()
        await task
