from collections.abc import AsyncGenerator, Callable, Generator, Mapping
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager
from typing import Final

import httpx
from fastapi import HTTPException, Request
from pydantic import ConfigDict, TypeAdapter
from typing_extensions import override

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.integrations.otel.logger import OpenTelemetryV2
from litellm.integrations.otel.model.config import ExporterSpec, OpenTelemetryV2Config
from litellm.tracing import TraceReceiver
from litellm.tracing.exporter import LensExporter
from litellm.tracing.remote import LensConnection, RemoteTraceStore
from litellm.tracing.storage import LensTraceStorage
from litellm.types.utils import CaptureMessageContent

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


@asynccontextmanager
async def manage_tracing(
    enabled: bool,
    receiver_factory: Callable[[], TraceReceiver] | None = None,
    settings: Mapping[str, object] | None = None,
    client_factory: Callable[[LensConnection], httpx.AsyncClient] = LensConnection.lifespan_client,
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
        async with AsyncExitStack() as exports:
            await exports.enter_async_context(_export_requests(LensExporter(client)))
            if await _accepts_gateway_traces(client):
                exports.enter_context(_export_traces(connection))
            yield tracing


async def _accepts_gateway_traces(client: httpx.AsyncClient) -> bool:
    """A Lens released before gateway traces has no route for them, and would fail every export."""
    try:
        response: Final = await client.post(
            LENS_TRACE_PATH, content=b"", headers={"Content-Type": "application/x-protobuf"}
        )
    except httpx.HTTPError:
        return True
    if response.status_code != httpx.codes.NOT_FOUND:
        return True
    verbose_proxy_logger.warning(
        "Lens does not accept gateway traces, so model calls reach it as cost only. Upgrade Lens to see them as traces."
    )
    return False


LENS_TRACE_PATH: Final = "/internal/traces"


LENS_TRACE_CALLBACK: Final = "lens"


class _LensTraceLogger(OpenTelemetryV2):
    """Model-call spans only, on a private provider, so a gateway's own OTel callbacks are left as they are."""

    @override
    def _init_otel_logger_on_litellm_proxy(self) -> None:
        return None


@contextmanager
def _export_traces(connection: LensConnection) -> Generator[None, None, None]:
    trace_logger: Final = _LensTraceLogger(
        config=OpenTelemetryV2Config(
            exporters=[
                ExporterSpec(
                    kind="otlp_http",
                    traces_endpoint=connection.endpoint(LENS_TRACE_PATH),
                    headers=f"Authorization=Bearer {connection.token}",
                )
            ],
            capture_message_content=CaptureMessageContent.SPAN_ONLY,
        ),
        callback_name=LENS_TRACE_CALLBACK,
    )
    manager: Final = litellm.logging_callback_manager
    manager.add_litellm_callback(trace_logger)
    try:
        yield None
    finally:
        manager.remove_callback_from_all_lists(trace_logger)


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
