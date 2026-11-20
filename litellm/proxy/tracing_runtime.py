import asyncio
from collections.abc import AsyncGenerator, Callable, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from contextvars import ContextVar
from datetime import datetime
from http import HTTPStatus
from typing import Final

import httpx
from fastapi import HTTPException, Request
from opentelemetry.context import Context
from opentelemetry.sdk.trace import Span as RecordingSpan
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.trace import Span
from pydantic import ConfigDict, TypeAdapter
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource
from typing_extensions import override

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.integrations.otel.logger import OpenTelemetryV2
from litellm.integrations.otel.model.config import ExporterSpec, OpenTelemetryV2Config
from litellm.integrations.otel.model.metadata import LLMCallEvent
from litellm.integrations.otel.model.utils import to_ns
from litellm.proxy.lens_agent_steps import AgentCall, agent_call
from litellm.tracing import TraceReceiver
from litellm.tracing.exporter import LensExporter
from litellm.tracing.remote import LensConnection, RemoteTraceStore
from litellm.tracing.storage import LensTraceStorage
from litellm.types.utils import CaptureMessageContent

_RECEIVER_ADAPTER: Final[TypeAdapter[TraceReceiver | None]] = TypeAdapter(
    TraceReceiver | None, config=ConfigDict(arbitrary_types_allowed=True)
)
_AGENT_NAME: Final = "gen_ai.agent.name"
_SESSION_ATTRIBUTE: Final = "gen_ai.conversation.id"
_TOOL_STEP_TRACER: Final = "litellm.lens.tool_steps"
_CALLING_AGENT: Final[ContextVar[str | None]] = ContextVar("lens_calling_agent", default=None)
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
            if await _accepts_gateway_traces(client, connection):
                await exports.enter_async_context(_export_traces(connection))
            yield tracing


async def _accepts_gateway_traces(client: httpx.AsyncClient, connection: LensConnection) -> bool:
    """A Lens released before gateway traces has no route for them, and would fail every export."""
    try:
        response: Final = await client.post(
            connection.endpoint(LENS_TRACE_PATH), content=b"", headers={"Content-Type": "application/x-protobuf"}
        )
    except httpx.HTTPError:
        return True
    if response.status_code != HTTPStatus.NOT_FOUND:
        return True
    verbose_proxy_logger.warning(
        "Lens does not accept gateway traces, so model calls reach it as cost only. Upgrade Lens to see them as traces."
    )
    return False


LENS_TRACE_PATH: Final = "/internal/traces"


LENS_TRACE_CALLBACK: Final = "lens"


class _LensTraceConfig(OpenTelemetryV2Config):
    """Built from its arguments alone, so the OTEL_* variables a gateway sets for its own tracing never reach Lens."""

    @classmethod
    @override
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings,)


class _LensTraceLogger(OpenTelemetryV2):
    """Model-call spans only, on a private provider, so a gateway's own OTel callbacks are left as they are."""

    @override
    def _init_otel_logger_on_litellm_proxy(self) -> None:
        return None

    @override
    def _close_llm_call(
        self,
        kwargs: Mapping[str, object],
        start_time: datetime | float | None,
        end_time: datetime | float | None,
    ) -> Span | None:
        call: Final = LLMCallEvent.from_dict(kwargs)
        made_by: Final = agent_call(call.payload)
        carrier: Final = self._open_llm_calls.get(call.call_id) if call.call_id else None
        if made_by.agent and carrier is not None and carrier.span is not None:
            carrier.span.set_attribute(_AGENT_NAME, made_by.agent)
        token: Final = _CALLING_AGENT.set(made_by.agent)
        try:
            span: Final = super()._close_llm_call(kwargs, start_time, end_time)
        finally:
            _CALLING_AGENT.reset(token)
        if span is not None and call.session_id:
            self._emit_tool_steps(made_by, call.session_id, to_ns(start_time))
        return span

    def _emit_tool_steps(self, made_by: AgentCall, session_id: str, at_ns: int | None) -> None:
        """One zero-length step per tool result, at the start of the model call whose prompt carries it."""
        tracer: Final = self.tracer_provider.get_tracer(_TOOL_STEP_TRACER)
        for exchange in made_by.tool_exchanges:
            tracer.start_span(
                f"execute_tool {exchange.name}",
                start_time=at_ns,
                attributes={
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": exchange.name,
                    "gen_ai.tool.call.id": exchange.call_id,
                    "gen_ai.tool.call.arguments": exchange.arguments,
                    "gen_ai.tool.call.result": exchange.result,
                    _SESSION_ATTRIBUTE: session_id,
                    **({_AGENT_NAME: made_by.agent} if made_by.agent else {}),
                },
            ).end(end_time=at_ns)


class _AgentNameProcessor(SpanProcessor):
    """Names the calling agent on a model-call span that is only created when the call closes."""

    @override
    def on_start(self, span: RecordingSpan, parent_context: Context | None = None) -> None:
        agent: Final = _CALLING_AGENT.get()
        if agent:
            span.set_attribute(_AGENT_NAME, agent)


@asynccontextmanager
async def _export_traces(connection: LensConnection) -> AsyncGenerator[None, None]:
    trace_logger: Final = _LensTraceLogger(
        config=_LensTraceConfig(
            exporters=[
                ExporterSpec(
                    kind="otlp_http",
                    traces_endpoint=connection.endpoint(LENS_TRACE_PATH),
                    headers=f"Authorization=Bearer {connection.token}",
                    resource_attributes={"lens.session.capture": "true", "lens.session.attribute": _SESSION_ATTRIBUTE},
                )
            ],
            capture_message_content=CaptureMessageContent.SPAN_ONLY,
        ),
        callback_name=LENS_TRACE_CALLBACK,
    )
    trace_logger.tracer_provider.add_span_processor(_AgentNameProcessor())
    manager: Final = litellm.logging_callback_manager
    manager.add_litellm_callback(trace_logger)
    try:
        yield None
    finally:
        manager.remove_callback_from_all_lists(trace_logger)
        await asyncio.to_thread(trace_logger.tracer_provider.shutdown)


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
