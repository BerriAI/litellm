"""``CustomLogger`` adapter on the OpenTelemetry span engine."""

from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Iterator, Mapping, Sequence, cast

from opentelemetry.context import Context, attach, get_current
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Span, Tracer, get_current_span, use_span

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.otel.model.baggage import promoted_baggage
from litellm.integrations.otel.model.config import OpenTelemetryV2Config
from litellm.integrations.otel.plumbing.context import (
    is_recordable_span,
    request_root_span,
    resolve_mcp_span_context,
    resolve_parent_context,
    resolve_request_span_context,
    set_request_baggage,
    set_request_root_span,
)
from litellm.integrations.otel.emitter import SpanEmitter
from litellm.integrations.otel.mappers import resolve_mappers
from litellm.integrations.otel.model.metadata import (
    LLMCallEvent,
    RequestIdentity,
    model_from_request_data,
)
from litellm.integrations.otel.model.payloads import (
    GuardrailSpanData,
    LLMCallSpanData,
    MCPListToolsSpanData,
    MCPToolCallSpanData,
    ServiceSpanData,
    SpanError,
    is_mcp_list_tools,
    is_mcp_tool_call,
)
from litellm.integrations.otel.plumbing.events import GenAIEventRecorder
from litellm.integrations.otel.plumbing.metrics import (
    GenAIMetricRecorder,
    create_genai_metrics,
)
from litellm.integrations.otel.plumbing.providers import (
    build_tracer_provider,
    get_event_logger,
    get_meter,
    get_tracer,
    resolve_logger_provider,
    resolve_meter_provider,
)
from litellm.integrations.otel.plumbing.routing import TenantTracerCache
from litellm.integrations.otel.model.spans import SpanRole, span_role_for_service
from litellm.integrations.otel.model.utils import to_ns

if TYPE_CHECKING:
    from litellm.types.utils import (
        StandardLoggingGuardrailInformation,
        StandardLoggingPayload,
    )

LITELLM_TRACER_NAME = "litellm"

# Any callback whose class belongs to one of these modules is "the OTel
# callback" for proxy-global-registration purposes.
_OTEL_MODULES = (
    "litellm.integrations.otel",
    "litellm.integrations.opentelemetry",
)


# Cap on the open-call carrier map. A span opened at ``pre_call`` that never
# reaches a success/failure callback (e.g. a stream that only fires stream
# events) would otherwise linger; bounding the map evicts the oldest so memory
# stays flat on a long-running proxy while covering every concurrent in-flight
# call.
_OPEN_CALLS_MAX = 10_000


class _LLMCallSpan:
    """The state carried from the ``pre_call`` boundary to span close.

    ``span`` is the live span when it could be opened at the boundary (the server
    span was ambient), or ``None`` when creation was deferred because no ambient
    parent was visible — in which case the async callback creates it against its
    own (worker-copied) ambient context using ``start_time_ns``. The presence of
    a carrier for a call at all is the proof that ``pre_call`` ran, i.e. that an
    upstream call was actually attempted.
    """

    __slots__ = ("span", "start_time_ns")

    def __init__(self, span: "Span | None", start_time_ns: int | None) -> None:
        self.span = span
        self.start_time_ns = start_time_ns


class OpenTelemetryV2(CustomLogger):
    """The ``CustomLogger`` for OpenTelemetry."""

    def __init__(
        self,
        config: OpenTelemetryV2Config | None = None,
        callback_name: str | None = None,
        tracer_provider: TracerProvider | None = None,
        logger_provider: LoggerProvider | None = None,
        meter_provider: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.config: OpenTelemetryV2Config = config or OpenTelemetryV2Config(**kwargs)
        self.callback_name = callback_name
        self._tracer_provider: TracerProvider = (
            tracer_provider if tracer_provider is not None else build_tracer_provider(self.config)
        )
        self.tracer: Tracer = get_tracer(self._tracer_provider, LITELLM_TRACER_NAME)
        self._metrics_recorder = self._init_metrics(meter_provider)
        self._metric_filter_error_logged = False
        self._emitter = SpanEmitter(
            self.tracer,
            self.config,
            mappers=resolve_mappers(self.config.mapper_names),
            event_recorder=self._init_events(logger_provider),
        )
        self._tenant_tracers = TenantTracerCache(self.config, callback_name, LITELLM_TRACER_NAME)
        self._open_llm_calls: "OrderedDict[str, _LLMCallSpan]" = OrderedDict()
        self._init_otel_logger_on_litellm_proxy()

    def _init_metrics(self, meter_provider: Any | None) -> "GenAIMetricRecorder | None":
        """Create the six GenAI histograms when metrics are enabled, else ``None``.

        ``meter_provider`` is an explicit override (tests inject one); otherwise the
        provider is resolved from the OTel global so the operator's configured
        readers/exporters receive the metrics, building and registering one only
        when no global provider is set.
        """
        if not self.config.enable_metrics:
            return None
        provider = resolve_meter_provider(self.config, meter_provider)
        meter = get_meter(provider, LITELLM_TRACER_NAME)
        return GenAIMetricRecorder(create_genai_metrics(meter), self.callback_name)

    def _init_events(self, logger_provider: LoggerProvider | None) -> "GenAIEventRecorder | None":
        """Create the GenAI event recorder when events are enabled, else ``None``.

        ``logger_provider`` is an explicit override (tests inject one); otherwise the
        provider is resolved from the OTel global so an operator-configured logs
        pipeline receives the events, building and registering one only when no
        global provider is set. A ``None`` resolution means the operator opted out
        of the logs signal, so no recorder is built.
        """
        if not self.config.enable_events:
            return None
        provider = resolve_logger_provider(self.config, logger_provider)
        if provider is None:
            return None
        return GenAIEventRecorder(get_event_logger(provider, LITELLM_TRACER_NAME))

    # ====================================================================== #
    #  Proxy global registration
    # ====================================================================== #

    def _register_in_callback_list(self, callbacks: list) -> None:
        already_otel = any(
            cb.__class__.__module__.startswith(_OTEL_MODULES) for cb in callbacks if hasattr(cb, "__class__")
        )
        if not already_otel:
            callbacks.append(self)

    def _init_otel_logger_on_litellm_proxy(self) -> None:
        try:
            from litellm.proxy import proxy_server
        except Exception:
            return
        try:
            self._register_in_callback_list(litellm.service_callback)
            self._register_in_callback_list(litellm.input_callback)
            self._register_in_callback_list(litellm._async_success_callback)
            self._register_in_callback_list(litellm._async_failure_callback)
        except Exception:
            pass
        if getattr(proxy_server, "open_telemetry_logger", None) is None:
            setattr(proxy_server, "open_telemetry_logger", self)

    # ====================================================================== #
    #  LLM-call callbacks — the span is opened at the ``pre_call`` boundary and
    #  closed here. See ``log_pre_api_call``.
    # ====================================================================== #

    def log_pre_api_call(self, model, messages, kwargs):
        """Open the LLM-call span at the call boundary.

        Runs synchronously inside the request task, before the upstream call —
        the one place where the live server span is genuinely the ambient OTel
        context — so the span parents to it natively, with no span threaded
        through a metadata dict. The open span is stashed on the per-request
        ``LiteLLMLoggingObj`` (a typed object) and closed in the async callback.

        When no recordable parent is visible (``pre_call`` was driven from a thread
        pool for a sync-only provider, where contextvars — and so the anchor —
        don't follow), creation is deferred: only the start time is recorded, and
        the async callback — whose worker context was copied from the request task
        and so still carries the anchor — creates the span then.

        Synthetic proxy-gate error logs (auth/rate-limit rejections) also fire this
        hook but never made an upstream call; they are tagged and skipped so no
        phantom LLM-call span is produced.
        """
        call = LLMCallEvent.from_dict(kwargs)
        if call.is_no_upstream_call:
            return
        call_id = call.call_id
        if call_id is None:
            return
        # Idempotent: a retried call may re-enter ``pre_call`` with the same
        # call id; keep the first span so its start time is the true one.
        if call_id in self._open_llm_calls:
            return
        start_time_ns = to_ns(datetime.now())
        span: Span | None = None
        # Parent to the request's anchored root span (stable across the request),
        # falling back to ambient on the SDK path. Open the span live only when
        # that resolves to a recordable parent; otherwise defer to the close
        # callback (the thread-pool case, where the anchor isn't visible here).
        parent_context = resolve_request_span_context()
        if is_recordable_span(get_current_span(parent_context)):
            span = self._emitter.start_span(
                SpanRole.LLM_CALL,
                call.provisional_span_name,
                parent_context=parent_context,
                start_time_ns=start_time_ns,
                tracer=self._tenant_tracers.tracer_for(self.tracer, call.dynamic_params),
            )
        self._open_llm_calls[call_id] = _LLMCallSpan(span=span, start_time_ns=start_time_ns)
        # Evict the oldest open call if the map is over budget. A call that opens
        # but never closes (a stream that only fires stream events) would linger
        # otherwise; the evicted span is simply dropped (never exported).
        if len(self._open_llm_calls) > _OPEN_CALLS_MAX:
            self._open_llm_calls.popitem(last=False)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        if self._emit_mcp_tool_call(kwargs, start_time, end_time):
            return
        if self._emit_mcp_list_tools(kwargs, start_time, end_time):
            return
        self._close_llm_call(kwargs, start_time, end_time)
        self._record_metrics(kwargs, response_obj, start_time, end_time)

    def _record_metrics(self, kwargs, response_obj, start_time, end_time) -> None:
        """Record the GenAI metrics for a successful LLM call. Best-effort: a
        recording failure (e.g. a malformed payload) must never break the span
        close or the request itself."""
        if self._metrics_recorder is None:
            return
        try:
            self._metrics_recorder.record(kwargs, response_obj, start_time, end_time)
        except ValueError as exc:
            if not self._metric_filter_error_logged:
                verbose_logger.error(
                    "OpenTelemetryV2: invalid otel.attributes metric filter, metrics disabled: %s",
                    exc,
                )
                self._metric_filter_error_logged = True
        except Exception as exc:
            verbose_logger.debug("OpenTelemetryV2: metric recording failed: %s", exc)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        if self._emit_mcp_tool_call(kwargs, start_time, end_time):
            return
        if self._emit_mcp_list_tools(kwargs, start_time, end_time):
            return
        self._close_llm_call(kwargs, start_time, end_time)

    def _seed_identity_baggage(self, identity: RequestIdentity, model: str | None, context: Context) -> Context:
        """Seed authenticated request-identity Baggage onto ``context`` so the Baggage
        processor stamps team/key/metadata onto the span. Identity is read from the
        parsed payload, never the client's ``params._meta`` carrier, so it can't be
        spoofed."""
        bag = promoted_baggage(
            identity,
            model,
            promoted_keys=tuple(self.config.baggage_promoted_keys),
            metadata_keys=tuple(self.config.baggage_metadata_keys),
            team_metadata_keys=tuple(self.config.baggage_team_metadata_keys),
        )
        return set_request_baggage(bag, context=context) if bag else context

    def _emit_mcp_tool_call(
        self,
        kwargs: Mapping[str, Any],
        start_time: datetime | float | None,
        end_time: datetime | float | None,
    ) -> bool:
        """Emit an MCP tool-call span when the closed request was a tool call.

        MCP tool calls reach the success/failure callbacks like any other request
        (with ``call_type`` ``call_mcp_tool``), but they are not LLM calls and have
        no ``pre_call`` carrier — so they get their own CLIENT span here. Per the MCP
        semconv it parents to the trace context the client propagated in
        ``params._meta`` (or starts a new root) and links the transport span, rather
        than nesting under the HTTP/session span. Returns whether it handled the
        event, so the caller skips the LLM-call path. The whole span is emitted at
        once (there is no boundary to open it at), deduped on the call id.
        """
        raw_payload = kwargs.get("standard_logging_object")
        if not raw_payload or not is_mcp_tool_call(cast(Mapping[str, object], raw_payload)):
            return False
        payload = cast("StandardLoggingPayload", raw_payload)
        data = MCPToolCallSpanData.from_standard_logging_payload(
            payload, capture_content=self.config.capture_span_content
        )
        # A stray LLM carrier from a ``pre_call`` that mis-fired for this id would
        # otherwise linger until evicted; drop it so it's neither leaked nor closed
        # as a phantom LLM span.
        if data.identity.call_id:
            self._open_llm_calls.pop(data.identity.call_id, None)
        parent_context, links = resolve_mcp_span_context()
        parent_context = self._seed_identity_baggage(data.identity, None, parent_context)
        self._emitter.emit(
            SpanRole.MCP_TOOL_CALL,
            data,
            parent_context=parent_context,
            start_time_ns=to_ns(start_time),
            end_time_ns=to_ns(end_time),
            links=links,
        )
        return True

    def _emit_mcp_list_tools(
        self,
        kwargs: Mapping[str, object],
        start_time: datetime | float | None,
        end_time: datetime | float | None,
    ) -> bool:
        """Emit an MCP ``tools/list`` span when the closed request was a discovery call.

        Like a tool call, listing reaches the success/failure callbacks (here with
        ``call_type`` ``list_mcp_tools``) with no ``pre_call`` carrier, so it gets its
        own CLIENT span. Per the MCP semconv it parents to the ``params._meta`` trace
        context (or starts a new root) and links the transport span, rather than
        nesting under the HTTP/session span. Returns whether it handled the event so
        the caller skips the LLM-call path.
        """
        raw_payload = kwargs.get("standard_logging_object")
        if not raw_payload or not is_mcp_list_tools(cast(Mapping[str, object], raw_payload)):
            return False
        payload = cast("StandardLoggingPayload", raw_payload)
        data = MCPListToolsSpanData.from_standard_logging_payload(
            payload, capture_content=self.config.capture_span_content
        )
        if data.identity.call_id:
            self._open_llm_calls.pop(data.identity.call_id, None)
        parent_context, links = resolve_mcp_span_context()
        parent_context = self._seed_identity_baggage(data.identity, None, parent_context)
        self._emitter.emit(
            SpanRole.MCP_LIST_TOOLS,
            data,
            parent_context=parent_context,
            start_time_ns=to_ns(start_time),
            end_time_ns=to_ns(end_time),
            links=links,
        )
        return True

    def _close_llm_call(
        self,
        kwargs: Mapping[str, Any],
        start_time: datetime | float | None,
        end_time: datetime | float | None,
    ) -> Span | None:
        """Finish the LLM-call span opened at ``pre_call`` (or create it deferred).

        No carrier for this call id means ``pre_call`` never ran — the request was
        rejected at the gate or blocked by a pre-call guardrail before any upstream
        call — so there is nothing to record and no phantom span.
        """
        call = LLMCallEvent.from_dict(kwargs)
        call_id = call.call_id
        # ``pop`` is the dedup: this method runs from both the success and failure
        # paths, and whichever fires first removes the carrier and closes the span.
        carrier = self._open_llm_calls.pop(call_id, None) if call_id else None
        if carrier is None:
            return None
        payload = call.payload
        if payload is None:
            if carrier.span is not None:
                # Opened at the boundary but the payload never materialized — end
                # it (named provisionally) so it isn't leaked as an open span.
                carrier.span.end(end_time=to_ns(end_time))
            return None
        data = LLMCallSpanData.from_standard_logging_payload(
            payload,
            capture_content=self.config.capture_span_content,
            time_to_first_chunk_seconds=call.time_to_first_chunk_seconds,
        )
        end_time_ns = to_ns(end_time)
        if carrier.span is not None:
            # Born at the boundary: stamp attributes from the typed payload, set
            # status, and end it. Its parent (the server span) was captured at
            # creation from real ambient context.
            self._emitter.finish_span(SpanRole.LLM_CALL, carrier.span, data, end_time_ns=end_time_ns)
            return carrier.span
        # Deferred: ``pre_call`` saw no recordable parent, so create the span now.
        # The worker copied the request task's context, which carries the anchored
        # root span — parent to it (ambient fallback on the SDK path). Seed identity
        # Baggage so the span — and the SDK path, which has none — is labeled
        # consistently.
        parent_ctx = self._seed_identity_baggage(data.identity, data.request_model, resolve_request_span_context())
        return self._emitter.emit(
            SpanRole.LLM_CALL,
            data,
            parent_context=parent_ctx,
            start_time_ns=carrier.start_time_ns,
            end_time_ns=end_time_ns,
            tracer=self._tenant_tracers.tracer_for(self.tracer, call.dynamic_params),
        )

    # ====================================================================== #
    #  Service hooks
    # ====================================================================== #

    async def async_service_success_hook(
        self,
        payload: Any,
        parent_otel_span: Span | None = None,
        start_time: datetime | float | None = None,
        end_time: datetime | float | None = None,
        event_metadata: dict | None = None,
    ) -> None:
        self._emit_service(
            payload,
            parent_otel_span=parent_otel_span,
            start_time=start_time,
            end_time=end_time,
            event_metadata=event_metadata,
            error_override=None,
        )

    async def async_service_failure_hook(
        self,
        payload: Any,
        error: str | None = "",
        parent_otel_span: Span | None = None,
        start_time: datetime | float | None = None,
        end_time: datetime | float | None = None,
        event_metadata: dict | None = None,
    ) -> None:
        self._emit_service(
            payload,
            parent_otel_span=parent_otel_span,
            start_time=start_time,
            end_time=end_time,
            event_metadata=event_metadata,
            error_override=error or "error",
        )

    def _emit_service(
        self,
        payload: Any,
        *,
        parent_otel_span: Span | None,
        start_time: datetime | float | None,
        end_time: datetime | float | None,
        event_metadata: dict | None,
        error_override: str | None,
    ) -> Span | None:
        data = ServiceSpanData.from_payload(payload, event_metadata=event_metadata)
        # Decide whether this service call is a span at all, and of what kind.
        # ``None`` means metrics-only (framework instrumentation that duplicates a
        # gen-AI span — ``self``/``router``/``proxy_pre_call`` — or ``auth``, which
        # gets a live phase span instead). Those still feed Prometheus/Datadog via
        # their own hooks; they just never enter the trace.
        role = span_role_for_service(data.service_name)
        if role is None:
            return None
        # A metrics-only ping with neither timing nor a parent (in-memory queue
        # gauges) is not a traceable operation; a span for it would be a
        # zero-duration root with no context, so skip it. Real background work
        # (budget/reset jobs, spend flush) passes start/end times and still emits
        # as a root; anything with a parent emits regardless.
        if error_override is None and start_time is None and end_time is None and parent_otel_span is None:
            return None
        if error_override is not None and data.error is None:
            data = ServiceSpanData(
                service_name=data.service_name,
                call_type=data.call_type,
                error=SpanError(message=error_override),
                event_metadata=data.event_metadata,
            )
        # Parent like every other span: ambient context first (so identity Baggage
        # rides along and the call nests under whatever request phase is active —
        # e.g. a DB lookup under the live ``auth`` span), falling back to the
        # server span the proxy threaded as ``parent_otel_span``. A background
        # service call has neither, so it starts its own root trace.
        parent_context = resolve_parent_context(threaded=parent_otel_span)
        return self._emitter.emit(
            role,
            data,
            parent_context=parent_context,
            start_time_ns=to_ns(start_time),
            end_time_ns=to_ns(end_time),
        )

    # ====================================================================== #
    #  async_post_call_* hooks — emit guardrail spans. The server span's status
    #  / errors are the FastAPI instrumentor's job, so we don't touch it here.
    # ====================================================================== #

    def seed_request_identity(self, user_api_key_dict: Any, model: Any = None) -> None:
        """Attach request-identity Baggage to the current context + server span.

        Seeding identity into Baggage makes **every** span emitted afterwards for
        this request — LLM call, guardrail, DB call — inherit it via
        ``LiteLLMBaggageSpanProcessor``. Called once at the auth boundary (as soon
        as the key resolves) so post-auth spans are labeled consistently; the
        Baggage rides the request task's contextvar from there on. Auth-internal
        DB lookups that run before the key is known stay unlabeled — identity
        isn't determined yet, which is correct.
        """
        try:
            identity = RequestIdentity.from_user_api_key_auth(user_api_key_dict)
            bag = promoted_baggage(
                identity,
                model,
                promoted_keys=tuple(self.config.baggage_promoted_keys),
                metadata_keys=tuple(self.config.baggage_metadata_keys),
                team_metadata_keys=tuple(self.config.baggage_team_metadata_keys),
            )
            if bag:
                # Attach (no detach): the contextvar is scoped to this request's
                # asyncio task and is reclaimed when the task ends.
                attach(set_request_baggage(bag, context=get_current()))
                # The server span was started by the instrumentor before this ran,
                # so the Baggage processor (which only fires at span start) won't
                # backfill it — stamp identity on it directly. Prefer the anchored
                # root span over the ambient one so identity still lands on the
                # server span when seeding from inside the live ``auth`` phase span
                # (the auth-failure path), where ``get_current_span`` is the phase
                # span, not the request's root.
                server_span = request_root_span() or get_current_span()
                if is_recordable_span(server_span):
                    # Re-capture the anchor here too: this runs post-auth with the
                    # server span active and covers entrypoints that bypass
                    # ``create_litellm_proxy_request_started_span`` (e.g. the SDK
                    # path's ``async_pre_call_hook``). Idempotent.
                    set_request_root_span(server_span)
                    for key, value in bag.items():
                        server_span.set_attribute(key, value)
        except Exception:
            pass

    @contextmanager
    def start_phase_span(self, name: str) -> "Iterator[Span]":
        span = self._emitter.start_span(SpanRole.SERVICE, name)
        with use_span(span, end_on_exit=True):
            yield span

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict,
        call_type: Any,
    ) -> dict:
        self.seed_request_identity(
            user_api_key_dict,
            model=model_from_request_data(data),
        )
        return data

    def emit_guardrail_span(self, entry: "StandardLoggingGuardrailInformation") -> None:
        # Emitted by the guardrail-recording code the moment a guardrail finishes,
        # not from a post-call hook — that hook does not fire on every path (a
        # pass-through request that passes its guardrails never reaches it), which
        # left passing guardrails without a span.
        #
        # A guardrail is a sibling of the LLM call under the request's root span,
        # so parent it to the explicit anchor — never the active span, which during
        # a pre_call guardrail can be the live ``auth`` phase span. Emit with the
        # guardrail's actual execution window so a pre_call guardrail is placed
        # before the LLM call rather than at emission time. One entry in, one span
        # out — the module-level entry point routes each entry to this single
        # registered logger so a guardrail is never emitted more than once.
        data = GuardrailSpanData.from_logging_entry(entry)
        self._emitter.emit(
            SpanRole.GUARDRAIL,
            data,
            parent_context=resolve_request_span_context(),
            start_time_ns=to_ns(data.start_time),
            end_time_ns=to_ns(data.end_time),
        )

    def create_litellm_proxy_request_started_span(
        self, start_time: datetime, headers: Mapping[str, str] | None
    ) -> Span | None:
        span = get_current_span()
        if not is_recordable_span(span):
            return None
        set_request_root_span(span)
        return span


def select_global_otel_v2_logger(
    in_memory_loggers: Sequence[object],
    registered: "OpenTelemetryV2 | None" = None,
) -> "OpenTelemetryV2":
    """The single ``OpenTelemetryV2`` whose provider should become the OTel global.

    The callback factory designates one logger as canonical the moment it builds
    the first one (``_init_otel_logger_on_litellm_proxy`` sets
    ``proxy_server.open_telemetry_logger``), and every other v2 entry point —
    guardrail, identity seeding, phase spans — already routes through that same
    ``registered`` owner. Reuse it here too so the global provider has one source
    of truth instead of a second, independently-derived guess; this is the logger
    a preset (arize, langfuse, …) folds the ``OTEL_*`` base exporter and its own
    exporter into, so the FastAPI server span and the gen-ai spans share one
    provider and one trace.

    Fall back to ``in_memory_loggers`` for the SDK path, where no proxy global is
    set (selecting from there, not ``service_callback``, which a preset logger does
    not always reach), and build a generic logger from ``OTEL_*`` only when none was
    configured at all. Each fallback still avoids the second generic logger that
    orphaned the gen-ai spans onto a different backend than the server span.
    """
    if registered is not None:
        return registered
    existing = next((cb for cb in in_memory_loggers if isinstance(cb, OpenTelemetryV2)), None)
    return existing if existing is not None else OpenTelemetryV2()


def publish_global_otel_v2_provider(
    in_memory_loggers: Sequence[object],
    set_global_provider: Callable[[TracerProvider], None],
    registered: "OpenTelemetryV2 | None" = None,
) -> "OpenTelemetryV2":
    """Select the single v2 logger and publish its provider as the OTel global.

    The proxy calls this once at startup, after callbacks are initialized, so the
    preset logger already exists; it passes ``registered`` (the canonical owner the
    factory designated as ``proxy_server.open_telemetry_logger``) so the global
    provider reuses the same logger the rest of the v2 code emits through (see
    :func:`select_global_otel_v2_logger`). Both ``registered`` and
    ``set_global_provider`` (the proxy passes
    ``opentelemetry.trace.set_tracer_provider``) are injected so the publish step is
    unit-testable without reading or mutating real global OTel state. Returns the
    logger whose provider was published.
    """
    logger = select_global_otel_v2_logger(in_memory_loggers, registered=registered)
    set_global_provider(logger._tracer_provider)
    return logger


def _registered_v2_logger() -> "OpenTelemetryV2 | None":
    try:
        from litellm.proxy import proxy_server
    except Exception:
        return None
    logger = getattr(proxy_server, "open_telemetry_logger", None)
    return logger if isinstance(logger, OpenTelemetryV2) else None


def emit_guardrail_span(entry: "StandardLoggingGuardrailInformation") -> None:
    """Emit a guardrail span on the registered v2 OTel logger.

    Called by the guardrail-recording code the moment a guardrail finishes, so a
    span is produced regardless of whether a post-call hook later runs (it does
    not on the pass-through allow path). Routes through the single canonical
    logger — the same one every other v2 entry point uses — so a guardrail
    recorded once yields exactly one span; fanning out across every reachable
    ``OpenTelemetryV2`` instance double-emits the same entry. Best-effort: span
    emission must never break guardrail evaluation.
    """
    logger = _registered_v2_logger()
    if logger is None:
        return
    try:
        logger.emit_guardrail_span(entry)
    except Exception:
        pass


def seed_request_identity(user_api_key_dict: Any, model: Any = None) -> None:
    logger = _registered_v2_logger()
    if logger is not None:
        logger.seed_request_identity(user_api_key_dict, model=model)


@contextmanager
def phase_span(name: str) -> "Iterator[Span | None]":
    logger = _registered_v2_logger()
    if logger is None:
        yield None
        return
    with logger.start_phase_span(name) as span:
        yield span
