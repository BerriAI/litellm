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
from litellm.integrations.otel.emitter import SpanEmitter, stamp_error
from litellm.integrations.otel.mappers import resolve_mappers
from litellm.integrations.otel.model.baggage import promoted_baggage
from litellm.integrations.otel.model.config import OpenTelemetryV2Config
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
from litellm.integrations.otel.model.spans import SpanRole, span_role_for_service
from litellm.integrations.otel.model.utils import to_ns
from litellm.integrations.otel.plumbing.context import (
    is_recordable_span,
    mcp_message_transport_span,
    request_root_span,
    resolve_mcp_span_context,
    resolve_parent_context,
    resolve_request_span_context,
    set_request_baggage,
    set_request_root_span,
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

if TYPE_CHECKING:
    from litellm.integrations.otel.model.destination import OtelDestination
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.utils import (
        StandardLoggingGuardrailInformation,
        StandardLoggingPayload,
    )

LITELLM_TRACER_NAME = "litellm"


def _span_error_from_exception(
    exception: "Exception | None",
    *,
    status_code: int | None = None,
    traceback_str: str | None = None,
) -> SpanError:
    """A ``SpanError`` for a proxy-level failure that never produced a ``StandardLoggingPayload``
    (auth/validation/malformed-body rejections). ``status_code`` pins ``error.code`` to the real
    response status."""
    from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup

    info = StandardLoggingPayloadSetup.get_error_information(
        original_exception=exception,
        traceback_str=traceback_str,
    )
    return SpanError(
        error_type=info.get("error_class") or info.get("error_code") or None,
        message=info.get("error_message") or None,
        code=str(status_code) if status_code is not None else (info.get("error_code") or None),
        stack_trace=info.get("traceback") or None,
        llm_provider=info.get("llm_provider") or None,
    )


# Any callback whose class belongs to one of these modules is "the OTel
# callback" for proxy-global-registration purposes.
_OTEL_MODULES = (
    "litellm.integrations.otel",
    "litellm.integrations.opentelemetry",
)


# Cap on the open-call carrier map: a span opened at ``pre_call`` that never reaches a close
# callback (a stream that only fires stream events) would otherwise linger. Evicts the oldest.
_OPEN_CALLS_MAX = 10_000


class _LLMCallSpan:
    """The state carried from the ``pre_call`` boundary to span close.

    ``spans`` is one live span per destination Resource group, opened at the boundary; empty when
    creation was deferred (no ambient parent visible), where the async callback creates them from
    ``start_time_ns``. A carrier existing at all proves ``pre_call`` ran (an upstream call was attempted).
    """

    __slots__ = ("spans", "start_time_ns")

    def __init__(self, spans: "tuple[Span, ...]", start_time_ns: int | None) -> None:
        self.spans = spans
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
            tracer_provider
            if tracer_provider is not None
            else build_tracer_provider(
                self.config,
                tenant_fan_out_owner=callback_name,
                attach_tenant_fan_out=True,
            )
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
        self._open_llm_calls: OrderedDict[str, _LLMCallSpan] = OrderedDict()
        self._closed_call_ids: OrderedDict[str, None] = OrderedDict()  # mutable-ok: bounded LRU of emitted call ids
        self._init_otel_logger_on_litellm_proxy()

    def _init_metrics(self, meter_provider: Any | None) -> "GenAIMetricRecorder | None":
        """Create the six GenAI histograms when metrics are enabled, else ``None``.

        ``meter_provider`` is an explicit override (tests inject one); otherwise it is resolved
        from the OTel global so the operator's readers/exporters receive the metrics.
        """
        if not self.config.enable_metrics:
            return None
        provider = resolve_meter_provider(self.config, meter_provider)
        meter = get_meter(provider, LITELLM_TRACER_NAME)
        return GenAIMetricRecorder(create_genai_metrics(meter), self.callback_name)

    def _init_events(self, logger_provider: LoggerProvider | None) -> "GenAIEventRecorder | None":
        """Create the GenAI event recorder when events are enabled, else ``None``.

        ``logger_provider`` is an explicit override (tests inject one); otherwise it is resolved
        from the OTel global. A ``None`` resolution means the operator opted out of logs, so no
        recorder is built.
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

    def _destinations_for_backend(self, call: "LLMCallEvent") -> tuple:
        """The call's admin-resolved destinations tagged with THIS logger's callback_name,
        so each backend's span keeps its own attribute vocabulary."""
        return tuple(d for d in call.otel_destinations if d.callback_name == self.callback_name)

    # ====================================================================== #
    #  LLM-call callbacks — the span is opened at the ``pre_call`` boundary and
    #  closed here. See ``log_pre_api_call``.
    # ====================================================================== #

    def log_pre_api_call(self, model, messages, kwargs):
        """Open the LLM-call span at the call boundary.

        Runs synchronously in the request task, where the live server span is the ambient OTel
        context, so the span parents to it natively; it is stashed and closed in the async
        callback. When no recordable parent is visible (a thread-pool sync-only provider, where
        the anchor doesn't follow), creation is deferred to the close callback. Synthetic
        proxy-gate logs (auth/rate-limit rejections) made no upstream call and are skipped.
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
        spans: tuple[Span, ...] = ()
        # Parent to the anchored root span (ambient on the SDK path); open live only when that
        # resolves to a recordable parent, else defer to the close callback (the thread-pool case).
        parent_context = resolve_request_span_context()
        if is_recordable_span(get_current_span(parent_context)):
            spans = tuple(
                self._emitter.start_span(
                    SpanRole.LLM_CALL,
                    call.provisional_span_name,
                    parent_context=parent_context,
                    start_time_ns=start_time_ns,
                    tracer=tracer,
                )
                for tracer in self._tenant_tracers.tracers_for(self.tracer, self._destinations_for_backend(call))
            )
        self._open_llm_calls[call_id] = _LLMCallSpan(spans=spans, start_time_ns=start_time_ns)
        # Evict the oldest open call if over budget; a call that opens but never closes would
        # linger otherwise (the evicted span is dropped, never exported).
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
        """Seed authenticated request-identity Baggage onto ``context`` so the Baggage processor
        stamps team/key/metadata onto the span. Read from the parsed payload, never the client's
        ``params._meta`` carrier, so it can't be spoofed."""
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

        MCP tool calls reach the callbacks with no ``pre_call`` carrier, so they get their own
        CLIENT span here, emitted at once and deduped on the call id. Per the MCP semconv it
        parents to the ``params._meta`` trace context (or a new root) and links the transport
        span. Returns whether it handled the event, so the caller skips the LLM-call path.
        """
        raw_payload = kwargs.get("standard_logging_object")
        if not raw_payload or not is_mcp_tool_call(cast(Mapping[str, object], raw_payload)):
            return False
        payload = cast("StandardLoggingPayload", raw_payload)
        data = MCPToolCallSpanData.from_standard_logging_payload(
            payload, capture_content=self.config.capture_span_content
        )
        # Drop any stray LLM carrier for this id so it's neither leaked nor closed as a phantom span.
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

        Like a tool call, listing has no ``pre_call`` carrier, so it gets its own CLIENT span.
        Per the MCP semconv it parents to the ``params._meta`` trace context (or a new root) and
        links the transport span. Returns whether it handled the event so the caller skips the
        LLM-call path.
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

        A missing carrier means either ``pre_call`` never ran (rejected at the gate or by a
        pre-call guardrail, no payload, so dropping is correct) or this v2 instance was lazily
        activated after ``pre_call`` (the destination-resolver path), where the payload IS set and
        names this backend, so a deferred span is emitted with the success event's start time.
        """
        call = LLMCallEvent.from_dict(kwargs)
        call_id = call.call_id

        if call_id and call_id in self._closed_call_ids:
            return None

        carrier = self._open_llm_calls.pop(call_id, None) if call_id else None
        payload = call.payload

        if carrier is None:
            destinations = self._destinations_for_backend(call)
            if call.is_no_upstream_call or payload is None or not destinations:
                return None
            self._mark_closed(call_id)
            return self._emit_deferred_llm_call(
                payload,
                destinations,
                to_ns(start_time),
                to_ns(end_time),
                call.time_to_first_chunk_seconds,
            )

        end_time_ns = to_ns(end_time)
        self._mark_closed(call_id)
        if payload is None:
            for span in carrier.spans:
                span.end(end_time=end_time_ns)
            return None

        data = LLMCallSpanData.from_standard_logging_payload(
            payload,
            capture_content=self.config.capture_span_content,
            time_to_first_chunk_seconds=call.time_to_first_chunk_seconds,
        )
        if carrier.spans:
            for span in carrier.spans:
                self._emitter.finish_span(SpanRole.LLM_CALL, span, data, end_time_ns=end_time_ns)
            return carrier.spans[0]
        return self._emit_deferred_llm_call(
            payload,
            self._destinations_for_backend(call),
            carrier.start_time_ns,
            end_time_ns,
            call.time_to_first_chunk_seconds,
        )

    def _mark_closed(self, call_id: str | None) -> None:
        """Remember a call_id has been closed so a duplicate callback no-ops (bounded FIFO)."""
        if not call_id:
            return
        self._closed_call_ids[call_id] = None
        if len(self._closed_call_ids) > _OPEN_CALLS_MAX:
            self._closed_call_ids.popitem(last=False)

    def _emit_deferred_llm_call(
        self,
        payload: "StandardLoggingPayload",
        destinations: "tuple[OtelDestination, ...]",
        start_time_ns: int | None,
        end_time_ns: int | None,
        time_to_first_chunk_seconds: float | None = None,
    ) -> Span | None:
        """Emit an LLM-call span outside the ``pre_call`` boundary.

        Two callers: the SDK thread-pool path and the destination-resolver path. Both anchor to
        the request's root span via the worker-copied context and seed identity Baggage.
        """
        data = LLMCallSpanData.from_standard_logging_payload(
            payload,
            capture_content=self.config.capture_span_content,
            time_to_first_chunk_seconds=time_to_first_chunk_seconds,
        )
        parent_ctx = self._seed_identity_baggage(data.identity, data.request_model, resolve_request_span_context())
        return self._emitter.emit_fanout(
            SpanRole.LLM_CALL,
            data,
            parent_context=parent_ctx,
            start_time_ns=start_time_ns,
            end_time_ns=end_time_ns,
            tracers=self._tenant_tracers.tracers_for(self.tracer, destinations),
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
        # ``None`` role means metrics-only (framework instrumentation that duplicates a gen-AI
        # span, or ``auth`` which gets a live phase span instead); it never enters the trace.
        role = span_role_for_service(data.service_name)
        if role is None:
            return None
        # Skip a ping with neither timing nor a parent (in-memory queue gauges): a span would be a
        # zero-duration root. Real background work passes start/end times; anything with a parent emits.
        if error_override is None and start_time is None and end_time is None and parent_otel_span is None:
            return None
        if error_override is not None and data.error is None:
            data = ServiceSpanData(
                service_name=data.service_name,
                call_type=data.call_type,
                error=SpanError(message=error_override),
                event_metadata=data.event_metadata,
            )
        # Parent ambient-first (so the call nests under the active request phase, e.g. a DB lookup
        # under ``auth``), falling back to the threaded ``parent_otel_span``; a background call has
        # neither and starts its own root trace.
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

        Called once at the auth boundary so every span emitted afterwards inherits identity via
        ``LiteLLMBaggageSpanProcessor``. Auth-internal DB lookups before the key resolves stay
        unlabeled, which is correct.
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
                # The instrumentor started the server span before this ran, so the Baggage
                # processor (fires only at span start) won't backfill it; stamp identity directly.
                # Prefer the anchored root over ambient so identity lands on the server span even
                # when seeding from inside the live ``auth`` phase span.
                server_span = request_root_span() or get_current_span()
                if is_recordable_span(server_span):
                    # Re-capture the anchor here too, covering entrypoints that bypass
                    # ``create_litellm_proxy_request_started_span`` (e.g. the SDK path). Idempotent.
                    set_request_root_span(server_span)
                    for key, value in bag.items():
                        server_span.set_attribute(key, value)
        except Exception:
            pass

    @contextmanager
    def start_phase_span(self, name: str) -> "Iterator[Span]":
        span = self._emitter.start_span(SpanRole.SERVICE, name)
        with use_span(span, end_on_exit=True):
            try:
                yield span
            except Exception as exc:
                if is_recordable_span(span):
                    stamp_error(span, _span_error_from_exception(exc), record_event=False, set_status=False)
                raise

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

    def record_error_attributes_on_span(
        self,
        span: "Span | None",
        exception: "Exception | None",
        status_code: int,
    ) -> None:
        """Stamp v2 error.* attributes on the FastAPI-owned SERVER span for a failure that dies
        before any LLM-call span exists (malformed body, auth/validation rejection).

        Only decorates the span (never sets status, ends it, or emits an exception event); the
        instrumentor still owns the span's lifecycle, matching v1's SERVER-span behavior.
        """
        if span is None or not is_recordable_span(span):
            return
        stamp_error(
            span,
            _span_error_from_exception(exception, status_code=status_code),
            record_event=False,
            set_status=False,
        )

    async def async_post_call_failure_hook(
        self,
        request_data: dict,
        original_exception: Exception,
        user_api_key_dict: "UserAPIKeyAuth",
        traceback_str: "str | None" = None,
    ) -> None:
        """Stamp error.* on the request's root SERVER span for a proxy-level failure that never
        reached an LLM call (empty body rejected, auth failure), so it carries the same error keys
        a failed LLM call does.

        For an MCP message the session-task anchor is whatever request opened the session, so
        prefer the transport the gateway published for this specific message.
        """
        span = mcp_message_transport_span() or request_root_span() or user_api_key_dict.parent_otel_span
        if span is None or not is_recordable_span(span):
            return None
        stamp_error(span, _span_error_from_exception(original_exception, traceback_str=traceback_str))
        return None

    def emit_guardrail_span(self, entry: "StandardLoggingGuardrailInformation") -> None:
        # A guardrail is a sibling of the LLM call under the request's root span, so parent it to
        # the explicit anchor, never the active span (which during a pre_call guardrail can be the
        # live ``auth`` phase span). Emit with the guardrail's actual execution window so a
        # pre_call guardrail is placed before the LLM call rather than at emission time.
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

    Prefer the canonical ``registered`` owner every other v2 entry point routes through, so the
    server span and gen-ai spans share one provider and one trace. Fall back to a v2 logger in
    ``in_memory_loggers`` (the SDK path), then build a generic one from ``OTEL_*`` only when none
    was configured; each fallback avoids the second generic logger that orphaned the gen-ai spans.
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

    Called once at startup; ``registered`` (the canonical owner) makes the global reuse the
    logger the rest of the v2 code emits through, and both it and ``set_global_provider`` are
    injected so the publish step is unit-testable without touching real global OTel state.
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

    Called when a guardrail finishes, so a span is produced even when no post-call hook runs (the
    pass-through allow path). Routes through the single canonical logger so a guardrail yields
    exactly one span. Best-effort: emission must never break guardrail evaluation.
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
