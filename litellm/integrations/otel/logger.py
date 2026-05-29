"""``CustomLogger`` adapter on the OpenTelemetry span engine.

Thin adapter: it translates litellm's logging callbacks into typed ``*SpanData``
and hands them to the engine (:mod:`emitter`), with multi-tenant tracer routing
in :mod:`routing`. It emits the gen-ai spans (LLM call, guardrail, service).

The proxy server span is NOT owned here. It is created by the FastAPI
instrumentation mounted in ``proxy_server``'s startup event, which stamps the
``http.*`` attributes and handles inbound context propagation. The proxy-span
methods below are therefore no-ops: routes never modify spans.

Gen-ai spans parent to that server span via the ambient OTel context rather
than an explicitly threaded span. litellm's async logging worker copies the
request's context at enqueue time, so ``async_log_success_event`` runs with the
server span active. Emission is therefore async-only — the sync callback runs
in an out-of-context thread, where there is no parent span, so it is a no-op.
"""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Mapping, cast

from opentelemetry.context import attach, get_current
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Span, Tracer, get_current_span

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.otel.baggage import promoted_baggage
from litellm.integrations.otel.config import OpenTelemetryV2Config
from litellm.integrations.otel.context import (
    context_from_span,
    is_recordable_span,
    set_request_baggage,
)
from litellm.integrations.otel.emitter import SpanEmitter
from litellm.integrations.otel.mappers import resolve_mappers
from litellm.integrations.otel.payloads import (
    GuardrailSpanData,
    LLMCallSpanData,
    RequestIdentity,
    ServiceSpanData,
    SpanError,
)
from litellm.integrations.otel.providers import build_tracer_provider, get_tracer
from litellm.integrations.otel.routing import TenantTracerCache
from litellm.integrations.otel.spans import SpanRole
from litellm.integrations.otel.utils import to_ns

if TYPE_CHECKING:
    from litellm.types.utils import StandardLoggingGuardrailInformation

LITELLM_TRACER_NAME = "litellm"
LITELLM_PROXY_REQUEST_SPAN_NAME = "Received Proxy Server Request"

# Any callback whose class belongs to one of these modules is "the OTel
# callback" for proxy-global-registration purposes.
_OTEL_MODULES = (
    "litellm.integrations.otel",
    "litellm.integrations.opentelemetry",
)


def _threaded_parent_span(kwargs: Mapping[str, Any]) -> Span | None:
    """The proxy SERVER span threaded through request metadata, if any.

    Normally the LLM-call span parents to the ambient OTel context (the active
    server span). But pass-through logging runs in a detached
    ``asyncio.create_task`` whose copied context may no longer carry that span,
    so the proxy also threads it explicitly as ``litellm_parent_otel_span`` (see
    ``litellm_pre_call_utils`` for proxy routes and the pass-through endpoint for
    catch-all routes). This reads it back so the call span can fall back to it.
    """
    litellm_params = kwargs.get("litellm_params")
    candidates: list[Any] = []
    if isinstance(litellm_params, Mapping):
        candidates.append(litellm_params.get("metadata"))
        candidates.append(litellm_params.get("litellm_metadata"))
    candidates.append(kwargs.get("metadata"))
    for meta in candidates:
        if isinstance(meta, Mapping):
            span = meta.get("litellm_parent_otel_span")
            if span is not None:
                return cast("Span", span)
    return None


def _pre_call_guardrail_blocked(payload: Mapping[str, Any]) -> bool:
    """True when a pre-call guardrail blocked the request (no LLM call happened).

    A blocked pre-call guardrail raises before the upstream call, yet litellm
    still emits a failure log — which would otherwise produce a phantom CLIENT
    span for a call that never occurred. We detect the case (request failed AND a
    ``pre_call`` guardrail intervened) so the caller can skip that span. A
    pre-call guardrail that merely *masks* lets the call proceed, so the request
    succeeds and this returns False — only genuine blocks fail the request.
    """
    if payload.get("status") != "failure":
        return False
    info = payload.get("guardrail_information")
    if not isinstance(info, list):
        return False
    for entry in info:
        if not isinstance(entry, dict):
            continue
        mode = entry.get("guardrail_mode")
        is_pre_call = mode == "pre_call" or (
            isinstance(mode, (list, tuple)) and "pre_call" in mode
        )
        if is_pre_call and entry.get("guardrail_status") == "guardrail_intervened":
            return True
    return False


class OpenTelemetryV2(CustomLogger):
    """The ``CustomLogger`` for OpenTelemetry.

    The constructor accepts an optional config, callback name, and pre-built
    OTel providers; when a provider is omitted it is built from the config.
    ``logger_provider`` and ``meter_provider`` are accepted but reserved for
    future OTel logs and metrics support.
    """

    def __init__(
        self,
        config: OpenTelemetryV2Config | None = None,
        callback_name: str | None = None,
        tracer_provider: TracerProvider | None = None,
        logger_provider: Any | None = None,  # reserved for OTel logs
        meter_provider: Any | None = None,  # reserved for metrics
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # Build the config from any settings passed through ``kwargs`` so
        # ``callback_settings.otel.*`` in config.yaml (e.g. ``baggage_promoted_keys``,
        # ``capture_message_content``) configures the logger. ``OpenTelemetryV2Config``
        # ignores extra keys, so unrelated kwargs are dropped harmlessly.
        self.config: OpenTelemetryV2Config = config or OpenTelemetryV2Config(**kwargs)
        self.callback_name = callback_name
        self._tracer_provider: TracerProvider = (
            tracer_provider
            if tracer_provider is not None
            else build_tracer_provider(self.config)
        )
        self.tracer: Tracer = get_tracer(self._tracer_provider, LITELLM_TRACER_NAME)
        self._emitter = SpanEmitter(
            self.tracer, self.config, mappers=resolve_mappers(self.config.mapper_names)
        )
        self._tenant_tracers = TenantTracerCache(
            self.config, callback_name, LITELLM_TRACER_NAME
        )
        self._init_otel_logger_on_litellm_proxy()

    # ====================================================================== #
    #  Proxy global registration
    # ====================================================================== #

    def _init_otel_logger_on_litellm_proxy(self) -> None:
        """Claim ``proxy_server.open_telemetry_logger`` if no one else has."""
        try:
            from litellm.proxy import proxy_server
        except Exception:
            return
        try:
            # Mutate ``litellm.service_callback`` in place. ``getattr(..) or []``
            # would bind a throwaway local when the list is empty (an empty list
            # is falsy), so the append would never reach the global and service
            # spans (Redis, Postgres, …) would be silently dropped from traces.
            service_callback = litellm.service_callback
            already_otel = any(
                cb.__class__.__module__.startswith(_OTEL_MODULES)
                for cb in service_callback
                if hasattr(cb, "__class__")
            )
            if not already_otel:
                service_callback.append(self)
        except Exception:
            pass
        if getattr(proxy_server, "open_telemetry_logger", None) is None:
            setattr(proxy_server, "open_telemetry_logger", self)

    # ====================================================================== #
    #  LLM-call callbacks
    # ====================================================================== #

    # Async-only: the async path runs inside the request's restored OTel context
    # (the logging worker copies it at enqueue), so the span parents to the
    # instrumentor's server span via ambient context. The sync path runs in an
    # out-of-context thread with no parent span, so it is a no-op.

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        return None

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        return None

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._emit_llm_call(kwargs, start_time, end_time)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._emit_llm_call(kwargs, start_time, end_time)

    def _emit_llm_call(
        self,
        kwargs: Mapping[str, Any],
        start_time: datetime | float | None,
        end_time: datetime | float | None,
    ) -> Span | None:
        payload = kwargs.get("standard_logging_object")
        if not payload:
            return None
        if _pre_call_guardrail_blocked(cast("Mapping[str, Any]", payload)):
            # A pre-call guardrail blocked the request, so the upstream LLM was
            # never called — litellm still emits a failure log, but a CLIENT
            # "chat …" span for a call that didn't happen is misleading. Skip it;
            # the guardrail span (ERROR, with the verdict) is the real outcome.
            return None
        data = LLMCallSpanData.from_standard_logging_payload(
            cast("Any", payload), capture_content=self.config.capture_span_content
        )
        # Parent is the ambient context (the instrumentor's server span,
        # restored by the logging worker). When the ambient context has lost the
        # server span — e.g. pass-through logging fired from a detached task —
        # fall back to the server span the proxy threaded through metadata so the
        # call span still nests under the request instead of being dropped.
        parent_ctx = get_current()
        if not is_recordable_span(get_current_span(parent_ctx)):
            threaded_parent = _threaded_parent_span(kwargs)
            if is_recordable_span(threaded_parent):
                parent_ctx = context_from_span(
                    cast("Span", threaded_parent), context=parent_ctx
                )
        # Write identity into Baggage so child spans (guardrails, services)
        # inherit it.
        bag = promoted_baggage(
            data.identity,
            data.request_model,
            promoted_keys=tuple(self.config.baggage_promoted_keys),
            metadata_keys=tuple(self.config.baggage_metadata_keys),
        )
        if bag:
            parent_ctx = set_request_baggage(bag, context=parent_ctx)
        return self._emitter.emit(
            SpanRole.LLM_CALL,
            data,
            parent_context=parent_ctx,
            start_time_ns=to_ns(start_time),
            end_time_ns=to_ns(end_time),
            tracer=self._tenant_tracers.tracer_for(
                self.tracer, kwargs.get("standard_callback_dynamic_params")
            ),
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
        if not is_recordable_span(parent_otel_span):
            return None
        data = ServiceSpanData.from_payload(payload, event_metadata=event_metadata)
        if error_override is not None and data.error is None:
            data = ServiceSpanData(
                service_name=data.service_name,
                call_type=data.call_type,
                error=SpanError(message=error_override),
                event_metadata=data.event_metadata,
            )
        # Parent to the server span, but layer it over the ambient context so the
        # identity Baggage seeded in ``async_pre_call_hook`` rides along and the
        # service span gets the same identity attributes as the LLM-call span.
        parent_context = context_from_span(
            cast(Span, parent_otel_span), context=get_current()
        )
        return self._emitter.emit(
            SpanRole.SERVICE,
            data,
            parent_context=parent_context,
            start_time_ns=to_ns(start_time),
            end_time_ns=to_ns(end_time),
        )

    # ====================================================================== #
    #  async_post_call_* hooks — emit guardrail spans. The server span's status
    #  / errors are the FastAPI instrumentor's job, so we don't touch it here.
    # ====================================================================== #

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict,
        call_type: Any,
    ) -> dict:
        """Seed request identity into Baggage at the start of the request.

        This runs in the request task (the server span is the ambient context),
        so attaching the identity Baggage here makes **every** span emitted for
        the request — LLM call, guardrail, and service — inherit it via
        ``LiteLLMBaggageSpanProcessor``. Without this, only the LLM-call span got
        identity (it promoted Baggage locally) and the guardrail/service spans,
        which parent to the server span, had none. The async logging worker
        copies this context at enqueue time, so the LLM-call span inherits it too.
        """
        try:
            identity = RequestIdentity.from_user_api_key_auth(user_api_key_dict)
            bag = promoted_baggage(
                identity,
                data.get("model") if isinstance(data, dict) else None,
                promoted_keys=tuple(self.config.baggage_promoted_keys),
                metadata_keys=tuple(self.config.baggage_metadata_keys),
            )
            if bag:
                # Attach (no detach): the contextvar is scoped to this request's
                # asyncio task and is reclaimed when the task ends.
                attach(set_request_baggage(bag, context=get_current()))
                # The server span was started by the instrumentor before this
                # hook ran, so the Baggage processor (which only fires at span
                # start) won't backfill it — stamp identity on it directly.
                server_span = get_current_span()
                if is_recordable_span(server_span):
                    for key, value in bag.items():
                        server_span.set_attribute(key, value)
        except Exception:
            pass
        return data

    async def async_post_call_success_hook(
        self,
        data: Mapping[str, Any],
        user_api_key_dict: Any,
        response: Any,
    ) -> Any:
        self._emit_guardrail_spans(data)
        return response

    async def async_post_call_failure_hook(
        self,
        request_data: Mapping[str, Any],
        original_exception: BaseException | None,
        user_api_key_dict: Any,
        traceback_str: str | None = None,
    ) -> None:
        self._emit_guardrail_spans(request_data)

    def _emit_guardrail_spans(self, request_data: Mapping[str, Any]) -> None:
        # Post-call hooks run in the request task, so the ambient context is the
        # server span; guardrail spans parent to it implicitly via that context.
        metadata = request_data.get("metadata")
        guardrails: list[Any] = []
        if isinstance(metadata, dict):
            info = metadata.get("standard_logging_guardrail_information")
            if isinstance(info, list):
                guardrails = info
            elif isinstance(info, dict):
                guardrails = [info]
        for entry in guardrails:
            if not isinstance(entry, dict):
                continue
            self._emitter.emit(
                SpanRole.GUARDRAIL,
                GuardrailSpanData.from_logging_entry(
                    cast("StandardLoggingGuardrailInformation", entry)
                ),
            )

    # ====================================================================== #
    #  Management endpoint hooks — no-ops. Management endpoints are ordinary
    #  FastAPI routes, so the mounted instrumentor already spans them.
    # ====================================================================== #

    async def async_management_endpoint_success_hook(
        self,
        logging_payload: Any,
        parent_otel_span: Span | None = None,
    ) -> None:
        return None

    async def async_management_endpoint_failure_hook(
        self,
        logging_payload: Any,
        parent_otel_span: Span | None = None,
    ) -> None:
        return None

    # ====================================================================== #
    #  Proxy SERVER-span API — no-ops. The FastAPI instrumentor owns the server
    #  span (creation, http.* attributes, inbound propagation) and gen-ai spans
    #  parent to it via ambient context. These methods are the surface the
    #  proxy and auth call sites invoke; they intentionally do nothing.
    # ====================================================================== #

    def create_litellm_proxy_request_started_span(
        self, start_time: datetime, headers: Mapping[str, str] | None
    ) -> Span | None:
        """Return the active server span instead of creating one.

        The FastAPI instrumentor owns the server span, so V2 creates nothing
        here. But the proxy threads this return value as ``litellm_parent_otel_span``
        — and service logging (Redis, Postgres, …) only invokes the OTel service
        hook when that parent is non-None. Returning the ambient server span lets
        service spans nest under it. The proxy must NOT ``.end()`` this span (the
        instrumentor does); ``_close_dangling_otel_server_span`` skips it under V2.
        """
        span = get_current_span()
        return span if is_recordable_span(span) else None

    @staticmethod
    def set_proxy_request_route_attributes(
        span: Span | None,
        *,
        url_path: str | None = None,
        http_route: str | None = None,
    ) -> None:
        """No-op: the FastAPI instrumentor stamps ``http.route`` / ``url.path``."""

    @staticmethod
    def set_response_status_code_attribute(
        span: Span | None, status_code: int | None
    ) -> None:
        """No-op: the FastAPI instrumentor stamps ``http.response.status_code``."""

    @staticmethod
    def set_preprocessing_duration_attribute(span: Span | None, container: Any) -> None:
        """No-op: the server span belongs to the FastAPI instrumentor."""
