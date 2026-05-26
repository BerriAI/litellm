"""The span engine: turn typed span data + the registry into emitted spans.

Driven entirely by ``spans.SPAN_REGISTRY`` (kind/hierarchy) and the mapper chain
(attributes). No untyped ``kwargs`` or god-objects cross its boundary.
"""

from typing import List, Optional, Sequence, Set, Tuple

from opentelemetry.context import Context
from opentelemetry.trace import Span, Tracer
from opentelemetry.trace.status import Status, StatusCode

from litellm.integrations.otel.config import OpenTelemetryV2Config
from litellm.integrations.otel.mappers.base import AttributeMapper, AttrValue
from litellm.integrations.otel.mappers.genai import GenAIMapper
from litellm.integrations.otel.mappers.legacy import LegacyMapper
from litellm.integrations.otel.payloads import (
    GuardrailSpanData,
    LLMCallSpanData,
    ServiceSpanData,
)
from litellm.integrations.otel.providers import to_otel_span_kind
from litellm.integrations.otel.semconv import Error, LiteLLM
from litellm.integrations.otel.spans import (
    SPAN_REGISTRY,
    SpanRole,
    guardrail_span_name,
    llm_call_span_name,
    service_span_name,
)


def default_mappers(config: OpenTelemetryV2Config) -> List[AttributeMapper]:
    """The canonical mapper chain: GenAI always; Legacy during the dual-emit window."""
    mappers: List[AttributeMapper] = [GenAIMapper()]
    if config.legacy_compat:
        mappers.append(LegacyMapper())
    return mappers


class SpanEmitter:
    def __init__(
        self,
        tracer: Tracer,
        config: OpenTelemetryV2Config,
        mappers: Optional[Sequence[AttributeMapper]] = None,
    ) -> None:
        self._tracer = tracer
        self._config = config
        self._mappers: List[AttributeMapper] = (
            list(mappers) if mappers is not None else default_mappers(config)
        )
        self._emitted: Set[Tuple[str, SpanRole]] = set()

    # -- low-level helpers --------------------------------------------------- #

    def _start(
        self,
        role: SpanRole,
        name: str,
        parent_context: Optional[Context] = None,
        start_time_ns: Optional[int] = None,
    ) -> Span:
        return self._tracer.start_span(
            name,
            context=parent_context,
            kind=to_otel_span_kind(SPAN_REGISTRY[role].kind),
            start_time=start_time_ns,
        )

    @staticmethod
    def _set(span: Span, key: str, value: Optional[AttrValue]) -> None:
        if value is None:
            return
        span.set_attribute(key, value)

    def _seen(self, dedup_key: Optional[str], role: SpanRole) -> bool:
        """Idempotency guard for the streaming sync+async dual-fire."""
        if not dedup_key:
            return False
        marker = (dedup_key, role)
        if marker in self._emitted:
            return True
        self._emitted.add(marker)
        return False

    # -- span emitters ------------------------------------------------------- #

    def emit_llm_call(
        self,
        data: LLMCallSpanData,
        parent_context: Optional[Context] = None,
        start_time_ns: Optional[int] = None,
        end_time_ns: Optional[int] = None,
    ) -> Optional[Span]:
        if self._seen(data.identity.call_id, SpanRole.LLM_CALL):
            return None
        span = self._start(
            SpanRole.LLM_CALL,
            llm_call_span_name(data),
            parent_context=parent_context,
            start_time_ns=start_time_ns,
        )
        for mapper in self._mappers:
            for key, value in mapper.map_llm_call(data).items():
                span.set_attribute(key, value)
        if data.error and data.error.error_type:
            self._set(span, Error.TYPE, data.error.error_type)
            span.set_status(
                Status(StatusCode.ERROR, data.error.message or data.error.error_type)
            )
        else:
            span.set_status(Status(StatusCode.OK))
        span.end(end_time=end_time_ns)
        return span

    def emit_guardrail(
        self,
        data: GuardrailSpanData,
        parent_context: Optional[Context] = None,
    ) -> Span:
        span = self._start(
            SpanRole.GUARDRAIL,
            guardrail_span_name(data),
            parent_context=parent_context,
        )
        self._set(span, LiteLLM.GUARDRAIL_NAME, data.guardrail_name)
        self._set(span, LiteLLM.GUARDRAIL_MODE, data.mode)
        self._set(span, LiteLLM.GUARDRAIL_STATUS, data.status)
        span.set_status(Status(StatusCode.OK))
        span.end()
        return span

    def emit_service(
        self,
        data: ServiceSpanData,
        parent_context: Optional[Context] = None,
    ) -> Span:
        span = self._start(
            SpanRole.SERVICE,
            service_span_name(data),
            parent_context=parent_context,
        )
        self._set(span, LiteLLM.SERVICE_NAME, data.service_name)
        self._set(span, LiteLLM.SERVICE_CALL_TYPE, data.call_type)
        if data.error is not None:
            self._set(span, Error.TYPE, data.error.error_type or "error")
            span.set_status(Status(StatusCode.ERROR, data.error.message or "error"))
        else:
            span.set_status(Status(StatusCode.OK))
        span.end()
        return span
