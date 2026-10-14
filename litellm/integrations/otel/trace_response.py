from typing import Final

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.instrumentation.asgi import (  # pyright: ignore[reportMissingTypeStubs]  # the ASGI instrumentation ships no py.typed marker
    asgi_getter,
)
from opentelemetry.instrumentation.propagators import ResponsePropagator
from opentelemetry.propagators.textmap import CarrierT, Setter, default_setter
from opentelemetry.trace import Span, format_span_id, format_trace_id
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from litellm.integrations.otel.model.semconv import LiteLLM


def _response_header(message: dict[str, object], name: str) -> str | None:  # mutable-ok: asgi_getter takes a dict
    values: Final = asgi_getter.get(message, name)  # pyright: ignore[reportUnknownMemberType]  # the ASGI instrumentation ships no py.typed marker
    return values[0] if values else None


def _http_response_start(message: object) -> dict[str, object] | None:  # mutable-ok: the propagator writes into it
    if isinstance(message, dict) and message.get("type") == "http.response.start":  # pyright: ignore[reportUnknownMemberType]  # an ASGI message is an untyped dict
        return message  # pyright: ignore[reportUnknownVariableType]  # an ASGI message is an untyped dict
    return None


def _link_a_restarted_upstream_trace(span: Span, upstream_traceresponse: str) -> None:
    upstream: Final = trace.get_current_span(
        TraceContextTextMapPropagator().extract({"traceparent": upstream_traceresponse})
    ).get_span_context()
    if upstream.trace_id != span.get_span_context().trace_id:
        span.add_link(upstream)


class UpstreamFirstTraceResponsePropagator(ResponsePropagator):
    def inject(
        self,
        carrier: CarrierT,
        context: Context | None = None,
        setter: Setter[CarrierT] = default_setter,
    ) -> None:
        start: Final = _http_response_start(carrier)
        if start is None:
            return
        span: Final = trace.get_current_span(context)
        call_id: Final = _response_header(start, "x-litellm-call-id")
        if call_id:
            span.set_attribute(LiteLLM.CALL_ID, call_id)
        upstream_traceresponse: Final = _response_header(start, "traceresponse")
        if upstream_traceresponse is not None:
            _link_a_restarted_upstream_trace(span, upstream_traceresponse)
            return
        own: Final = span.get_span_context()
        if not own.trace_flags.sampled:
            return
        sent: Final = start.get("headers")
        if isinstance(sent, tuple):
            start["headers"] = [*sent]
        setter.set(
            carrier,
            "traceresponse",
            f"00-{format_trace_id(own.trace_id)}-{format_span_id(own.span_id)}-{own.trace_flags:02x}",
        )
