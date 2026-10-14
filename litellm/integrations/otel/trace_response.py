from typing import Final

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.instrumentation.asgi import (  # pyright: ignore[reportMissingTypeStubs]  # the ASGI instrumentation ships no py.typed marker
    asgi_getter,
)
from opentelemetry.instrumentation.propagators import TraceResponsePropagator
from opentelemetry.propagators.textmap import CarrierT, Setter, default_setter

from litellm.integrations.otel.model.semconv import LiteLLM


def _response_header(message: dict[str, object], name: str) -> str | None:  # mutable-ok: asgi_getter takes a dict
    values: Final = asgi_getter.get(message, name)  # pyright: ignore[reportUnknownMemberType]  # the ASGI instrumentation ships no py.typed marker
    return values[0] if values else None


def _http_response_start(message: object) -> dict[str, object] | None:  # mutable-ok: the propagator writes into it
    if isinstance(message, dict) and message.get("type") == "http.response.start":  # pyright: ignore[reportUnknownMemberType]  # an ASGI message is an untyped dict
        return message  # pyright: ignore[reportUnknownVariableType]  # an ASGI message is an untyped dict
    return None


class UpstreamFirstTraceResponsePropagator(TraceResponsePropagator):
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
        if not span.get_span_context().trace_flags.sampled or _response_header(start, "traceresponse") is not None:
            return
        sent: Final = start.get("headers")
        if isinstance(sent, tuple):
            start["headers"] = [*sent]
        super().inject(carrier, context, setter)  # pyright: ignore[reportUnknownMemberType]  # upstream leaves its Setter parameter unparameterized
