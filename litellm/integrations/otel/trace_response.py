from typing import Final

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.instrumentation.asgi import (  # pyright: ignore[reportMissingTypeStubs]  # the ASGI instrumentation ships no py.typed marker
    asgi_getter,
)
from opentelemetry.instrumentation.propagators import TraceResponsePropagator
from opentelemetry.propagators.textmap import CarrierT, Setter, default_setter

from litellm.integrations.otel.model.semconv import LiteLLM


def _response_header(message: object, name: str) -> str | None:
    if not isinstance(message, dict):
        return None
    values: Final = asgi_getter.get(message, name)  # pyright: ignore[reportUnknownMemberType]  # the ASGI instrumentation ships no py.typed marker
    return values[0] if values else None


class UpstreamFirstTraceResponsePropagator(TraceResponsePropagator):
    def inject(
        self,
        carrier: CarrierT,
        context: Context | None = None,
        setter: Setter[CarrierT] = default_setter,
    ) -> None:
        call_id: Final = _response_header(carrier, "x-litellm-call-id")
        if call_id:
            trace.get_current_span(context).set_attribute(LiteLLM.CALL_ID, call_id)
        if _response_header(carrier, "traceresponse") is None:
            super().inject(carrier, context, setter)  # pyright: ignore[reportUnknownMemberType]  # upstream leaves its Setter parameter unparameterized
