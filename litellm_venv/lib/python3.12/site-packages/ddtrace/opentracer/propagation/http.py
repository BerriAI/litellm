from typing import Dict  # noqa:F401

from opentracing import InvalidCarrierException

from ddtrace.propagation.http import HTTPPropagator as DDHTTPPropagator

from ...internal.logger import get_logger
from ..span_context import SpanContext
from .propagator import Propagator


log = get_logger(__name__)

_HTTP_BAGGAGE_PREFIX = "ot-baggage-"
_HTTP_BAGGAGE_PREFIX_LEN = len(_HTTP_BAGGAGE_PREFIX)


class HTTPPropagator(Propagator):
    """OpenTracing compatible HTTP_HEADER and TEXT_MAP format propagator.

    `HTTPPropagator` provides compatibility by using existing OpenTracing
    compatible methods from the ddtracer along with new logic supporting the
    outstanding OpenTracing-defined functionality.
    """

    @staticmethod
    def inject(span_context, carrier):
        # type: (SpanContext, Dict[str, str]) -> None
        """Inject a span context into a carrier.

        *span_context* is injected into the carrier by first using an
        :class:`ddtrace.propagation.http.HTTPPropagator` to inject the ddtracer
        specific fields.

        Then the baggage is injected into *carrier*.

        :param span_context: span context to inject.

        :param carrier: carrier to inject into.
        """
        if not isinstance(carrier, dict):
            raise InvalidCarrierException("propagator expects carrier to be a dict")

        DDHTTPPropagator.inject(span_context._dd_context, carrier)

        # Add the baggage
        if span_context.baggage is not None:
            for key in span_context.baggage:
                carrier[_HTTP_BAGGAGE_PREFIX + key] = span_context.baggage[key]

    @staticmethod
    def extract(carrier):
        # type: (Dict[str, str]) -> SpanContext
        """Extract a span context from a carrier.

        :class:`ddtrace.propagation.http.HTTPPropagator` is used to extract
        ddtracer supported fields into a `ddtrace.Context` context which is
        combined with new logic to extract the baggage which is returned in an
        OpenTracing compatible span context.

        :param carrier: carrier to extract from.

        :return: extracted span context.
        """
        if not isinstance(carrier, dict):
            raise InvalidCarrierException("propagator expects carrier to be a dict")

        ddspan_ctx = DDHTTPPropagator.extract(carrier)
        baggage = {}
        for key in carrier:
            if key.startswith(_HTTP_BAGGAGE_PREFIX):
                baggage[key[_HTTP_BAGGAGE_PREFIX_LEN:]] = carrier[key]

        return SpanContext(ddcontext=ddspan_ctx, baggage=baggage)
