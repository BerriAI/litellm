from datetime import datetime, timezone
from time import time
from typing import TYPE_CHECKING, cast

from opentelemetry.context import get_value
from opentelemetry.sdk.trace import SpanProcessor, ReadableSpan as OTelSpan
from opentelemetry.semconv.trace import SpanAttributes
from opentelemetry.trace import (
    format_span_id,
    format_trace_id,
    get_current_span,
    SpanKind,
)
from opentelemetry.trace.span import (
    INVALID_SPAN_ID,
    INVALID_TRACE_ID,
)
from sentry_sdk import get_client, start_transaction
from sentry_sdk.consts import INSTRUMENTER, SPANSTATUS
from sentry_sdk.integrations.opentelemetry.consts import (
    SENTRY_BAGGAGE_KEY,
    SENTRY_TRACE_KEY,
)
from sentry_sdk.scope import add_global_event_processor
from sentry_sdk.tracing import Transaction, Span as SentrySpan
from sentry_sdk.utils import Dsn

from urllib3.util import parse_url as urlparse

if TYPE_CHECKING:
    from typing import Any, Optional, Union
    from opentelemetry import context as context_api
    from sentry_sdk._types import Event, Hint

OPEN_TELEMETRY_CONTEXT = "otel"
SPAN_MAX_TIME_OPEN_MINUTES = 10
SPAN_ORIGIN = "auto.otel"


def link_trace_context_to_error_event(event, otel_span_map):
    # type: (Event, dict[str, Union[Transaction, SentrySpan]]) -> Event
    client = get_client()

    if client.options["instrumenter"] != INSTRUMENTER.OTEL:
        return event

    if hasattr(event, "type") and event["type"] == "transaction":
        return event

    otel_span = get_current_span()
    if not otel_span:
        return event

    ctx = otel_span.get_span_context()

    if ctx.trace_id == INVALID_TRACE_ID or ctx.span_id == INVALID_SPAN_ID:
        return event

    sentry_span = otel_span_map.get(format_span_id(ctx.span_id), None)
    if not sentry_span:
        return event

    contexts = event.setdefault("contexts", {})
    contexts.setdefault("trace", {}).update(sentry_span.get_trace_context())

    return event


class SentrySpanProcessor(SpanProcessor):
    """
    Converts OTel spans into Sentry spans so they can be sent to the Sentry backend.
    """

    # The mapping from otel span ids to sentry spans
    otel_span_map = {}  # type: dict[str, Union[Transaction, SentrySpan]]

    # The currently open spans. Elements will be discarded after SPAN_MAX_TIME_OPEN_MINUTES
    open_spans = {}  # type: dict[int, set[str]]

    def __new__(cls):
        # type: () -> SentrySpanProcessor
        if not hasattr(cls, "instance"):
            cls.instance = super().__new__(cls)

        return cls.instance

    def __init__(self):
        # type: () -> None
        @add_global_event_processor
        def global_event_processor(event, hint):
            # type: (Event, Hint) -> Event
            return link_trace_context_to_error_event(event, self.otel_span_map)

    def _prune_old_spans(self):
        # type: (SentrySpanProcessor) -> None
        """
        Prune spans that have been open for too long.
        """
        current_time_minutes = int(time() / 60)
        for span_start_minutes in list(
            self.open_spans.keys()
        ):  # making a list because we change the dict
            # prune empty open spans buckets
            if self.open_spans[span_start_minutes] == set():
                self.open_spans.pop(span_start_minutes)

            # prune old buckets
            elif current_time_minutes - span_start_minutes > SPAN_MAX_TIME_OPEN_MINUTES:
                for span_id in self.open_spans.pop(span_start_minutes):
                    self.otel_span_map.pop(span_id, None)

    def on_start(self, otel_span, parent_context=None):
        # type: (OTelSpan, Optional[context_api.Context]) -> None
        client = get_client()

        if not client.dsn:
            return

        try:
            _ = Dsn(client.dsn)
        except Exception:
            return

        if client.options["instrumenter"] != INSTRUMENTER.OTEL:
            return

        if not otel_span.get_span_context().is_valid:
            return

        if self._is_sentry_span(otel_span):
            return

        trace_data = self._get_trace_data(otel_span, parent_context)

        parent_span_id = trace_data["parent_span_id"]
        sentry_parent_span = (
            self.otel_span_map.get(parent_span_id) if parent_span_id else None
        )

        start_timestamp = None
        if otel_span.start_time is not None:
            start_timestamp = datetime.fromtimestamp(
                otel_span.start_time / 1e9, timezone.utc
            )  # OTel spans have nanosecond precision

        sentry_span = None
        if sentry_parent_span:
            sentry_span = sentry_parent_span.start_child(
                span_id=trace_data["span_id"],
                name=otel_span.name,
                start_timestamp=start_timestamp,
                instrumenter=INSTRUMENTER.OTEL,
                origin=SPAN_ORIGIN,
            )
        else:
            sentry_span = start_transaction(
                name=otel_span.name,
                span_id=trace_data["span_id"],
                parent_span_id=parent_span_id,
                trace_id=trace_data["trace_id"],
                baggage=trace_data["baggage"],
                start_timestamp=start_timestamp,
                instrumenter=INSTRUMENTER.OTEL,
                origin=SPAN_ORIGIN,
            )

        self.otel_span_map[trace_data["span_id"]] = sentry_span

        if otel_span.start_time is not None:
            span_start_in_minutes = int(
                otel_span.start_time / 1e9 / 60
            )  # OTel spans have nanosecond precision
            self.open_spans.setdefault(span_start_in_minutes, set()).add(
                trace_data["span_id"]
            )

        self._prune_old_spans()

    def on_end(self, otel_span):
        # type: (OTelSpan) -> None
        client = get_client()

        if client.options["instrumenter"] != INSTRUMENTER.OTEL:
            return

        span_context = otel_span.get_span_context()
        if not span_context.is_valid:
            return

        span_id = format_span_id(span_context.span_id)
        sentry_span = self.otel_span_map.pop(span_id, None)
        if not sentry_span:
            return

        sentry_span.op = otel_span.name

        self._update_span_with_otel_status(sentry_span, otel_span)

        if isinstance(sentry_span, Transaction):
            sentry_span.name = otel_span.name
            sentry_span.set_context(
                OPEN_TELEMETRY_CONTEXT, self._get_otel_context(otel_span)
            )
            self._update_transaction_with_otel_data(sentry_span, otel_span)

        else:
            self._update_span_with_otel_data(sentry_span, otel_span)

        end_timestamp = None
        if otel_span.end_time is not None:
            end_timestamp = datetime.fromtimestamp(
                otel_span.end_time / 1e9, timezone.utc
            )  # OTel spans have nanosecond precision

        sentry_span.finish(end_timestamp=end_timestamp)

        if otel_span.start_time is not None:
            span_start_in_minutes = int(
                otel_span.start_time / 1e9 / 60
            )  # OTel spans have nanosecond precision
            self.open_spans.setdefault(span_start_in_minutes, set()).discard(span_id)

        self._prune_old_spans()

    def _is_sentry_span(self, otel_span):
        # type: (OTelSpan) -> bool
        """
        Break infinite loop:
        HTTP requests to Sentry are caught by OTel and send again to Sentry.
        """
        otel_span_url = None
        if otel_span.attributes is not None:
            otel_span_url = otel_span.attributes.get(SpanAttributes.HTTP_URL)
        otel_span_url = cast("Optional[str]", otel_span_url)

        dsn_url = None
        client = get_client()
        if client.dsn:
            dsn_url = Dsn(client.dsn).netloc

        if otel_span_url and dsn_url and dsn_url in otel_span_url:
            return True

        return False

    def _get_otel_context(self, otel_span):
        # type: (OTelSpan) -> dict[str, Any]
        """
        Returns the OTel context for Sentry.
        See: https://develop.sentry.dev/sdk/performance/opentelemetry/#step-5-add-opentelemetry-context
        """
        ctx = {}

        if otel_span.attributes:
            ctx["attributes"] = dict(otel_span.attributes)

        if otel_span.resource.attributes:
            ctx["resource"] = dict(otel_span.resource.attributes)

        return ctx

    def _get_trace_data(self, otel_span, parent_context):
        # type: (OTelSpan, Optional[context_api.Context]) -> dict[str, Any]
        """
        Extracts tracing information from one OTel span and its parent OTel context.
        """
        trace_data = {}  # type: dict[str, Any]
        span_context = otel_span.get_span_context()

        span_id = format_span_id(span_context.span_id)
        trace_data["span_id"] = span_id

        trace_id = format_trace_id(span_context.trace_id)
        trace_data["trace_id"] = trace_id

        parent_span_id = (
            format_span_id(otel_span.parent.span_id) if otel_span.parent else None
        )
        trace_data["parent_span_id"] = parent_span_id

        sentry_trace_data = get_value(SENTRY_TRACE_KEY, parent_context)
        sentry_trace_data = cast("dict[str, Union[str, bool, None]]", sentry_trace_data)
        trace_data["parent_sampled"] = (
            sentry_trace_data["parent_sampled"] if sentry_trace_data else None
        )

        baggage = get_value(SENTRY_BAGGAGE_KEY, parent_context)
        trace_data["baggage"] = baggage

        return trace_data

    def _update_span_with_otel_status(self, sentry_span, otel_span):
        # type: (SentrySpan, OTelSpan) -> None
        """
        Set the Sentry span status from the OTel span
        """
        if otel_span.status.is_unset:
            return

        if otel_span.status.is_ok:
            sentry_span.set_status(SPANSTATUS.OK)
            return

        sentry_span.set_status(SPANSTATUS.INTERNAL_ERROR)

    def _update_span_with_otel_data(self, sentry_span, otel_span):
        # type: (SentrySpan, OTelSpan) -> None
        """
        Convert OTel span data and update the Sentry span with it.
        This should eventually happen on the server when ingesting the spans.
        """
        sentry_span.set_data("otel.kind", otel_span.kind)

        op = otel_span.name
        description = otel_span.name

        if otel_span.attributes is not None:
            for key, val in otel_span.attributes.items():
                sentry_span.set_data(key, val)

            http_method = otel_span.attributes.get(SpanAttributes.HTTP_METHOD)
            http_method = cast("Optional[str]", http_method)

            db_query = otel_span.attributes.get(SpanAttributes.DB_SYSTEM)

            if http_method:
                op = "http"

                if otel_span.kind == SpanKind.SERVER:
                    op += ".server"
                elif otel_span.kind == SpanKind.CLIENT:
                    op += ".client"

                description = http_method

                peer_name = otel_span.attributes.get(SpanAttributes.NET_PEER_NAME, None)
                if peer_name:
                    description += " {}".format(peer_name)

                target = otel_span.attributes.get(SpanAttributes.HTTP_TARGET, None)
                if target:
                    description += " {}".format(target)

                if not peer_name and not target:
                    url = otel_span.attributes.get(SpanAttributes.HTTP_URL, None)
                    url = cast("Optional[str]", url)
                    if url:
                        parsed_url = urlparse(url)
                        url = "{}://{}{}".format(
                            parsed_url.scheme, parsed_url.netloc, parsed_url.path
                        )
                        description += " {}".format(url)

                status_code = otel_span.attributes.get(
                    SpanAttributes.HTTP_STATUS_CODE, None
                )
                status_code = cast("Optional[int]", status_code)
                if status_code:
                    sentry_span.set_http_status(status_code)

            elif db_query:
                op = "db"
                statement = otel_span.attributes.get(SpanAttributes.DB_STATEMENT, None)
                statement = cast("Optional[str]", statement)
                if statement:
                    description = statement

        sentry_span.op = op
        sentry_span.description = description

    def _update_transaction_with_otel_data(self, sentry_span, otel_span):
        # type: (SentrySpan, OTelSpan) -> None
        if otel_span.attributes is None:
            return

        http_method = otel_span.attributes.get(SpanAttributes.HTTP_METHOD)

        if http_method:
            status_code = otel_span.attributes.get(SpanAttributes.HTTP_STATUS_CODE)
            status_code = cast("Optional[int]", status_code)
            if status_code:
                sentry_span.set_http_status(status_code)

            op = "http"

            if otel_span.kind == SpanKind.SERVER:
                op += ".server"
            elif otel_span.kind == SpanKind.CLIENT:
                op += ".client"

            sentry_span.op = op
