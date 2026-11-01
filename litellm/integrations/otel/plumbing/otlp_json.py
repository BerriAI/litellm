"""OTLP/HTTP span exporter that sends the OTLP/JSON encoding instead of protobuf.

The SDK only ships a protobuf OTLP/HTTP exporter; this reuses its transport and
retry loop and swaps the payload for OTLP/JSON (enums as integers, ids as hex).
"""

import base64
import json
import logging
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final, TypeAlias

import requests
from google.protobuf.json_format import MessageToDict
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.exporter.otlp.proto.http._common import RequestPayloadTooLargeError
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.semconv.attributes.http_attributes import HTTP_RESPONSE_STATUS_CODE

_logger: Final = logging.getLogger(__name__)
JSON_CONTENT_TYPE: Final = "application/json"
_HEX_ID_KEYS: Final = frozenset({"traceId", "spanId", "parentSpanId"})

_JsonValue: TypeAlias = "Mapping[str, _JsonValue] | Sequence[_JsonValue] | str | int | float | bool | None"
_JsonObject: TypeAlias = Mapping[str, "_JsonValue"]


def _objects(node: _JsonObject, key: str) -> tuple[_JsonObject, ...]:
    items: Final = node.get(key)
    if isinstance(items, str) or not isinstance(items, Sequence):
        return ()
    return tuple(item for item in items if isinstance(item, Mapping))


def _hex_ids(node: _JsonObject) -> _JsonObject:
    return MappingProxyType(
        {
            key: base64.b64decode(item).hex() if key in _HEX_ID_KEYS and isinstance(item, str) else item
            for key, item in node.items()
        }
    )


def _hex_span(span: _JsonObject) -> _JsonObject:
    links: Final = _objects(span, "links")
    if not links:
        return _hex_ids(span)
    return MappingProxyType({**_hex_ids(span), "links": tuple(_hex_ids(link) for link in links)})


def _hex_scope_spans(scope: _JsonObject) -> _JsonObject:
    return MappingProxyType({**scope, "spans": tuple(_hex_span(span) for span in _objects(scope, "spans"))})


def _hex_resource_spans(resource: _JsonObject) -> _JsonObject:
    scope_spans: Final = tuple(_hex_scope_spans(scope) for scope in _objects(resource, "scopeSpans"))
    return MappingProxyType({**resource, "scopeSpans": scope_spans})


def encode_spans_json(spans: Sequence[ReadableSpan]) -> bytes:
    payload: Final[_JsonObject] = MessageToDict(encode_spans(spans), use_integers_for_enums=True)
    resource_spans: Final = tuple(_hex_resource_spans(resource) for resource in _objects(payload, "resourceSpans"))
    hexed: Final[_JsonObject] = MappingProxyType({**payload, "resourceSpans": resource_spans})
    return json.dumps(hexed, default=dict, separators=(",", ":")).encode()


class OTLPJsonSpanExporter(OTLPSpanExporter):
    def __init__(
        self,
        endpoint: str | None,
        headers: Mapping[str, str],
        certificate_file: str | None = None,
        session: "requests.Session | None" = None,
        *,
        max_request_size: int | None = None,
    ) -> None:
        super().__init__(
            endpoint=endpoint,
            headers={**{key.lower(): value for key, value in headers.items()}, "content-type": JSON_CONTENT_TYPE},
            certificate_file=certificate_file,
            session=session,
            max_request_size=max_request_size,
        )

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        if self._shutdown:
            _logger.warning("Exporter already shutdown, ignoring batch")
            return SpanExportResult.FAILURE

        with self._metrics.export_operation(len(spans)) as result:
            try:
                serialized_data: Final = encode_spans_json(spans)
            except Exception as error:  # noqa: BLE001  # SDK export contract reports encoding failures as results
                _logger.error("Failed to encode span batch: %s", error)
                result.error = error
                return SpanExportResult.FAILURE

            if 0 < self._max_request_size < len(serialized_data):
                result.error = RequestPayloadTooLargeError(
                    f"Serialized span request size {len(serialized_data)} bytes "
                    f"exceeds max_request_size {self._max_request_size} bytes."
                )
                _logger.warning("Dropping span batch: %s", result.error)
                return SpanExportResult.FAILURE

            export_result: Final = self._client.export(serialized_data)
            if not export_result.success:
                result.error = export_result.error
                result.error_attrs = (
                    {HTTP_RESPONSE_STATUS_CODE: export_result.status_code}
                    if export_result.status_code is not None
                    else None
                )
                return SpanExportResult.FAILURE

        return SpanExportResult.SUCCESS
