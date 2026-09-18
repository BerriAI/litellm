"""OTLP/HTTP span exporter that sends the OTLP/JSON encoding instead of protobuf.

The SDK only ships a protobuf OTLP/HTTP exporter; this reuses its transport and
retry loop and swaps the payload for OTLP/JSON (enums as integers, ids as hex).
"""

import base64
import json
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final, TypeAlias

from google.protobuf.json_format import MessageToDict
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import ReadableSpan

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
    def __init__(self, endpoint: str | None, headers: dict[str, str]) -> None:  # mutable-ok: SDK __init__ takes Dict
        super().__init__(endpoint=endpoint, headers=headers)
        self._session.headers["Content-Type"] = JSON_CONTENT_TYPE

    def _serialize_spans(self, spans: Sequence[ReadableSpan]) -> bytes:
        return encode_spans_json(spans)
