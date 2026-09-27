"""OTLP/HTTP span exporter that sends the OTLP/JSON encoding instead of protobuf.

The SDK only ships a protobuf OTLP/HTTP exporter; this reuses its transport and
retry loop and swaps the payload for OTLP/JSON (enums as integers, ids as hex).
"""

import base64
import json
import threading
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from types import MappingProxyType
from typing import Final, TypeAlias

import requests
from google.protobuf.json_format import MessageToDict
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.exporter.otlp.proto.http import trace_exporter as _http_trace_exporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult

JSON_CONTENT_TYPE: Final = "application/json"
_HEX_ID_KEYS: Final = frozenset({"traceId", "spanId", "parentSpanId"})
# SDK 1.44 serializes inside OTLPSpanExporter.export and no longer calls
# _serialize_spans. The protobuf encode_spans name is the only seam left, and a
# context flag keeps a concurrent protobuf export on another thread on protobuf.
_EXPORT_JSON: Final[ContextVar[bool]] = ContextVar("litellm_otlp_json_export", default=False)
_JSON_HOOK_LOCK: Final = threading.Lock()


class _JsonPayloadHook:
    installed: bool = False


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


def _install_json_payload_hook() -> None:
    with _JSON_HOOK_LOCK:
        if _JsonPayloadHook.installed:
            return
        protobuf_encode: Final = _http_trace_exporter.encode_spans

        def encode_spans(spans: Sequence[ReadableSpan]) -> object:
            if not _EXPORT_JSON.get():
                return protobuf_encode(spans)
            payload: Final = encode_spans_json(spans)

            class _JsonMessage:
                def SerializePartialToString(self) -> bytes:
                    return payload

            return _JsonMessage()

        _http_trace_exporter.encode_spans = encode_spans
        _JsonPayloadHook.installed = True


def encode_spans_json(spans: Sequence[ReadableSpan]) -> bytes:
    payload: Final[_JsonObject] = MessageToDict(encode_spans(spans), use_integers_for_enums=True)
    resource_spans: Final = tuple(_hex_resource_spans(resource) for resource in _objects(payload, "resourceSpans"))
    hexed: Final[_JsonObject] = MappingProxyType({**payload, "resourceSpans": resource_spans})
    return json.dumps(hexed, default=dict, separators=(",", ":")).encode()


class OTLPJsonSpanExporter(OTLPSpanExporter):
    def __init__(
        self,
        endpoint: str | None,
        headers: dict[str, str],  # mutable-ok: SDK __init__ takes Dict
        certificate_file: str | None = None,
        session: "requests.Session | None" = None,
    ) -> None:
        super().__init__(endpoint=endpoint, headers=headers, certificate_file=certificate_file, session=session)
        self._session.headers["Content-Type"] = JSON_CONTENT_TYPE
        _install_json_payload_hook()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        token: Final = _EXPORT_JSON.set(True)
        try:
            return super().export(spans)
        finally:
            _EXPORT_JSON.reset(token)
