"""OTLP/HTTP span exporter that sends the OTLP/JSON encoding instead of protobuf.

The SDK only ships a protobuf OTLP/HTTP exporter; this reuses its transport and
retry loop and swaps the payload for OTLP/JSON (enums as integers, ids as hex).
"""

import base64
import gzip
import json
import zlib
from collections.abc import Mapping, MutableMapping, Sequence
from types import MappingProxyType
from typing import Final, TypeAlias

import requests
from google.protobuf.json_format import MessageToDict
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.sdk.trace import ReadableSpan
from typing_extensions import ReadOnly, TypedDict, Unpack

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
    return _encode_trace_request_json(encode_spans(spans))


def _encode_trace_request_json(request: ExportTraceServiceRequest) -> bytes:
    payload: Final[_JsonObject] = MessageToDict(request, use_integers_for_enums=True)
    resource_spans: Final = tuple(_hex_resource_spans(resource) for resource in _objects(payload, "resourceSpans"))
    hexed: Final[_JsonObject] = MappingProxyType({**payload, "resourceSpans": resource_spans})
    return json.dumps(hexed, default=dict, separators=(",", ":")).encode()


class _SendOptions(TypedDict, total=False):
    stream: ReadOnly[bool | None]
    verify: ReadOnly[bool | str | None]
    proxies: ReadOnly[MutableMapping[str, str] | None]  # mutable-ok: requests.Session.request requires MutableMapping
    cert: ReadOnly[str | tuple[str, str] | None]
    timeout: ReadOnly[float | tuple[float | None, float | None] | None]
    allow_redirects: ReadOnly[bool]


def _json_request_body(body: bytes, compression: str) -> bytes:
    match compression:
        case "gzip":
            return gzip.compress(_json_request_body(gzip.decompress(body), ""))
        case "deflate":
            return zlib.compress(_json_request_body(zlib.decompress(body), ""))
        case _:
            return _encode_trace_request_json(ExportTraceServiceRequest.FromString(body))


class _JsonSession(requests.Session):
    def __init__(self, session: requests.Session) -> None:
        super().__init__()
        self._delegate = session
        self.headers.clear()
        self.trust_env = False
        self.verify = session.verify
        self.cert = session.cert
        self.stream = session.stream

    def send(
        self,
        request: requests.PreparedRequest,
        **kwargs: Unpack[_SendOptions],  # kwargs-ok: requests.Session.send requires forwarding transport options
    ) -> requests.Response:
        if not isinstance(request.body, bytes):
            raise TypeError("OTLP protobuf request body must be bytes")
        body: Final = _json_request_body(request.body, request.headers.get("Content-Encoding", ""))
        headers: Final = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in {"content-type", "content-length"}
        }
        return self._delegate.request(
            method=request.method or "POST",
            url=request.url or "",
            data=body,
            headers={**headers, "Content-Type": JSON_CONTENT_TYPE},
            **kwargs,
        )

    def close(self) -> None:
        self._delegate.close()
        super().close()


class OTLPJsonSpanExporter(OTLPSpanExporter):
    def __init__(
        self,
        endpoint: str | None,
        headers: dict[str, str],  # mutable-ok: SDK __init__ takes Dict
        certificate_file: str | None = None,
        session: "requests.Session | None" = None,
    ) -> None:
        json_session: Final = _JsonSession(session if session is not None else requests.Session())
        super().__init__(endpoint=endpoint, headers=headers, certificate_file=certificate_file, session=json_session)
