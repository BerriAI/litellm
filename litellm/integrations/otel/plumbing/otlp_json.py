"""OTLP/HTTP span exporter that sends the OTLP/JSON encoding instead of protobuf.

The SDK only ships a protobuf OTLP/HTTP exporter; this reuses its transport and
retry loop and swaps the payload for OTLP/JSON (enums as integers, ids as hex).
"""

import base64
import json
from collections.abc import Mapping, Sequence
from typing import Final, TypeAlias

from google.protobuf.json_format import MessageToDict
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import ReadableSpan

JSON_CONTENT_TYPE: Final = "application/json"
_HEX_ID_KEYS: Final = frozenset({"traceId", "spanId", "parentSpanId"})

_Json: TypeAlias = "Mapping[str, _Json] | Sequence[_Json] | str | int | float | bool | None"


def _hex_ids(value: _Json) -> _Json:
    if isinstance(value, dict):
        return {  # mutable-ok: json.dumps rejects MappingProxyType, so the payload must stay a real dict
            key: base64.b64decode(item).hex() if key in _HEX_ID_KEYS and isinstance(item, str) else _hex_ids(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return tuple(_hex_ids(item) for item in value)
    return value


def encode_spans_json(spans: Sequence[ReadableSpan]) -> bytes:
    payload: Final[Mapping[str, _Json]] = MessageToDict(encode_spans(spans), use_integers_for_enums=True)
    return json.dumps(_hex_ids(payload), separators=(",", ":")).encode()


class OTLPJsonSpanExporter(OTLPSpanExporter):
    def __init__(self, endpoint: str | None, headers: dict[str, str]) -> None:  # mutable-ok: SDK __init__ takes Dict
        super().__init__(endpoint=endpoint, headers=headers)
        self._session.headers["Content-Type"] = JSON_CONTENT_TYPE

    def _serialize_spans(self, spans: Sequence[ReadableSpan]) -> bytes:
        return encode_spans_json(spans)
