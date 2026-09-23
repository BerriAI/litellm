"""OTLP/HTTP trace sink: records exported spans and exposes them over a control API.

Accepts ``application/x-protobuf`` ``ExportTraceServiceRequest`` bodies and OTLP
``http/json`` bodies on any path. Tests read spans through ``recorded_spans`` and
steer the sink through ``configure``; the process can also be frozen with
``SIGSTOP``/``SIGCONT`` after reading its pid from ``/__pid``.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Final
from urllib.parse import urlparse

import httpx
from pydantic import JsonValue

INTERNAL_MARKERS: Final = ("gen_ai.operation.name", "mcp.method.name", "litellm.guardrail_name")


def _proto_spans(body: bytes) -> list[dict[str, JsonValue]]:
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

    def scalar(value: object) -> JsonValue:
        which: Final = value.WhichOneof("value")  # type: ignore[attr-defined]  # protobuf AnyValue
        if which is None:
            return None
        raw: Final = getattr(value, which)
        if which == "array_value":
            return [scalar(item) for item in raw.values]
        if which == "kvlist_value":
            return {pair.key: scalar(pair.value) for pair in raw.values}
        return raw

    request: Final = ExportTraceServiceRequest()
    request.ParseFromString(body)
    return [
        {
            "trace_id": span.trace_id.hex(),
            "span_id": span.span_id.hex(),
            "parent_span_id": span.parent_span_id.hex(),
            "kind": span.kind,
            "name": span.name,
            "attributes": {attribute.key: scalar(attribute.value) for attribute in span.attributes},
            "resource": {attribute.key: scalar(attribute.value) for attribute in resource.resource.attributes},
        }
        for resource in request.resource_spans
        for scope in resource.scope_spans
        for span in scope.spans
    ]


def _json_spans(body: bytes) -> list[dict[str, JsonValue]]:
    payload: Final = json.loads(body)

    def scalar(value: object) -> JsonValue:
        if not isinstance(value, dict):
            return value if isinstance(value, (str, int, float, bool)) or value is None else str(value)
        for key in ("stringValue", "intValue", "doubleValue", "boolValue", "bytesValue"):
            if key in value:
                return value[key]
        if "arrayValue" in value:
            return [scalar(item) for item in value["arrayValue"].get("values", [])]
        if "kvlistValue" in value:
            return {pair["key"]: scalar(pair["value"]) for pair in value["kvlistValue"].get("values", [])}
        return None

    return [
        {
            "trace_id": span.get("traceId", ""),
            "span_id": span.get("spanId", ""),
            "parent_span_id": span.get("parentSpanId", ""),
            "kind": span.get("kind", 0),
            "name": span.get("name", ""),
            "attributes": {attribute["key"]: scalar(attribute.get("value")) for attribute in span.get("attributes", [])},
            "resource": {
                attribute["key"]: scalar(attribute.get("value"))
                for attribute in resource.get("resource", {}).get("attributes", [])
            },
        }
        for resource in payload.get("resourceSpans", [])
        for scope in resource.get("scopeSpans", [])
        for span in scope.get("spans", [])
    ]


def decode_spans(body: bytes, content_type: str) -> list[dict[str, JsonValue]]:
    if "protobuf" in content_type:
        return _proto_spans(body)
    return _json_spans(body)


def span_class(span: dict[str, JsonValue]) -> str:
    if span["kind"] == 2:
        return "root"
    attributes: Final = span.get("attributes") or {}
    if any(marker in attributes for marker in INTERNAL_MARKERS):  # type: ignore[operator]  # attributes is a dict
        return "tenant"
    return "internal"


def spans_for_trace(spans: tuple[dict[str, JsonValue], ...], trace_id: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(span for span in spans if span["trace_id"] == trace_id)


@dataclass(slots=True)
class _State:
    spans: list[dict[str, JsonValue]] = None  # type: ignore[assignment]  # initialized in __post_init__
    requests: list[dict[str, JsonValue]] = None  # type: ignore[assignment]
    status: int = 200
    delay_seconds: float = 0.0
    pause: threading.Event = threading.Event()

    def __post_init__(self) -> None:
        self.spans = []
        self.requests = []
        self.pause.set()


class _Handler(BaseHTTPRequestHandler):
    state: _State
    protocol_version = "HTTP/1.1"

    def _read_body(self) -> bytes:
        return self.rfile.read(int(self.headers.get("content-length", "0")))

    def _send_json(self, payload: object, status: int = 200) -> None:
        body: Final = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _record(self) -> None:
        body: Final = self._read_body()
        self.state.pause.wait(timeout=120)
        if self.state.delay_seconds > 0:
            time.sleep(self.state.delay_seconds)
        recorded: Final = decode_spans(body, self.headers.get("content-type", ""))
        self.state.spans.extend(recorded)
        self.state.requests.append({"path": self.path, "count": len(recorded)})
        self._send_json({"recorded": len(recorded)}, status=self.state.status)

    do_POST = _record
    do_PUT = _record

    def do_GET(self) -> None:
        parsed: Final = urlparse(self.path)
        if parsed.path == "/__spans":
            since: Final = int(dict(part.split("=", 1) for part in parsed.query.split("&") if part).get("since", "0"))
            self._send_json({"next": len(self.state.spans), "spans": self.state.spans[since:]})
            return
        if parsed.path == "/__pid":
            import os

            self._send_json({"pid": os.getpid()})
            return
        if parsed.path == "/__requests":
            self._send_json({"requests": self.state.requests})
            return
        self._send_json({"error": "unknown"}, status=404)

    def do_DELETE(self) -> None:
        if urlparse(self.path).path == "/__spans":
            self.state.spans.clear()
            self.state.requests.clear()
            self._send_json({"cleared": True})
            return
        self._send_json({"error": "unknown"}, status=404)

    def do_PATCH(self) -> None:
        if urlparse(self.path).path != "/__control":
            self._send_json({"error": "unknown"}, status=404)
            return
        fields: Final = json.loads(self._read_body() or b"{}")
        if "status" in fields:
            self.state.status = int(fields["status"])
        if "delay_seconds" in fields:
            self.state.delay_seconds = float(fields["delay_seconds"])
        if fields.get("paused") is True:
            self.state.pause.clear()
        if fields.get("paused") is False:
            self.state.pause.set()
        self._send_json({"status": self.state.status, "delay_seconds": self.state.delay_seconds})

    def log_message(self, format: str, *args: object) -> None:
        pass


def recorded_spans(url: str, since: int = 0) -> tuple[int, tuple[dict[str, JsonValue], ...]]:
    response: Final = httpx.get(f"{url}/__spans", params={"since": since}, trust_env=False, timeout=15)
    response.raise_for_status()
    payload: Final = response.json()
    return int(payload["next"]), tuple(payload["spans"])


def configure_sink(url: str, **fields: JsonValue) -> None:
    httpx.request("PATCH", f"{url}/__control", json=dict(fields), trust_env=False, timeout=15).raise_for_status()


def reset_sink(url: str) -> None:
    httpx.delete(f"{url}/__spans", trust_env=False, timeout=15).raise_for_status()


def sink_pid(url: str) -> int:
    return int(httpx.get(f"{url}/__pid", trust_env=False, timeout=15).json()["pid"])


def main() -> None:
    parser: Final = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    arguments: Final = parser.parse_args()

    class BoundHandler(_Handler):
        state = _State()

    server: Final = ThreadingHTTPServer(("127.0.0.1", arguments.port), BoundHandler)
    server.daemon_threads = True
    server.serve_forever()


if __name__ == "__main__":
    main()
