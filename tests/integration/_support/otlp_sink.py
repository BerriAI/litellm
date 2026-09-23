"""OTLP/HTTP trace sink: records exported spans and exposes them over a control API.

Accepts ``application/x-protobuf`` ``ExportTraceServiceRequest`` bodies and OTLP
``http/json`` bodies on any path. Tests read spans through ``recorded_spans`` and
steer the sink through ``configure``; the process can also be frozen with
``SIGSTOP``/``SIGCONT`` after reading its pid from ``/__pid``.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Final
from urllib.parse import urlparse

import httpx
import psutil
from pydantic import JsonValue, TypeAdapter
from typing_extensions import ReadOnly, TypedDict

INTERNAL_MARKERS: Final = ("gen_ai.operation.name", "mcp.method.name", "litellm.guardrail_name")


class Span(TypedDict):
    trace_id: ReadOnly[str]
    span_id: ReadOnly[str]
    parent_span_id: ReadOnly[str]
    kind: ReadOnly[int]
    name: ReadOnly[str]
    attributes: ReadOnly[Mapping[str, JsonValue]]
    resource: ReadOnly[Mapping[str, JsonValue]]


class _SpanListing(TypedDict):
    next: ReadOnly[int]
    spans: ReadOnly[list[Span]]


_SPAN_LISTING: Final = TypeAdapter(_SpanListing)


def _proto_spans(body: bytes) -> list[Span]:
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
    from opentelemetry.proto.common.v1.common_pb2 import AnyValue

    def scalar(value: AnyValue) -> JsonValue:
        match value.WhichOneof("value"):
            case "string_value":
                return value.string_value
            case "bool_value":
                return value.bool_value
            case "int_value":
                return int(value.int_value)
            case "double_value":
                return value.double_value
            case "bytes_value":
                return value.bytes_value.decode("utf-8", errors="replace")
            case "array_value":
                return [scalar(item) for item in value.array_value.values]
            case "kvlist_value":
                return {pair.key: scalar(pair.value) for pair in value.kvlist_value.values}
            case _:
                return None

    request: Final = ExportTraceServiceRequest()
    request.ParseFromString(body)
    return [
        Span(
            trace_id=span.trace_id.hex(),
            span_id=span.span_id.hex(),
            parent_span_id=span.parent_span_id.hex(),
            kind=span.kind,
            name=span.name,
            attributes={attribute.key: scalar(attribute.value) for attribute in span.attributes},
            resource={attribute.key: scalar(attribute.value) for attribute in resource.resource.attributes},
        )
        for resource in request.resource_spans
        for scope in resource.scope_spans
        for span in scope.spans
    ]


def _json_spans(body: bytes) -> list[Span]:
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
        Span(
            trace_id=str(span.get("traceId", "")),
            span_id=str(span.get("spanId", "")),
            parent_span_id=str(span.get("parentSpanId", "")),
            kind=int(span.get("kind", 0)),
            name=str(span.get("name", "")),
            attributes={attribute["key"]: scalar(attribute.get("value")) for attribute in span.get("attributes", [])},
            resource={
                attribute["key"]: scalar(attribute.get("value"))
                for attribute in resource.get("resource", {}).get("attributes", [])
            },
        )
        for resource in payload.get("resourceSpans", [])
        for scope in resource.get("scopeSpans", [])
        for span in scope.get("spans", [])
    ]


def decode_spans(body: bytes, content_type: str) -> list[Span]:
    if "protobuf" in content_type:
        return _proto_spans(body)
    return _json_spans(body)


def span_class(span: Span) -> str:
    if span["kind"] == 2:
        return "root"
    if any(marker in span["attributes"] for marker in INTERNAL_MARKERS):
        return "tenant"
    return "internal"


def spans_for_trace(spans: tuple[Span, ...], trace_id: str) -> tuple[Span, ...]:
    return tuple(span for span in spans if span["trace_id"] == trace_id)


@dataclass(slots=True)
class _State:
    spans: list[Span] = field(default_factory=list)
    requests: list[dict[str, JsonValue]] = field(default_factory=list)
    status: int = 200
    delay_seconds: float = 0.0
    pause: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self) -> None:
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


def recorded_spans(url: str, since: int = 0) -> tuple[int, tuple[Span, ...]]:
    response: Final = httpx.get(f"{url}/__spans", params={"since": since}, trust_env=False, timeout=15)
    response.raise_for_status()
    listing: Final = _SPAN_LISTING.validate_python(response.json())
    return listing["next"], tuple(listing["spans"])


def configure_sink(url: str, **fields: JsonValue) -> None:
    httpx.request("PATCH", f"{url}/__control", json=dict(fields), trust_env=False, timeout=15).raise_for_status()


def reset_sink(url: str) -> None:
    httpx.delete(f"{url}/__spans", trust_env=False, timeout=15).raise_for_status()


def sink_pid(url: str) -> int:
    return int(httpx.get(f"{url}/__pid", trust_env=False, timeout=15).json()["pid"])


@dataclass(frozen=True, slots=True)
class SpanSinks:
    operator: str
    tenant: str
    arize: str


def _free_port() -> int:
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        return int(reserve.getsockname()[1])


def _pid_reachable(url: str) -> bool:
    try:
        return httpx.get(f"{url}/__pid", trust_env=False, timeout=2).status_code == 200
    except httpx.TransportError:
        return False


@contextmanager
def owned_sinks(directory: Path) -> Iterator[SpanSinks]:
    from integration._support.process import group_members, signal_group, stop_root_process

    directory.mkdir(parents=True, exist_ok=True)
    ports: Final = tuple(_free_port() for _ in range(3))
    root: Final = Path(__file__).resolve().parents[3]
    with ExitStack() as stack:
        processes: Final = tuple(
            subprocess.Popen(
                [sys.executable, "-m", "integration._support.otlp_sink", "--port", str(port)],
                cwd=root,
                stdout=stack.enter_context((directory / f"otlp-sink-{port}.log").open("w")),
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            for port in ports
        )
        try:
            urls: Final = tuple(f"http://127.0.0.1:{port}" for port in ports)
            deadline: Final = time.monotonic() + 30
            while True:
                alive: Final = all(process.poll() is None for process in processes)
                assert alive, "OTLP sink exited before readiness"
                if all(_pid_reachable(url) for url in urls):
                    break
                assert time.monotonic() < deadline, "OTLP sink readiness deadline exceeded"
                time.sleep(0.05)
            yield SpanSinks(operator=urls[0], tenant=urls[1], arize=urls[2])
        finally:
            for process in processes:
                stopped: Final = stop_root_process(process)
                residual: Final = group_members(process.pid)
                if residual:
                    signal_group(process.pid, signal.SIGKILL)
                    psutil.wait_procs(residual, timeout=5)
                survivors: Final = group_members(process.pid)
                assert not survivors and stopped, "OTLP sink required forced cleanup"


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
