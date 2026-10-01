"""Profile OTLP decoding and insertion against a loopback HTTP sink in a fresh process.

Run from the checkout being measured, with its matching extension installed:
LITELLM_RUST=1 python scripts/benchmark_otlp_copies.py --case fanout --concurrency 2
Use --legacy for the original per-row tenant stamping and Pydantic insert validation.
The parent samples RSS while native code holds the GIL. Times include debug-build overhead
unless the installed extension was compiled with optimization.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import gzip
import io
import json
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import SimpleQueue
from typing import Final, Protocol, cast

import psutil
from pydantic import BaseModel

CASES: Final = (
    "normal",
    "small-fanout",
    "fanout",
    "unique",
    "oversized-insert",
    "scope-fanout",
    "escaped-unique",
    "escaped-4m",
)


@dataclass(frozen=True, slots=True)
class Timing:
    decode_bridge_ms: float = 0
    normalize_stamp_ms: float = 0
    validation_ms: float = 0
    insert_conversion_ms: float = 0
    encode_transport_ms: float = 0
    error: str = ""


class Options(BaseModel):
    case: str
    concurrency: int
    legacy: bool
    worker: bool


class Measurement(BaseModel):
    case: str
    concurrency: int
    body_bytes: int
    content_type: str
    initial_rss_mib: float
    elapsed_ms: float
    requests: list[Timing]
    insert_bytes: list[int]
    extension: str | None


class MemoryInfo(Protocol):
    @property
    def rss(self) -> int: ...


def resident_bytes(process: psutil.Process) -> int:
    return cast(MemoryInfo, process.memory_info()).rss  # cast-ok: psutil's platform-specific namedtuple has rss


class Sink(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), Handler)
        self.sizes: Final = SimpleQueue[int]()


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        raw: Final = self.rfile.read(int(self.headers["Content-Length"]))
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
            size: Final = sum(len(chunk) for chunk in iter(lambda: stream.read(65536), b""))
        cast(Sink, self.server).sizes.put(size)  # cast-ok: Sink installs this handler
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        pass


def payload(case: str) -> tuple[bytes, str]:
    if case == "normal":
        fixture: Final = Path("tests/test_litellm/tracing/fixtures/langsmith_deep_agent_export.json")
        return fixture.read_bytes(), "application/json"
    if case in ("escaped-unique", "escaped-4m"):
        from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
        from opentelemetry.proto.common.v1.common_pb2 import AnyValue, ArrayValue, KeyValue
        from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, ScopeSpans, Span

        request: Final = ExportTraceServiceRequest(
            resource_spans=[
                ResourceSpans(
                    scope_spans=[
                        ScopeSpans(
                            spans=[
                                Span(
                                    trace_id=b"\x01" * 16,
                                    span_id=index.to_bytes(8, "big"),
                                    attributes=[
                                        KeyValue(
                                            key="escaped",
                                            value=AnyValue(
                                                array_value=ArrayValue(
                                                    values=[
                                                        AnyValue(
                                                            string_value="\x00"
                                                            * (4000 if case == "escaped-4m" else 16_000)
                                                        )
                                                    ]
                                                )
                                            ),
                                        )
                                    ],
                                )
                                for index in range(1, 1025)
                            ]
                        )
                    ]
                )
            ]
        )
        return request.SerializeToString(), "application/x-protobuf"
    size: Final = {
        "small-fanout": 8192,
        "fanout": 16384,
        "unique": 16000,
        "oversized-insert": 8 * 1024**2,
        "scope-fanout": 8 * 1024**2,
    }[case]
    spans: Final = [
        {
            "traceId": "01" * 16,
            "spanId": f"{index + 1:016x}",
            "name": "span",
            "startTimeUnixNano": "1",
            "endTimeUnixNano": "2",
            **(
                {"attributes": [{"key": "unique", "value": {"stringValue": f"{index:04}" + "x" * (size - 4)}}]}
                if case == "unique"
                else {}
            ),
        }
        for index in range(1024)
    ]
    resource: Final = (
        {}
        if case in ("unique", "scope-fanout")
        else {"resource": {"attributes": [{"key": "shared", "value": {"stringValue": "x" * size}}]}}
    )
    scope: Final = {"name": "x" * size} if case == "scope-fanout" else {}
    return json.dumps(
        {"resourceSpans": [{**resource, "scopeSpans": [{"scope": scope, "spans": spans}]}]}
    ).encode(), "application/json"


def worker(case: str, concurrency: int, legacy: bool) -> None:
    from pydantic import JsonValue, TypeAdapter

    from litellm.rust_bridge import _native
    from litellm.tracing.decode import (
        _span_row,  # pyright: ignore[reportPrivateUsage]  # measures normalization separately
    )
    from litellm.tracing.receiver import Tenant
    from litellm.tracing.types import SpanRow

    body, content_type = payload(case)
    gc.collect()
    initial: Final = resident_bytes(psutil.Process()) / 1024**2
    server: Final = Sink()
    thread: Final = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    storage: Final = _native.NativeTraceStorage("benchmark", f"http://127.0.0.1:{server.server_port}")
    tenant: Final = Tenant("team", "key", "org")
    validator: Final = TypeAdapter(list[dict[str, JsonValue]])

    def decode() -> tuple[tuple[SpanRow, ...], float, float]:
        start: Final = time.perf_counter()
        spans: Final = _native.trace_decode_otlp(body, content_type)
        decoded: Final = time.perf_counter()
        rows: Final = tuple(_span_row(span) for span in spans)
        stamped: Final = tuple(tenant.stamp(row) for row in rows) if legacy else tenant.stamp_rows(rows)
        return stamped, (decoded - start) * 1000, (time.perf_counter() - decoded) * 1000

    async def run() -> Timing:
        try:
            rows, decode_ms, normalize_ms = await asyncio.to_thread(decode)
            validate_start: Final = time.perf_counter()
            validated: Final = validator.validate_python(rows) if legacy else rows
            convert_start: Final = time.perf_counter()
            future: Final = storage.insert_rows("otel_traces", validated)
            encode_start: Final = time.perf_counter()
            await future
            return Timing(
                decode_ms,
                normalize_ms,
                (convert_start - validate_start) * 1000,
                (encode_start - convert_start) * 1000,
                (time.perf_counter() - encode_start) * 1000,
            )
        except (OverflowError, ValueError) as error:
            return Timing(error=str(error))

    async def batch() -> list[Timing]:
        return await asyncio.gather(*(run() for _ in range(concurrency)))

    start: Final = time.perf_counter()
    try:
        results: Final = asyncio.run(batch())
        sys.stdout.write(
            json.dumps(
                {
                    "case": case,
                    "concurrency": concurrency,
                    "body_bytes": len(body),
                    "content_type": content_type,
                    "initial_rss_mib": initial,
                    "elapsed_ms": (time.perf_counter() - start) * 1000,
                    "requests": [asdict(result) for result in results],
                    "insert_bytes": [server.sizes.get_nowait() for _ in range(server.sizes.qsize())],
                    "extension": _native.__file__,
                }
            )
            + "\n"
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def memory_samples(child: subprocess.Popen[str]) -> Iterator[int]:
    process: Final = psutil.Process(child.pid)
    while child.poll() is None:
        try:
            yield resident_bytes(process)
        except psutil.NoSuchProcess:
            return
        time.sleep(0.002)


def main() -> None:
    parser: Final = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, default="normal")
    parser.add_argument("--concurrency", type=int, choices=(1, 2), default=1)
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args: Final = Options.model_validate(vars(parser.parse_args()))
    if args.worker:
        worker(args.case, args.concurrency, args.legacy)
        return
    with subprocess.Popen(
        [sys.executable, __file__, *sys.argv[1:], "--worker"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    ) as child:
        peak: Final = max(memory_samples(child), default=0) / 1024**2
        output, error = child.communicate()
        if child.returncode:
            sys.stderr.write(error)
            raise SystemExit(child.returncode)
        result: Final = Measurement.model_validate_json(output)
        sys.stdout.write(
            json.dumps({**result.model_dump(), "peak_rss_mib": peak, "rss_increase_mib": peak - result.initial_rss_mib})
            + "\n"
        )
        sys.stderr.write(error)


if __name__ == "__main__":
    main()
