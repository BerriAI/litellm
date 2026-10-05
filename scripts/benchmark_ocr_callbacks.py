#!/usr/bin/env python3
"""Measure serial sync/async OCR latency through a loopback HTTP provider

Run each callback mode in a fresh process against an installed release wheel:
python -I scripts/benchmark_ocr_callbacks.py --callbacks none --label before \
    --expected-transport rust --iterations 200 --warmup 20 --output before-none.json
Repeat with --callbacks noop and with the candidate wheel in a separate venv
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import importlib.metadata
import json
import statistics
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Final, cast

SIZES: Final = (
    1024,
    4 * 1024,
    16 * 1024,
    64 * 1024,
    256 * 1024,
    1024 * 1024,
)
MODEL: Final = "mistral/mistral-ocr-latest"
EXPECTED_MARKDOWN: Final = "mock remote OCR response"
RESPONSE: Final = json.dumps(
    {
        "pages": [{"index": 0, "markdown": EXPECTED_MARKDOWN, "images": [], "dimensions": None}],
        "model": "mistral-ocr-latest",
        "usage_info": {"pages_processed": 1},
    },
    separators=(",", ":"),
).encode()


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), Handler)
        self.user_agents: set[str] = set()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        server: Final = cast(Server, self.server)
        server.user_agents.add(self.headers.get("User-Agent", ""))
        length: Final = int(self.headers["Content-Length"])
        body: Final = self.rfile.read(length)
        request: Final = json.loads(body)
        if self.path != "/v1/ocr" or request.get("model") != "mistral-ocr-latest":
            self.send_error(400)
            return
        document: Final = request.get("document", {})
        if not isinstance(document, dict) or not str(document.get("document_url", "")).startswith(
            "data:application/pdf;base64,"
        ):
            self.send_error(400)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(RESPONSE)))
        self.end_headers()
        self.wfile.write(RESPONSE)

    def log_message(self, format: str, *args: object) -> None:
        return


@dataclass(frozen=True, slots=True)
class Result:
    label: str
    mode: str
    size: int
    iterations: int
    median_ms: float
    mean_ms: float
    p95_ms: float
    requests_per_second: float


def document(size: int) -> dict[str, str]:
    payload: Final = b"%PDF-1.4\n" + b"x" * max(0, size - 9)
    encoded: Final = base64.b64encode(payload[:size]).decode("ascii")
    return {"type": "document_url", "document_url": f"data:application/pdf;base64,{encoded}"}


def percentile(values: Sequence[float], quantile: float) -> float:
    ordered: Final = sorted(values)
    index: Final = min(len(ordered) - 1, round((len(ordered) - 1) * quantile))
    return ordered[index]


def verify(response: object) -> None:
    pages: Final = getattr(response, "pages", ())
    if len(pages) != 1 or getattr(pages[0], "markdown", None) != EXPECTED_MARKDOWN:
        raise RuntimeError(f"unexpected OCR response: {response!r}")


def summarize(label: str, mode: str, size: int, samples: Sequence[float]) -> Result:
    median: Final = statistics.median(samples)
    return Result(
        label=label,
        mode=mode,
        size=size,
        iterations=len(samples),
        median_ms=median * 1000,
        mean_ms=statistics.fmean(samples) * 1000,
        p95_ms=percentile(samples, 0.95) * 1000,
        requests_per_second=1 / median,
    )


def sync_samples(litellm: object, url: str, request_document: dict[str, str], count: int) -> tuple[float, ...]:
    samples: list[float] = []
    for _ in range(count):
        started: Final = time.perf_counter()
        response: Final = litellm.ocr(
            model=MODEL, document=request_document, api_base=url, api_key="mock-key", timeout=30
        )
        samples.append(time.perf_counter() - started)
        verify(response)
    return tuple(samples)


async def async_samples(litellm: object, url: str, request_document: dict[str, str], count: int) -> tuple[float, ...]:
    samples: list[float] = []
    for _ in range(count):
        started: Final = time.perf_counter()
        response: Final = await litellm.aocr(
            model=MODEL, document=request_document, api_base=url, api_key="mock-key", timeout=30
        )
        samples.append(time.perf_counter() - started)
        verify(response)
    return tuple(samples)


async def main() -> int:
    parser: Final = argparse.ArgumentParser(description="E2E OCR benchmark against a local remote-style HTTP server")
    parser.add_argument("--callbacks", choices=("none", "noop"), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--expected-transport", choices=("python", "rust"), required=True)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    parser.add_argument("--output", type=Path, required=True)
    args: Final = parser.parse_args()

    import litellm
    from litellm.integrations.custom_logger import CustomLogger

    class NoopCallback(CustomLogger):
        def __init__(self) -> None:
            super().__init__()
            self.pre_calls = 0
            self.sync_successes = 0
            self.async_successes = 0

        def log_pre_api_call(self, model, messages, kwargs):
            self.pre_calls += 1

        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            self.sync_successes += 1

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            self.async_successes += 1

    registry_names: Final = (
        "callbacks",
        "input_callback",
        "success_callback",
        "failure_callback",
        "_async_input_callback",
        "_async_success_callback",
        "_async_failure_callback",
    )
    if any(getattr(litellm, name) for name in registry_names):
        raise RuntimeError("benchmark requires initially empty callback registrations")
    callback: Final = NoopCallback()
    if args.callbacks == "noop":
        litellm.callbacks.append(callback)

    rust_toggle: Final = getattr(litellm, "rust", None)
    if callable(rust_toggle):
        rust_toggle(False)
    package: Final = Path(litellm.__file__).resolve()
    version: Final = importlib.metadata.version("litellm")
    native_path: str | None = None
    native_sha256: str | None = None
    try:
        from litellm.rust_bridge import _native

        native: Final = Path(_native.__file__).resolve()
        native_path = str(native)
        native_sha256 = hashlib.file_digest(native.open("rb"), "sha256").hexdigest()
    except ImportError:
        pass

    server: Final = Server()
    thread: Final = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url: Final = f"http://127.0.0.1:{server.server_port}"
    results: list[Result] = []
    try:
        for size in args.sizes:
            request_document: Final = document(size)
            sync_samples(litellm, url, request_document, args.warmup)
            sync_result: Final = summarize(
                args.label, "sync", size, sync_samples(litellm, url, request_document, args.iterations)
            )
            results.append(sync_result)
            await async_samples(litellm, url, request_document, args.warmup)
            async_result: Final = summarize(
                args.label,
                "async",
                size,
                await async_samples(litellm, url, request_document, args.iterations),
            )
            results.append(async_result)
            sys.stdout.write(json.dumps(asdict(sync_result)) + "\n")
            sys.stdout.write(json.dumps(asdict(async_result)) + "\n")
            sys.stdout.flush()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    from litellm.litellm_core_utils.litellm_logging import executor
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    await GLOBAL_LOGGING_WORKER.flush()
    await asyncio.to_thread(executor.shutdown, wait=True)
    per_mode: Final = len(args.sizes) * (args.iterations + args.warmup)
    if args.callbacks == "noop":
        if (callback.pre_calls, callback.sync_successes, callback.async_successes) != (
            2 * per_mode,
            per_mode,
            per_mode,
        ):
            raise RuntimeError(f"callback delivery mismatch: {vars(callback)}")
    elif any(getattr(litellm, name) for name in registry_names):
        raise RuntimeError("callback registrations appeared in the no-callback case")
    await GLOBAL_LOGGING_WORKER.stop()

    user_agents: Final = tuple(sorted(server.user_agents))
    python_transport: Final = any(
        value.startswith("python-httpx") or value.startswith("litellm/") for value in user_agents
    )
    if (args.expected_transport == "python") != python_transport:
        raise RuntimeError(f"unexpected transport for {args.label}: user_agents={user_agents}")
    artifact: Final = {
        "label": args.label,
        "callbacks": args.callbacks,
        "python": sys.executable,
        "callback_counts": {
            "pre": callback.pre_calls,
            "sync_success": callback.sync_successes,
            "async_success": callback.async_successes,
        },
        "version": version,
        "package": str(package),
        "native": native_path,
        "native_sha256": native_sha256,
        "user_agents": user_agents,
        "results": tuple(asdict(result) for result in results),
    }
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    sys.stdout.write(json.dumps({key: artifact[key] for key in ("label", "version", "package", "user_agents")}) + "\n")
    sys.stdout.write(f"results={args.output}\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
