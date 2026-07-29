#!/usr/bin/env python3
"""Benchmark LiteLLM proxy /v1/chat/completions overhead and streaming TTFT.

The script can run a local OpenAI-compatible mock provider plus a LiteLLM proxy
from any checkout. That makes it useful for comparing tags/commits without
depending on real provider latency.

Example:
    uv run python scripts/benchmark_chat_completions_perf.py \
        --label current --requests 500 --concurrency 100

Compare another checkout:
    uv run python scripts/benchmark_chat_completions_perf.py \
        --label v1.83.14-stable --litellm-dir /tmp/litellm-v1.83.14-stable
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import aiohttp
from aiohttp import web


DEFAULT_MODEL = "perf-test-model"
DEFAULT_API_KEY = "sk-1234"


@dataclass
class RequestSample:
    success: bool
    latency_ms: float
    status_code: int
    overhead_header_ms: Optional[float] = None
    error: str = ""


@dataclass
class SummaryStats:
    requests: int
    failures: int
    rps: float
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    overhead_header_mean_ms: Optional[float] = None
    overhead_header_p50_ms: Optional[float] = None
    overhead_header_p95_ms: Optional[float] = None


class MockOpenAIProvider:
    def __init__(
        self,
        host: str,
        port: int,
        first_token_delay_ms: float,
        stream_content_chunks: int,
    ) -> None:
        self.host = host
        self.port = port
        self.first_token_delay_ms = first_token_delay_ms
        self.stream_content_chunks = stream_content_chunks
        self.runner: Optional[web.AppRunner] = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    async def start(self) -> None:
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self.handle_chat_completions)
        self.runner = web.AppRunner(app, access_log=None)
        await self.runner.setup()
        site = web.TCPSite(self.runner, self.host, self.port)
        await site.start()

    async def stop(self) -> None:
        if self.runner is not None:
            await self.runner.cleanup()

    async def handle_chat_completions(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        if body.get("stream"):
            return await self._streaming_response(request=request, body=body)
        return self._json_response(body)

    def _json_response(self, body: dict[str, Any]) -> web.Response:
        now = int(time.time())
        payload = {
            "id": "chatcmpl-perf",
            "object": "chat.completion",
            "created": now,
            "model": body.get("model", DEFAULT_MODEL),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        }
        return web.json_response(payload)

    async def _streaming_response(
        self, request: web.Request, body: dict[str, Any]
    ) -> web.StreamResponse:
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
            },
        )
        await response.prepare(request)
        if self.first_token_delay_ms > 0:
            await asyncio.sleep(self.first_token_delay_ms / 1000)

        created = int(time.time())
        chunks = [{"role": "assistant"}]
        chunks.extend({"content": "hello"} for _ in range(self.stream_content_chunks))
        for delta in chunks:
            event = {
                "id": "chatcmpl-perf",
                "object": "chat.completion.chunk",
                "created": created,
                "model": body.get("model", DEFAULT_MODEL),
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
            await response.write(f"data: {json.dumps(event)}\n\n".encode())

        done_event = {
            "id": "chatcmpl-perf",
            "object": "chat.completion.chunk",
            "created": created,
            "model": body.get("model", DEFAULT_MODEL),
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        await response.write(f"data: {json.dumps(done_event)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(int(len(sorted_values) * pct / 100), len(sorted_values) - 1)
    return sorted_values[index]


def summarize(samples: list[RequestSample], wall_time_s: float) -> SummaryStats:
    latencies = [sample.latency_ms for sample in samples if sample.success]
    overhead_headers = [
        sample.overhead_header_ms
        for sample in samples
        if sample.success and sample.overhead_header_ms is not None
    ]
    failures = len(samples) - len(latencies)
    return SummaryStats(
        requests=len(samples),
        failures=failures,
        rps=(len(latencies) / wall_time_s) if wall_time_s > 0 else 0.0,
        mean_ms=statistics.mean(latencies) if latencies else 0.0,
        p50_ms=percentile(latencies, 50),
        p95_ms=percentile(latencies, 95),
        p99_ms=percentile(latencies, 99),
        overhead_header_mean_ms=(
            statistics.mean(overhead_headers) if overhead_headers else None
        ),
        overhead_header_p50_ms=(
            percentile(overhead_headers, 50) if overhead_headers else None
        ),
        overhead_header_p95_ms=(
            percentile(overhead_headers, 95) if overhead_headers else None
        ),
    )


def format_optional_ms(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def get_git_revision(litellm_dir: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=litellm_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def write_proxy_config(config_path: Path, provider_base_url: str, api_key: str) -> None:
    config_path.write_text(
        f"""model_list:
  - model_name: {DEFAULT_MODEL}
    litellm_params:
      model: openai/{DEFAULT_MODEL}
      api_key: fake-provider-key
      api_base: {provider_base_url}/v1

general_settings:
  master_key: {api_key}

litellm_settings:
  drop_params: true
  telemetry: false
""",
        encoding="utf-8",
    )


async def wait_for_proxy(base_url: str, timeout_s: float) -> None:
    deadline = time.perf_counter() + timeout_s
    last_error = ""
    async with aiohttp.ClientSession() as session:
        while time.perf_counter() < deadline:
            try:
                async with session.get(f"{base_url}/health") as response:
                    if response.status < 500:
                        return
                    last_error = f"HTTP {response.status}: {await response.text()}"
            except Exception as exc:
                last_error = str(exc)
            await asyncio.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for proxy at {base_url}: {last_error}")


def start_proxy_process(
    litellm_dir: Path,
    proxy_command: str,
    config_path: Path,
    port: int,
    log_path: Path,
) -> subprocess.Popen:
    command = shlex.split(proxy_command) + [
        "--config",
        str(config_path),
        "--port",
        str(port),
    ]
    env = {
        **os.environ,
        "LITELLM_TELEMETRY": "False",
        "PYTHONUNBUFFERED": "1",
    }
    log_file = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(
        command,
        cwd=litellm_dir,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def stop_proxy_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except Exception:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except Exception:
            pass


def extract_overhead_header(headers: aiohttp.typedefs.LooseHeaders) -> Optional[float]:
    raw_value = headers.get("x-litellm-overhead-duration-ms")  # type: ignore[union-attr]
    if raw_value is None:
        return None
    try:
        return float(raw_value)
    except ValueError:
        return None


async def post_non_streaming(
    session: aiohttp.ClientSession,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> RequestSample:
    async with semaphore:
        start = time.perf_counter()
        try:
            async with session.post(url, headers=headers, json=payload) as response:
                body = await response.read()
                latency_ms = (time.perf_counter() - start) * 1000
                if response.status != 200:
                    return RequestSample(
                        success=False,
                        latency_ms=latency_ms,
                        status_code=response.status,
                        error=body.decode("utf-8", errors="ignore")[:200],
                    )
                return RequestSample(
                    success=True,
                    latency_ms=latency_ms,
                    status_code=response.status,
                    overhead_header_ms=extract_overhead_header(response.headers),
                )
        except Exception as exc:
            return RequestSample(
                success=False,
                latency_ms=(time.perf_counter() - start) * 1000,
                status_code=0,
                error=str(exc)[:200],
            )


async def run_non_streaming_benchmark(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    requests: int,
    concurrency: int,
    warmup: int,
    timeout_s: float,
) -> SummaryStats:
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    connector = aiohttp.TCPConnector(
        limit=max(concurrency * 2, 10),
        limit_per_host=max(concurrency, 10),
        force_close=False,
    )
    semaphore = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        if warmup > 0:
            await asyncio.gather(
                *[
                    post_non_streaming(session, url, headers, payload, semaphore)
                    for _ in range(warmup)
                ]
            )
        wall_start = time.perf_counter()
        samples = await asyncio.gather(
            *[
                post_non_streaming(session, url, headers, payload, semaphore)
                for _ in range(requests)
            ]
        )
        wall_time_s = time.perf_counter() - wall_start
    return summarize(samples, wall_time_s)


async def measure_stream_ttft(
    session: aiohttp.ClientSession,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> RequestSample:
    async with semaphore:
        start = time.perf_counter()
        try:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    body = await response.read()
                    return RequestSample(
                        success=False,
                        latency_ms=(time.perf_counter() - start) * 1000,
                        status_code=response.status,
                        error=body.decode("utf-8", errors="ignore")[:200],
                    )

                while raw_line := await response.content.readline():
                    line = raw_line.strip()
                    if not line or not line.startswith(b"data:"):
                        continue
                    event_payload = line[5:].strip()
                    if event_payload == b"[DONE]":
                        break
                    event = json.loads(event_payload)
                    choice = (event.get("choices") or [{}])[0]
                    delta = choice.get("delta") or {}
                    content = delta.get("content") or choice.get("text")
                    if content:
                        return RequestSample(
                            success=True,
                            latency_ms=(time.perf_counter() - start) * 1000,
                            status_code=response.status,
                            overhead_header_ms=extract_overhead_header(
                                response.headers
                            ),
                        )
                return RequestSample(
                    success=False,
                    latency_ms=(time.perf_counter() - start) * 1000,
                    status_code=response.status,
                    error="stream ended before a content token",
                )
        except Exception as exc:
            return RequestSample(
                success=False,
                latency_ms=(time.perf_counter() - start) * 1000,
                status_code=0,
                error=str(exc)[:200],
            )


async def run_streaming_ttft_benchmark(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    requests: int,
    concurrency: int,
    warmup: int,
    timeout_s: float,
) -> SummaryStats:
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    connector = aiohttp.TCPConnector(
        limit=max(concurrency * 2, 10),
        limit_per_host=max(concurrency, 10),
        force_close=False,
    )
    semaphore = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        if warmup > 0:
            await asyncio.gather(
                *[
                    measure_stream_ttft(session, url, headers, payload, semaphore)
                    for _ in range(warmup)
                ]
            )
        wall_start = time.perf_counter()
        samples = await asyncio.gather(
            *[
                measure_stream_ttft(session, url, headers, payload, semaphore)
                for _ in range(requests)
            ]
        )
        wall_time_s = time.perf_counter() - wall_start
    return summarize(samples, wall_time_s)


async def measure_stream_full_response(
    session: aiohttp.ClientSession,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> RequestSample:
    async with semaphore:
        start = time.perf_counter()
        try:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    body = await response.read()
                    return RequestSample(
                        success=False,
                        latency_ms=(time.perf_counter() - start) * 1000,
                        status_code=response.status,
                        error=body.decode("utf-8", errors="ignore")[:200],
                    )

                saw_content = False
                while raw_line := await response.content.readline():
                    line = raw_line.strip()
                    if not line or not line.startswith(b"data:"):
                        continue
                    event_payload = line[5:].strip()
                    if event_payload == b"[DONE]":
                        return RequestSample(
                            success=saw_content,
                            latency_ms=(time.perf_counter() - start) * 1000,
                            status_code=response.status,
                            overhead_header_ms=extract_overhead_header(
                                response.headers
                            ),
                            error="" if saw_content else "stream ended without content",
                        )
                    if b'"content"' in event_payload or b'"text"' in event_payload:
                        saw_content = True

                return RequestSample(
                    success=False,
                    latency_ms=(time.perf_counter() - start) * 1000,
                    status_code=response.status,
                    error="stream ended before [DONE]",
                )
        except Exception as exc:
            return RequestSample(
                success=False,
                latency_ms=(time.perf_counter() - start) * 1000,
                status_code=0,
                error=str(exc)[:200],
            )


async def run_streaming_full_benchmark(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    requests: int,
    concurrency: int,
    warmup: int,
    timeout_s: float,
) -> SummaryStats:
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    connector = aiohttp.TCPConnector(
        limit=max(concurrency * 2, 10),
        limit_per_host=max(concurrency, 10),
        force_close=False,
    )
    semaphore = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        if warmup > 0:
            await asyncio.gather(
                *[
                    measure_stream_full_response(
                        session, url, headers, payload, semaphore
                    )
                    for _ in range(warmup)
                ]
            )
        wall_start = time.perf_counter()
        samples = await asyncio.gather(
            *[
                measure_stream_full_response(session, url, headers, payload, semaphore)
                for _ in range(requests)
            ]
        )
        wall_time_s = time.perf_counter() - wall_start
    return summarize(samples, wall_time_s)


def stats_to_dict(stats: SummaryStats) -> dict[str, Any]:
    return {
        "requests": stats.requests,
        "failures": stats.failures,
        "rps": stats.rps,
        "mean_ms": stats.mean_ms,
        "p50_ms": stats.p50_ms,
        "p95_ms": stats.p95_ms,
        "p99_ms": stats.p99_ms,
        "overhead_header_mean_ms": stats.overhead_header_mean_ms,
        "overhead_header_p50_ms": stats.overhead_header_p50_ms,
        "overhead_header_p95_ms": stats.overhead_header_p95_ms,
    }


def _median_run(
    runs: list[tuple[SummaryStats, SummaryStats, SummaryStats, Optional[SummaryStats]]],
) -> tuple[SummaryStats, SummaryStats, SummaryStats, Optional[SummaryStats]]:
    # Pick the run whose proxy non-stream p50 is the median across repeats.
    # Choosing a single representative run (rather than aggregating each metric
    # separately) keeps related metrics from the same execution context so
    # client-overhead deltas stay internally consistent.
    sorted_runs = sorted(runs, key=lambda r: r[1].p50_ms)
    return sorted_runs[len(sorted_runs) // 2]


def print_summary(
    label: str,
    revision: str,
    direct: SummaryStats,
    proxy: SummaryStats,
    stream: SummaryStats,
    stream_full: Optional[SummaryStats],
) -> None:
    client_overhead_p50 = proxy.p50_ms - direct.p50_ms
    client_overhead_p95 = proxy.p95_ms - direct.p95_ms
    print("\n=== Benchmark summary ===")
    print(f"Label: {label}")
    print(f"Revision: {revision}")
    print(f"Direct provider non-stream p50: {direct.p50_ms:.2f} ms")
    print(f"Proxy non-stream p50: {proxy.p50_ms:.2f} ms")
    print(f"Proxy non-stream p95: {proxy.p95_ms:.2f} ms")
    print(f"Proxy non-stream RPS: {proxy.rps:.2f}")
    print(f"Client-observed overhead p50: {client_overhead_p50:.2f} ms")
    print(f"Client-observed overhead p95: {client_overhead_p95:.2f} ms")
    print(
        "x-litellm-overhead-duration-ms p50: "
        f"{format_optional_ms(proxy.overhead_header_p50_ms)} ms"
    )
    print(f"Streaming TTFT p50: {stream.p50_ms:.2f} ms")
    print(f"Streaming TTFT p95: {stream.p95_ms:.2f} ms")
    print(f"Streaming TTFT RPS: {stream.rps:.2f}")
    if stream_full is not None:
        print(f"Streaming full response p50: {stream_full.p50_ms:.2f} ms")
        print(f"Streaming full response p95: {stream_full.p95_ms:.2f} ms")
        print(f"Streaming full response RPS: {stream_full.rps:.2f}")
    print("\nMarkdown row:")
    print(
        "| "
        + " | ".join(
            [
                label,
                revision,
                f"{stream.p50_ms:.2f}",
                f"{stream.p95_ms:.2f}",
                f"{proxy.rps:.2f}",
                f"{client_overhead_p50:.2f}",
                f"{client_overhead_p95:.2f}",
                format_optional_ms(proxy.overhead_header_p50_ms),
                f"{stream_full.p50_ms:.2f}" if stream_full is not None else "n/a",
                f"{stream_full.rps:.2f}" if stream_full is not None else "n/a",
            ]
        )
        + " |"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="current", help="Label for this run")
    parser.add_argument(
        "--litellm-dir",
        default=str(Path.cwd()),
        help="Checkout directory used to start the LiteLLM proxy",
    )
    parser.add_argument(
        "--proxy-command",
        default="uv run litellm",
        help="Command used to start the proxy inside --litellm-dir",
    )
    parser.add_argument("--proxy-host", default="127.0.0.1")
    parser.add_argument("--proxy-port", type=int, default=4000)
    parser.add_argument("--provider-host", default="127.0.0.1")
    parser.add_argument("--provider-port", type=int, default=8099)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=100)
    parser.add_argument("--stream-requests", type=int, default=200)
    parser.add_argument("--stream-concurrency", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--stream-warmup", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--proxy-start-timeout", type=float, default=90)
    parser.add_argument("--provider-first-token-delay-ms", type=float, default=0)
    parser.add_argument(
        "--provider-stream-content-chunks",
        type=int,
        default=20,
        help="Streaming chunks the mock provider emits. Default 20 (realistic).",
    )
    parser.add_argument(
        "--measure-full-stream",
        action="store_true",
        default=True,
        help="Measure time to consume the complete streaming response (on by default).",
    )
    parser.add_argument(
        "--no-measure-full-stream",
        dest="measure_full_stream",
        action="store_false",
        help="Skip the full-stream RPS measurement.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Run the entire suite N times against the same proxy and report the median run.",
    )
    parser.add_argument(
        "--no-start-proxy",
        action="store_true",
        help="Benchmark an already-running proxy at --proxy-host/--proxy-port",
    )
    parser.add_argument(
        "--provider-url",
        help="Use an already-running provider instead of starting the mock provider",
    )
    parser.add_argument("--output-json", help="Write machine-readable results")
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    litellm_dir = Path(args.litellm_dir).resolve()
    revision = get_git_revision(litellm_dir)
    proxy_base_url = f"http://{args.proxy_host}:{args.proxy_port}"
    proxy_url = f"{proxy_base_url}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {args.api_key}",
        "Content-Type": "application/json",
    }
    provider_headers = {
        "Authorization": "Bearer fake-provider-key",
        "Content-Type": "application/json",
    }
    non_stream_payload = {
        "model": DEFAULT_MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
    }
    stream_payload = {**non_stream_payload, "stream": True}

    provider: Optional[MockOpenAIProvider] = None
    proxy_process: Optional[subprocess.Popen] = None
    with tempfile.TemporaryDirectory(prefix="litellm-perf-") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        proxy_log_path = tmp_dir / "proxy.log"
        if args.provider_url:
            provider_base_url = args.provider_url.rstrip("/")
        else:
            provider = MockOpenAIProvider(
                host=args.provider_host,
                port=args.provider_port,
                first_token_delay_ms=args.provider_first_token_delay_ms,
                stream_content_chunks=args.provider_stream_content_chunks,
            )
            await provider.start()
            provider_base_url = provider.base_url

        config_path = tmp_dir / "config.yaml"
        write_proxy_config(config_path, provider_base_url, args.api_key)

        try:
            if not args.no_start_proxy:
                proxy_process = start_proxy_process(
                    litellm_dir=litellm_dir,
                    proxy_command=args.proxy_command,
                    config_path=config_path,
                    port=args.proxy_port,
                    log_path=proxy_log_path,
                )
            await wait_for_proxy(proxy_base_url, args.proxy_start_timeout)

            runs: list[
                tuple[
                    SummaryStats,
                    SummaryStats,
                    SummaryStats,
                    Optional[SummaryStats],
                ]
            ] = []
            for run_idx in range(max(1, args.repeats)):
                if args.repeats > 1:
                    print(f"\n--- Run {run_idx + 1}/{args.repeats} ---")
                _direct = await run_non_streaming_benchmark(
                    url=f"{provider_base_url}/v1/chat/completions",
                    headers=provider_headers,
                    payload=non_stream_payload,
                    requests=args.requests,
                    concurrency=args.concurrency,
                    warmup=args.warmup,
                    timeout_s=args.timeout,
                )
                _proxy = await run_non_streaming_benchmark(
                    url=proxy_url,
                    headers=headers,
                    payload=non_stream_payload,
                    requests=args.requests,
                    concurrency=args.concurrency,
                    warmup=args.warmup,
                    timeout_s=args.timeout,
                )
                _stream = await run_streaming_ttft_benchmark(
                    url=proxy_url,
                    headers=headers,
                    payload=stream_payload,
                    requests=args.stream_requests,
                    concurrency=args.stream_concurrency,
                    warmup=args.stream_warmup,
                    timeout_s=args.timeout,
                )
                _stream_full = (
                    await run_streaming_full_benchmark(
                        url=proxy_url,
                        headers=headers,
                        payload=stream_payload,
                        requests=args.stream_requests,
                        concurrency=args.stream_concurrency,
                        warmup=args.stream_warmup,
                        timeout_s=args.timeout,
                    )
                    if args.measure_full_stream
                    else None
                )
                runs.append((_direct, _proxy, _stream, _stream_full))
                if args.repeats > 1:
                    print(
                        f"  run {run_idx + 1}: non-stream p50={_proxy.p50_ms:.2f}ms "
                        f"rps={_proxy.rps:.2f} | TTFT p50={_stream.p50_ms:.2f}ms "
                        f"full RPS="
                        + (f"{_stream_full.rps:.2f}" if _stream_full else "n/a")
                    )

            direct, proxy, stream, stream_full = _median_run(runs)
        finally:
            if proxy_process is not None:
                stop_proxy_process(proxy_process)
            if provider is not None:
                await provider.stop()

        print_summary(args.label, revision, direct, proxy, stream, stream_full)

        if args.output_json:
            output = {
                "label": args.label,
                "revision": revision,
                "direct_non_streaming": stats_to_dict(direct),
                "proxy_non_streaming": stats_to_dict(proxy),
                "proxy_streaming_ttft": stats_to_dict(stream),
                "proxy_streaming_full": (
                    stats_to_dict(stream_full) if stream_full is not None else None
                ),
                "client_observed_overhead_p50_ms": proxy.p50_ms - direct.p50_ms,
                "client_observed_overhead_p95_ms": proxy.p95_ms - direct.p95_ms,
                "proxy_log_path": str(proxy_log_path),
            }
            Path(args.output_json).write_text(
                json.dumps(output, indent=2, sort_keys=True), encoding="utf-8"
            )


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
