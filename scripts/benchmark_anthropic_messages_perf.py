#!/usr/bin/env python3
"""Benchmark LiteLLM proxy /v1/messages (Anthropic Messages API) streaming.

Measures the two metrics that matter for an interactive streaming proxy:

  * TTFT  - time to first streamed token (first ``content_block_delta``)
  * TPM   - sustained output token throughput (tokens / second) once the
            full stream is consumed, plus request throughput (RPS)

It boots a local mock Anthropic provider that speaks the real Anthropic
streaming SSE wire format (``message_start`` -> ``content_block_delta`` ->
``message_stop``) and a LiteLLM proxy from any checkout, so commits/branches
can be compared without depending on real provider latency.

Example:
    uv run python scripts/benchmark_anthropic_messages_perf.py \
        --label baseline --proxy-command ".venv/bin/litellm"

Compare an already-running proxy:
    uv run python scripts/benchmark_anthropic_messages_perf.py \
        --no-start-proxy --label current
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
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import aiohttp
from aiohttp import web

DEFAULT_MODEL = "claude-perf-test"
DEFAULT_API_KEY = "sk-1234"


@dataclass
class StreamSample:
    success: bool
    ttft_ms: float
    total_ms: float
    output_tokens: int
    status_code: int
    error: str = ""


@dataclass
class SummaryStats:
    requests: int
    failures: int
    rps: float
    ttft_mean_ms: float
    ttft_p50_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    total_p50_ms: float
    total_p95_ms: float
    tokens_per_sec: float


class MockAnthropicProvider:
    """Minimal Anthropic Messages API server (real streaming SSE format)."""

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
        app.router.add_post("/v1/messages", self.handle_messages)
        self.runner = web.AppRunner(app, access_log=None)
        await self.runner.setup()
        site = web.TCPSite(self.runner, self.host, self.port)
        await site.start()

    async def stop(self) -> None:
        if self.runner is not None:
            await self.runner.cleanup()

    async def handle_messages(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        if body.get("stream"):
            return await self._streaming_response(request, body)
        return self._json_response(body)

    def _json_response(self, body: dict[str, Any]) -> web.Response:
        payload = {
            "id": "msg_perf",
            "type": "message",
            "role": "assistant",
            "model": body.get("model", DEFAULT_MODEL),
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 8, "output_tokens": 1},
        }
        return web.json_response(payload)

    @staticmethod
    def _sse(event: str, data: dict[str, Any]) -> bytes:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()

    async def _streaming_response(
        self, request: web.Request, body: dict[str, Any]
    ) -> web.StreamResponse:
        model = body.get("model", DEFAULT_MODEL)
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
            },
        )
        await response.prepare(request)

        await response.write(
            self._sse(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_perf",
                        "type": "message",
                        "role": "assistant",
                        "model": model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 8, "output_tokens": 0},
                    },
                },
            )
        )
        await response.write(
            self._sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            )
        )

        if self.first_token_delay_ms > 0:
            await asyncio.sleep(self.first_token_delay_ms / 1000)

        for _ in range(self.stream_content_chunks):
            await response.write(
                self._sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": "hello "},
                    },
                )
            )

        await response.write(
            self._sse("content_block_stop", {"type": "content_block_stop", "index": 0})
        )
        await response.write(
            self._sse(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": self.stream_content_chunks},
                },
            )
        )
        await response.write(self._sse("message_stop", {"type": "message_stop"}))
        await response.write_eof()
        return response


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(int(len(sorted_values) * pct / 100), len(sorted_values) - 1)
    return sorted_values[index]


def summarize(samples: list[StreamSample], wall_time_s: float) -> SummaryStats:
    ok = [s for s in samples if s.success]
    ttfts = [s.ttft_ms for s in ok]
    totals = [s.total_ms for s in ok]
    total_tokens = sum(s.output_tokens for s in ok)
    return SummaryStats(
        requests=len(samples),
        failures=len(samples) - len(ok),
        rps=(len(ok) / wall_time_s) if wall_time_s > 0 else 0.0,
        ttft_mean_ms=statistics.mean(ttfts) if ttfts else 0.0,
        ttft_p50_ms=percentile(ttfts, 50),
        ttft_p95_ms=percentile(ttfts, 95),
        ttft_p99_ms=percentile(ttfts, 99),
        total_p50_ms=percentile(totals, 50),
        total_p95_ms=percentile(totals, 95),
        # Aggregate output-token throughput: total tokens delivered across all
        # successful requests divided by wall-clock time. This is the true
        # server TPM and (unlike tokens / summed-per-request-latency) scales
        # correctly with concurrency.
        tokens_per_sec=(total_tokens / wall_time_s) if wall_time_s > 0 else 0.0,
    )


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
      model: anthropic/{DEFAULT_MODEL}
      api_key: fake-provider-key
      api_base: {provider_base_url}

general_settings:
  master_key: {api_key}

litellm_settings:
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
                async with session.get(f"{base_url}/health/liveliness") as response:
                    if response.status < 500:
                        return
                    last_error = f"HTTP {response.status}"
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


async def measure_stream(
    session: aiohttp.ClientSession,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
) -> StreamSample:
    start = time.perf_counter()
    ttft_ms = 0.0
    output_tokens = 0
    try:
        async with session.post(url, headers=headers, json=payload) as response:
            if response.status != 200:
                body = await response.read()
                return StreamSample(
                    success=False,
                    ttft_ms=0.0,
                    total_ms=(time.perf_counter() - start) * 1000,
                    output_tokens=0,
                    status_code=response.status,
                    error=body.decode("utf-8", errors="ignore")[:200],
                )
            async for raw_line in response.content:
                line = raw_line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                etype = event.get("type")
                if etype == "content_block_delta":
                    if ttft_ms == 0.0:
                        ttft_ms = (time.perf_counter() - start) * 1000
                    output_tokens += 1
                elif etype == "message_stop":
                    break
        total_ms = (time.perf_counter() - start) * 1000
        if ttft_ms == 0.0:
            return StreamSample(
                success=False,
                ttft_ms=0.0,
                total_ms=total_ms,
                output_tokens=0,
                status_code=response.status,
                error="stream ended before a content token",
            )
        return StreamSample(
            success=True,
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            output_tokens=output_tokens,
            status_code=response.status,
        )
    except Exception as exc:
        return StreamSample(
            success=False,
            ttft_ms=0.0,
            total_ms=(time.perf_counter() - start) * 1000,
            output_tokens=0,
            status_code=0,
            error=str(exc)[:200],
        )


async def run_benchmark(
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

    async def worker(
        session: aiohttp.ClientSession,
        counter: list[int],
        budget: int,
        sink: list[StreamSample],
    ) -> None:
        # Steady-state load: exactly `concurrency` workers, each pulling the
        # next request slot as soon as its previous one finishes. Keeps
        # in-flight concurrency constant (vs. a gather-all + semaphore burst)
        # which removes the thundering-herd variance that otherwise swamps a
        # 10% signal.
        while True:
            idx = counter[0]
            if idx >= budget:
                return
            counter[0] = idx + 1
            sink.append(await measure_stream(session, url, headers, payload))

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        if warmup > 0:
            wcounter = [0]
            await asyncio.gather(
                *[worker(session, wcounter, warmup, []) for _ in range(concurrency)]
            )
        samples: list[StreamSample] = []
        counter = [0]
        wall_start = time.perf_counter()
        await asyncio.gather(
            *[worker(session, counter, requests, samples) for _ in range(concurrency)]
        )
        wall_time_s = time.perf_counter() - wall_start
    return summarize(samples, wall_time_s)


def stats_to_dict(stats: SummaryStats) -> dict[str, Any]:
    return {
        "requests": stats.requests,
        "failures": stats.failures,
        "rps": stats.rps,
        "ttft_mean_ms": stats.ttft_mean_ms,
        "ttft_p50_ms": stats.ttft_p50_ms,
        "ttft_p95_ms": stats.ttft_p95_ms,
        "ttft_p99_ms": stats.ttft_p99_ms,
        "total_p50_ms": stats.total_p50_ms,
        "total_p95_ms": stats.total_p95_ms,
        "tokens_per_sec": stats.tokens_per_sec,
    }


def print_summary(label: str, revision: str, stats: SummaryStats) -> None:
    print("\n=== Anthropic /v1/messages streaming benchmark ===")
    print(f"Label: {label}")
    print(f"Revision: {revision}")
    print(f"Requests: {stats.requests}  Failures: {stats.failures}")
    print(f"TTFT mean:  {stats.ttft_mean_ms:.2f} ms")
    print(f"TTFT p50:   {stats.ttft_p50_ms:.2f} ms")
    print(f"TTFT p95:   {stats.ttft_p95_ms:.2f} ms")
    print(f"TTFT p99:   {stats.ttft_p99_ms:.2f} ms")
    print(f"Full p50:   {stats.total_p50_ms:.2f} ms")
    print(f"Full p95:   {stats.total_p95_ms:.2f} ms")
    print(f"Throughput: {stats.rps:.2f} req/s")
    print(f"TPM:        {stats.tokens_per_sec:.1f} output tokens/s")
    print("\nMarkdown row:")
    print(
        "| "
        + " | ".join(
            [
                label,
                revision,
                f"{stats.ttft_p50_ms:.2f}",
                f"{stats.ttft_p95_ms:.2f}",
                f"{stats.tokens_per_sec:.1f}",
                f"{stats.rps:.2f}",
            ]
        )
        + " |"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="current")
    parser.add_argument("--litellm-dir", default=str(Path.cwd()))
    parser.add_argument("--proxy-command", default="uv run litellm")
    parser.add_argument("--proxy-host", default="127.0.0.1")
    parser.add_argument("--proxy-port", type=int, default=4000)
    parser.add_argument("--provider-host", default="127.0.0.1")
    parser.add_argument("--provider-port", type=int, default=8098)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--proxy-start-timeout", type=float, default=90)
    parser.add_argument("--provider-first-token-delay-ms", type=float, default=0)
    parser.add_argument(
        "--provider-stream-content-chunks",
        type=int,
        default=64,
        help="Number of text delta chunks the mock emits (default 64).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Run the suite N times against the same proxy; report the median run.",
    )
    parser.add_argument(
        "--no-start-proxy",
        action="store_true",
        help="Benchmark an already-running proxy at --proxy-host/--proxy-port",
    )
    parser.add_argument(
        "--provider-url",
        help="Use an already-running Anthropic-compatible provider",
    )
    parser.add_argument("--output-json", help="Write machine-readable results")
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    litellm_dir = Path(args.litellm_dir).resolve()
    revision = get_git_revision(litellm_dir)
    proxy_base_url = f"http://{args.proxy_host}:{args.proxy_port}"
    proxy_url = f"{proxy_base_url}/v1/messages"
    headers = {
        "Authorization": f"Bearer {args.api_key}",
        "Content-Type": "application/json",
    }
    stream_payload = {
        "model": DEFAULT_MODEL,
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }

    provider: Optional[MockAnthropicProvider] = None
    proxy_process: Optional[subprocess.Popen] = None
    with tempfile.TemporaryDirectory(prefix="litellm-anthropic-perf-") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        proxy_log_path = tmp_dir / "proxy.log"
        if args.provider_url:
            provider_base_url = args.provider_url.rstrip("/")
        else:
            provider = MockAnthropicProvider(
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

            runs: list[SummaryStats] = []
            for run_idx in range(max(1, args.repeats)):
                if args.repeats > 1:
                    print(f"\n--- Run {run_idx + 1}/{args.repeats} ---")
                stats = await run_benchmark(
                    url=proxy_url,
                    headers=headers,
                    payload=stream_payload,
                    requests=args.requests,
                    concurrency=args.concurrency,
                    warmup=args.warmup,
                    timeout_s=args.timeout,
                )
                runs.append(stats)
                if args.repeats > 1:
                    print(
                        f"  run {run_idx + 1}: TTFT p50={stats.ttft_p50_ms:.2f}ms "
                        f"TPM={stats.tokens_per_sec:.1f} tok/s RPS={stats.rps:.2f}"
                    )

            stats = sorted(runs, key=lambda s: s.ttft_p50_ms)[len(runs) // 2]
        finally:
            if proxy_process is not None:
                stop_proxy_process(proxy_process)
            if provider is not None:
                await provider.stop()

        print_summary(args.label, revision, stats)

        if args.output_json:
            Path(args.output_json).write_text(
                json.dumps(
                    {
                        "label": args.label,
                        "revision": revision,
                        "proxy_streaming": stats_to_dict(stats),
                        "proxy_log_path": str(proxy_log_path),
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
