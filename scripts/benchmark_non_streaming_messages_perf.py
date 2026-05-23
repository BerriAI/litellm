#!/usr/bin/env python3
"""Benchmark LiteLLM proxy /v1/messages (Anthropic Messages API) non-streaming.

Measures pure proxy overhead for non-streaming requests across three provider
paths: Anthropic native, Bedrock Invoke, and Bedrock Converse.

The script boots a local mock provider that returns a valid Anthropic
/v1/messages JSON response immediately, so measured latency reflects
LiteLLM overhead only, not real provider latency.

Metrics captured per run:
  * p50 / p95 / p99 total latency
  * Throughput (requests per second)
  * Failure rate

Example — baseline vs optimised:
    uv run python scripts/benchmark_non_streaming_messages_perf.py \\
        --label baseline --requests 500 --concurrency 20

Compare an already-running proxy:
    uv run python scripts/benchmark_non_streaming_messages_perf.py \\
        --no-start-proxy --label current

Provider paths are selected via --provider:
    anthropic    (default) anthropic/claude-perf-test
    bedrock-invoke         bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0
    bedrock-converse       bedrock_converse/anthropic.claude-3-5-sonnet-20241022-v2:0
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

DEFAULT_API_KEY = "sk-1234"

# Provider paths available for benchmarking
PROVIDER_CONFIGS = {
    "anthropic": {
        "model": "claude-perf-test",
        "litellm_params_model": "anthropic/claude-perf-test",
        "description": "Anthropic native path",
    },
    "bedrock-invoke": {
        "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "litellm_params_model": "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
        "description": "Bedrock Invoke native path",
    },
    "bedrock-converse": {
        "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "litellm_params_model": "bedrock_converse/anthropic.claude-3-5-sonnet-20241022-v2:0",
        "description": "Bedrock Converse adapter path",
    },
}


@dataclass
class NonStreamSample:
    success: bool
    total_ms: float
    status_code: int
    error: str = ""


@dataclass
class SummaryStats:
    requests: int
    failures: int
    rps: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float


class MockAnthropicProvider:
    """Minimal Anthropic /v1/messages provider returning a fixed non-streaming response."""

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.runner: Optional[web.AppRunner] = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    async def start(self) -> None:
        app = web.Application()
        app.router.add_post("/v1/messages", self._handle)
        # Bedrock invoke uses a different path
        app.router.add_post("/model/{model_id}/invoke", self._handle_bedrock_invoke)
        self.runner = web.AppRunner(app, access_log=None)
        await self.runner.setup()
        site = web.TCPSite(self.runner, self.host, self.port)
        await site.start()

    async def stop(self) -> None:
        if self.runner is not None:
            await self.runner.cleanup()

    async def _handle(self, request: web.Request) -> web.Response:
        body = await request.json()
        model = body.get("model", "claude-perf-test")
        return web.json_response(self._make_response(model))

    async def _handle_bedrock_invoke(self, request: web.Request) -> web.Response:
        body = await request.json()
        model = body.get("model", "claude-3-5-sonnet")
        return web.json_response(self._make_response(model))

    @staticmethod
    def _make_response(model: str) -> dict:
        return {
            "id": "msg_perf",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 8, "output_tokens": 1},
        }


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sv = sorted(values)
    idx = min(int(len(sv) * pct / 100), len(sv) - 1)
    return sv[idx]


def summarize(samples: list[NonStreamSample], wall_time_s: float) -> SummaryStats:
    ok = [s for s in samples if s.success]
    latencies = [s.total_ms for s in ok]
    return SummaryStats(
        requests=len(samples),
        failures=len(samples) - len(ok),
        rps=(len(ok) / wall_time_s) if wall_time_s > 0 else 0.0,
        p50_ms=percentile(latencies, 50),
        p95_ms=percentile(latencies, 95),
        p99_ms=percentile(latencies, 99),
        mean_ms=statistics.mean(latencies) if latencies else 0.0,
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


def write_proxy_config(
    config_path: Path,
    provider_base_url: str,
    api_key: str,
    provider: str,
) -> None:
    cfg = PROVIDER_CONFIGS[provider]
    model_name = cfg["model"]
    litellm_model = cfg["litellm_params_model"]

    if provider == "bedrock-invoke":
        # Bedrock invoke uses the direct bedrock-runtime endpoint style
        litellm_params = f"""
      model: {litellm_model}
      api_base: {provider_base_url}
      aws_access_key_id: fake-key
      aws_secret_access_key: fake-secret
      aws_region_name: us-east-1"""
    elif provider == "bedrock-converse":
        litellm_params = f"""
      model: {litellm_model}
      api_base: {provider_base_url}
      aws_access_key_id: fake-key
      aws_secret_access_key: fake-secret
      aws_region_name: us-east-1"""
    else:
        litellm_params = f"""
      model: {litellm_model}
      api_key: fake-provider-key
      api_base: {provider_base_url}"""

    config_path.write_text(
        f"""model_list:
  - model_name: {model_name}
    litellm_params:{litellm_params}

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
                async with session.get(f"{base_url}/health/liveliness") as resp:
                    if resp.status < 500:
                        return
                    last_error = f"HTTP {resp.status}"
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
    env = {**os.environ, "LITELLM_TELEMETRY": "False", "PYTHONUNBUFFERED": "1"}
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


async def measure_non_stream(
    session: aiohttp.ClientSession,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
) -> NonStreamSample:
    start = time.perf_counter()
    try:
        async with session.post(url, headers=headers, json=payload) as response:
            body = await response.read()
            total_ms = (time.perf_counter() - start) * 1000
            if response.status != 200:
                return NonStreamSample(
                    success=False,
                    total_ms=total_ms,
                    status_code=response.status,
                    error=body.decode("utf-8", errors="ignore")[:200],
                )
            return NonStreamSample(success=True, total_ms=total_ms, status_code=200)
    except Exception as exc:
        return NonStreamSample(
            success=False,
            total_ms=(time.perf_counter() - start) * 1000,
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
        sink: list[NonStreamSample],
    ) -> None:
        while True:
            idx = counter[0]
            if idx >= budget:
                return
            counter[0] = idx + 1
            sink.append(await measure_non_stream(session, url, headers, payload))

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        if warmup > 0:
            wc = [0]
            await asyncio.gather(
                *[worker(session, wc, warmup, []) for _ in range(concurrency)]
            )
        samples: list[NonStreamSample] = []
        counter = [0]
        wall_start = time.perf_counter()
        await asyncio.gather(
            *[worker(session, counter, requests, samples) for _ in range(concurrency)]
        )
        wall_time_s = time.perf_counter() - wall_start

    return summarize(samples, wall_time_s)


def stats_row(label: str, revision: str, provider: str, stats: SummaryStats) -> str:
    return (
        f"| {label} | {revision} | {provider} | "
        f"{stats.p50_ms:.2f} | {stats.p95_ms:.2f} | {stats.p99_ms:.2f} | "
        f"{stats.mean_ms:.2f} | {stats.rps:.2f} | {stats.failures} |"
    )


def print_summary(
    label: str, revision: str, provider: str, stats: SummaryStats
) -> None:
    desc = PROVIDER_CONFIGS[provider]["description"]
    print("\n=== LiteLLM /v1/messages non-streaming benchmark ===")
    print(f"Label:     {label}")
    print(f"Revision:  {revision}")
    print(f"Provider:  {provider} ({desc})")
    print(f"Requests:  {stats.requests}  Failures: {stats.failures}")
    print(f"p50:       {stats.p50_ms:.2f} ms")
    print(f"p95:       {stats.p95_ms:.2f} ms")
    print(f"p99:       {stats.p99_ms:.2f} ms")
    print(f"mean:      {stats.mean_ms:.2f} ms")
    print(f"RPS:       {stats.rps:.2f}")
    print("\nMarkdown table row:")
    print(
        "| label | revision | provider | p50_ms | p95_ms | p99_ms | mean_ms | rps | failures |"
    )
    print(
        "|-------|----------|----------|--------|--------|--------|---------|-----|----------|"
    )
    print(stats_row(label, revision, provider, stats))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", default="current")
    p.add_argument("--litellm-dir", default=str(Path.cwd()))
    p.add_argument("--proxy-command", default="uv run litellm")
    p.add_argument("--proxy-host", default="127.0.0.1")
    p.add_argument("--proxy-port", type=int, default=4000)
    p.add_argument("--provider-host", default="127.0.0.1")
    p.add_argument("--provider-port", type=int, default=8099)
    p.add_argument("--api-key", default=DEFAULT_API_KEY)
    p.add_argument("--requests", type=int, default=300)
    p.add_argument("--concurrency", type=int, default=20)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--timeout", type=float, default=30)
    p.add_argument("--proxy-start-timeout", type=float, default=90)
    p.add_argument(
        "--provider",
        choices=list(PROVIDER_CONFIGS.keys()),
        default="anthropic",
        help="Provider path to benchmark",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Run N times against the same proxy; report the median run.",
    )
    p.add_argument("--no-start-proxy", action="store_true")
    p.add_argument("--provider-url", help="Use an already-running provider")
    p.add_argument("--output-json", help="Write machine-readable results to this path")
    return p.parse_args()


async def async_main() -> None:
    args = parse_args()
    litellm_dir = Path(args.litellm_dir).resolve()
    revision = get_git_revision(litellm_dir)
    proxy_base_url = f"http://{args.proxy_host}:{args.proxy_port}"
    proxy_url = f"{proxy_base_url}/v1/messages"
    cfg = PROVIDER_CONFIGS[args.provider]
    headers = {
        "Authorization": f"Bearer {args.api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": cfg["model"],
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "hi"}],
        # non-streaming: no "stream" key (or stream=false)
    }

    provider: Optional[MockAnthropicProvider] = None
    proxy_process: Optional[subprocess.Popen] = None

    with tempfile.TemporaryDirectory(prefix="litellm-nonstream-perf-") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        proxy_log_path = tmp_dir / "proxy.log"

        if args.provider_url:
            provider_base_url = args.provider_url.rstrip("/")
        else:
            provider = MockAnthropicProvider(
                host=args.provider_host, port=args.provider_port
            )
            await provider.start()
            provider_base_url = provider.base_url

        config_path = tmp_dir / "config.yaml"
        write_proxy_config(config_path, provider_base_url, args.api_key, args.provider)

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
                    payload=payload,
                    requests=args.requests,
                    concurrency=args.concurrency,
                    warmup=args.warmup,
                    timeout_s=args.timeout,
                )
                runs.append(stats)
                if args.repeats > 1:
                    print(
                        f"  run {run_idx + 1}: p50={stats.p50_ms:.2f}ms "
                        f"p99={stats.p99_ms:.2f}ms RPS={stats.rps:.2f}"
                    )

            stats = sorted(runs, key=lambda s: s.p50_ms)[len(runs) // 2]
        finally:
            if proxy_process is not None:
                stop_proxy_process(proxy_process)
            if provider is not None:
                await provider.stop()

        print_summary(args.label, revision, args.provider, stats)

        if args.output_json:
            output = {
                "label": args.label,
                "revision": revision,
                "provider": args.provider,
                "non_streaming": {
                    "requests": stats.requests,
                    "failures": stats.failures,
                    "rps": stats.rps,
                    "p50_ms": stats.p50_ms,
                    "p95_ms": stats.p95_ms,
                    "p99_ms": stats.p99_ms,
                    "mean_ms": stats.mean_ms,
                },
            }
            Path(args.output_json).write_text(
                json.dumps(output, indent=2, sort_keys=True), encoding="utf-8"
            )


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
