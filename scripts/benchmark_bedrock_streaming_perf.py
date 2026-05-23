#!/usr/bin/env python3
"""Benchmark LiteLLM proxy Bedrock Converse + Invoke streaming hot paths.

Measures the two metrics that matter for an interactive streaming proxy:

  * TTFT  - time to first streamed token (first ``data:`` chunk with content)
  * TPM   - sustained output token throughput (tokens/second) once the full
            stream is consumed, plus request throughput (RPS)

It boots a local mock Bedrock endpoint that speaks the real AWS Event Stream
binary framing format (contentBlockDelta events) and a LiteLLM proxy from any
checkout, so commits/branches can be compared without AWS credentials or
network access.

Example — Bedrock Converse:
    uv run python scripts/benchmark_bedrock_streaming_perf.py \\
        --label baseline --variant converse

Example — Bedrock Invoke (Nova):
    uv run python scripts/benchmark_bedrock_streaming_perf.py \\
        --label baseline --variant invoke

Compare an already-running proxy:
    uv run python scripts/benchmark_bedrock_streaming_perf.py \\
        --no-start-proxy --label current --variant converse
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import json
import os
import shlex
import signal
import statistics
import struct
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import aiohttp
from aiohttp import web

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_API_KEY = "sk-1234"
CONVERSE_MODEL = "bedrock-perf-converse"
INVOKE_MODEL = "bedrock-perf-invoke"


# ---------------------------------------------------------------------------
# AWS Event Stream binary frame encoder
# ---------------------------------------------------------------------------


def _encode_string_headers(headers: dict) -> bytes:
    """Encode a dict of string→string headers into AWS Event Stream header bytes."""
    buf = b""
    for name, value in headers.items():
        name_b = name.encode("utf-8")
        value_b = value.encode("utf-8")
        buf += struct.pack("!B", len(name_b))  # 1-byte name length
        buf += name_b
        buf += struct.pack("!B", 7)  # header value type: string (7)
        buf += struct.pack("!H", len(value_b))  # 2-byte value length
        buf += value_b
    return buf


def encode_bedrock_chunk_event(converse_payload: dict) -> bytes:
    """
    Encode a Bedrock converse payload dict as a single AWS Event Stream frame.

    The outer frame is a ``chunk`` event whose JSON payload wraps the converse
    JSON as a base64-encoded ``bytes`` field — matching the format that
    botocore's ``EventStreamJSONParser`` + the litellm
    ``_parse_message_from_event`` helper expect.
    """
    converse_json = json.dumps(converse_payload).encode("utf-8")
    outer_payload = json.dumps(
        {"bytes": base64.b64encode(converse_json).decode()}
    ).encode("utf-8")

    headers = {
        ":event-type": "chunk",
        ":content-type": "application/json",
        ":message-type": "event",
    }
    header_bytes = _encode_string_headers(headers)

    total_len = 4 + 4 + 4 + len(header_bytes) + len(outer_payload) + 4
    prelude = struct.pack("!II", total_len, len(header_bytes))
    prelude_crc = struct.pack("!I", binascii.crc32(prelude) & 0xFFFFFFFF)
    msg_body = prelude + prelude_crc + header_bytes + outer_payload
    msg_crc = struct.pack("!I", binascii.crc32(msg_body) & 0xFFFFFFFF)
    return msg_body + msg_crc


def _converse_stream_frames(n_text_chunks: int) -> List[bytes]:
    """Return a list of binary event stream frames for a Converse streaming response."""
    frames: List[bytes] = []
    # Content block start (text block)
    frames.append(encode_bedrock_chunk_event({"contentBlockIndex": 0, "start": {}}))
    # N text delta chunks
    for _ in range(n_text_chunks):
        frames.append(
            encode_bedrock_chunk_event(
                {"contentBlockIndex": 0, "delta": {"text": "hi "}}
            )
        )
    # Content block stop
    frames.append(encode_bedrock_chunk_event({"contentBlockIndex": 0}))
    # Stop reason
    frames.append(encode_bedrock_chunk_event({"stopReason": "end_turn"}))
    # Usage (metadata)
    frames.append(
        encode_bedrock_chunk_event(
            {
                "usage": {
                    "inputTokens": 8,
                    "outputTokens": n_text_chunks,
                    "totalTokens": n_text_chunks + 8,
                }
            }
        )
    )
    return frames


def _invoke_nova_stream_frames(n_text_chunks: int) -> List[bytes]:
    """
    Return binary event stream frames for a Bedrock Invoke (Nova) streaming
    response.  Nova wraps each content delta under ``contentBlockDelta``.
    """
    frames: List[bytes] = []
    for _ in range(n_text_chunks):
        frames.append(
            encode_bedrock_chunk_event(
                {
                    "contentBlockDelta": {
                        "contentBlockIndex": 0,
                        "delta": {"text": "hi "},
                    }
                }
            )
        )
    frames.append(
        encode_bedrock_chunk_event(
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"stopReason": "end_turn"},
                }
            }
        )
    )
    return frames


# ---------------------------------------------------------------------------
# Sample / stats types
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Mock Bedrock provider
# ---------------------------------------------------------------------------


class MockBedrockProvider:
    """
    Minimal Bedrock Converse/Invoke streaming endpoint.

    Returns real AWS Event Stream binary frames over an HTTP chunked response.
    The LiteLLM proxy connects here instead of AWS, giving apples-to-apples
    overhead measurements.
    """

    def __init__(
        self,
        host: str,
        port: int,
        stream_content_chunks: int,
        first_token_delay_ms: float,
        variant: str,
    ) -> None:
        self.host = host
        self.port = port
        self.stream_content_chunks = stream_content_chunks
        self.first_token_delay_ms = first_token_delay_ms
        self.variant = variant  # "converse" or "invoke"
        self.runner: Optional[web.AppRunner] = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    async def start(self) -> None:
        app = web.Application()
        # Bedrock Converse stream: POST /model/{modelId}/converse-stream
        app.router.add_post(
            r"/model/{model_id:.*}/converse-stream", self.handle_converse_stream
        )
        # Bedrock Invoke stream: POST /model/{modelId}/invoke-with-response-stream
        app.router.add_post(
            r"/model/{model_id:.*}/invoke-with-response-stream",
            self.handle_invoke_stream,
        )
        self.runner = web.AppRunner(app, access_log=None)
        await self.runner.setup()
        site = web.TCPSite(self.runner, self.host, self.port)
        await site.start()

    async def stop(self) -> None:
        if self.runner is not None:
            await self.runner.cleanup()

    async def _send_frames(
        self, request: web.Request, frames: List[bytes]
    ) -> web.StreamResponse:
        response = web.StreamResponse(
            status=200,
            headers={"Content-Type": "application/vnd.amazon.eventstream"},
        )
        await response.prepare(request)
        if self.first_token_delay_ms > 0:
            await asyncio.sleep(self.first_token_delay_ms / 1000)
        for frame in frames:
            await response.write(frame)
        await response.write_eof()
        return response

    async def handle_converse_stream(self, request: web.Request) -> web.StreamResponse:
        frames = _converse_stream_frames(self.stream_content_chunks)
        return await self._send_frames(request, frames)

    async def handle_invoke_stream(self, request: web.Request) -> web.StreamResponse:
        frames = _invoke_nova_stream_frames(self.stream_content_chunks)
        return await self._send_frames(request, frames)


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    sv = sorted(values)
    idx = min(int(len(sv) * pct / 100), len(sv) - 1)
    return sv[idx]


def summarize(samples: List[StreamSample], wall_time_s: float) -> SummaryStats:
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
        tokens_per_sec=(total_tokens / wall_time_s) if wall_time_s > 0 else 0.0,
    )


# ---------------------------------------------------------------------------
# LiteLLM proxy management
# ---------------------------------------------------------------------------


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
    variant: str,
) -> None:
    """Write a minimal litellm proxy config for benchmarking Bedrock."""
    if variant == "converse":
        # Bedrock Converse: use model prefix bedrock/converse/
        model_name = CONVERSE_MODEL
        litellm_model = "bedrock/converse/anthropic.claude-3-sonnet-20240229-v1:0"
    else:
        # Bedrock Invoke (Nova): use bedrock/ prefix
        model_name = INVOKE_MODEL
        litellm_model = "bedrock/amazon.nova-pro-v1:0"

    config_path.write_text(
        f"""model_list:
  - model_name: {model_name}
    litellm_params:
      model: {litellm_model}
      aws_access_key_id: fake-key-id
      aws_secret_access_key: fake-secret-key
      aws_region_name: us-east-1
      aws_bedrock_runtime_endpoint: {provider_base_url}

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


# ---------------------------------------------------------------------------
# Load measurement
# ---------------------------------------------------------------------------


async def measure_stream(
    session: aiohttp.ClientSession,
    url: str,
    headers: dict,
    payload: dict,
) -> StreamSample:
    """Send one streaming chat-completions request and record TTFT + token count."""
    start = time.perf_counter()
    ttft_ms = 0.0
    output_tokens = 0
    try:
        async with session.post(url, headers=headers, json=payload) as resp:
            if resp.status != 200:
                body = await resp.read()
                return StreamSample(
                    success=False,
                    ttft_ms=0.0,
                    total_ms=(time.perf_counter() - start) * 1000,
                    output_tokens=0,
                    status_code=resp.status,
                    error=body.decode("utf-8", errors="ignore")[:300],
                )
            async for raw_line in resp.content:
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
                choices = event.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        if ttft_ms == 0.0:
                            ttft_ms = (time.perf_counter() - start) * 1000
                        output_tokens += 1
        total_ms = (time.perf_counter() - start) * 1000
        if ttft_ms == 0.0:
            return StreamSample(
                success=False,
                ttft_ms=0.0,
                total_ms=total_ms,
                output_tokens=0,
                status_code=200,
                error="stream ended before a content token",
            )
        return StreamSample(
            success=True,
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            output_tokens=output_tokens,
            status_code=200,
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
    headers: dict,
    payload: dict,
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
        counter: list,
        budget: int,
        sink: list,
    ) -> None:
        while True:
            idx = counter[0]
            if idx >= budget:
                return
            counter[0] = idx + 1
            sink.append(await measure_stream(session, url, headers, payload))

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        if warmup > 0:
            wc = [0]
            await asyncio.gather(
                *[worker(session, wc, warmup, []) for _ in range(concurrency)]
            )
        samples: list = []
        counter = [0]
        wall_start = time.perf_counter()
        await asyncio.gather(
            *[worker(session, counter, requests, samples) for _ in range(concurrency)]
        )
        wall_time_s = time.perf_counter() - wall_start
    return summarize(samples, wall_time_s)


# ---------------------------------------------------------------------------
# Output / reporting
# ---------------------------------------------------------------------------


def stats_to_dict(stats: SummaryStats) -> dict:
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


def print_summary(label: str, revision: str, variant: str, stats: SummaryStats) -> None:
    variant_label = (
        "Bedrock Converse" if variant == "converse" else "Bedrock Invoke (Nova)"
    )
    print(f"\n=== {variant_label} streaming benchmark ===")
    print(f"Label:    {label}")
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
                f"{stats.ttft_p99_ms:.2f}",
                f"{stats.tokens_per_sec:.1f}",
                f"{stats.rps:.2f}",
            ]
        )
        + " |"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", default="current")
    p.add_argument(
        "--variant",
        choices=["converse", "invoke"],
        default="converse",
        help="Which Bedrock path to benchmark (default: converse)",
    )
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
    p.add_argument("--provider-first-token-delay-ms", type=float, default=0)
    p.add_argument(
        "--provider-stream-content-chunks",
        type=int,
        default=256,
        help="Number of text delta chunks the mock emits (default 256).",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Run the suite N times against the same proxy; report the median run.",
    )
    p.add_argument("--no-start-proxy", action="store_true")
    p.add_argument(
        "--provider-url", help="Use an already-running Bedrock-compatible endpoint"
    )
    p.add_argument("--output-json", help="Write machine-readable results to this file")
    return p.parse_args()


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
    model_name = CONVERSE_MODEL if args.variant == "converse" else INVOKE_MODEL
    stream_payload = {
        "model": model_name,
        "max_tokens": args.provider_stream_content_chunks + 16,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }

    provider: Optional[MockBedrockProvider] = None
    proxy_process: Optional[subprocess.Popen] = None

    with tempfile.TemporaryDirectory(prefix="litellm-bedrock-perf-") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        proxy_log_path = tmp_dir / "proxy.log"

        if args.provider_url:
            provider_base_url = args.provider_url.rstrip("/")
        else:
            provider = MockBedrockProvider(
                host=args.provider_host,
                port=args.provider_port,
                stream_content_chunks=args.provider_stream_content_chunks,
                first_token_delay_ms=args.provider_first_token_delay_ms,
                variant=args.variant,
            )
            await provider.start()
            provider_base_url = provider.base_url
            print(f"Mock Bedrock provider started at {provider_base_url}")

        if not args.no_start_proxy:
            config_path = tmp_dir / "config.yaml"
            write_proxy_config(
                config_path, provider_base_url, args.api_key, args.variant
            )
            proxy_process = start_proxy_process(
                litellm_dir=litellm_dir,
                proxy_command=args.proxy_command,
                config_path=config_path,
                port=args.proxy_port,
                log_path=proxy_log_path,
            )
            print(f"Proxy started (pid={proxy_process.pid}), waiting for readiness …")
            try:
                await wait_for_proxy(proxy_base_url, args.proxy_start_timeout)
                print("Proxy ready.")
            except TimeoutError as exc:
                print(f"ERROR: {exc}")
                if proxy_process:
                    stop_proxy_process(proxy_process)
                return

        try:
            all_stats: list = []
            for run_idx in range(args.repeats):
                if args.repeats > 1:
                    print(f"\nRun {run_idx + 1}/{args.repeats} …")
                stats = await run_benchmark(
                    url=proxy_url,
                    headers=headers,
                    payload=stream_payload,
                    requests=args.requests,
                    concurrency=args.concurrency,
                    warmup=args.warmup,
                    timeout_s=args.timeout,
                )
                all_stats.append(stats)

            # Pick median run by TTFT p50
            best = sorted(all_stats, key=lambda s: s.ttft_p50_ms)[len(all_stats) // 2]
            print_summary(args.label, revision, args.variant, best)

            if args.output_json:
                import json as _json

                Path(args.output_json).write_text(
                    _json.dumps(
                        {
                            "label": args.label,
                            "revision": revision,
                            "variant": args.variant,
                            **stats_to_dict(best),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                print(f"\nResults written to {args.output_json}")
        finally:
            if proxy_process:
                stop_proxy_process(proxy_process)
            if provider:
                await provider.stop()


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
