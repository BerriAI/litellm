#!/usr/bin/env python3
"""Benchmark CustomStreamWrapper per-chunk overhead.

Drives CustomStreamWrapper directly with synthetic in-memory chunks for
Anthropic (GenericStreamingChunk), Bedrock Invoke (GenericStreamingChunk),
and Bedrock Converse (ModelResponseStream). A full proxy benchmark adds
FastAPI, HTTP, and TCP latency, which dilutes the per-chunk CPU signal.

Example:
    uv run python scripts/benchmark_streaming_chunk_overhead.py \\
        --streams 500 --chunks 200 --warmup 50 --repeats 5
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import logging
import os
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional
from unittest.mock import MagicMock

# Silence litellm's "Provider List" warnings emitted by get_llm_provider
# when it sees synthetic model names — we're not exercising provider
# routing, only the per-chunk wrapper hot path.
os.environ.setdefault("LITELLM_LOG", "ERROR")
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

import litellm  # noqa: E402

litellm.suppress_debug_info = True

from litellm.litellm_core_utils.streaming_handler import (
    CustomStreamWrapper,
)  # noqa: E402
from litellm.types.utils import (  # noqa: E402
    Delta,
    GenericStreamingChunk as GChunk,
    ModelResponseStream,
    StreamingChoices,
    Usage,
)

# ---------------------------------------------------------------------------
# Synthetic chunk fixtures
# ---------------------------------------------------------------------------


def _make_logging_obj(provider: str) -> MagicMock:
    logging_obj = MagicMock()
    logging_obj.model_call_details = {
        "custom_llm_provider": provider,
        "litellm_params": {},
    }
    logging_obj.call_type = "completion"
    logging_obj.stream_options = None
    logging_obj.messages = [{"role": "user", "content": "hi"}]
    logging_obj.completion_start_time = None
    logging_obj._llm_caching_handler = None
    return logging_obj


def _make_generic_chunk(
    text: str,
    is_finished: bool = False,
    finish_reason: str = "",
    usage: Optional[dict] = None,
) -> GChunk:
    return GChunk(
        text=text,
        is_finished=is_finished,
        finish_reason=finish_reason,
        usage=usage,
        index=0,
        tool_use=None,
    )


def _make_converse_chunk(
    text: str = "",
    finish_reason: str = "",
    usage: Optional[Usage] = None,
) -> ModelResponseStream:
    return ModelResponseStream(
        choices=[
            StreamingChoices(
                finish_reason=finish_reason or None,
                index=0,
                delta=Delta(content=text, role="assistant"),
            )
        ],
        id="msg-bench",
        model="anthropic.claude-3-5-sonnet",
        usage=usage,
    )


# ---------------------------------------------------------------------------
# Provider stream factories
# ---------------------------------------------------------------------------


def anthropic_chunks(n: int) -> List[GChunk]:
    out: List[GChunk] = [_make_generic_chunk(f"tok{i} ") for i in range(n)]
    out.append(
        _make_generic_chunk(
            "",
            is_finished=True,
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": n, "total_tokens": 10 + n},
        )
    )
    return out


def bedrock_invoke_chunks(n: int) -> List[GChunk]:
    # Bedrock Invoke surfaces GChunk-shaped dicts, same shape as Anthropic.
    return anthropic_chunks(n)


def bedrock_converse_chunks(n: int) -> List[ModelResponseStream]:
    out: List[ModelResponseStream] = [
        _make_converse_chunk(f"tok{i} ") for i in range(n)
    ]
    out.append(
        _make_converse_chunk(
            text="",
            finish_reason="stop",
            usage=Usage(prompt_tokens=10, completion_tokens=n, total_tokens=10 + n),
        )
    )
    return out


PROVIDERS: dict[str, tuple[str, Callable[[int], list]]] = {
    "anthropic": ("anthropic", anthropic_chunks),
    "bedrock_invoke": ("bedrock", bedrock_invoke_chunks),
    "bedrock_converse": ("bedrock", bedrock_converse_chunks),
}


# ---------------------------------------------------------------------------
# Drive a single stream end-to-end
# ---------------------------------------------------------------------------


def _make_wrapper(
    chunks: list, provider: str, async_stream: bool
) -> CustomStreamWrapper:
    logging_obj = _make_logging_obj(provider)
    if async_stream:

        async def _agen():
            for c in chunks:
                yield c

        stream = _agen()
    else:
        stream = iter(chunks)
    return CustomStreamWrapper(
        completion_stream=stream,
        model="claude-3-5-sonnet",
        logging_obj=logging_obj,
        custom_llm_provider=provider,
    )


def drive_sync(provider_key: str, chunks_per_stream: int, n_streams: int) -> float:
    provider, factory = PROVIDERS[provider_key]
    # Pre-build the chunk lists; we only measure wrapper iteration cost.
    chunk_lists = [factory(chunks_per_stream) for _ in range(n_streams)]
    gc.collect()
    gc.disable()
    try:
        start = time.perf_counter()
        for chunks in chunk_lists:
            wrapper = _make_wrapper(chunks, provider, async_stream=False)
            for _ in wrapper:
                pass
        elapsed = time.perf_counter() - start
    finally:
        gc.enable()
    return elapsed


async def drive_async(
    provider_key: str, chunks_per_stream: int, n_streams: int
) -> float:
    provider, factory = PROVIDERS[provider_key]
    chunk_lists = [factory(chunks_per_stream) for _ in range(n_streams)]
    gc.collect()
    gc.disable()
    try:
        start = time.perf_counter()
        for chunks in chunk_lists:
            wrapper = _make_wrapper(chunks, provider, async_stream=True)
            async for _ in wrapper:
                pass
        elapsed = time.perf_counter() - start
    finally:
        gc.enable()
    return elapsed


# ---------------------------------------------------------------------------
# Repeat × take-min runner
# ---------------------------------------------------------------------------


@dataclass
class Result:
    label: str
    provider: str
    mode: str
    streams: int
    chunks_per_stream: int
    total_chunks: int
    elapsed_min_s: float
    elapsed_median_s: float
    per_chunk_us: float
    chunks_per_sec: float
    streams_per_sec: float


def run_case(
    label: str,
    provider_key: str,
    mode: str,
    chunks_per_stream: int,
    n_streams: int,
    repeats: int,
    warmup: int,
) -> Result:
    if mode == "sync":
        # Warmup runs amortize import-time and JIT-y caches.
        for _ in range(warmup):
            drive_sync(provider_key, chunks_per_stream, max(1, n_streams // 10))
        samples = [
            drive_sync(provider_key, chunks_per_stream, n_streams)
            for _ in range(repeats)
        ]
    elif mode == "async":

        async def _warm():
            for _ in range(warmup):
                await drive_async(
                    provider_key, chunks_per_stream, max(1, n_streams // 10)
                )

        asyncio.run(_warm())
        samples = [
            asyncio.run(drive_async(provider_key, chunks_per_stream, n_streams))
            for _ in range(repeats)
        ]
    else:
        raise ValueError(f"unknown mode {mode!r}")

    elapsed_min = min(samples)
    elapsed_median = statistics.median(samples)
    # Each stream emits chunks_per_stream text chunks + 1 finish/usage chunk.
    total_chunks = n_streams * (chunks_per_stream + 1)
    per_chunk_us = (elapsed_min * 1_000_000) / total_chunks
    chunks_per_sec = total_chunks / elapsed_min if elapsed_min > 0 else 0.0
    streams_per_sec = n_streams / elapsed_min if elapsed_min > 0 else 0.0

    return Result(
        label=label,
        provider=provider_key,
        mode=mode,
        streams=n_streams,
        chunks_per_stream=chunks_per_stream,
        total_chunks=total_chunks,
        elapsed_min_s=elapsed_min,
        elapsed_median_s=elapsed_median,
        per_chunk_us=per_chunk_us,
        chunks_per_sec=chunks_per_sec,
        streams_per_sec=streams_per_sec,
    )


def format_result(r: Result) -> str:
    return (
        f"  {r.provider:18s} {r.mode:5s}: "
        f"min={r.elapsed_min_s*1000:8.2f} ms  "
        f"median={r.elapsed_median_s*1000:8.2f} ms  "
        f"per-chunk={r.per_chunk_us:7.2f} μs  "
        f"chunks/s={r.chunks_per_sec:>10,.0f}  "
        f"streams/s={r.streams_per_sec:>8,.1f}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--label", required=True, help="Run label (e.g. baseline / optimized)"
    )
    ap.add_argument("--streams", type=int, default=500, help="Streams per run")
    ap.add_argument(
        "--chunks",
        type=int,
        default=200,
        help="Text chunks per stream (excl. finish chunk)",
    )
    ap.add_argument("--warmup", type=int, default=2, help="Warmup runs")
    ap.add_argument(
        "--repeats", type=int, default=5, help="Measured runs (we report min)"
    )
    ap.add_argument(
        "--providers",
        default="anthropic,bedrock_invoke,bedrock_converse",
        help="Comma-separated provider list",
    )
    ap.add_argument(
        "--modes",
        default="sync,async",
        help="Comma-separated iteration modes (sync/async)",
    )
    ap.add_argument(
        "--json", dest="json_out", help="Write results as JSON to this path"
    )
    args = ap.parse_args()

    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    for p in providers:
        if p not in PROVIDERS:
            raise SystemExit(f"unknown provider {p!r}; choose from {list(PROVIDERS)}")
    for m in modes:
        if m not in {"sync", "async"}:
            raise SystemExit(f"unknown mode {m!r}; choose from sync/async")

    print(
        f"\n=== label={args.label}  streams={args.streams}  chunks/stream={args.chunks}  "
        f"warmup={args.warmup}  repeats={args.repeats} (min reported) ==="
    )
    results: List[Result] = []
    for provider_key in providers:
        for mode in modes:
            r = run_case(
                label=args.label,
                provider_key=provider_key,
                mode=mode,
                chunks_per_stream=args.chunks,
                n_streams=args.streams,
                repeats=args.repeats,
                warmup=args.warmup,
            )
            results.append(r)
            print(format_result(r))

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"\nWrote {len(results)} results to {args.json_out}")


if __name__ == "__main__":
    main()
