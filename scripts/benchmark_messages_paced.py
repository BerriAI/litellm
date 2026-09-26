"""Serve paced Anthropic SSE or measure a local proxy against it.

Each synthetic token is a text delta containing its sequence and monotonic send
stamp. Run server and client as separate processes on the same machine.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import asdict, dataclass
from itertools import chain
from pathlib import Path
from typing import Final, Literal

import aiohttp
import psutil
from aiohttp import web
from pydantic import BaseModel, ConfigDict, JsonValue


class Options(BaseModel):
    action: Literal["serve", "bench"]
    port: int
    chunks: int
    rate: float
    url: str
    model: str
    mode: Literal["direct", "python", "rust"]
    pid: int | None
    concurrency: int
    repeats: int
    output: str


class RequestBody(BaseModel):
    model: str
    stream: Literal[True]
    max_tokens: int


class Delta(BaseModel):
    text: str = ""


class Usage(BaseModel):
    output_tokens: int


class Event(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: str
    delta: Delta | None = None
    usage: Usage | None = None


@dataclass(frozen=True, slots=True)
class Received:
    event: Event
    at_ns: int


@dataclass(frozen=True, slots=True)
class Sample:
    ttft_ms: float
    total_ms: float
    lag_ms: tuple[float, ...]
    gaps_ms: tuple[float, ...]
    chunks: int
    rust_header: str | None


def frame(event: str, **fields: JsonValue) -> bytes:
    return f"event: {event}\ndata: {json.dumps({'type': event, **fields})}\n\n".encode()


class Mock:
    def __init__(self, chunks: int, rate: float) -> None:
        self.chunks = chunks
        self.rate = rate

    async def messages(self, request: web.Request) -> web.StreamResponse:
        body: Final = RequestBody.model_validate_json(await request.read())
        count: Final = min(self.chunks, body.max_tokens)
        response: Final = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
        await response.prepare(request)
        await response.write(
            frame(
                "message_start",
                message={
                    "id": "msg_paced",
                    "type": "message",
                    "role": "assistant",
                    "model": body.model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 8, "output_tokens": 0},
                },
            )
        )
        await response.write(frame("content_block_start", index=0, content_block={"type": "text", "text": ""}))
        start: Final = time.perf_counter()
        for index in range(count):
            await asyncio.sleep(max(0, start + (index + 1) / self.rate - time.perf_counter()))
            await response.write(
                frame(
                    "content_block_delta",
                    index=0,
                    delta={
                        "type": "text_delta",
                        "text": f"{index}:{time.perf_counter_ns()} The stream continues.\n",
                    },
                )
            )
        await response.write(frame("content_block_stop", index=0))
        await response.write(
            frame(
                "message_delta",
                delta={"stop_reason": "end_turn", "stop_sequence": None},
                usage={"output_tokens": count},
            )
        )
        await response.write(frame("message_stop"))
        await response.write_eof()
        return response


async def events(response: aiohttp.ClientResponse) -> AsyncIterator[Received]:
    async for line in response.content:
        if line.startswith(b"data:"):
            at_ns: Final = time.perf_counter_ns()
            yield Received(Event.model_validate_json(line[5:].strip()), at_ns)


async def sample(session: aiohttp.ClientSession, url: str, model: str, chunks: int, mode: str) -> Sample:
    start: Final = time.perf_counter_ns()
    async with session.post(
        url,
        json={
            "model": model,
            "max_tokens": chunks,
            "stream": True,
            "messages": [{"role": "user", "content": "Please generate a long streaming response."}],
        },
        headers={"Authorization": "Bearer sk-benchmark", "anthropic-version": "2023-06-01"},
    ) as response:
        response.raise_for_status()
        rust_header: Final = response.headers.get("x-litellm-rust")
        assert (rust_header == "true") == (mode == "rust"), (mode, rust_header)
        received: Final = tuple([event async for event in events(response)])
    end: Final = time.perf_counter_ns()
    deltas: Final = tuple(item for item in received if item.event.type == "content_block_delta")
    texts: Final = tuple(item.event.delta.text for item in deltas if item.event.delta is not None)
    sequences: Final = tuple(int(text.split(":", 1)[0]) for text in texts)
    stamps: Final = tuple(int(text.split(":", 1)[1].split(" ", 1)[0]) for text in texts)
    assert sequences == tuple(range(chunks)), "Missing, duplicate, or reordered deltas"
    assert received[0].event.type == "message_start", "Missing message_start"
    assert received[-1].event.type == "message_stop", "Truncated stream"
    usages: Final = tuple(item.event.usage.output_tokens for item in received if item.event.usage is not None)
    assert usages == (chunks,), ("Unexpected output usage", usages)
    return Sample(
        (deltas[0].at_ns - start) / 1e6,
        (end - start) / 1e6,
        tuple((item.at_ns - stamp) / 1e6 for item, stamp in zip(deltas, stamps, strict=True)),
        tuple((right.at_ns - left.at_ns) / 1e6 for left, right in zip(deltas, deltas[1:])),
        len(deltas),
        rust_header,
    )


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered: Final = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def cpu_seconds(process: psutil.Process | None) -> float:
    if process is None:
        return 0.0
    usage: Final = process.cpu_times()
    return usage.user + usage.system


async def rss_samples(process: psutil.Process, stop: asyncio.Event) -> AsyncIterator[int]:
    while not stop.is_set():
        yield process.memory_info().rss
        await asyncio.sleep(0.1)


async def peak_rss(process: psutil.Process | None, stop: asyncio.Event) -> float | None:
    if process is None:
        return None
    return max([rss async for rss in rss_samples(process, stop)], default=0) / 2**20


async def benchmark(args: Options) -> None:
    process: Final = psutil.Process(args.pid) if args.pid else None
    idle_rss: Final = process.memory_info().rss / 2**20 if process else None
    connector: Final = aiohttp.TCPConnector(limit=max(args.concurrency, 100), limit_per_host=args.concurrency)
    async with aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(total=120)) as session:
        await asyncio.gather(*(sample(session, args.url, args.model, 5, args.mode) for _ in range(args.concurrency)))
        await asyncio.sleep(1)
        for repeat in range(args.repeats):
            stop: Final = asyncio.Event()
            memory: Final = asyncio.create_task(peak_rss(process, stop))
            rss_before: Final = process.memory_info().rss / 2**20 if process else None
            cpu_start: Final = cpu_seconds(process)
            wall_start: Final = time.perf_counter()
            samples: Final = tuple(
                await asyncio.gather(
                    *(sample(session, args.url, args.model, args.chunks, args.mode) for _ in range(args.concurrency))
                )
            )
            wall_s: Final = time.perf_counter() - wall_start
            cpu_s: Final = cpu_seconds(process) - cpu_start
            stop.set()
            rss_peak: Final = await memory
            lag: Final = tuple(chain.from_iterable(item.lag_ms for item in samples))
            gaps: Final = tuple(chain.from_iterable(item.gaps_ms for item in samples))
            rss_after: Final = process.memory_info().rss / 2**20 if process else None
            result: Final = {
                "mode": args.mode,
                "concurrency": args.concurrency,
                "repeat": repeat,
                "requests": len(samples),
                "chunks_per_request": args.chunks,
                "wall_s": wall_s,
                "requests_per_s": len(samples) / wall_s,
                "deltas_per_s": args.chunks * len(samples) / wall_s,
                "proxy_cpu_percent_one_core": 100 * cpu_s / wall_s if process else None,
                "proxy_idle_rss_mib": idle_rss,
                "proxy_before_rss_mib": rss_before,
                "proxy_peak_rss_mib": max(rss_peak, rss_after)
                if rss_peak is not None and rss_after is not None
                else None,
                "ttft_p50_ms": statistics.median(item.ttft_ms for item in samples),
                "ttft_p95_ms": percentile(tuple(item.ttft_ms for item in samples), 0.95),
                "total_p50_ms": statistics.median(item.total_ms for item in samples),
                "lag_p50_ms": statistics.median(lag),
                "lag_p95_ms": percentile(lag, 0.95),
                "lag_p99_ms": percentile(lag, 0.99),
                "gap_p50_ms": statistics.median(gaps),
                "gap_p99_ms": percentile(gaps, 0.99),
                "proxy_cpu_s": cpu_s if process else None,
                "proxy_cpu_ms_per_request": cpu_s * 1000 / len(samples) if process else None,
                "proxy_cpu_us_per_chunk": cpu_s * 1e6 / (args.chunks * len(samples)) if process else None,
                "proxy_rss_mib": rss_after,
                "samples": tuple(asdict(item) for item in samples),
            }
            with Path(args.output).open("a") as output:
                output.write(json.dumps(result) + "\n")
            sys.stdout.write(json.dumps({key: value for key, value in result.items() if key != "samples"}) + "\n")
            sys.stdout.flush()


def main() -> None:
    parser: Final = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("serve", "bench"))
    parser.add_argument("--port", type=int, default=18098)
    parser.add_argument("--chunks", type=int, default=1500)
    parser.add_argument("--rate", type=float, default=50)
    parser.add_argument("--url", default="http://127.0.0.1:18099/v1/messages")
    parser.add_argument("--model", default="paced-mock")
    parser.add_argument("--mode", choices=("direct", "python", "rust"), default="direct")
    parser.add_argument("--pid", type=int)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", default="messages-paced.jsonl")
    args: Final = Options.model_validate(vars(parser.parse_args()))
    if min(args.chunks, args.rate, args.concurrency, args.repeats) <= 0:
        parser.error("chunks, rate, concurrency, and repeats must be positive")
    if args.action == "serve":
        app: Final = web.Application()
        app.router.add_post("/v1/messages", Mock(args.chunks, args.rate).messages)
        web.run_app(app, host="127.0.0.1", port=args.port, access_log=None)
        return
    asyncio.run(benchmark(args))


if __name__ == "__main__":
    main()
