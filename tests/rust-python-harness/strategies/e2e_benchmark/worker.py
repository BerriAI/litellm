from __future__ import annotations

import asyncio
import gc
import hashlib
import json
import platform
import sys
from collections.abc import Awaitable, Callable, Iterator
from itertools import count
from pathlib import Path
from time import perf_counter_ns, process_time_ns
from typing import Final, cast

from litellm.llms.base_llm.ocr.transformation import OCRResponse

from .models import Invocation, Ready, Timing


def file_sha256(path: Path) -> str:
    digest: Final = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(256 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_ready(response: OCRResponse) -> Ready:
    from litellm.rust_bridge import get_native_bridge
    from litellm.rust_bridge.configuration import rust_enabled

    bridge: Final = get_native_bridge() if rust_enabled() else None
    if rust_enabled() and bridge is None:
        raise RuntimeError("native Rust bridge is unavailable; build it with maturin develop --release")
    native_path: Final = bridge.__file__ if bridge is not None else None
    return Ready(
        response_digest=hashlib.sha256(json.dumps(response.model_dump(), sort_keys=True).encode()).hexdigest(),
        python_version=platform.python_version(),
        native_sha256=file_sha256(Path(native_path)) if native_path else None,
    )


def _publish(value: Ready | Timing, destination: Path) -> None:
    temporary: Final = destination.with_suffix(".tmp")
    temporary.write_text(value.model_dump_json())
    temporary.replace(destination)


def _handshake(ready: Ready, directory: Path) -> None:
    gc.collect()
    _publish(ready, directory / "ready.json")
    if sys.stdin.readline().strip() != "go":
        raise RuntimeError("benchmark controller disconnected before measurement")


def _finish(timing: Timing, directory: Path) -> None:
    gc.collect()
    _publish(timing, directory / "timing.json")
    sys.stdin.readline()


def _sync_sample(call: Callable[[], OCRResponse]) -> float:
    start: Final = perf_counter_ns()
    call()
    return (perf_counter_ns() - start) / 1e6


async def _async_sample(call: Callable[[], Awaitable[OCRResponse]]) -> float:
    start: Final = perf_counter_ns()
    await call()
    return (perf_counter_ns() - start) / 1e6


def sample_indices(invocation: Invocation, start_ns: int) -> Iterator[int]:
    deadline: Final = start_ns + invocation.min_time * 1e9
    for index in count():
        if index >= invocation.iterations and perf_counter_ns() >= deadline:
            return
        yield index


def measure_sync(call: Callable[[], OCRResponse], invocation: Invocation) -> Timing:
    if invocation.phase == "memory":
        for _ in range(invocation.iterations):
            call()
        return Timing(latency_ms=(), cpu_ms=0, elapsed_ms=0)
    cpu_start: Final = process_time_ns()
    wall_start: Final = perf_counter_ns()
    samples: Final = tuple(_sync_sample(call) for _ in sample_indices(invocation, wall_start))
    elapsed: Final = perf_counter_ns() - wall_start
    return Timing(latency_ms=samples, cpu_ms=(process_time_ns() - cpu_start) / 1e6, elapsed_ms=elapsed / 1e6)


async def measure_async(call: Callable[[], Awaitable[OCRResponse]], invocation: Invocation) -> Timing:
    if invocation.phase == "memory":
        for _ in range(invocation.iterations):
            await call()
        return Timing(latency_ms=(), cpu_ms=0, elapsed_ms=0)
    cpu_start: Final = process_time_ns()
    wall_start: Final = perf_counter_ns()
    samples: Final = tuple([await _async_sample(call) for _ in sample_indices(invocation, wall_start)])
    elapsed: Final = perf_counter_ns() - wall_start
    return Timing(latency_ms=samples, cpu_ms=(process_time_ns() - cpu_start) / 1e6, elapsed_ms=elapsed / 1e6)


async def _run_async(call: Callable[[], Awaitable[OCRResponse]], invocation: Invocation, directory: Path) -> None:
    for _ in range(invocation.warmup):
        await call()
    ready: Final = capture_ready(await call())
    _handshake(ready, directory)
    _finish(await measure_async(call, invocation), directory)


def run_worker(invocation: Invocation, directory: Path) -> None:
    import litellm

    kwargs: Final = {
        "model": invocation.model,
        "document": {"type": "document_url", "document_url": invocation.document_url},
        "api_key": "benchmark-local-only",
        "api_base": invocation.provider_url,
        "timeout": 10,
        "num_retries": 0,
    }
    if invocation.route == "aocr":
        async_route: Final = cast(Callable[..., Awaitable[OCRResponse]], litellm.aocr)
        asyncio.run(_run_async(lambda: async_route(**kwargs), invocation, directory))
        return
    sync_route: Final = cast(Callable[..., OCRResponse], litellm.ocr)
    call: Final = lambda: sync_route(**kwargs)
    for _ in range(invocation.warmup):
        call()
    ready: Final = capture_ready(call())
    _handshake(ready, directory)
    _finish(measure_sync(call, invocation), directory)


if __name__ == "__main__":
    case_file: Final = Path(sys.argv[1])
    run_worker(Invocation.model_validate_json(case_file.read_bytes()), case_file.parent)
