from __future__ import annotations

import hashlib
import itertools
import json
import logging
import os
import random
import resource
import subprocess
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, TextIO

import httpx
import psutil
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from run import Variant

LOGGER: Final = logging.getLogger(__name__)


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class MemoryStats(FrozenModel):
    rss: int


class Stats(FrozenModel):
    requests: int
    rate: float
    success_rate: float
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: float
    elapsed_seconds: float
    throughput_rps: float
    first_event_p50_ms: float
    first_event_p95_ms: float
    first_event_p99_ms: float
    status_code_counts: Mapping[str, int]
    drop_reasons: Mapping[str, int]


class Result(FrozenModel):
    litellm: Stats


class Counters(FrozenModel):
    requests: int
    active: int
    completed: int
    cancelled: int


class Settings(FrozenModel):
    binary: Path
    suite_root: Path
    output: Path
    seconds: int
    repetitions: int
    python_body: str
    rust_client: str
    python_client: str
    http_parser: str
    sizes: tuple[int, ...]
    concurrencies: tuple[int, ...]
    responses: tuple[bool, ...] = (False, True)
    event_loop: str = "uvloop"
    profile: str = "litellm-current"
    seed: int = 20261019
    warmup_seconds: int = 2


def counters(client: httpx.Client) -> Counters:
    return Counters.model_validate_json(client.get("http://127.0.0.1:19400/metrics").content)


def processes_for(variant: Variant, roots: Mapping[str, psutil.Process]) -> tuple[psutil.Process, ...]:
    selected: Final = tuple(roots[name] for name in variant.process_names)
    return selected + tuple(itertools.chain.from_iterable(process.children(recursive=True) for process in selected))


def cpu_seconds(processes: Sequence[psutil.Process]) -> float:
    return sum(process.cpu_times().user + process.cpu_times().system for process in processes)


def execute(
    args: tuple[str, ...], group: tuple[psutil.Process, ...], settings: Settings, port: int
) -> tuple[float, int, float, float]:
    before: Final = cpu_seconds(group)
    children_before: Final = resource.getrusage(resource.RUSAGE_CHILDREN)
    started: Final = time.perf_counter()
    with (
        (settings.output.parent / "runtime-bifrost.log").open("w") as log,
        subprocess.Popen(
            args,
            cwd=settings.suite_root,
            stdout=log,
            stderr=log,
            env=dict(os.environ) | {"LITELLM_PORT": str(port), "NO_PROXY": "127.0.0.1,localhost"},
        ) as process,
    ):
        samples: Final = tuple(sample_rss(process, group))
        status: Final = process.wait(timeout=40)
        assert status == 0, (settings.output.parent / "runtime-bifrost.log").read_text()
    wall: Final = time.perf_counter() - started
    children_after: Final = resource.getrusage(resource.RUSAGE_CHILDREN)
    loadgen_cpu: Final = (
        children_after.ru_utime + children_after.ru_stime - children_before.ru_utime - children_before.ru_stime
    )
    return cpu_seconds(group) - before, max(samples, default=0), wall, loadgen_cpu


def sample_rss(process: subprocess.Popen[bytes], group: tuple[psutil.Process, ...]) -> Iterator[int]:
    while process.poll() is None:
        yield sum(MemoryStats.model_validate(item.memory_info(), from_attributes=True).rss for item in group)
        time.sleep(0.1)


@dataclass(frozen=True, slots=True)
class Cell:
    size: int
    streaming: bool
    concurrency: int
    variant: Variant


@dataclass(frozen=True, slots=True)
class Trial:
    cell: Cell
    index: int
    total: int
    settings: Settings
    group: tuple[psutil.Process, ...]
    args: tuple[str, ...]
    stats_path: Path
    body_bytes: int


def run_bifrost(roots: Mapping[str, psutil.Process], variants: Sequence[Variant], settings: Settings) -> None:
    runtime: Final = settings.output.resolve().parent / "bifrost-runtime"
    runtime.mkdir(exist_ok=True)
    cells: Final = tuple(itertools.product(settings.sizes, settings.responses, settings.concurrencies, variants))
    ordered: Final = random.Random(settings.seed).sample(cells, len(cells))
    with settings.output.open("w") as output, httpx.Client(trust_env=False, timeout=5) as client:
        for cell_index, (size, streaming, concurrency, variant) in enumerate(ordered):
            run_cell(
                roots,
                settings,
                Cell(size, streaming, concurrency, variant),
                runtime,
                client,
                output,
                cell_index,
                len(ordered),
            )


def run_cell(
    roots: Mapping[str, psutil.Process],
    settings: Settings,
    cell: Cell,
    runtime: Path,
    client: httpx.Client,
    output: TextIO,
    cell_index: int,
    total: int,
) -> None:
    size, streaming, concurrency, variant = cell.size, cell.streaming, cell.concurrency, cell.variant
    payload: Final = {
        "model": "fixture-model",
        "max_tokens": 32,
        "stream": streaming,
        "messages": [{"role": "user", "content": "q" * size}],
    }
    data: Final = json.dumps(payload, separators=(",", ":")).encode()
    headers: Final = {
        "x-bench-expected-sha256": hashlib.sha256(data).hexdigest(),
        "x-bench-delay-ms": "5",
    } | ({"x-bench-python": "1"} if variant.delegate else {})
    request_path: Final = runtime / "request.json"
    header_path: Final = runtime / "headers.json"
    stats_path: Final = runtime / "result.json"
    request_path.write_bytes(data)
    header_path.write_text(json.dumps(headers))
    group: Final = processes_for(variant, roots)
    base_args: Final = (
        str(settings.binary),
        "-provider",
        "litellm",
        "-host",
        "127.0.0.1",
        "-path",
        "messages",
        "-model",
        "fixture-model",
        "-max-output-tokens",
        "32",
        "-users",
        str(concurrency),
        "-request-file",
        str(request_path),
        "-headers-file",
        str(header_path),
        "-timeout",
        str(settings.seconds + 30),
        "-cooldown",
        "0",
        "-output",
        str(stats_path),
    )
    trial: Final = Trial(cell, cell_index, total, settings, group, base_args, stats_path, len(data))
    for repetition in range(-1, settings.repetitions):
        run_trial(trial, repetition, client, output)


def run_trial(trial: Trial, repetition: int, client: httpx.Client, output: TextIO) -> None:
    settings, group, base_args, stats_path = trial.settings, trial.group, trial.args, trial.stats_path
    size, streaming, concurrency, variant = (
        trial.cell.size,
        trial.cell.streaming,
        trial.cell.concurrency,
        trial.cell.variant,
    )
    cell_index, total = trial.index, trial.total
    before: Final = counters(client)
    seconds: Final = settings.warmup_seconds if repetition < 0 else settings.seconds
    cpu, rss, wall, loadgen_cpu = execute(base_args + ("-duration", str(seconds)), group, settings, variant.port)
    stats: Final = Result.model_validate_json(stats_path.read_bytes()).litellm
    after: Final = counters(client)
    assert stats.requests > 0
    assert stats.requests * stats.mean_latency_ms / 1000 <= concurrency * stats.elapsed_seconds + 1e-6
    assert stats.status_code_counts.get("200", 0) == stats.requests
    assert after.requests - before.requests == stats.requests
    assert after.cancelled == before.cancelled
    assert after.active == before.active
    assert 0 < stats.p50_latency_ms <= stats.p95_latency_ms <= stats.p99_latency_ms <= stats.max_latency_ms
    if streaming:
        assert after.completed - before.completed == stats.requests
        assert 0 < stats.first_event_p50_ms <= stats.p50_latency_ms
    if repetition < 0:
        return
    row: Final = {
        "generator": "bifrost",
        "variant": variant.name,
        "prompt_bytes": size,
        "body_bytes": trial.body_bytes,
        "stream": streaming,
        "concurrency": concurrency,
        "repetition": repetition,
        "python_body": settings.python_body,
        "python_client": settings.python_client,
        "rust_client": settings.rust_client,
        "http_parser": settings.http_parser,
        "event_loop": settings.event_loop,
        "profile": settings.profile,
        "gateway_cpu_s": cpu,
        "gateway_cpu_cores": cpu / wall,
        "gateway_peak_rss_bytes": rss,
        "loadgen_cpu_cores": loadgen_cpu / wall,
        "stats": stats.model_dump(),
        "backend_completed": after.completed - before.completed,
        "backend_requests": after.requests - before.requests,
        "backend_cancelled": after.cancelled - before.cancelled,
    }
    output.write(json.dumps(row) + "\n")
    output.flush()
    LOGGER.info(
        "%s/%s %s c=%s size=%s stream=%s rep=%s %.0f rps",
        cell_index + 1,
        total,
        variant.name,
        concurrency,
        size,
        streaming,
        repetition,
        stats.throughput_rps,
    )
