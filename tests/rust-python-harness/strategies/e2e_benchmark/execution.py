from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from collections.abc import Generator, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from time import monotonic, sleep
from typing import TYPE_CHECKING, Final, TextIO, cast

import psutil

from .models import PREFIX, Backend, BenchmarkModel, Invocation, Measurement, Memory, Options, Ready, Route, Timing
from .provider import PYTHON_SENTINEL, provider_process

if TYPE_CHECKING:
    from .workloads import Workload

WORKER_MODULE: Final = "tests.rust-python-harness.strategies.e2e_benchmark.worker"


class ProcessMemory(BenchmarkModel):
    rss: int


def rss_bytes(process: psutil.Process) -> int:
    return ProcessMemory.model_validate(process.memory_info(), from_attributes=True).rss


def _read_message(stream: TextIO) -> str:
    for line in stream:
        if line.startswith(PREFIX):
            return line.removeprefix(PREFIX)
    raise RuntimeError("SDK worker exited without returning a measurement")


@contextmanager
def sdk_process(case_file: Path, backend: Backend, repo_root: Path, log: TextIO) -> Generator[subprocess.Popen[str]]:
    process: Final = subprocess.Popen(
        (sys.executable, "-m", WORKER_MODULE, str(case_file)),
        cwd=repo_root,
        env={
            **os.environ,
            "LITELLM_RUST": "1" if backend == "rust" else "0",
            "LITELLM_USER_AGENT": PYTHON_SENTINEL,
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
            "PYTHONPATH": str(repo_root),
        },
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=log,
        text=True,
    )
    try:
        yield process
    finally:
        if process.stdin is not None:
            process.stdin.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        if process.stdout is not None:
            process.stdout.close()


def sample_rss(process: psutil.Process, completed: Future[str], interval: float, timeout: float) -> Iterator[int]:
    deadline: Final = monotonic() + timeout
    while not completed.done():
        if monotonic() >= deadline:
            raise TimeoutError("memory measurement timed out")
        yield rss_bytes(process)
        sleep(interval)


def execute_phase(
    invocation: Invocation, backend: Backend, options: Options, repo_root: Path
) -> tuple[Ready, Timing, Memory]:
    with tempfile.TemporaryDirectory(prefix="litellm-benchmark-") as raw_directory:
        directory: Final = Path(raw_directory)
        case_file: Final = directory / "invocation.json"
        case_file.write_text(invocation.model_dump_json())
        with (directory / "worker.log").open("w+") as log:
            try:
                with ThreadPoolExecutor(max_workers=1) as reader:
                    with sdk_process(case_file, backend, repo_root, log) as child:
                        assert child.stdout is not None and child.stdin is not None
                        stdout: Final = cast(TextIO, child.stdout)
                        ready: Final = Ready.model_validate_json(
                            reader.submit(_read_message, stdout).result(timeout=options.timeout)
                        )
                        process: Final = psutil.Process(child.pid)
                        baseline: Final = rss_bytes(process) if invocation.phase == "memory" else 0
                        child.stdin.write("go\n")
                        child.stdin.flush()
                        result: Final = reader.submit(_read_message, stdout)
                        samples: Final = (
                            tuple(sample_rss(process, result, options.sample_interval_ms / 1000, options.timeout))
                            if invocation.phase == "memory"
                            else ()
                        )
                        timing: Final = Timing.model_validate_json(result.result(timeout=options.timeout))
                        retained: Final = rss_bytes(process) if invocation.phase == "memory" else 0
                        memory: Final = Memory(
                            baseline_rss_bytes=baseline,
                            sampled_peak_rss_bytes=max((baseline, retained, *samples)),
                            retained_rss_bytes=retained,
                            samples=len(samples),
                        )
                        return ready, timing, memory
            except (RuntimeError, OSError, ValueError, TimeoutError) as error:
                log.seek(0)
                raise RuntimeError(f"{backend}/{invocation.phase}: {error}\n{log.read()[-6000:]}") from error


def benchmark(
    workload: Workload, route: Route, backend: Backend, repeat: int, options: Options, repo_root: Path
) -> Measurement:
    with provider_process(workload.response, backend) as url:
        invocation: Final = Invocation(
            model=workload.model,
            document_url=workload.document_url,
            route=route,
            provider_url=url,
            iterations=options.iterations,
            warmup=options.warmup,
            phase="timing",
        )
        ready, timing, _ = execute_phase(invocation, backend, options, repo_root)
        memory_ready, _, memory = execute_phase(
            invocation.model_copy(update={"phase": "memory"}), backend, options, repo_root
        )
        if ready != memory_ready:
            raise ValueError("timing and memory workers returned different preflight results")
        return Measurement(
            backend=backend,
            repeat=repeat,
            profile=workload.profile,
            route=route,
            document_bytes=workload.document_bytes,
            response_bytes=len(workload.response),
            response_pages=workload.response_pages,
            fixture_sha256=workload.fixture_sha256,
            ready=ready,
            timing=timing,
            memory=memory,
        )
