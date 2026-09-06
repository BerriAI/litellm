from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from collections.abc import Generator, Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from time import monotonic, sleep
from typing import TYPE_CHECKING, Final, TextIO

import psutil

from .constants import PYTHON_SENTINEL
from .models import Backend, BenchmarkModel, Invocation, Measurement, Memory, Options, Ready, Route, Timing
from .provider import provider_process

if TYPE_CHECKING:
    from .workloads import Workload

WORKER_MODULE: Final = "tests.rust-python-harness.strategies.e2e_benchmark.worker"


class ProcessMemory(BenchmarkModel):
    rss: int


def rss_bytes(process: psutil.Process) -> int:
    return ProcessMemory.model_validate(process.memory_info(), from_attributes=True).rss


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
        stdout=log,
        stderr=log,
        text=True,
    )
    try:
        yield process
    except BaseException:
        process.terminate()
        raise
    finally:
        if process.stdin is not None:
            with suppress(BrokenPipeError):
                process.stdin.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def wait_for_output(
    output: Path, child: subprocess.Popen[str], options: Options, *, sample_memory: bool = False
) -> Iterator[int]:
    deadline: Final = monotonic() + options.timeout
    process: Final = psutil.Process(child.pid) if sample_memory else None
    interval: Final = options.sample_interval_ms / 1000 if sample_memory else 0.01
    while not output.exists():
        if child.poll() is not None:
            raise RuntimeError(f"SDK worker exited with code {child.returncode} before writing {output.name}")
        if monotonic() >= deadline:
            raise TimeoutError(f"SDK worker timed out waiting for {output.name}")
        if process is not None:
            yield rss_bytes(process)
        sleep(min(interval, max(0, deadline - monotonic())))


def execute_phase(
    invocation: Invocation, backend: Backend, options: Options, repo_root: Path
) -> tuple[Ready, Timing, Memory]:
    with tempfile.TemporaryDirectory(prefix="litellm-benchmark-") as raw_directory:
        directory: Final = Path(raw_directory)
        case_file: Final = directory / "invocation.json"
        case_file.write_text(invocation.model_dump_json())
        ready_file: Final = directory / "ready.json"
        timing_file: Final = directory / "timing.json"
        with (directory / "worker.log").open("w+") as log:
            try:
                with sdk_process(case_file, backend, repo_root, log) as child:
                    tuple(wait_for_output(ready_file, child, options))
                    ready: Final = Ready.model_validate_json(ready_file.read_bytes())
                    if invocation.phase == "timing":
                        child.communicate(input="go\n", timeout=options.timeout)
                        if child.returncode != 0:
                            raise RuntimeError(f"SDK worker exited with code {child.returncode}")
                        return (
                            ready,
                            Timing.model_validate_json(timing_file.read_bytes()),
                            Memory(baseline_rss_bytes=0, sampled_peak_rss_bytes=0, retained_rss_bytes=0, samples=0),
                        )
                    process: Final = psutil.Process(child.pid)
                    baseline: Final = rss_bytes(process)
                    assert child.stdin is not None
                    child.stdin.write("go\n")
                    child.stdin.flush()
                    samples: Final = tuple(wait_for_output(timing_file, child, options, sample_memory=True))
                    retained: Final = rss_bytes(process)
                    return (
                        ready,
                        Timing.model_validate_json(timing_file.read_bytes()),
                        Memory(
                            baseline_rss_bytes=baseline,
                            sampled_peak_rss_bytes=max((baseline, retained, *samples)),
                            retained_rss_bytes=retained,
                            samples=len(samples),
                        ),
                    )
            except (RuntimeError, OSError, ValueError, TimeoutError, subprocess.TimeoutExpired, psutil.Error) as error:
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
            min_time=options.min_time,
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
