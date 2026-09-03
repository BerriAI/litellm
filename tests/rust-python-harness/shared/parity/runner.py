from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from collections import deque
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TextIO, cast

from pydantic import TypeAdapter, ValidationError

from .models import (
    Execution,
    SDKCommand,
    WorkerFailure,
    WorkerResult,
    WorkerSuccess,
)
from .recorded_http import RecordedResponse
from .replay import ReplayServer, replay_server

WORKER_RESULT_PREFIX: Final = "LITELLM_PARITY_RESULT "
WORKER_RESULT_ADAPTER: Final[TypeAdapter[WorkerResult]] = TypeAdapter(WorkerResult)


@dataclass(frozen=True, slots=True)
class SubprocessRunner:
    entrypoint: Path
    baseline_user_agent: str
    route_label: str

    def command(self, provider_url: str) -> tuple[str, ...]:
        return (
            sys.executable,
            "-m",
            ".".join(
                self.entrypoint.resolve().relative_to(Path(__file__).resolve().parents[4]).with_suffix("").parts
            ),
            "--parity-worker",
            provider_url,
        )


@dataclass(frozen=True, slots=True)
class ExecutionVariant:
    name: str
    environment: tuple[tuple[str, str], ...]


class SubprocessWorker:
    def __init__(self, runner: SubprocessRunner, provider: ReplayServer, variant: ExecutionVariant) -> None:
        project_root: Final = str(Path(__file__).resolve().parents[4])
        existing_pythonpath: Final = os.environ.get("PYTHONPATH")
        env: Final = {
            **os.environ,
            **dict(variant.environment),
            "LITELLM_USER_AGENT": runner.baseline_user_agent,
            "PYTHONPATH": os.pathsep.join(path for path in (project_root, existing_pythonpath) if path),
        }
        self.mode: Final = variant.name
        self.route_label: Final = runner.route_label
        self.provider: Final = provider
        self.process: Final = subprocess.Popen(
            runner.command(provider.url),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        self.output_reader: Final = ThreadPoolExecutor(max_workers=1)
        self.recent_output: Final[deque[str]] = deque(maxlen=100)

    def execute(
        self,
        case_file: Path,
        route: str,
        responses: tuple[RecordedResponse, ...],
    ) -> Execution:
        stdin: Final = self.process.stdin
        if stdin is None or self.process.poll() is not None:
            raise AssertionError(f"{self.mode} {self.route_label} worker exited before processing {case_file}")
        for response in responses:
            self.provider.enqueue_response(response)
        command: Final = SDKCommand(case_file=str(case_file), route=route)
        try:
            stdin.write(f"{command.model_dump_json()}\n")
            stdin.flush()
            result: Final = self.output_reader.submit(self._read_result).result(timeout=60)
        except TimeoutError as error:
            self.provider.reset()
            self.close()
            raise AssertionError(
                f"{self.mode} {self.route_label} worker timed out after 60s while processing {case_file}"
            ) from error
        except AssertionError:
            self.provider.reset()
            raise
        except (BrokenPipeError, OSError) as error:
            self.provider.reset()
            raise AssertionError(self._failure_message(f"worker pipe failed while processing {case_file}")) from error
        if isinstance(result, WorkerFailure):
            self.provider.reset()
            raise AssertionError(
                f"{self.mode} {self.route_label} worker failed while processing {case_file}:\n{result.error}"
            )
        assert isinstance(result, WorkerSuccess)
        try:
            return Execution(requests=self.provider.take_requests(len(responses)), report=result.report)
        except AssertionError:
            self.provider.reset()
            raise

    def _read_result(self) -> WorkerResult:
        process_stdout: Final = self.process.stdout
        if process_stdout is None:
            raise AssertionError(self._failure_message("worker stdout is unavailable"))
        stdout: Final = cast(TextIO, process_stdout)
        line: Final = stdout.readline()
        if not line:
            raise AssertionError(self._failure_message("worker exited without returning a result"))
        stripped: Final = line.rstrip()
        if not stripped.startswith(WORKER_RESULT_PREFIX):
            self.recent_output.append(stripped)
            return self._read_result()
        payload: Final = stripped.removeprefix(WORKER_RESULT_PREFIX)
        try:
            return WORKER_RESULT_ADAPTER.validate_json(payload)
        except ValidationError as error:
            raise AssertionError(self._failure_message("worker returned an invalid result")) from error

    def _failure_message(self, message: str) -> str:
        output: Final = "\n".join(self.recent_output)
        prefix: Final = f"{self.mode} {self.route_label}"
        return f"{prefix} {message}" if not output else f"{prefix} {message}\noutput:\n{output}"

    def close(self) -> None:
        stdin: Final = self.process.stdin
        if stdin is not None and not stdin.closed:
            stdin.close()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=10)
        self.output_reader.shutdown(wait=True, cancel_futures=True)


@contextmanager
def execution_worker(
    runner: SubprocessRunner,
    variant: ExecutionVariant,
) -> Generator[SubprocessWorker]:
    with replay_server() as provider:
        worker: Final = SubprocessWorker(runner, provider, variant)
        try:
            yield worker
        finally:
            worker.close()


def run_execution(
    worker: SubprocessWorker,
    case_file: Path,
    route: str,
    responses: tuple[RecordedResponse, ...],
) -> Execution:
    return worker.execute(case_file, route, responses)


@contextmanager
def execution_worker_pair(
    runner: SubprocessRunner,
    baseline: ExecutionVariant,
    candidate: ExecutionVariant,
) -> Generator[tuple[SubprocessWorker, SubprocessWorker]]:
    with execution_worker(runner, baseline) as baseline_worker:
        with execution_worker(runner, candidate) as candidate_worker:
            yield baseline_worker, candidate_worker


def parity_worker_main(
    execute_command: Callable[[str, str, asyncio.AbstractEventLoop], WorkerResult],
    mock_url: str,
) -> None:
    event_loop: Final = asyncio.new_event_loop()
    try:
        for line in sys.stdin:
            sys.stdout.write(f"{WORKER_RESULT_PREFIX}{execute_command(line, mock_url, event_loop).model_dump_json()}\n")
            sys.stdout.flush()
    finally:
        event_loop.close()
