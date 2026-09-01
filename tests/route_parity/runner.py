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

from tests.route_parity.models import (
    Execution,
    SDKCommand,
    WorkerFailure,
    WorkerResult,
    WorkerSuccess,
)
from tests.route_parity.recorded_http import RecordedResponse
from tests.route_parity.replay import ReplayServer, replay_server

WORKER_RESULT_PREFIX: Final = "LITELLM_PARITY_RESULT "
WORKER_RESULT_ADAPTER: Final[TypeAdapter[WorkerResult]] = TypeAdapter(WorkerResult)


@dataclass(frozen=True, slots=True)
class PythonScriptRunner:
    entrypoint: Path
    rust_env_var: str
    python_user_agent: str
    route_label: str

    def command(self, provider_url: str) -> tuple[str, ...]:
        return (
            sys.executable,
            str(self.entrypoint.resolve()),
            "--parity-worker",
            provider_url,
        )


class PythonScriptWorker:
    def __init__(self, runner: PythonScriptRunner, provider: ReplayServer, rust_enabled: bool) -> None:
        project_root: Final = str(runner.entrypoint.resolve().parents[3])
        existing_pythonpath: Final = os.environ.get("PYTHONPATH")
        env: Final = {
            **os.environ,
            runner.rust_env_var: "1" if rust_enabled else "0",
            "LITELLM_USER_AGENT": runner.python_user_agent,
            "PYTHONPATH": os.pathsep.join(path for path in (project_root, existing_pythonpath) if path),
        }
        self.mode: Final = "Rust" if rust_enabled else "Python"
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
    runner: PythonScriptRunner,
    rust_enabled: bool,
) -> Generator[PythonScriptWorker]:
    with replay_server() as provider:
        worker: Final = PythonScriptWorker(runner, provider, rust_enabled)
        try:
            yield worker
        finally:
            worker.close()


def run_execution(
    worker: PythonScriptWorker,
    case_file: Path,
    route: str,
    responses: tuple[RecordedResponse, ...],
) -> Execution:
    return worker.execute(case_file, route, responses)


@contextmanager
def execution_worker_pair(
    runner: PythonScriptRunner,
) -> Generator[tuple[PythonScriptWorker, PythonScriptWorker]]:
    with execution_worker(runner, rust_enabled=False) as python_worker:
        with execution_worker(runner, rust_enabled=True) as accelerated_worker:
            yield python_worker, accelerated_worker


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
