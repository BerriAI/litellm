"""Sends a case's plan through the gateway from a separate process: the plan is written to disk as
wire requests, Locust (`locustfile.py`) replays it at the planned rate, and the records it
wrote come back as `WireRecord`s."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Final

from e2e_config import PROXY_BASE_URL
from pydantic import TypeAdapter

from setup.performance_test_case import PerformanceTestCase
from setup.plan import PlannedRequest
from setup.provision import CaseSetup
from setup.wire import WireRecord, WireRequest, WireShape

_LOCUSTFILE: Final = Path(__file__).with_name("locustfile.py")
_PERF_DIR: Final = Path(__file__).resolve().parents[1]
_PLAN_ADAPTER: Final = TypeAdapter(list[WireRequest])
_RECORD_ADAPTER: Final = TypeAdapter(WireRecord)
_PATHS: Final = {"chat_completions": "/v1/chat/completions", "responses": "/v1/responses", "messages": "/v1/messages"}


@dataclass(frozen=True, slots=True)
class RunOutcome:
    records: tuple[WireRecord, ...]
    locust_exit: int
    locust_output: str

    @property
    def wall_seconds(self) -> float:
        if not self.records:
            return 0.0
        first: Final = min(record.started_at for record in self.records)
        last: Final = max(record.started_at + record.total_s for record in self.records)
        return last - first


def wire_request(item: PlannedRequest, setup: CaseSetup) -> WireRequest:
    return WireRequest(
        seq=item.seq,
        marker=item.marker,
        caller=item.caller.label,
        endpoint=item.endpoint,
        path=_PATHS[item.endpoint],
        model=item.model_group,
        shape=WireShape(
            name=item.shape.name,
            input_tokens=item.input_tokens,
            tools=item.shape.tools,
            output_tokens=item.shape.output_tokens,
        ),
        authorization=setup.authorization(item.caller),
        token_source=setup.token_source(item.caller),
        stream=item.stream,
        blocked=item.blocked,
        send_at_s=item.send_at_s,
    )


def _keep_records_artifact(records_path: Path) -> None:
    """Copy the per-request records next to the JUnit file when the runner exposes an artifact
    directory (`E2E_ARTIFACT_DIR`), so a red lane run can be broken down after the pod is gone."""
    artifact_dir: Final = os.environ.get("E2E_ARTIFACT_DIR")
    if not artifact_dir or not Path(artifact_dir).is_dir():
        return
    _ = shutil.copyfile(records_path, Path(artifact_dir) / "perf-records.jsonl")


class RecordStream:
    def __init__(self, source: BinaryIO) -> None:
        self.source: Final = source
        self.partial = b""

    def read(self) -> tuple[WireRecord, ...]:
        chunks: Final = (self.partial + self.source.read()).split(b"\n")
        self.partial = chunks[-1]
        return tuple(_RECORD_ADAPTER.validate_json(line) for line in chunks[:-1] if line)


def run_until(
    command: list[str],
    env: dict[str, str],
    records_path: Path,
    stop_path: Path,
    stop_when: Callable[[tuple[WireRecord, ...]], bool],
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    with records_path.open("rb") as records, records_path.with_suffix(".log").open("w+") as output:
        with subprocess.Popen(command, env=env, stdout=output, stderr=subprocess.STDOUT, text=True) as process:
            deadline: Final = time.monotonic() + timeout
            reader: Final = RecordStream(records)
            try:
                while process.poll() is None:
                    if stop_when(reader.read()):
                        stop_path.touch()
                    if time.monotonic() >= deadline:
                        raise subprocess.TimeoutExpired(command, timeout)
                    try:
                        _ = process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        pass
            finally:
                stop_path.touch()
                try:
                    _ = process.wait(timeout=35)
                except subprocess.TimeoutExpired:
                    process.kill()
                    _ = process.wait(timeout=10)
                _keep_records_artifact(records_path)
        _ = output.seek(0)
        return subprocess.CompletedProcess(command, process.returncode, output.read(), "")


def run_plan(
    case: PerformanceTestCase,
    plan: tuple[PlannedRequest, ...],
    setup: CaseSetup,
    *,
    stop_when: Callable[[tuple[WireRecord, ...]], bool] | None = None,
    request_timeout: float = 120.0,
) -> RunOutcome:
    """Replay `plan` with Locust in a child process and collect its records. The run-time ceiling
    is well past the planned duration so Locust's own clock never cuts a slow run short; the
    locustfile quits itself once the last record is written."""
    wire: Final = [wire_request(item, setup) for item in plan]
    ceiling_s: Final = int(case.traffic.expected_seconds * 2 + 180)
    with tempfile.TemporaryDirectory(prefix="e2e-perf-") as directory:
        plan_path: Final = Path(directory) / "plan.json"
        records_path: Final = Path(directory) / "records.jsonl"
        stop_path: Final = Path(directory) / "stop"
        _ = plan_path.write_bytes(_PLAN_ADAPTER.dump_json(wire))
        records_path.touch()
        env: Final = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join(filter(None, (str(_PERF_DIR), os.environ.get("PYTHONPATH")))),
            "PERF_PLAN_FILE": str(plan_path),
            "PERF_RECORDS_FILE": str(records_path),
            "PERF_RPS": str(case.traffic.rps),
            "PERF_USERS": str(case.traffic.concurrency),
            "PERF_REQUEST_TIMEOUT": str(request_timeout),
            **({"PERF_STOP_FILE": str(stop_path)} if stop_when is not None else {}),
        }
        command: Final = [
            sys.executable,
            "-P",
            "-m",
            "locust",
            "--headless",
            "--only-summary",
            "--locustfile",
            str(_LOCUSTFILE),
            "--host",
            PROXY_BASE_URL,
            "--users",
            str(case.traffic.concurrency),
            "--spawn-rate",
            str(case.traffic.concurrency),
            "--run-time",
            f"{ceiling_s}s",
            "--exit-code-on-error",
            "0",
        ]
        completed: Final = (
            run_until(command, env, records_path, stop_path, stop_when, ceiling_s + 120)
            if stop_when is not None
            else subprocess.run(command, env=env, capture_output=True, text=True, timeout=ceiling_s + 120, check=False)
        )
        records: Final = tuple(
            _RECORD_ADAPTER.validate_json(line)
            for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        _keep_records_artifact(records_path)
    output: Final = (completed.stdout + completed.stderr)[-6000:]
    return RunOutcome(records=records, locust_exit=completed.returncode, locust_output=output)
