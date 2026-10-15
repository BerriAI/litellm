"""Sends a case's plan through the gateway from a separate process: the plan is written to disk as
wire requests, Locust (`locustfile.py`) replays it at the planned rate, and the records it
wrote come back as `WireRecord`s."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

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
    )


def run_plan(case: PerformanceTestCase, plan: tuple[PlannedRequest, ...], setup: CaseSetup) -> RunOutcome:
    """Replay `plan` with Locust in a child process and collect its records. The run-time ceiling
    is well past the planned duration so Locust's own clock never cuts a slow run short; the
    locustfile quits itself once the last record is written."""
    wire: Final = [wire_request(item, setup) for item in plan]
    ceiling_s: Final = int(case.traffic.expected_seconds * 2 + 180)
    with tempfile.TemporaryDirectory(prefix="e2e-perf-") as directory:
        plan_path: Final = Path(directory) / "plan.json"
        records_path: Final = Path(directory) / "records.jsonl"
        _ = plan_path.write_bytes(_PLAN_ADAPTER.dump_json(wire))
        records_path.touch()
        env: Final = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join(filter(None, (str(_PERF_DIR), os.environ.get("PYTHONPATH")))),
            "PERF_PLAN_FILE": str(plan_path),
            "PERF_RECORDS_FILE": str(records_path),
            "PERF_RPS": str(case.traffic.rps),
            "PERF_USERS": str(case.traffic.concurrency),
        }
        completed: Final = subprocess.run(
            [
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
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=ceiling_s + 120,
            check=False,
        )
        records: Final = tuple(
            _RECORD_ADAPTER.validate_json(line)
            for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    output: Final = (completed.stdout + completed.stderr)[-6000:]
    return RunOutcome(records=records, locust_exit=completed.returncode, locust_output=output)
