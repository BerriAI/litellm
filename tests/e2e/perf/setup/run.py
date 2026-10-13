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
from setup.wire import WireRecord, WireRequest

_LOCUSTFILE: Final = Path(__file__).with_name("locustfile.py")
_PERF_DIR: Final = Path(__file__).resolve().parents[1]
_PLAN_ADAPTER: Final = TypeAdapter(list[WireRequest])
_RECORD_ADAPTER: Final = TypeAdapter(WireRecord)
_PATHS: Final = {"chat_completions": "/v1/chat/completions", "responses": "/v1/responses", "messages": "/v1/messages"}
_TOOLS: Final[tuple[tuple[str, str, dict[str, object]], ...]] = (
    (
        "get_weather",
        "Current weather for a city",
        {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    ),
    (
        "search_orders",
        "Find a customer's orders, optionally by status",
        {
            "type": "object",
            "properties": {"customer_id": {"type": "string"}, "status": {"type": "string"}},
            "required": ["customer_id"],
        },
    ),
)


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


def _openai_tools() -> list[dict[str, object]]:
    return [
        {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}
        for name, description, parameters in _TOOLS
    ]


def _responses_tools() -> list[dict[str, object]]:
    return [
        {"type": "function", "name": name, "description": description, "parameters": parameters}
        for name, description, parameters in _TOOLS
    ]


def _anthropic_tools() -> list[dict[str, object]]:
    return [
        {"name": name, "description": description, "input_schema": parameters}
        for name, description, parameters in _TOOLS
    ]


def _turns(item: PlannedRequest) -> list[dict[str, object]]:
    """The history as alternating user and assistant turns, then the final user message."""
    earlier: Final[list[dict[str, object]]] = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": text} for index, text in enumerate(item.history)
    ]
    return [*earlier, {"role": "user", "content": item.prompt}]


def request_body(item: PlannedRequest) -> dict[str, object]:
    """The request in the endpoint's own wire format, sized by the planned `RequestShape`."""
    if item.endpoint == "responses":
        return {
            "model": item.model_group,
            "input": _turns(item),
            "max_output_tokens": item.max_tokens,
            "stream": item.stream,
            **({"instructions": item.system} if item.system else {}),
            **({"tools": _responses_tools()} if item.shape.tools else {}),
        }
    if item.endpoint == "messages":
        return {
            "model": item.model_group,
            "messages": _turns(item),
            "max_tokens": item.max_tokens,
            "stream": item.stream,
            **({"system": item.system} if item.system else {}),
            **({"tools": _anthropic_tools()} if item.shape.tools else {}),
        }
    return {
        "model": item.model_group,
        "messages": [*([{"role": "system", "content": item.system}] if item.system else []), *_turns(item)],
        "max_tokens": item.max_tokens,
        "stream": item.stream,
        **({"tools": _openai_tools()} if item.shape.tools else {}),
    }


def wire_request(item: PlannedRequest, setup: CaseSetup, *, post_guardrail: bool) -> WireRequest:
    return WireRequest(
        seq=item.seq,
        marker=item.marker,
        caller=item.caller.label,
        endpoint=item.endpoint,
        path=_PATHS[item.endpoint],
        body=request_body(item),
        authorization=setup.authorization(item.caller),
        token_source=setup.token_source(item.caller),
        stream=item.stream,
        expected_text=item.expected_text(post_guardrail=post_guardrail),
        blocked=item.blocked,
    )


def run_plan(case: PerformanceTestCase, plan: tuple[PlannedRequest, ...], setup: CaseSetup) -> RunOutcome:
    """Replay `plan` with Locust in a child process and collect its records. The run-time ceiling
    is well past the planned duration so Locust's own clock never cuts a slow run short; the
    locustfile quits itself once the last record is written."""
    wire: Final = [wire_request(item, setup, post_guardrail=case.has_post_guardrail) for item in plan]
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
