from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, TypeAdapter

_LOCUSTFILE = Path(__file__).with_name("locustfile.py")


class _LocustStatEntry(BaseModel):
    num_requests: int
    num_failures: int
    start_time: float
    last_request_timestamp: float


_STATS_ADAPTER: TypeAdapter[list[_LocustStatEntry]] = TypeAdapter(list[_LocustStatEntry])


@dataclass(frozen=True, slots=True)
class LoadResult:
    requests: int
    failures: int
    requests_per_second: float
    failure_reasons: tuple[tuple[str, int], ...] = ()

    @property
    def failure_ratio(self) -> float:
        return self.failures / self.requests if self.requests else 1.0

    def format_failure_reasons(self, *, limit: int = 10) -> str:
        if not self.failure_reasons:
            return "(no per-status breakdown; Locust did not write a failure report)"
        lines = [f"  {reason}: {count}" for reason, count in self.failure_reasons[:limit]]
        return "\n".join(lines)


def _aggregate(
    entries: list[_LocustStatEntry],
    failure_reasons: tuple[tuple[str, int], ...],
) -> LoadResult:
    requests = sum(entry.num_requests for entry in entries)
    failures = sum(entry.num_failures for entry in entries)
    if not entries or requests == 0:
        return LoadResult(
            requests=requests,
            failures=failures,
            requests_per_second=0.0,
            failure_reasons=failure_reasons,
        )
    elapsed = max(entry.last_request_timestamp for entry in entries) - min(
        entry.start_time for entry in entries
    )
    rps = requests / elapsed if elapsed > 0 else 0.0
    return LoadResult(
        requests=requests,
        failures=failures,
        requests_per_second=rps,
        failure_reasons=failure_reasons,
    )


def _read_failure_report(path: Path) -> tuple[tuple[str, int], ...]:
    if not path.is_file():
        return ()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    if not isinstance(raw, dict):
        return ()
    pairs: list[tuple[str, int]] = []
    for key, value in raw.items():
        if isinstance(key, str) and isinstance(value, int):
            pairs.append((key, value))
    pairs.sort(key=lambda item: item[1], reverse=True)
    return tuple(pairs)


def run_chat_load(
    *,
    base_url: str,
    api_key: str,
    model: str,
    users: int,
    spawn_rate: float,
    duration_seconds: float,
) -> LoadResult:
    with tempfile.TemporaryDirectory(prefix="e2e-load-") as tmp:
        report_path = Path(tmp) / "failure_reasons.json"
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "locust",
                "--headless",
                "--json",
                "--locustfile",
                str(_LOCUSTFILE),
                "--host",
                base_url,
                "--users",
                str(users),
                "--spawn-rate",
                str(spawn_rate),
                "--run-time",
                f"{int(duration_seconds)}s",
                "--exit-code-on-error",
                "0",
            ],
            env={
                **os.environ,
                "LOAD_API_KEY": api_key,
                "LOAD_MODEL": model,
                "LOAD_FAILURE_REPORT_PATH": str(report_path),
            },
            capture_output=True,
            text=True,
            timeout=duration_seconds + 120,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"locust exited {completed.returncode} before it could report throughput "
                f"(a startup failure, not request failures, which are folded into the JSON "
                f"summary via --exit-code-on-error 0):\n{completed.stderr}"
            )
        try:
            entries = _STATS_ADAPTER.validate_json(completed.stdout)
        except ValueError as exc:
            raise RuntimeError(
                f"locust exited 0 but did not print a parseable --json throughput summary "
                f"on stdout; got stdout={completed.stdout!r}, stderr={completed.stderr!r}"
            ) from exc
        return _aggregate(entries, _read_failure_report(report_path))
