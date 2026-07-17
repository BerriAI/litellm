from __future__ import annotations

import os
import subprocess
import sys
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

    @property
    def failure_ratio(self) -> float:
        return self.failures / self.requests if self.requests else 1.0


def _aggregate(entries: list[_LocustStatEntry]) -> LoadResult:
    requests = sum(entry.num_requests for entry in entries)
    failures = sum(entry.num_failures for entry in entries)
    if not entries or requests == 0:
        return LoadResult(requests=requests, failures=failures, requests_per_second=0.0)
    elapsed = max(entry.last_request_timestamp for entry in entries) - min(entry.start_time for entry in entries)
    rps = requests / elapsed if elapsed > 0 else 0.0
    return LoadResult(requests=requests, failures=failures, requests_per_second=rps)


def run_chat_load(
    *,
    base_url: str,
    api_key: str,
    model: str,
    users: int,
    spawn_rate: float,
    duration_seconds: float,
) -> LoadResult:
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
        ],
        env={**os.environ, "LOAD_API_KEY": api_key, "LOAD_MODEL": model},
        capture_output=True,
        text=True,
        timeout=duration_seconds + 120,
        check=True,
    )
    return _aggregate(_STATS_ADAPTER.validate_json(completed.stdout))
