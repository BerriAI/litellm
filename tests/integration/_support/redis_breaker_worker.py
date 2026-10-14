"""One SDK worker process for the Redis circuit-breaker metric cells: a Redis-cached completion against an owned
peer, then a report of the breaker state gauge on that process's own default registry. Run as a script by
``run_breaker_worker`` so ``PROMETHEUS_MULTIPROC_DIR`` is in place before ``prometheus_client`` is imported."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, TypeAlias

from pydantic import TypeAdapter

Scenario: TypeAlias = Literal["callbacks_before_cache", "callbacks_after_cache", "no_callbacks"]

STATE_GAUGE: Final = "litellm_redis_circuit_breaker_state"
WORKER_TIMEOUT_SECONDS: Final = 120


@dataclass(frozen=True, slots=True)
class WorkerReport:
    pid: int
    answer: str
    state_series: tuple[tuple[tuple[tuple[str, str], ...], float], ...]


_REPORT: Final = TypeAdapter(WorkerReport)


def run_breaker_worker(
    scenario: Scenario, api_base: str, prom_dir: Path, environment: Mapping[str, str] = os.environ
) -> WorkerReport:
    completed: Final = subprocess.run(
        (sys.executable, "-I", __file__, scenario, api_base, environment["REDIS_HOST"], environment["REDIS_PORT"]),
        env={**environment, "PROMETHEUS_MULTIPROC_DIR": str(prom_dir)},
        capture_output=True,
        text=True,
        timeout=WORKER_TIMEOUT_SECONDS,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    return _REPORT.validate_json(completed.stdout.splitlines()[-1])


def _state_series_on_default_registry() -> tuple[tuple[tuple[tuple[str, str], ...], float], ...]:
    from prometheus_client import REGISTRY

    return tuple(sorted(_series(REGISTRY.collect())))


def _series(families: Iterator[object]) -> Iterator[tuple[tuple[tuple[str, str], ...], float]]:
    for family in families:
        if getattr(family, "name", None) != STATE_GAUGE:
            continue
        for sample in getattr(family, "samples", ()):
            yield tuple(sorted(sample.labels.items())), float(sample.value)


def main(scenario: str, api_base: str, redis_host: str, redis_port: str) -> None:
    import litellm
    from litellm.caching.caching import Cache

    if scenario == "callbacks_before_cache":
        litellm.callbacks = ["prometheus"]  # test-quality-ok: this worker process exits right after the report
    litellm.cache = Cache(type="redis", host=redis_host, port=redis_port)  # test-quality-ok: throwaway worker process
    if scenario == "callbacks_after_cache":
        litellm.callbacks = ["prometheus"]  # test-quality-ok: throwaway worker process
    response: Final = litellm.completion(
        model="openai/gpt-5.6",
        api_base=api_base,
        api_key="synthetic-breaker-worker-key",
        messages=[{"role": "user", "content": f"breaker worker {os.getpid()}"}],
    )
    report: Final = WorkerReport(
        pid=os.getpid(),
        answer=str(response.choices[0].message.content),
        state_series=_state_series_on_default_registry(),
    )
    sys.stdout.write(json.dumps(_REPORT.dump_python(report)) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:5])
