from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Final

from ...shared.reporting.models import HarnessCase, HarnessRun, RunStatus
from ...shared.reporting.strategy import ModuleCaseSpec, UpdateCallback


@dataclass(frozen=True, slots=True)
class E2ECheck:
    name: str
    execute: Callable[[], None]


@dataclass(frozen=True, slots=True)
class E2ELoadFailure:
    message: str


def _load_checks(reference: str) -> tuple[E2ECheck, ...] | E2ELoadFailure:
    try:
        module: Final = importlib.import_module(reference)
        factory: Final = getattr(module, "parity_checks", None)
        if not callable(factory):
            return E2ELoadFailure(f"{reference} must export parity_checks()")
        checks: Final = factory()
    except Exception as error:
        return E2ELoadFailure(f"cannot load {reference}: {type(error).__name__}: {error}")
    if not isinstance(checks, tuple) or not all(isinstance(check, E2ECheck) for check in checks):
        return E2ELoadFailure(f"{reference}.parity_checks() must return tuple[E2ECheck, ...]")
    return checks


def run_e2e_cases(
    cases: Sequence[HarnessCase],
    repo_root: Path,
    on_update: UpdateCallback,
    runner_args: Sequence[str] = (),
) -> tuple[int, HarnessRun]:
    del repo_root, runner_args
    run: Final = HarnessRun.from_cases(cases)
    for harness_case in cases:
        result: Final = run.results[harness_case.key]
        spec: Final = harness_case.spec
        if not isinstance(spec, ModuleCaseSpec) or spec.module is None:
            result.finalize()
            continue
        loaded: Final = _load_checks(spec.module)
        if isinstance(loaded, E2ELoadFailure):
            nodeid: Final = f"e2e:{harness_case.surface}:{harness_case.sdk_function}:load"
            result.collected.add(nodeid)
            result.record(nodeid, RunStatus.ERROR)
            run.failures.append((nodeid, loaded.message))
            on_update(run)
            continue
        nodeids: Final = tuple((check, f"e2e:{harness_case.surface}:{harness_case.sdk_function}:{check.name}") for check in loaded)
        result.collected.update(nodeid for _, nodeid in nodeids)
        if not nodeids:
            result.status = RunStatus.SKIPPED
            on_update(run)
            continue
        result.status = RunStatus.RUNNING
        on_update(run)
        for check, nodeid in nodeids:
            started_at: Final = monotonic()
            try:
                check.execute()
            except Exception as error:
                result.record(nodeid, RunStatus.FAILED, monotonic() - started_at)
                run.failures.append((nodeid, f"{type(error).__name__}: {error}"))
            else:
                result.record(nodeid, RunStatus.PASSED, monotonic() - started_at)
            on_update(run)
    run.finished_at = monotonic()
    on_update(run)
    failed: Final = any(
        result.status in {RunStatus.ERROR, RunStatus.FAILED, RunStatus.MISSING} for result in run.results.values()
    )
    return int(failed), run
