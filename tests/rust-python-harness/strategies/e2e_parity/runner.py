from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Final, cast

from ...shared.reporting.models import CaseResult, HarnessCase, HarnessRun, RunStatus
from ...shared.reporting.strategy import ModuleCaseSpec, UpdateCallback


@dataclass(frozen=True, slots=True)
class E2ECheck:
    name: str
    execute: Callable[[], None]


@dataclass(frozen=True, slots=True)
class E2ELoadFailure:
    message: str


def _load_checks(reference: str) -> AbstractContextManager[object] | E2ELoadFailure:
    try:
        module: Final = importlib.import_module(reference)
        factory_value: Final[object] = getattr(module, "parity_checks", None)
        if not callable(factory_value):
            return E2ELoadFailure(f"{reference} must export parity_checks()")
        factory: Final = cast(Callable[[], object], factory_value)
        checks_value: Final = factory()
    except Exception as error:
        return E2ELoadFailure(f"cannot load {reference}: {type(error).__name__}: {error}")
    if isinstance(checks_value, tuple):
        return nullcontext(cast(object, checks_value))
    if isinstance(checks_value, AbstractContextManager):
        return cast(AbstractContextManager[object], checks_value)
    return E2ELoadFailure(
        f"{reference}.parity_checks() must return tuple[E2ECheck, ...] or a context manager yielding one"
    )


def _validate_checks(reference: str, checks_value: object) -> tuple[E2ECheck, ...] | E2ELoadFailure:
    if not isinstance(checks_value, tuple):
        return E2ELoadFailure(f"{reference}.parity_checks() context manager must yield tuple[E2ECheck, ...]")
    untyped_checks: Final = cast(tuple[object, ...], checks_value)
    if not all(isinstance(check, E2ECheck) for check in untyped_checks):
        return E2ELoadFailure(f"{reference}.parity_checks() must return tuple[E2ECheck, ...]")
    return cast(tuple[E2ECheck, ...], untyped_checks)


def _run_check(
    run: HarnessRun,
    result: CaseResult,
    check: E2ECheck,
    nodeid: str,
    on_update: UpdateCallback,
) -> None:
    started_at: Final = monotonic()
    try:
        check.execute()
    except Exception as error:
        result.record(nodeid, RunStatus.FAILED, monotonic() - started_at)
        run.failures.append((nodeid, f"{type(error).__name__}: {error}"))
    else:
        result.record(nodeid, RunStatus.PASSED, monotonic() - started_at)
    on_update(run)


def _run_case(run: HarnessRun, harness_case: HarnessCase, on_update: UpdateCallback) -> None:
    result: Final = run.results[harness_case.key]
    spec: Final = harness_case.spec
    if not isinstance(spec, ModuleCaseSpec):
        return
    loaded: Final = _load_checks(spec.module)
    if isinstance(loaded, E2ELoadFailure):
        load_nodeid: Final = f"e2e:{harness_case.surface}:{harness_case.sdk_function}:load"
        result.collected.add(load_nodeid)
        result.record(load_nodeid, RunStatus.ERROR)
        run.failures.append((load_nodeid, loaded.message))
        on_update(run)
        return
    try:
        with loaded as checks_value:
            checks: Final = _validate_checks(spec.module, checks_value)
            if isinstance(checks, E2ELoadFailure):
                raise TypeError(checks.message)
            nodeids: Final = tuple(
                (check, f"e2e:{harness_case.surface}:{harness_case.sdk_function}:{check.name}") for check in checks
            )
            result.collected.update(nodeid for _, nodeid in nodeids)
            if not nodeids:
                result.status = RunStatus.SKIPPED
                on_update(run)
                return
            result.status = RunStatus.RUNNING
            on_update(run)
            for check, nodeid in nodeids:
                _run_check(run, result, check, nodeid, on_update)
    except Exception as error:
        session_nodeid: Final = f"e2e:{harness_case.surface}:{harness_case.sdk_function}:session"
        result.collected.add(session_nodeid)
        result.record(session_nodeid, RunStatus.ERROR)
        run.failures.append((session_nodeid, f"{type(error).__name__}: {error}"))
        on_update(run)


def run_e2e_cases(
    cases: Sequence[HarnessCase],
    repo_root: Path,
    on_update: UpdateCallback,
    runner_args: Sequence[str] = (),
) -> tuple[int, HarnessRun]:
    del repo_root, runner_args
    run: Final = HarnessRun.from_cases(cases)
    for harness_case in cases:
        _run_case(run, harness_case, on_update)
    run.finished_at = monotonic()
    on_update(run)
    failed: Final = any(
        result.status in {RunStatus.ERROR, RunStatus.FAILED, RunStatus.MISSING} for result in run.results.values()
    )
    return int(failed), run
