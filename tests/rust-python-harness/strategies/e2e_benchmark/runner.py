from __future__ import annotations

import argparse
import platform
import subprocess
from collections.abc import Sequence
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Final

from ...shared.reporting.models import CaseResult, HarnessCase, HarnessRun, ResultArtifact, RunStatus
from ...shared.reporting.strategy import ModuleCaseSpec, UpdateCallback
from .execution import benchmark
from .models import Backend, BenchmarkModel, Measurement, Options, Profile, Route
from .reporting import ARTIFACT_KIND, MEASUREMENTS, measurements

if TYPE_CHECKING:
    from .workloads import Workload


class Report(BenchmarkModel):
    schema_version: int = 1
    revision: str
    working_tree_dirty: bool
    platform: str
    options: Options
    measurements: tuple[Measurement, ...]
    failures: tuple[tuple[str, str], ...]


def parse_options(arguments: Sequence[str]) -> Options:
    parser: Final = argparse.ArgumentParser(prog="e2e_benchmark", exit_on_error=False)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile", dest="profiles", action="append", default=argparse.SUPPRESS)
    parser.add_argument("--route", dest="routes", action="append", default=argparse.SUPPRESS)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--sample-interval-ms", type=float, default=5)
    parser.add_argument("--output")
    parsed, unknown = parser.parse_known_args(arguments)
    if unknown:
        raise ValueError(f"unknown benchmark arguments: {' '.join(unknown)}")
    return Options.model_validate(vars(parsed))


def _run_pair(
    workload: Workload, route: Route, repeat: int, options: Options, repo_root: Path
) -> tuple[Measurement, ...]:
    order: Final[tuple[Backend, Backend]] = ("python", "rust") if repeat % 2 == 0 else ("rust", "python")
    pair: Final = tuple(benchmark(workload, route, backend, repeat, options, repo_root) for backend in order)
    if pair[0].ready.response_digest != pair[1].ready.response_digest:
        raise ValueError("Python and Rust preflight SDK responses differ; run e2e_parity before comparing performance")
    if any(len(value.timing.latency_ms) != options.iterations for value in pair):
        raise ValueError("SDK worker returned an incomplete measurement")
    return pair


def _run_job(
    result: CaseResult,
    run: HarnessRun,
    options: Options,
    repo_root: Path,
    job: tuple[Profile, Route, int, str],
) -> bool:
    from .workloads import ocr_workload

    profile, route, repeat, nodeid = job
    start: Final = monotonic()
    try:
        pair: Final = _run_pair(ocr_workload(profile), route, repeat, options, repo_root)
    except Exception as error:
        result.record(nodeid, RunStatus.ERROR, monotonic() - start)
        run.failures.append((nodeid, f"{type(error).__name__}: {error}"))
        return False
    result.record(
        nodeid,
        RunStatus.PASSED,
        monotonic() - start,
        artifacts=(ResultArtifact(ARTIFACT_KIND, MEASUREMENTS.dump_json(pair).decode()),),
    )
    return True


def _run_case(result: CaseResult, run: HarnessRun, options: Options, repo_root: Path, update: UpdateCallback) -> None:
    jobs: Final[tuple[tuple[Profile, Route, int, str], ...]] = tuple(
        (profile, route, repeat, f"benchmark:{route}:{profile}:{repeat}")
        for profile in dict.fromkeys(options.profiles)
        for route in dict.fromkeys(options.routes)
        for repeat in range(options.repeats)
    )
    result.collected.update(nodeid for _, _, _, nodeid in jobs)
    for job in jobs:
        result.status = RunStatus.RUNNING
        run.current_nodeid = job[3]
        update(run)
        if not _run_job(result, run, options, repo_root, job):
            update(run)
            return
        update(run)


def run_benchmark_cases(
    cases: Sequence[HarnessCase],
    repo_root: Path,
    on_update: UpdateCallback,
    runner_args: Sequence[str] = (),
) -> tuple[int, HarnessRun]:
    options: Final = parse_options(runner_args)
    run: Final = HarnessRun.from_cases(cases)
    for result in run.results.values():
        if isinstance(result.case.spec, ModuleCaseSpec):
            _run_case(result, run, options, repo_root, on_update)
    run.finished_at = monotonic()
    if options.output:
        revision: Final = subprocess.run(
            ("git", "rev-parse", "HEAD"), cwd=repo_root, capture_output=True, text=True, check=True
        ).stdout.strip()
        report: Final = Report(
            revision=revision,
            working_tree_dirty=bool(
                subprocess.run(
                    ("git", "status", "--porcelain"), cwd=repo_root, capture_output=True, text=True, check=True
                ).stdout.strip()
            ),
            platform=platform.platform(),
            options=options,
            measurements=measurements(tuple(run.results.values())),
            failures=tuple(run.failures),
        )
        Path(options.output).write_text(report.model_dump_json(indent=2) + "\n")
    on_update(run)
    return int(bool(run.failures)), run
