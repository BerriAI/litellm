from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, cast

from pluggy import HookimplMarker
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    import pytest

hookimpl: Final = HookimplMarker("pytest")

Backend = Literal["python", "rust"]


class PythonReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    backend: Backend
    verified: bool
    tests: tuple[str, ...] = ()
    outcomes: tuple[tuple[str, str, str], ...] = ()
    exit_code: int
    problems: tuple[str, ...] = ()


class BackendSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    environment_variable: str
    probe: str = ""


class WorkerArgs(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    backend: Backend
    probe: str
    output: Path
    pytest_args: tuple[str, ...]


class ResultPlugin:
    def __init__(self, backend: Backend, probe: Callable[[], object] | None) -> None:
        self.backend: Final = backend
        self.probe: Final = probe
        self.tests: tuple[str, ...] = ()
        self.outcomes: tuple[tuple[str, str, str], ...] = ()
        self.problems: tuple[str, ...] = ()

    def verify(self) -> None:
        if self.probe is not None and self.probe() != self.backend:
            raise RuntimeError(f"backend probe did not select {self.backend}")

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        self.tests = tuple(item.nodeid for item in session.items)

    @hookimpl(tryfirst=True)
    def pytest_runtest_call(self, item: pytest.Item) -> None:
        del item
        self.verify()

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        if report.failed:
            self.problems = (*self.problems, str(report.longrepr))

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        self.outcomes = (*self.outcomes, (report.nodeid, report.when, report.outcome))
        if report.failed:
            self.problems = (*self.problems, str(report.longrepr))


def run_python_tests(
    selectors: Sequence[str],
    repo_root: Path,
    backend: Backend,
    spec: BackendSpec,
    pytest_args: Sequence[str] = (),
) -> PythonReport:
    with tempfile.TemporaryDirectory(prefix="litellm-unit-tests-") as directory:
        output: Final = Path(directory) / "report.json"
        command: Final = (
            sys.executable,
            "-m",
            __name__,
            "--backend",
            backend,
            *(("--probe", spec.probe) if spec.probe else ()),
            "--output",
            str(output),
            "--",
            *selectors,
            *pytest_args,
        )
        env: Final = {
            **os.environ,
            spec.environment_variable: "1" if backend == "rust" else "0",
            "PYTHONPATH": os.pathsep.join((str(repo_root), os.environ.get("PYTHONPATH", ""))),
        }
        try:
            result: Final = subprocess.run(
                command, cwd=repo_root, env=env, capture_output=True, text=True, timeout=600, check=False
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            return PythonReport(backend=backend, verified=False, exit_code=1, problems=(str(error),))
        if not output.exists():
            return PythonReport(
                backend=backend,
                verified=False,
                exit_code=result.returncode or 1,
                problems=(result.stdout + result.stderr,),
            )
        report: Final = PythonReport.model_validate_json(output.read_text())
        process_output: Final = (result.stdout + result.stderr).strip()
        if result.returncode and not report.problems and process_output:
            return report.model_copy(update={"problems": (process_output,)})
        if report.exit_code != result.returncode:
            return report.model_copy(
                update={
                    "exit_code": result.returncode or 1,
                    "problems": (*report.problems, "worker exit code differs from report"),
                }
            )
        return report


def compare_python_runs(python: PythonReport, rust: PythonReport) -> tuple[str, ...]:
    python_only: Final = tuple(sorted(set(python.outcomes) - set(rust.outcomes)))
    rust_only: Final = tuple(sorted(set(rust.outcomes) - set(python.outcomes)))
    return (
        *(("backend selection was not verified",) if not python.verified or not rust.verified else ()),
        *(("Python run used the wrong backend",) if python.backend != "python" else ()),
        *(("Rust run used the wrong backend",) if rust.backend != "rust" else ()),
        *(("Python/Rust test inventories differ",) if python.tests != rust.tests else ()),
        *(("Python/Rust test outcomes differ",) if python_only or rust_only else ()),
        *(f"Python only: {nodeid} [{stage}] {outcome}" for nodeid, stage, outcome in python_only),
        *(f"Rust only: {nodeid} [{stage}] {outcome}" for nodeid, stage, outcome in rust_only),
        *(f"Python run: {problem}" for problem in python.problems if not python.verified or not python.tests),
        *(f"Rust run: {problem}" for problem in rust.problems if not rust.verified or not rust.tests),
        *(("no Python tests collected",) if not python.tests else ()),
        *(("Python/Rust exit codes differ",) if python.exit_code != rust.exit_code else ()),
    )


def _load_probe(reference: str) -> Callable[[], object] | None:
    if not reference:
        return None
    module, name = reference.rsplit(":", 1)
    return cast(Callable[[], object], getattr(importlib.import_module(module), name))


def main(argv: Sequence[str] | None = None) -> int:
    import pytest

    parser: Final = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=("python", "rust"))
    parser.add_argument("--probe", default="")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    namespace: Final = parser.parse_args(argv)
    args: Final = WorkerArgs.model_validate(vars(namespace))
    try:
        plugin: Final = ResultPlugin(args.backend, _load_probe(args.probe))
        plugin.verify()
        code: Final = int(
            pytest.main(["-o", "consider_namespace_packages=true", *args.pytest_args[1:]], plugins=[plugin])
        )
        report: Final = PythonReport(
            backend=args.backend,
            verified=True,
            tests=plugin.tests,
            outcomes=plugin.outcomes,
            exit_code=code,
            problems=plugin.problems,
        )
    except Exception as error:
        failure: Final = PythonReport(backend=args.backend, verified=False, exit_code=1, problems=(str(error),))
        args.output.write_text(failure.model_dump_json())
        return 1
    args.output.write_text(report.model_dump_json())
    return report.exit_code


def contract_nodeid(nodeid: str) -> str:
    owner, separator, test = nodeid.rpartition("::")
    function: Final = test.partition("[")[0]
    if not separator or not function.startswith("test_"):
        raise ValueError(f"Unrecognized pytest node id: {nodeid}")
    return f"{owner}::{function}"


def collect_python_tests(selectors: Sequence[str], repo_root: Path) -> frozenset[str]:
    report: Final = run_python_tests(
        selectors,
        repo_root,
        "python",
        BackendSpec(environment_variable="LITELLM_RUST"),
        ("--collect-only", "-p", "no:cacheprovider"),
    )
    if report.exit_code or report.problems:
        details: Final = "\n".join(report.problems) or f"pytest exited with code {report.exit_code}"
        raise ValueError(f"Python test collection failed:\n{details}")
    tests: Final = frozenset(contract_nodeid(nodeid) for nodeid in report.tests)
    if not tests:
        raise ValueError(f"pytest collected no tests for: {', '.join(selectors)}")
    return tests


if __name__ == "__main__":
    raise SystemExit(main())
