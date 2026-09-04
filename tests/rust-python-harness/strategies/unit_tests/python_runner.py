from __future__ import annotations

import argparse
import ast
import importlib
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final, Literal, cast

import pytest
from pydantic import BaseModel, ConfigDict

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
    probe: str


def ocr_backend() -> Backend:
    from litellm.rust_bridge import native_bridge_available
    from litellm.rust_bridge.configuration import rust_ocr_enabled

    if not rust_ocr_enabled():
        return "python"
    if not native_bridge_available():
        raise RuntimeError("Rust OCR was enabled but the native extension is unavailable")
    return "rust"


class ResultPlugin:
    def __init__(self, backend: Backend, probe: Callable[[], object]) -> None:
        self.backend: Final = backend
        self.probe: Final = probe
        self.tests: tuple[str, ...] = ()
        self.outcomes: tuple[tuple[str, str, str], ...] = ()
        self.problems: tuple[str, ...] = ()

    def verify(self) -> None:
        if self.probe() != self.backend:
            raise RuntimeError(f"backend probe did not select {self.backend}")

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        self.tests = tuple(item.nodeid for item in session.items)

    @pytest.hookimpl(tryfirst=True)
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
            "--probe",
            spec.probe,
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
        if report.exit_code != result.returncode:
            return report.model_copy(
                update={
                    "exit_code": result.returncode or 1,
                    "problems": (*report.problems, "worker exit code differs from report"),
                }
            )
        return report


def compare_python_runs(python: PythonReport, rust: PythonReport) -> tuple[str, ...]:
    return (
        *(("backend selection was not verified",) if not python.verified or not rust.verified else ()),
        *(("Python run used the wrong backend",) if python.backend != "python" else ()),
        *(("Rust run used the wrong backend",) if rust.backend != "rust" else ()),
        *(("Python/Rust test inventories differ",) if python.tests != rust.tests else ()),
        *(("Python/Rust test outcomes differ",) if sorted(python.outcomes) != sorted(rust.outcomes) else ()),
        *(("no Python tests collected",) if not python.tests else ()),
        *(
            ("Python tests did not all pass",)
            if set(python.tests)
            != {node for node, phase, status in python.outcomes if phase == "call" and status == "passed"}
            else ()
        ),
        *(
            ("Rust-enabled Python tests did not all pass",)
            if set(rust.tests)
            != {node for node, phase, status in rust.outcomes if phase == "call" and status == "passed"}
            else ()
        ),
        *(("Python test run failed",) if python.exit_code else ()),
        *(("Rust-enabled Python test run failed",) if rust.exit_code else ()),
        *python.problems,
        *rust.problems,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser: Final = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=("python", "rust"))
    parser.add_argument("--probe", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args: Final = parser.parse_args(argv)
    try:
        module, name = args.probe.rsplit(":", 1)
        probe: Final = cast(Callable[[], object], getattr(importlib.import_module(module), name))
        plugin: Final = ResultPlugin(args.backend, probe)
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


def enumerate_python_tests(repo_root: Path, relative_path: str) -> frozenset[str]:
    source = (repo_root / relative_path).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=relative_path)

    module_level: list[str] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            module_level.append(node.name)
        elif isinstance(node, ast.ClassDef):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name.startswith(
                    "test_"
                ):
                    module_level.append(f"{node.name}::{child.name}")

    return frozenset(module_level)


if __name__ == "__main__":
    raise SystemExit(main())
