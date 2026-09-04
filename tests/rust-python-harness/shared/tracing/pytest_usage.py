from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import os
import subprocess
import sys
import tempfile
import warnings
from collections.abc import Generator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final

from pluggy import HookimplMarker
from pydantic import BaseModel, ConfigDict

from .profiler import profile_python_function_usage

if TYPE_CHECKING:
    import pytest

hookimpl: Final = HookimplMarker("pytest")


class PythonFunctionIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    file: str
    line: int
    qualname: str

    @property
    def raw(self) -> str:
        return f"{self.file}:{self.line} {self.qualname}"

    @property
    def key(self) -> str:
        return f"{self.file}::{self.qualname}"

    @classmethod
    def from_trace(cls, raw: str) -> PythonFunctionIdentity:
        location, separator, qualname = raw.partition(" ")
        file, line_separator, line = location.rpartition(":")
        if not separator or not line_separator or not file or not line.isdigit() or not qualname:
            raise ValueError(f"Unrecognized Python trace function: {raw}")
        return cls(file=file, line=int(line), qualname=qualname)


class PythonFunctionReference(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    module: str
    qualname: str

    @property
    def owner(self) -> str:
        return self.qualname.partition(".")[0]

    def resolve(self, source_root: Path) -> PythonFunctionIdentity:
        value: object = importlib.import_module(self.module)
        for component in self.qualname.split("."):
            value = getattr(value, component)
        function: Final = inspect.unwrap(value)
        code: Final = getattr(function, "__code__", None)
        if code is None:
            raise ValueError(f"Python function has no code object: {self.module}:{self.qualname}")
        source: Final = Path(code.co_filename).resolve()
        try:
            relative: Final = source.relative_to(source_root.resolve())
        except ValueError as error:
            raise ValueError(f"Python function is outside {source_root}: {source}") from error
        return PythonFunctionIdentity(
            file=relative.as_posix(),
            line=code.co_firstlineno,
            qualname=code.co_qualname,
        )


class RustFunctionIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    file: str
    line: int
    module_path: str
    function: str

    @property
    def test_module(self) -> str:
        _, separator, module = self.module_path.partition("::")
        if not separator:
            raise ValueError(f"Rust function has no crate-qualified module: {self.module_path}")
        return f"{module}::tests"

    @classmethod
    def from_trace(cls, raw: str) -> RustFunctionIdentity:
        location, separator, qualified = raw.partition(" ")
        file, line_separator, line = location.rpartition(":")
        module_path, function_separator, function = qualified.rpartition("::")
        if (
            not separator
            or not line_separator
            or not function_separator
            or not file
            or not line.isdigit()
            or not module_path
            or not function
        ):
            raise ValueError(f"Unrecognized Rust trace function: {raw}")
        return cls(file=file, line=int(line), module_path=module_path, function=function)


class PythonFunctionUsage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    function: PythonFunctionIdentity
    tests: tuple[str, ...]


class PythonUsageReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    usages: tuple[PythonFunctionUsage, ...]
    collected_tests: tuple[str, ...]
    exit_code: int
    problems: tuple[str, ...] = ()


def candidate_test_files(
    functions: Sequence[PythonFunctionReference | PythonFunctionIdentity],
    search_roots: Sequence[str],
    repo_root: Path,
    *,
    exclude_roots: Sequence[str] = (),
) -> tuple[str, ...]:
    owners: Final = frozenset(
        function.owner if isinstance(function, PythonFunctionReference) else function.qualname.partition(".")[0]
        for function in functions
        if "." in function.qualname
        and (
            isinstance(function, PythonFunctionReference)
            or function.file.startswith("ocr/")
            or "/ocr/" in function.file
        )
    )
    top_level_functions: Final = frozenset(
        function.qualname for function in functions if "." not in function.qualname and function.qualname.isidentifier()
    )
    candidates: Final = tuple(
        path.relative_to(repo_root).as_posix()
        for root in search_roots
        for path in sorted((repo_root / root).rglob("test*.py"))
        if not any(
            path == repo_root / excluded or path.is_relative_to(repo_root / excluded) for excluded in exclude_roots
        )
        if _references_function(path, owners, top_level_functions)
    )
    return tuple(dict.fromkeys(candidates))


def _references_function(path: Path, owners: frozenset[str], top_level_functions: frozenset[str]) -> bool:
    contents: Final = path.read_text(errors="ignore")
    if any(owner in contents for owner in owners):
        return True
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree: Final = ast.parse(contents)
    except SyntaxError:
        return False
    aliases: Final = frozenset(
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if alias.name in top_level_functions
    )
    names: Final = top_level_functions | aliases
    return any(
        isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id in names)
            or (isinstance(node.func, ast.Attribute) and node.func.attr in top_level_functions)
        )
        for node in ast.walk(tree)
    )


class _WorkerConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    functions: tuple[PythonFunctionIdentity, ...]
    source_root: Path
    output: Path
    pytest_args: tuple[str, ...]


class _FunctionUsagePlugin:
    def __init__(self, functions: tuple[PythonFunctionIdentity, ...], source_root: Path) -> None:
        self._functions: Final = functions
        self._function_names: Final = frozenset(function.raw for function in functions)
        self._source_root: Final = source_root
        self._tests_by_function: Final[dict[str, set[str]]] = {function.raw: set() for function in functions}
        self.collected_tests: tuple[str, ...] = ()
        self.problems: tuple[str, ...] = ()

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        self.collected_tests = tuple(item.nodeid for item in session.items)

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        if report.failed:
            self.problems = (*self.problems, str(report.longrepr))

    @hookimpl(hookwrapper=True)
    def pytest_runtest_protocol(self, item: pytest.Item, nextitem: pytest.Item | None) -> Generator[None, object, None]:
        del nextitem
        with profile_python_function_usage(self._source_root, self._function_names, threads=True) as profiler:
            yield
        for function in self._functions:
            if function.raw in profiler.called:
                self._tests_by_function[function.raw].add(item.nodeid)

    def usages(self) -> tuple[PythonFunctionUsage, ...]:
        return tuple(
            PythonFunctionUsage(
                function=function,
                tests=tuple(sorted(self._tests_by_function[function.raw])),
            )
            for function in self._functions
        )


def collect_python_function_tests(
    functions: Sequence[PythonFunctionIdentity],
    selectors: Sequence[str],
    repo_root: Path,
    *,
    source_root: Path | None = None,
    exclusions: Sequence[str] = (),
) -> PythonUsageReport:
    selected_functions: Final = tuple(dict.fromkeys(functions))
    if not selected_functions:
        raise ValueError("Python function discovery needs at least one function")
    if not selectors:
        raise ValueError("Python function discovery needs at least one test selector")
    with tempfile.TemporaryDirectory(prefix="litellm-function-tests-") as directory:
        temporary: Final = Path(directory)
        config_path: Final = temporary / "config.json"
        output_path: Final = temporary / "report.json"
        config: Final = _WorkerConfig(
            functions=selected_functions,
            source_root=source_root or repo_root / "litellm",
            output=output_path,
            pytest_args=tuple(
                (
                    "-o",
                    "consider_namespace_packages=true",
                    "-p",
                    "no:cacheprovider",
                    *selectors,
                    *(f"--deselect={nodeid}" for nodeid in exclusions),
                )
            ),
        )
        config_path.write_text(config.model_dump_json())
        import_roots: Final = tuple(
            dict.fromkeys(
                (
                    str(repo_root),
                    str(source_root or repo_root / "litellm"),
                    *(
                        str(path.parent if path.suffix == ".py" else path)
                        for selector in selectors
                        if (path := repo_root / selector.partition("::")[0]).exists()
                    ),
                    os.environ.get("PYTHONPATH", ""),
                )
            )
        )
        env: Final = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join(import_roots),
        }
        try:
            result: Final = subprocess.run(
                (sys.executable, "-m", __name__, str(config_path)),
                cwd=repo_root,
                env=env,
                capture_output=True,
                text=True,
                timeout=600,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            return PythonUsageReport(usages=(), collected_tests=(), exit_code=1, problems=(str(error),))
        if not output_path.exists():
            return PythonUsageReport(
                usages=(),
                collected_tests=(),
                exit_code=result.returncode or 1,
                problems=((result.stdout + result.stderr).strip(),),
            )
        report: Final = PythonUsageReport.model_validate_json(output_path.read_text())
        process_output: Final = (result.stdout + result.stderr).strip()
        if result.returncode and not report.problems and process_output:
            return report.model_copy(update={"problems": (process_output,)})
        return report


def _run_worker(config: _WorkerConfig) -> int:
    import pytest

    plugin: Final = _FunctionUsagePlugin(config.functions, config.source_root)
    exit_code: Final = int(pytest.main(list(config.pytest_args), plugins=[plugin]))
    report: Final = PythonUsageReport(
        usages=plugin.usages(),
        collected_tests=plugin.collected_tests,
        exit_code=exit_code,
        problems=plugin.problems,
    )
    config.output.write_text(report.model_dump_json())
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    parser: Final = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    namespace: Final = parser.parse_args(argv)
    config: Final = _WorkerConfig.model_validate_json(namespace.config.read_text())
    return _run_worker(config)


if __name__ == "__main__":
    raise SystemExit(main())
