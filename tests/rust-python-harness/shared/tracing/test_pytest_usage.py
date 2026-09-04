from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest

from .pytest_usage import (
    PythonFunctionIdentity,
    PythonFunctionReference,
    RustFunctionIdentity,
    candidate_test_files,
    collect_python_function_tests,
)


def test_collects_parameterized_tests_that_execute_function(tmp_path: Path) -> None:
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    (tmp_path / "source.py").write_text("def target():\n    return 1\n\ndef other():\n    return 2\n")
    (tmp_path / "test_source.py").write_text(
        "import pytest\n"
        "from source import other, target\n"
        "@pytest.mark.parametrize('value', [1, 2])\n"
        "def test_target(value): assert target() + value > 0\n"
        "def test_other(): assert other() == 2\n"
    )
    target: Final = PythonFunctionIdentity(file="source.py", line=1, qualname="target")

    report: Final = collect_python_function_tests(
        (target,),
        ("test_source.py",),
        tmp_path,
        source_root=tmp_path,
    )

    assert report.exit_code == 0, report.problems
    assert report.usages[0].tests == (
        "test_source.py::test_target[1]",
        "test_source.py::test_target[2]",
    )


def test_collects_async_and_threaded_function_calls(tmp_path: Path) -> None:
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    (tmp_path / "source.py").write_text(
        "async def async_target():\n    return 1\n\ndef threaded_target():\n    return 2\n"
    )
    (tmp_path / "test_source.py").write_text(
        "import asyncio\n"
        "from threading import Thread\n"
        "from source import async_target, threaded_target\n"
        "def test_async(): assert asyncio.run(async_target()) == 1\n"
        "def test_thread():\n"
        "    thread = Thread(target=threaded_target)\n"
        "    thread.start()\n"
        "    thread.join()\n"
    )
    functions: Final = (
        PythonFunctionIdentity(file="source.py", line=1, qualname="async_target"),
        PythonFunctionIdentity(file="source.py", line=4, qualname="threaded_target"),
    )

    report: Final = collect_python_function_tests(
        functions,
        ("test_source.py",),
        tmp_path,
        source_root=tmp_path,
    )

    assert report.exit_code == 0, report.problems
    assert report.usages[0].tests == ("test_source.py::test_async",)
    assert report.usages[1].tests == ("test_source.py::test_thread",)


def test_adds_candidate_directory_to_worker_import_path(tmp_path: Path) -> None:
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    source: Final = tmp_path / "source"
    tests: Final = tmp_path / "tests"
    source.mkdir()
    tests.mkdir()
    (source / "implementation.py").write_text("def target():\n    return 1\n")
    (tests / "helper.py").write_text("VALUE = 1\n")
    (tests / "test_source.py").write_text(
        "from helper import VALUE\nfrom implementation import target\ndef test_target(): assert target() == VALUE\n"
    )
    target: Final = PythonFunctionIdentity(file="implementation.py", line=1, qualname="target")

    report: Final = collect_python_function_tests(
        (target,),
        ("tests/test_source.py",),
        tmp_path,
        source_root=source,
    )

    assert report.exit_code == 0, report.problems
    assert report.usages[0].tests == ("tests/test_source.py::test_target",)


def test_parses_function_identity_from_trace() -> None:
    function: Final = PythonFunctionIdentity.from_trace("llms/mistral/ocr/transformation.py:72 Config.map")

    assert function.file == "llms/mistral/ocr/transformation.py"
    assert function.line == 72
    assert function.qualname == "Config.map"


def test_resolves_function_and_finds_candidate_test_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package: Final = tmp_path / "package"
    tests: Final = tmp_path / "tests"
    package.mkdir()
    tests.mkdir()
    (package / "__init__.py").write_text("")
    (package / "implementation.py").write_text("class Config:\n    def transform(self):\n        return 1\n")
    (tests / "test_implementation.py").write_text("from package.implementation import Config\n")
    (tests / "test_unrelated.py").write_text("def test_other(): pass\n")
    monkeypatch.syspath_prepend(tmp_path)
    reference: Final = PythonFunctionReference(module="package.implementation", qualname="Config.transform")

    function: Final = reference.resolve(tmp_path)
    candidates: Final = candidate_test_files((reference,), ("tests",), tmp_path)

    assert function.file == "package/implementation.py"
    assert function.qualname == "Config.transform"
    assert candidates == ("tests/test_implementation.py",)


def test_candidate_test_files_excludes_harness_roots(tmp_path: Path) -> None:
    tests: Final = tmp_path / "tests"
    harness: Final = tests / "harness"
    harness.mkdir(parents=True)
    (tests / "test_implementation.py").write_text("from package.implementation import Config\n")
    (harness / "test_fixture.py").write_text("from package.implementation import Config\n")
    function: Final = PythonFunctionReference(module="package.implementation", qualname="Config.transform")

    candidates: Final = candidate_test_files(
        (function,),
        ("tests",),
        tmp_path,
        exclude_roots=("tests/harness",),
    )

    assert candidates == ("tests/test_implementation.py",)


def test_candidate_test_files_finds_top_level_calls_and_import_aliases(tmp_path: Path) -> None:
    tests: Final = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_attribute.py").write_text("import package\ndef test_call(): package.ocr()\n")
    (tests / "test_alias.py").write_text("from package import ocr as run_ocr\ndef test_call(): run_ocr()\n")
    (tests / "test_unrelated.py").write_text("def test_call(): return 'ocr'\n")
    function: Final = PythonFunctionIdentity(file="ocr/main.py", line=1, qualname="ocr")

    candidates: Final = candidate_test_files((function,), ("tests",), tmp_path)

    assert candidates == (
        "tests/test_alias.py",
        "tests/test_attribute.py",
    )


def test_parses_rust_function_identity_and_derives_test_module() -> None:
    function: Final = RustFunctionIdentity.from_trace(
        "crates/core/src/providers/mistral/ocr/transformation.rs:73 "
        "litellm_core::providers::mistral::ocr::transformation::supported_ocr_params"
    )

    assert function.file == "crates/core/src/providers/mistral/ocr/transformation.rs"
    assert function.test_module == "providers::mistral::ocr::transformation::tests"
