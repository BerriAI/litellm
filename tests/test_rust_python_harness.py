from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Final

import pytest

catalog = importlib.import_module("tests.rust-python-harness.catalog")
cli = importlib.import_module("tests.rust-python-harness.cli")
models = importlib.import_module("tests.rust-python-harness.shared.reporting.models")
runner = importlib.import_module("tests.rust-python-harness.shared.reporting.pytest_runner")
ui = importlib.import_module("tests.rust-python-harness.shared.reporting.ui")
ledger_module = importlib.import_module("tests.rust-python-harness.shared.parity.ledger")
mapping_validator = importlib.import_module(
    "tests.rust-python-harness.strategies.unit_tests.mapping_validator"
)

load_catalog = catalog.load_catalog
load_ledger = ledger_module.load_ledger
ledger_path_for = mapping_validator.ledger_path_for
REPO_ROOT = mapping_validator.REPO_ROOT
audit_ledger = mapping_validator.audit_ledger
build_function_report = mapping_validator.build_function_report
_pick_values = cli._pick_values
_coverage_pytest_args = cli._coverage_pytest_args
_select = cli._select
_validate_ledger = cli._validate_ledger
CaseResult = models.CaseResult
Coverage = models.Coverage
HarnessCase = models.HarnessCase
HarnessRun = models.HarnessRun
RunStatus = models.RunStatus
SDK_FUNCTIONS = models.SDK_FUNCTIONS
section_confidence = models.section_confidence
run_pytest = runner.run_pytest
runnable_selectors = runner.runnable_selectors
selector_matches_node = runner.selector_matches_node
_format_duration = ui._format_duration
_rerun_command = ui._rerun_command
_summary = ui._summary


def _case(
    *, selectors: tuple[str, ...] = (), coverage: Coverage = Coverage.COMPLETE
) -> HarnessCase:
    return HarnessCase(
        strategy_id="example",
        strategy_label="Example",
        sdk_function="messages",
        coverage=coverage,
        selectors=selectors,
    )


def _manifest() -> dict[str, object]:
    return {
        "order": 1,
        "id": "example",
        "label": "Example strategy",
        "description": "Example description",
        "functions": {
            function: {"coverage": "planned", "selectors": []}
            for function in SDK_FUNCTIONS
        },
    }


def test_should_load_the_four_harness_strategies_in_order() -> None:
    strategies = load_catalog()

    assert [strategy.id for strategy in strategies] == [
        "e2e_parity",
        "trace_parity",
        "unit_tests",
        "existing_e2e_test_sdk",
    ]
    assert all(
        tuple(case.sdk_function for case in strategy.cases) == SDK_FUNCTIONS
        for strategy in strategies
    )


def test_should_reject_a_manifest_missing_an_sdk_function(tmp_path: Path) -> None:
    strategy_directory = tmp_path / "example"
    strategy_directory.mkdir()
    manifest = _manifest()
    del manifest["functions"]["count_tokens"]  # type: ignore[index]
    (strategy_directory / "strategy.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="functions must exactly match"):
        load_catalog(tmp_path)


@pytest.mark.parametrize(
    ("selector", "nodeid", "matches"),
    [
        ("tests/test_parity.py", "tests/test_parity.py::test_one", True),
        ("tests/test_parity.py::test_one", "tests/test_parity.py::test_one", True),
        (
            "tests/test_parity.py::test_one",
            "tests/test_parity.py::test_one[value]",
            True,
        ),
        ("tests/test_parity.py::test_one", "tests/test_parity.py::test_two", False),
        ("tests/ocr_tests/", "tests/ocr_tests/test_ocr_mistral.py::test_one", True),
        ("tests/ocr_tests/", "tests/other_tests/test_ocr_mistral.py::test_one", False),
    ],
)
def test_should_match_pytest_file_and_node_selectors(
    selector: str, nodeid: str, matches: bool
) -> None:
    assert selector_matches_node(selector, nodeid) is matches


def test_should_only_return_selectors_whose_files_exist(tmp_path: Path) -> None:
    existing = tmp_path / "tests" / "test_parity.py"
    existing.parent.mkdir()
    existing.write_text("", encoding="utf-8")
    case = _case(
        selectors=("tests/test_parity.py", "tests/test_missing.py::test_missing")
    )

    assert runnable_selectors((case,), tmp_path) == ("tests/test_parity.py",)


def test_should_treat_an_existing_folder_selector_as_runnable(tmp_path: Path) -> None:
    (tmp_path / "tests" / "ocr_tests").mkdir(parents=True)
    case = _case(selectors=("tests/ocr_tests/",))

    assert runnable_selectors((case,), tmp_path) == ("tests/ocr_tests/",)


def test_should_mark_planned_and_not_applicable_cases_without_running() -> None:
    planned = CaseResult(case=_case(coverage=Coverage.PLANNED))
    not_applicable = CaseResult(case=_case(coverage=Coverage.NOT_APPLICABLE))

    planned.set_initial_status()
    not_applicable.set_initial_status()

    assert planned.status is RunStatus.PLANNED
    assert not_applicable.status is RunStatus.NOT_APPLICABLE


def test_should_treat_an_all_planned_filtered_run_as_success(tmp_path: Path) -> None:
    exit_code, run = run_pytest(
        cases=(_case(coverage=Coverage.PLANNED),),
        repo_root=tmp_path,
        on_update=lambda _: None,
    )

    assert exit_code == 0
    assert next(iter(run.results.values())).status is RunStatus.PLANNED


@pytest.mark.parametrize("strategy_id", ("e2e_parity", "existing_e2e_test_sdk"))
def test_should_run_namespace_package_relative_imports(tmp_path: Path, strategy_id: str) -> None:
    package: Final = tmp_path / "manual_suite" / "relative-tests"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "values.py").write_text("ANSWER = 42\n", encoding="utf-8")
    (package / "test_relative.py").write_text(
        "from .values import ANSWER\n\ndef test_answer():\n    assert ANSWER == 42\n",
        encoding="utf-8",
    )
    result: Final = subprocess.run(
        (
            sys.executable,
            "-c",
            "import importlib\n"
            "from pathlib import Path\n"
            "cli = importlib.import_module('tests.rust-python-harness.cli')\n"
            "models = importlib.import_module('tests.rust-python-harness.shared.reporting.models')\n"
            f"case = models.HarnessCase(strategy_id={strategy_id!r}, strategy_label='Example', "
            "sdk_function='ocr', coverage=models.Coverage.COMPLETE, "
            "selectors=('manual_suite/relative-tests/',))\n"
            f"code, run = cli._resolve_runner({strategy_id!r})((case,), Path.cwd(), lambda _: None)\n"
            "assert code == 0, code\n"
            "assert next(iter(run.results.values())).passed == 1\n",
        ),
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join((str(tmp_path), str(Path(__file__).resolve().parents[1]))),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_should_finalize_a_fully_passing_case() -> None:
    result = CaseResult(case=_case(selectors=("tests/test_parity.py",)))
    result.set_initial_status()
    result.collected.update({"one", "two"})
    result.completed.update({"one", "two"})
    result.passed = 2

    result.finalize()

    assert result.status is RunStatus.PASSED


def test_should_replace_a_pass_with_a_teardown_error() -> None:
    result = CaseResult(case=_case(selectors=("tests/test_parity.py",)))
    result.set_initial_status()
    result.collected.add("one")

    result.record("one", RunStatus.PASSED, 0.1)
    result.record("one", RunStatus.ERROR, 0.2)

    assert result.status is RunStatus.ERROR
    assert result.passed == 0
    assert result.errors == 1
    assert result.duration == pytest.approx(0.3)


def test_should_filter_the_catalog_by_strategy_and_sdk_function() -> None:
    strategies = load_catalog()

    cases = _select(strategies, {"e2e_parity"}, {"messages"})

    assert len(cases) == 1
    assert cases[0].key == "e2e_parity:messages"


def test_should_reject_an_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="Unknown strategy"):
        _select(load_catalog(), {"not-real"}, set())


def test_should_pick_multiple_interactive_filters() -> None:
    answers = iter(["nope", "1, 3"])

    selected = _pick_values(
        "Examples",
        (("one", "One"), ("two", "Two"), ("three", "Three")),
        input_fn=lambda _: next(answers),
    )

    assert selected == {"one", "three"}


def test_should_format_developer_facing_run_context() -> None:
    run = HarnessRun.from_cases((_case(selectors=("tests/test_parity.py",)),))
    result = next(iter(run.results.values()))
    result.collected.add("tests/test_parity.py::test_one")
    result.record("tests/test_parity.py::test_one", RunStatus.PASSED, 1.25)

    assert _summary(run) == (1, 0, 0, 0)
    assert _format_duration(1.25) == "1.2s"
    assert _rerun_command("tests/test_parity.py::test_one") == (
        "poetry run pytest tests/test_parity.py::test_one -q -o consider_namespace_packages=true"
    )
    assert _rerun_command("tests/test_parity.py::test_one[value with spaces]") == (
        "poetry run pytest 'tests/test_parity.py::test_one[value with spaces]' -q -o consider_namespace_packages=true"
    )


def test_should_build_python_coverage_reports_below_the_target_directory(
    tmp_path: Path,
) -> None:
    args = _coverage_pytest_args(tmp_path)

    assert tmp_path.is_dir()
    assert "--cov=litellm" in args
    assert "--cov-context=test" in args
    assert f"--cov-report=json:{tmp_path / 'python.json'}" in args
    assert f"--cov-report=xml:{tmp_path / 'python.xml'}" in args
    assert f"--cov-report=html:{tmp_path / 'python-html'}" in args


def test_should_report_confidence_for_each_sdk_section() -> None:
    strategies = load_catalog()
    cases = tuple(case for strategy in strategies for case in strategy.cases)
    run = HarnessRun.from_cases(cases)
    passing = run.results["e2e_parity:responses"]
    passing.collected.add("tests/test_parity.py::test_one")
    passing.record("tests/test_parity.py::test_one", RunStatus.PASSED)

    scores = {
        score.sdk_function: score for score in section_confidence(run, strategies)
    }

    assert scores["responses"].verified_strategies == 1
    assert scores["responses"].required_strategies == 4
    assert scores["responses"].percentage == 25
    assert scores["responses"].level.value == "MEDIUM"
    assert scores["count_tokens"].percentage == 0
    assert scores["count_tokens"].level.value == "LOW"



def test_should_report_no_ledger_for_a_function_without_one() -> None:
    report = build_function_report("messages", repo_root=REPO_ROOT)

    assert report.has_ledger is False
    assert report.is_clean is True


def test_should_report_ocr_ledger_stats_and_a_clean_audit() -> None:
    ledger = load_ledger(ledger_path_for("ocr"))

    report = build_function_report("ocr", repo_root=REPO_ROOT)

    assert report.has_ledger is True
    assert report.ledger.mapped_count == ledger.mapped_count
    assert report.ledger.total_count == ledger.total_count
    assert report.is_clean is True


def test_should_scope_validate_ledger_to_the_requested_function(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = _validate_ledger({"messages"})

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "messages" in captured.out
    assert "no ledger yet" in captured.out
    assert "ocr" not in captured.out


@pytest.mark.parametrize("strategy_id", (None, "e2e_parity", "trace_parity", "unit_tests", "existing_e2e_test_sdk"))
def test_should_validate_chat_completions_ledger_from_each_runner(
    strategy_id: str | None, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code: Final = cli.main(
        ("--validate-ledger", "--function", "chat_completions"), strategy_id=strategy_id
    )

    captured: Final = capsys.readouterr()
    assert exit_code == 0
    assert "chat_completions" in captured.out
    assert "no ledger yet" in captured.out
    assert "ocr" not in captured.out


def test_should_have_every_python_and_rust_ocr_test_accounted_for_in_the_ledger() -> None:
    ledger = load_ledger(ledger_path_for("ocr"))

    report = audit_ledger(ledger, repo_root=REPO_ROOT)

    assert report.is_clean, (
        "\nOCR test-parity ledger is out of sync with the live test files.\n"
        f"Ledger references a Python test that no longer exists: {list(report.missing_python_tests)}\n"
        f"Python test exists but is not tracked in the ledger: {list(report.stale_python_tests)}\n"
        f"Ledger references a Rust test that no longer exists: {list(report.missing_rust_tests)}\n"
        f"Rust test exists but is not tracked in the ledger: {list(report.stale_rust_tests)}\n"
    )
