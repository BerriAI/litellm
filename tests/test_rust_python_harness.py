from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

catalog = importlib.import_module("tests.rust-python-harness.catalog")
cli = importlib.import_module("tests.rust-python-harness.cli")
ledger_module = importlib.import_module("tests.rust-python-harness.ledger")
models = importlib.import_module("tests.rust-python-harness.models")
runner = importlib.import_module("tests.rust-python-harness.runner")
ui = importlib.import_module("tests.rust-python-harness.ui")
validate_ledger = importlib.import_module("tests.rust-python-harness.validate_ledger")

load_catalog = catalog.load_catalog
load_ledger = ledger_module.load_ledger
LEDGER_PATH = ledger_module.LEDGER_PATH
REPO_ROOT = validate_ledger.REPO_ROOT
audit_ledger = validate_ledger.audit_ledger
run_audit = validate_ledger.run_audit
LedgerDriftError = validate_ledger.LedgerDriftError
_enumerate_python_tests = validate_ledger._enumerate_python_tests
_enumerate_rust_tests = validate_ledger._enumerate_rust_tests
_pick_values = cli._pick_values
_coverage_pytest_args = cli._coverage_pytest_args
_select = cli._select
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


def test_should_load_the_three_harness_strategies_in_order() -> None:
    strategies = load_catalog()

    assert [strategy.id for strategy in strategies] == [
        "e2e_fuzz_tests",
        "unit_tests_rust",
        "validate_sub_methods",
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

    cases = _select(strategies, {"e2e_fuzz_tests"}, {"messages"})

    assert len(cases) == 1
    assert cases[0].key == "e2e_fuzz_tests:messages"


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
        "poetry run pytest tests/test_parity.py::test_one -q"
    )
    assert _rerun_command("tests/test_parity.py::test_one[value with spaces]") == (
        "poetry run pytest 'tests/test_parity.py::test_one[value with spaces]' -q"
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
    passing = run.results["e2e_fuzz_tests:responses"]
    passing.collected.add("tests/test_parity.py::test_one")
    passing.record("tests/test_parity.py::test_one", RunStatus.PASSED)

    scores = {
        score.sdk_function: score for score in section_confidence(run, strategies)
    }

    assert scores["responses"].verified_strategies == 1
    assert scores["responses"].required_strategies == 3
    assert scores["responses"].percentage == 33
    assert scores["responses"].level.value == "MEDIUM"
    assert scores["count_tokens"].percentage == 0
    assert scores["count_tokens"].level.value == "LOW"


def test_should_load_the_ocr_ledger_and_compute_its_ratio() -> None:
    ledger = load_ledger()

    assert ledger.sdk_function == "ocr"
    assert ledger.total_count == 145
    assert ledger.mapped_count == 8
    assert ledger.percentage == 5.5


def test_should_reject_a_mapped_entry_missing_rust_test(tmp_path: Path) -> None:
    ledger_path = tmp_path / "bad_ledger.json"
    ledger_path.write_text(
        json.dumps(
            {
                "sdk_function": "ocr",
                "python_scope": ["tests/test_litellm/ocr/test_x.py"],
                "rust_scope": ["litellm-rust/crates/x.rs"],
                "entries": [
                    {
                        "python_file": "tests/test_litellm/ocr/test_x.py",
                        "python_test": "test_one",
                        "status": "mapped",
                        "rust_file": "litellm-rust/crates/x.rs",
                    }
                ],
                "rust_only_tests": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="rust_test"):
        load_ledger(ledger_path)


def test_should_reject_an_unmapped_entry_missing_a_reason(tmp_path: Path) -> None:
    ledger_path = tmp_path / "bad_ledger.json"
    ledger_path.write_text(
        json.dumps(
            {
                "sdk_function": "ocr",
                "python_scope": ["tests/test_litellm/ocr/test_x.py"],
                "rust_scope": [],
                "entries": [
                    {
                        "python_file": "tests/test_litellm/ocr/test_x.py",
                        "python_test": "test_one",
                        "status": "unmapped",
                    }
                ],
                "rust_only_tests": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reason"):
        load_ledger(ledger_path)


def test_should_enumerate_module_level_and_class_scoped_python_tests(
    tmp_path: Path,
) -> None:
    module = tmp_path / "tests" / "example"
    module.mkdir(parents=True)
    source_path = module / "test_example.py"
    source_path.write_text(
        "def test_top_level():\n"
        "    pass\n"
        "\n"
        "class TestGroup:\n"
        "    def test_one(self):\n"
        "        pass\n"
        "\n"
        "    def test_two(self):\n"
        "        pass\n"
        "\n"
        "    def helper(self):\n"
        "        pass\n",
        encoding="utf-8",
    )

    found = _enumerate_python_tests(tmp_path, "tests/example/test_example.py")

    assert found == frozenset(
        {"test_top_level", "TestGroup::test_one", "TestGroup::test_two"}
    )


def test_should_enumerate_rust_test_and_tokio_test_functions(tmp_path: Path) -> None:
    source_path = tmp_path / "example.rs"
    source_path.write_text(
        "#[test]\n"
        "fn plain_test() {\n"
        "    assert!(true);\n"
        "}\n"
        "\n"
        "#[tokio::test]\n"
        "async fn async_test() {\n"
        "    assert!(true);\n"
        "}\n"
        "\n"
        "fn not_a_test() {}\n",
        encoding="utf-8",
    )

    found = _enumerate_rust_tests(tmp_path, "example.rs")

    assert found == frozenset({"plain_test", "async_test"})


def test_should_flag_a_python_test_missing_from_the_ledger(tmp_path: Path) -> None:
    (tmp_path / "litellm-rust").mkdir()
    python_dir = tmp_path / "tests" / "example"
    python_dir.mkdir(parents=True)
    (python_dir / "test_example.py").write_text(
        "def test_tracked():\n    pass\n\n\ndef test_untracked():\n    pass\n",
        encoding="utf-8",
    )
    ledger = load_ledger(_write_ledger(tmp_path, mapped_python_test="test_tracked"))

    report = audit_ledger(ledger, repo_root=tmp_path)

    assert report.stale_python_tests == ("tests/example/test_example.py:test_untracked",)
    assert not report.is_clean


def test_should_flag_a_stale_ledger_entry_for_a_removed_python_test(
    tmp_path: Path,
) -> None:
    (tmp_path / "litellm-rust").mkdir()
    python_dir = tmp_path / "tests" / "example"
    python_dir.mkdir(parents=True)
    (python_dir / "test_example.py").write_text(
        "def test_still_here():\n    pass\n", encoding="utf-8"
    )
    ledger = load_ledger(_write_ledger(tmp_path, mapped_python_test="test_removed"))

    report = audit_ledger(ledger, repo_root=tmp_path)

    assert report.missing_python_tests == (
        "tests/example/test_example.py:test_removed",
    )
    assert not report.is_clean


def test_should_raise_ledger_drift_error_with_every_offending_test(
    tmp_path: Path,
) -> None:
    (tmp_path / "litellm-rust").mkdir()
    python_dir = tmp_path / "tests" / "example"
    python_dir.mkdir(parents=True)
    (python_dir / "test_example.py").write_text(
        "def test_still_here():\n    pass\n", encoding="utf-8"
    )
    ledger_path = _write_ledger(tmp_path, mapped_python_test="test_removed")

    with pytest.raises(LedgerDriftError, match="test_removed"):
        run_audit(ledger_path, repo_root=tmp_path)


def _write_ledger(tmp_path: Path, *, mapped_python_test: str) -> Path:
    rust_dir = tmp_path / "litellm-rust"
    (rust_dir / "example.rs").write_text(
        '#[test]\nfn rust_side() {\n    assert!(true);\n}\n', encoding="utf-8"
    )
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(
        json.dumps(
            {
                "sdk_function": "ocr",
                "python_scope": ["tests/example/test_example.py"],
                "rust_scope": ["litellm-rust/example.rs"],
                "entries": [
                    {
                        "python_file": "tests/example/test_example.py",
                        "python_test": mapped_python_test,
                        "status": "mapped",
                        "rust_file": "litellm-rust/example.rs",
                        "rust_test": "rust_side",
                        "justification": "example",
                    }
                ],
                "rust_only_tests": [],
            }
        ),
        encoding="utf-8",
    )
    return ledger_path


def test_should_report_the_real_ocr_ledger_as_clean_against_the_live_repo() -> None:
    ledger = load_ledger()

    report = audit_ledger(ledger, repo_root=REPO_ROOT)

    assert report.is_clean, (
        f"missing_python_tests={report.missing_python_tests} "
        f"stale_python_tests={report.stale_python_tests} "
        f"missing_rust_tests={report.missing_rust_tests} "
        f"stale_rust_tests={report.stale_rust_tests}"
    )
