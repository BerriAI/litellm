from __future__ import annotations

import importlib
import json
import shutil
import sys
from pathlib import Path
from typing import Final

import pytest

catalog = importlib.import_module("tests.rust-python-harness.catalog")
cli = importlib.import_module("tests.rust-python-harness.cli")
ledger_module = importlib.import_module("tests.rust-python-harness.shared.parity.ledger")
mapping_validator = importlib.import_module(
    "tests.rust-python-harness.strategies.unit_tests.mapping_validator"
)
models = importlib.import_module("tests.rust-python-harness.models")
runner = importlib.import_module("tests.rust-python-harness.runner")
rust_runner = importlib.import_module("tests.rust-python-harness.strategies.unit_tests.rust_runner")
ui = importlib.import_module("tests.rust-python-harness.ui")

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
        "e2e_fuzz_tests",
        "unit_tests_rust",
        "validate_sub_methods",
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
    assert scores["responses"].required_strategies == 4
    assert scores["responses"].percentage == 25
    assert scores["responses"].level.value == "MEDIUM"
    assert scores["count_tokens"].percentage == 0
    assert scores["count_tokens"].level.value == "LOW"



def test_should_report_no_ledger_for_a_function_without_one() -> None:
    report = build_function_report("messages", repo_root=REPO_ROOT)

    assert report.has_ledger is False
    assert report.passes_validation is False


def test_should_report_ocr_ledger_stats_and_unresolved_portable_contracts() -> None:
    ledger = load_ledger(ledger_path_for("ocr"))

    report = build_function_report("ocr", repo_root=REPO_ROOT, rust_inventory=lambda *_: ledger.rust_tests)

    assert report.has_ledger is True
    assert report.ledger.mapped_count == ledger.mapped_count
    assert report.ledger.total_count == ledger.total_count
    assert report.ledger.unresolved_portable_count == ledger.unresolved_portable_count
    assert report.ledger.python_only_count == ledger.python_only_count
    assert report.ledger.portable_count == (
        ledger.mapped_count + ledger.unresolved_portable_count
    )
    assert report.passes_validation is False


def test_should_scope_validate_ledger_to_the_requested_function(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = _validate_ledger({"messages"})

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "messages" in captured.out
    assert "no ledger yet" in captured.out
    assert "ocr" not in captured.out


def test_should_have_every_python_ocr_test_accounted_for_in_the_ledger() -> None:
    ledger = load_ledger(ledger_path_for("ocr"))

    report = audit_ledger(ledger, repo_root=REPO_ROOT, rust_inventory=lambda *_: ledger.rust_tests)

    assert report.is_clean, (
        "\nOCR test-parity ledger is out of sync with the live test files.\n"
        f"Ledger references a Python test that no longer exists: {list(report.missing_python_tests)}\n"
        f"Python test exists but is not tracked in the ledger: {list(report.stale_python_tests)}\n"
        f"Ledger references a Rust test that no longer exists: {list(report.missing_rust_tests)}\n"
        f"Rust test exists but is not tracked in the ledger: {list(report.stale_rust_tests)}\n"
    )


def test_should_reject_duplicate_rust_mapping_targets(tmp_path: Path) -> None:
    ledger_path: Final = tmp_path / "ledger.json"
    ledger_path.write_text(
        """{
          "schema_version": 2,
          "sdk_function": "ocr",
          "python_scope": ["tests/python.py"],
          "rust_scope": [{
            "target": {"package": "example", "name": "example", "kind": "lib"},
            "features": [],
            "default_features": true,
            "modules": ["ocr"]
          }],
          "entries": [
            {
              "python_file": "tests/python.py",
              "python_test": "test_one",
              "status": "mapped",
              "rust_file": "src/rust.rs",
              "rust": {
                "target": {"package": "example", "name": "example", "kind": "lib"},
                "name": "ocr::same_target"
              },
              "justification": "Same observable behavior"
            },
            {
              "python_file": "tests/python.py",
              "python_test": "test_two",
              "status": "mapped",
              "rust_file": "src/different.rs",
              "rust": {
                "target": {"package": "example", "name": "example", "kind": "lib"},
                "name": "ocr::same_target"
              },
              "justification": "Same observable behavior"
            }
          ],
          "rust_only_tests": []
        }""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Rust mapping targets contains duplicates"):
        load_ledger(ledger_path)


def test_should_reject_fields_from_another_entry_variant(tmp_path: Path) -> None:
    ledger_path: Final = tmp_path / "ledger.json"
    ledger_path.write_text(
        """{
          "schema_version": 2,
          "sdk_function": "ocr",
          "python_scope": ["tests/python.py"],
          "rust_scope": [],
          "entries": [{
            "python_file": "tests/python.py",
            "python_test": "test_one",
            "status": "python_only",
            "reason": "Owned by Python",
            "rust_test": "silently_ignored_before"
          }],
          "rust_only_tests": []
        }""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        load_ledger(ledger_path)


@pytest.mark.parametrize(
    ("status", "expected_clean"),
    [("python_only", True), ("unresolved_portable", False)],
)
def test_should_accept_exclusions_and_fail_unresolved_portable_contracts(
    tmp_path: Path, status: str, expected_clean: bool
) -> None:
    python_file: Final = tmp_path / "tests" / "python.py"
    python_file.parent.mkdir(parents=True)
    python_file.write_text("def test_one(): pass\n", encoding="utf-8")
    ledger_path: Final = ledger_path_for("ocr", tmp_path)
    ledger_path.parent.mkdir(parents=True)
    ledger_path.write_text(
        f"""{{
          "schema_version": 2,
          "sdk_function": "ocr",
          "python_scope": ["tests/python.py"],
          "rust_scope": [],
          "entries": [{{
            "python_file": "tests/python.py",
            "python_test": "test_one",
            "status": "{status}",
            "reason": "Classification rationale"
          }}],
          "rust_only_tests": []
        }}""",
        encoding="utf-8",
    )

    report: Final = build_function_report("ocr", tmp_path)

    assert report.audit is not None and report.audit.is_clean
    assert report.passes_audit is True
    assert report.passes_validation is expected_clean


def test_should_detect_missing_and_untracked_compiled_tests() -> None:
    ledger: Final = load_ledger(ledger_path_for("ocr"))
    removed: Final = next(iter(ledger.rust_tests))
    added: Final = ledger_module.RustTestIdentity(target=removed.target, name="ocr::new_compiled_test")
    inventory: Final = (ledger.rust_tests - frozenset((removed,))) | frozenset((added,))

    report: Final = audit_ledger(ledger, rust_inventory=lambda *_: inventory)

    assert report.is_clean is False
    assert report.missing_rust_tests == (removed.key,)
    assert report.stale_rust_tests == (added.key,)


def test_should_report_inventory_failure_without_claiming_a_clean_audit() -> None:
    def failed_inventory(*_: object) -> frozenset[object]:
        raise ValueError("Cargo build failed")

    report: Final = build_function_report("ocr", rust_inventory=failed_inventory)

    assert report.passes_audit is False
    assert report.passes_validation is False
    assert report.error == "Cargo build failed"


def test_should_fail_closed_when_an_inventory_command_exits_nonzero(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"failed \(3\)"):
        rust_runner.run_command((sys.executable, "-c", "raise SystemExit(3)"), tmp_path)


def test_should_reject_overlapping_rust_module_scopes() -> None:
    with pytest.raises(ValueError, match="Rust modules overlap"):
        ledger_module.RustTestScope(
            target=ledger_module.RustTarget(package="example", name="example", kind="lib"),
            features=(), default_features=True, modules=("ocr", "ocr::tests"),
        )


@pytest.mark.skipif(shutil.which("cargo") is None, reason="Cargo is required for compiled inventory regression tests")
@pytest.mark.parametrize(
    ("default_features", "features", "has_extra"),
    [(True, (), True), (False, (), False), (False, ("extra",), True)],
)
def test_should_inventory_compiled_modules_macros_and_features(
    tmp_path: Path, default_features: bool, features: tuple[str, ...], has_extra: bool
) -> None:
    workspace: Final = tmp_path / "litellm-rust"
    source: Final = workspace / "src"
    external: Final = source / "ocr" / "external.rs"
    external.parent.mkdir(parents=True)
    (workspace / "Cargo.toml").write_text(
        '[package]\nname = "inventory-fixture"\nversion = "0.1.0"\nedition = "2021"\n'
        '[features]\ndefault = ["extra"]\nextra = []\nignored = []\n', encoding="utf-8",
    )
    (source / "lib.rs").write_text(
        '''#[cfg(test)]
mod ocr {
    mod external;
    #[test] fn same_name() {}
    #[cfg(any())] #[test] fn compiled_out() {}
    #[cfg(feature = "extra")] #[test] fn extra_case() {}
    #[cfg(feature = "ignored")] #[test] #[ignore] fn ignored_case() {}
    macro_rules! generate_test { ($name:ident) => { #[test] fn $name() {} }; }
    generate_test!(generated_case);
}
''', encoding="utf-8",
    )
    external.write_text("#[test] fn same_name() {}\n", encoding="utf-8")
    rust_runner.run_command(("cargo", "generate-lockfile", "--offline"), workspace)
    target: Final = ledger_module.RustTarget(package="inventory-fixture", name="inventory_fixture", kind="lib")
    scope: Final = ledger_module.RustTestScope(
        target=target, features=features, default_features=default_features, modules=("ocr",)
    )

    inventory: Final = rust_runner.enumerate_rust_tests(tmp_path, (scope,))

    expected: Final = frozenset(("ocr::same_name", "ocr::external::same_name", "ocr::generated_case")) | (
        frozenset(("ocr::extra_case",)) if has_extra else frozenset()
    )
    assert frozenset(identity.name for identity in inventory) == expected
    ignored_scope: Final = ledger_module.RustTestScope(
        target=target, features=("ignored",), default_features=False, modules=("ocr",)
    )
    with pytest.raises(ValueError, match=r"Ignored Rust tests.*ocr::ignored_case"):
        rust_runner.enumerate_rust_tests(tmp_path, (ignored_scope,))
