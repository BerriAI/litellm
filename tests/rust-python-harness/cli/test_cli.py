from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Final

import pytest

from ..shared.reporting.models import SDK_FUNCTIONS, SURFACES, CaseDisposition, HarnessCase, HarnessRun, RunStatus, Strategy
from ..shared.reporting.strategy import NotImplementedCaseSpec, SkippedCaseSpec, StrategyDefinition
from ..shared.reporting.ui import final_report
from . import main
from .catalog import STRATEGIES_ROOT, load_catalog
from .commands import REPO_ROOT
from .selection import pick_values, select

def _strategy_source(
    *,
    strategy_id: str = "example",
    drop: tuple[str, str] | None = None,
    duplicate: tuple[str, str] | None = None,
    incompatible: tuple[str, str] | None = None,
) -> str:
    cells: Final = tuple(
        (surface, function)
        for surface in SURFACES
        for function in SDK_FUNCTIONS
        if (surface, function) != drop
    )
    definitions: Final = tuple(
        (
            f"strategy.CaseDefinition({surface!r}, {function!r}, "
            "strategy.ModuleCaseSpec(coverage=models.Coverage.COMPLETE, module='tests.example'))"
            if (surface, function) == incompatible
            else (
                f"strategy.CaseDefinition({surface!r}, {function!r}, "
                "strategy.NotImplementedCaseSpec(reason='Not implemented yet'))"
            )
        )
        for surface, function in (*cells, *((duplicate,) if duplicate is not None else ()))
    )
    return (
        "import importlib\n"
        "from pathlib import Path\n"
        "strategy = importlib.import_module('tests.rust-python-harness.shared.reporting.strategy')\n"
        "models = importlib.import_module('tests.rust-python-harness.shared.reporting.models')\n"
        "runner = importlib.import_module('tests.rust-python-harness.strategies.trace_parity.runner')\n"
        "rendering = importlib.import_module('tests.rust-python-harness.shared.reporting.rendering')\n"
        "def render(results):\n"
        "    return (rendering.ReportSection('Example outcomes', tuple(rendering.render_case_outcome(r) for r in results)),)\n"
        f"CASES = ({','.join(definitions)},)\n"
        "STRATEGY = strategy.StrategyDefinition("
        f"id={strategy_id!r}, order=1, label='Example strategy', description='Example description', "
        "directory=Path(__file__).parent, runnable_spec=strategy.SuiteCaseSpec, cases=CASES, "
        "run=runner.run_trace_cases, render=render)\n"
    )


def _write_strategy_folder(
    root: Path,
    name: str = "example",
    *,
    init_source: str | None = None,
) -> Path:
    folder: Final = root / name
    folder.mkdir(parents=True)
    (folder / "__init__.py").write_text(init_source or _strategy_source(), encoding="utf-8")
    return folder


def test_should_load_the_five_harness_strategies_in_order() -> None:
    strategies = load_catalog()

    assert [strategy.id for strategy in strategies] == [
        "e2e_parity",
        "trace_parity",
        "unit_tests_mapping",
        "unit_tests_parity",
        "unit_tests_rust",
    ]
    assert all(
        tuple((case.surface, case.sdk_function) for case in strategy.cases)
        == tuple((surface, function) for surface in SURFACES for function in SDK_FUNCTIONS)
        for strategy in strategies
    )


def test_every_strategy_folder_complies() -> None:
    strategies = load_catalog()

    folders = {
        path.name
        for path in STRATEGIES_ROOT.iterdir()
        if path.is_dir() and (path / "__init__.py").exists()
    }
    assert folders == {strategy.id for strategy in strategies}
    for strategy in strategies:
        definition = strategy.definition
        assert isinstance(definition, StrategyDefinition)
        assert definition.directory == strategy.directory
        assert not (strategy.directory / "strategy.json").exists()
        assert (strategy.directory / "AGENTS.md").exists()
        for case in strategy.cases:
            if case.spec.disposition is CaseDisposition.RUNNABLE:
                assert isinstance(case.spec, definition.runnable_spec)


def test_should_reject_a_registry_missing_a_matrix_cell(tmp_path: Path) -> None:
    _write_strategy_folder(tmp_path, init_source=_strategy_source(drop=("sdk", "count_tokens")))

    with pytest.raises(ValueError, match="must exactly match the harness matrix"):
        load_catalog(tmp_path)


def test_should_reject_a_duplicate_matrix_cell(tmp_path: Path) -> None:
    _write_strategy_folder(tmp_path, init_source=_strategy_source(duplicate=("sdk", "ocr")))

    with pytest.raises(ValueError, match="duplicate strategy cases"):
        load_catalog(tmp_path)


def test_should_reject_a_folder_without_a_strategy_definition(tmp_path: Path) -> None:
    folder: Final = tmp_path / "example"
    folder.mkdir()
    (folder / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="STRATEGY"):
        load_catalog(tmp_path)


def test_should_reject_a_strategy_id_that_differs_from_its_folder(tmp_path: Path) -> None:
    _write_strategy_folder(tmp_path, init_source=_strategy_source(strategy_id="other"))

    with pytest.raises(ValueError, match="must match folder name"):
        load_catalog(tmp_path)


def test_should_reject_a_runnable_case_incompatible_with_the_strategy(tmp_path: Path) -> None:
    _write_strategy_folder(tmp_path, init_source=_strategy_source(incompatible=("sdk", "ocr")))

    with pytest.raises(ValueError, match="runnable cases do not match SuiteCaseSpec"):
        load_catalog(tmp_path)


@pytest.mark.parametrize("case_type", (NotImplementedCaseSpec, SkippedCaseSpec))
def test_should_reject_an_unavailable_case_with_a_blank_reason(
    case_type: type[NotImplementedCaseSpec] | type[SkippedCaseSpec],
) -> None:
    with pytest.raises(ValueError, match="at least 1 character"):
        case_type(reason=" ")


def test_should_filter_the_catalog_by_strategy_and_sdk_function() -> None:
    strategies = load_catalog()

    cases = select(strategies, {"e2e_parity"}, {"messages"})

    assert tuple(case.key for case in cases) == (
        "e2e_parity:messages",
        "e2e_parity:gateway:messages",
    )


def test_should_filter_the_catalog_by_surface() -> None:
    strategies = load_catalog()

    cases = select(strategies, set(), set(), "gateway")

    assert len(cases) == len(strategies) * len(SDK_FUNCTIONS)
    assert all(case.surface == "gateway" for case in cases)
    assert select(strategies, {"e2e_parity"}, set(), "sdk")[0].surface == "sdk"


def test_every_matrix_cell_can_be_selected_exactly() -> None:
    strategies: Final = load_catalog()
    selections: Final = tuple(
        tuple((case.surface, case.sdk_function) for case in select(strategies, {strategy.id}, {sdk_function}, surface))
        for strategy in strategies
        for surface in SURFACES
        for sdk_function in SDK_FUNCTIONS
    )
    expected: Final = tuple(
        ((surface, sdk_function),)
        for _strategy in strategies
        for surface in SURFACES
        for sdk_function in SDK_FUNCTIONS
    )

    assert selections == expected


def _assert_unavailable_cell(strategy: Strategy, case: HarnessCase, section_title: str) -> None:
    spec: Final = case.spec
    assert isinstance(spec, (NotImplementedCaseSpec, SkippedCaseSpec))
    scoped: Final = replace(strategy, cases=(case,))
    exit_code, run = strategy.definition.run((case,), REPO_ROOT, lambda _: None)
    result: Final = run.results[case.key]
    expected: Final = (
        RunStatus.NOT_IMPLEMENTED
        if spec.disposition is CaseDisposition.NOT_IMPLEMENTED
        else RunStatus.SKIPPED
    )
    report: Final = final_report(run, exit_code, (scoped,))

    assert exit_code == 0
    assert result.status is expected
    assert spec.reason in report
    assert section_title in report
    assert f"Status: {'INCOMPLETE' if expected is RunStatus.NOT_IMPLEMENTED else 'SKIPPED'}" in report


def test_every_unavailable_matrix_cell_finishes_and_explains_itself() -> None:
    strategies: Final = load_catalog()
    section_titles: Final = {
        "e2e_parity": "End-to-end parity outcomes",
        "trace_parity": "Trace comparisons",
        "unit_tests_mapping": "Python/Rust unit-test mappings",
        "unit_tests_parity": "Python backend parity outcomes",
        "unit_tests_rust": "Native Rust unit-test outcomes",
    }

    unavailable: Final = tuple(
        (strategy, case)
        for strategy in strategies
        for case in strategy.cases
        if case.spec.disposition is not CaseDisposition.RUNNABLE
    )

    for strategy, case in unavailable:
        _assert_unavailable_cell(strategy, case, section_titles[strategy.id])


def test_should_reject_an_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="Unknown strategy"):
        select(load_catalog(), {"not-real"}, set())


def test_should_reject_an_empty_selection() -> None:
    with pytest.raises(ValueError, match="matched no harness cases"):
        select((), set(), set())


def test_should_pick_multiple_interactive_filters() -> None:
    answers = iter(["nope", "1, 3"])

    selected = pick_values(
        "Examples",
        (("one", "One"), ("two", "Two"), ("three", "Three")),
        input_fn=lambda _: next(answers),
    )

    assert selected == {"one", "three"}


def test_list_honors_strategy_and_function_filters(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["list", "--strategy", "e2e_parity", "--function", "ocr"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "e2e_parity" in captured.out
    assert "trace_parity" not in captured.out
    assert "messages" not in captured.out


def test_list_describes_an_unavailable_case(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code: Final = main(
        ["list", "--strategy", "e2e_parity", "--surface", "sdk", "--function", "messages"]
    )

    captured: Final = capsys.readouterr()
    assert exit_code == 0
    assert "sdk/messages" in captured.out
    assert "not_implemented" in captured.out
    assert "no standalone end-to-end parity case is registered" in captured.out


def test_run_reports_not_implemented_cells_as_incomplete(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["run", "--strategy", "trace_parity", "--surface", "gateway", "--plain"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Rust <-> Python parity report" in captured.out
    assert "Status: INCOMPLETE" in captured.out
    assert "Cases: 6 selected, 6 not implemented, 0 skipped" in captured.out
    assert "Exit code: 0" in captured.out
    assert "Trace: NOT IMPLEMENTED" in captured.out
    assert "No gateway OCR trace-parity case is registered." in captured.out
    assert "Port confidence" not in captured.out
    assert "Slowest tests" not in captured.out
    assert "Strategy × API" not in captured.out
    assert "╭" not in captured.out
    assert "✓" not in captured.out


def test_run_reports_skipped_only_cells_as_skipped(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code: Final = main(["run", "--strategy", "unit_tests_rust", "--surface", "gateway", "--plain"])

    captured: Final = capsys.readouterr()
    assert exit_code == 0
    assert "Status: SKIPPED" in captured.out
    assert "Cases: 6 selected, 0 not implemented, 6 skipped" in captured.out
    assert "Native Rust unit tests are not gateway execution." in captured.out


def test_run_rejects_an_unknown_strategy(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["run", "--strategy", "not-real"])

    captured: Final = capsys.readouterr()
    assert excinfo.value.code == 2
    assert "invalid choice: 'not-real'" in captured.err
    assert "e2e_parity" in captured.err


def test_run_help_lists_every_strategy(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["run", "--help"])

    captured: Final = capsys.readouterr()
    assert excinfo.value.code == 0
    assert (
        "--strategy {e2e_parity,trace_parity,unit_tests_mapping,unit_tests_parity,unit_tests_rust}"
        in captured.out
    )


def test_keyboard_interrupt_exits_cleanly(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli: Final = importlib.import_module("tests.rust-python-harness.cli")

    def interrupt() -> tuple[object, ...]:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "load_catalog", interrupt)

    exit_code: Final = main(["list"])

    captured: Final = capsys.readouterr()
    assert exit_code == 130
    assert captured.out == ""
    assert captured.err == "\nInterrupted\n"


def test_runner_interrupt_skips_the_completion_report(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    commands: Final = importlib.import_module("tests.rust-python-harness.cli.commands")

    def interrupt_run(
        strategies: Sequence[Strategy],
        repo_root: Path,
        on_update: Callable[[HarnessRun], None],
        runner_args: Sequence[str] = (),
    ) -> tuple[int, HarnessRun]:
        del repo_root, on_update, runner_args
        run: Final = HarnessRun.from_cases(
            case for strategy in strategies for case in strategy.cases
        )
        return 130, run

    monkeypatch.setattr(commands, "run_strategies", interrupt_run)

    exit_code: Final = main(
        ["run", "--strategy", "trace_parity", "--surface", "gateway", "--plain"]
    )

    captured: Final = capsys.readouterr()
    assert exit_code == 130
    assert "Rust <-> Python parity report" not in captured.out
    assert captured.err == "Interrupted\n"


def test_should_validate_the_chat_completions_ledger_through_check(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        ["check", "--strategy", "unit_tests_mapping", "--function", "chat_completions"]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "chat_completions" in captured.out
    assert "no ledger yet" in captured.out
    assert "ocr" not in captured.out


def test_check_scopes_the_ledger_report_to_the_requested_function(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["check", "--strategy", "unit_tests_mapping", "--function", "messages"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "messages" in captured.out
    assert "no ledger yet" in captured.out
    assert "ocr" not in captured.out


def test_check_reports_strategies_without_a_check(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["check", "--strategy", "trace_parity"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "trace_parity: no check defined" in captured.out
