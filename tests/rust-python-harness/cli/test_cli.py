from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Final

import pytest

from ..shared.reporting.models import (
    SDK_FUNCTIONS,
    SURFACES,
    CaseDisposition,
    HarnessCase,
    HarnessRun,
    RunStatus,
    Strategy,
)
from ..shared.reporting.strategy import NotImplementedCaseSpec, SkippedCaseSpec, StrategyDefinition
from ..shared.reporting.ui import PlainDashboard, final_report, make_dashboard
from ..strategies.unit_tests_mapping.mappings import UNIT_TEST_CONTRACTS
from ..strategies.unit_tests_parity import UNIT_PARITY_SUITES
from ..strategies.unit_tests_rust import RUST_SUITES
from . import main
from .catalog import STRATEGIES_ROOT, load_catalog
from .commands import REPO_ROOT, select_cases


def _strategy_source(
    *,
    strategy_id: str = "example",
    surfaces: tuple[str, ...] = (),
    drop: tuple[str | None, str] | None = None,
    duplicate: tuple[str | None, str] | None = None,
    incompatible: tuple[str | None, str] | None = None,
) -> str:
    cells: Final = tuple(
        (surface, function)
        for surface in (surfaces or (None,))
        for function in SDK_FUNCTIONS
        if (surface, function) != drop
    )
    definitions: Final = tuple(
        (
            f"strategy.CaseDefinition({function!r}, "
            "strategy.ModuleCaseSpec(coverage=models.Coverage.COMPLETE, module='tests.example'), "
            f"surface={surface!r})"
            if (surface, function) == incompatible
            else (
                f"strategy.CaseDefinition({function!r}, "
                "strategy.NotImplementedCaseSpec(reason='Not implemented yet'), "
                f"surface={surface!r})"
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
        "    return (rendering.ReportSection('Example outcomes', "
        "tuple(rendering.render_case_outcome(r) for r in results)),)\n"
        f"CASES = ({','.join(definitions)},)\n"
        "STRATEGY = strategy.StrategyDefinition("
        f"id={strategy_id!r}, order=1, label='Example strategy', description='Example description', "
        "directory=Path(__file__).parent, runnable_spec=strategy.SuiteCaseSpec, cases=CASES, "
        f"run=runner.run_trace_cases, render=render, surfaces={surfaces!r})\n"
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


def test_should_load_surface_aware_and_function_only_strategies() -> None:
    strategies: Final = load_catalog()

    assert [strategy.id for strategy in strategies] == [
        "e2e_parity",
        "trace_parity",
        "unit_tests_mapping",
        "unit_tests_parity",
        "unit_tests_rust",
    ]
    for strategy in strategies:
        expected: Final = tuple(
            (surface, function) for surface in (strategy.definition.surfaces or (None,)) for function in SDK_FUNCTIONS
        )
        assert tuple((case.surface, case.sdk_function) for case in strategy.cases) == expected


def test_unit_strategies_use_function_only_cases() -> None:
    strategies: Final = {
        strategy.id: strategy
        for strategy in load_catalog()
        if strategy.id in {"unit_tests_mapping", "unit_tests_parity", "unit_tests_rust"}
    }

    for sdk_function in SDK_FUNCTIONS:
        cases: Final = tuple(
            case for strategy in strategies.values() for case in strategy.cases if case.sdk_function == sdk_function
        )
        assert len(cases) == 3
        assert all(case.surface is None for case in cases)
        expected_mapping: Final = (
            CaseDisposition.RUNNABLE if sdk_function in UNIT_TEST_CONTRACTS else CaseDisposition.NOT_IMPLEMENTED
        )
        assert cases[0].spec.disposition is expected_mapping
        expected_parity: Final = (
            CaseDisposition.RUNNABLE if sdk_function in UNIT_PARITY_SUITES else CaseDisposition.NOT_IMPLEMENTED
        )
        expected_rust: Final = (
            CaseDisposition.RUNNABLE if sdk_function in RUST_SUITES else CaseDisposition.NOT_IMPLEMENTED
        )
        assert cases[1].spec.disposition is expected_parity
        assert cases[2].spec.disposition is expected_rust


def test_raw_dashboard_is_always_the_default() -> None:
    assert isinstance(make_dashboard(load_catalog()), PlainDashboard)


def test_every_strategy_folder_complies() -> None:
    strategies: Final = load_catalog()
    folders: Final = {
        path.name for path in STRATEGIES_ROOT.iterdir() if path.is_dir() and (path / "__init__.py").exists()
    }

    assert folders == {strategy.id for strategy in strategies}
    for strategy in strategies:
        definition: Final = strategy.definition
        assert isinstance(definition, StrategyDefinition)
        assert definition.directory == strategy.directory
        assert not (strategy.directory / "strategy.json").exists()
        assert (strategy.directory / "AGENTS.md").exists()
        for case in strategy.cases:
            if case.spec.disposition is CaseDisposition.RUNNABLE:
                assert isinstance(case.spec, definition.runnable_spec)


@pytest.mark.parametrize("surfaces", ((), SURFACES))
def test_should_reject_a_registry_missing_a_declared_matrix_cell(tmp_path: Path, surfaces: tuple[str, ...]) -> None:
    surface: Final = surfaces[0] if surfaces else None
    _write_strategy_folder(
        tmp_path,
        init_source=_strategy_source(surfaces=surfaces, drop=(surface, "count_tokens")),
    )

    with pytest.raises(ValueError, match="must exactly match its declared matrix"):
        load_catalog(tmp_path)


def test_should_reject_a_duplicate_matrix_cell(tmp_path: Path) -> None:
    _write_strategy_folder(tmp_path, init_source=_strategy_source(duplicate=(None, "ocr")))

    with pytest.raises(ValueError, match="duplicate strategy cases"):
        load_catalog(tmp_path)


def test_should_reject_invalid_declared_surfaces(tmp_path: Path) -> None:
    _write_strategy_folder(tmp_path, init_source=_strategy_source(surfaces=("sdk", "sdk")))

    with pytest.raises(ValueError, match="invalid strategy surfaces"):
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
    _write_strategy_folder(tmp_path, init_source=_strategy_source(incompatible=(None, "ocr")))

    with pytest.raises(ValueError, match="runnable cases do not match SuiteCaseSpec"):
        load_catalog(tmp_path)


@pytest.mark.parametrize("case_type", (NotImplementedCaseSpec, SkippedCaseSpec))
def test_should_reject_an_unavailable_case_with_a_blank_reason(
    case_type: type[NotImplementedCaseSpec] | type[SkippedCaseSpec],
) -> None:
    with pytest.raises(ValueError, match="at least 1 character"):
        case_type(reason=" ")


def test_should_select_functions_and_surfaces() -> None:
    strategy: Final = next(strategy for strategy in load_catalog() if strategy.id == "e2e_parity")

    assert tuple(case.key for case in select_cases((strategy,), {"messages"})) == (
        "e2e_parity:messages",
        "e2e_parity:gateway:messages",
    )
    assert tuple(case.display_name for case in select_cases((strategy,), {"ocr"}, "gateway")) == ("gateway/ocr",)


def _assert_unavailable_cell(strategy: Strategy, case: HarnessCase, section_title: str) -> None:
    spec: Final = case.spec
    assert isinstance(spec, (NotImplementedCaseSpec, SkippedCaseSpec))
    scoped: Final = replace(strategy, cases=(case,))
    exit_code, run = strategy.definition.run((case,), REPO_ROOT, lambda _: None)
    result: Final = run.results[case.key]
    expected: Final = (
        RunStatus.NOT_IMPLEMENTED if spec.disposition is CaseDisposition.NOT_IMPLEMENTED else RunStatus.SKIPPED
    )
    report: Final = final_report(run, exit_code, (scoped,))

    assert exit_code == 0
    assert result.status is expected
    assert spec.reason in report
    assert section_title in report
    expected_result: Final = "NOT RUN" if expected is RunStatus.NOT_IMPLEMENTED else "SKIPPED"
    expected_implemented: Final = 0 if expected is RunStatus.NOT_IMPLEMENTED else 1
    assert f"Result: {expected_result}" in report
    assert f"Harness support: {expected_implemented}/1 cases implemented" in report


def test_every_unavailable_case_finishes_and_explains_itself() -> None:
    section_titles: Final = {
        "e2e_parity": "End-to-end parity outcomes",
        "trace_parity": "trace comparisons",
        "unit_tests_mapping": "Python/Rust unit-test mappings",
        "unit_tests_parity": "Python backend parity outcomes",
        "unit_tests_rust": "Native Rust unit-test outcomes",
    }
    unavailable: Final = tuple(
        (strategy, case)
        for strategy in load_catalog()
        for case in strategy.cases
        if case.spec.disposition is not CaseDisposition.RUNNABLE
    )

    for strategy, case in unavailable:
        _assert_unavailable_cell(strategy, case, section_titles[strategy.id])


@pytest.mark.parametrize(
    ("strategy_id", "present", "absent"),
    (
        ("e2e_parity", "--surface", "--pytest-arg"),
        ("trace_parity", "--surface", "--pytest-arg"),
        ("unit_tests_parity", "--pytest-arg", "--surface"),
        ("unit_tests_mapping", "--detail", "--surface"),
        ("unit_tests_rust", "--function", "--surface"),
    ),
)
def test_strategy_help_only_lists_supported_options(
    strategy_id: str,
    present: str,
    absent: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code: Final = main(["run", strategy_id, "--help"])
    captured: Final = capsys.readouterr()

    assert exit_code == 0
    assert present in captured.out
    assert absent not in captured.out


def test_run_help_lists_all_and_every_strategy(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code: Final = main(["run", "--help"])
    captured: Final = capsys.readouterr()

    assert exit_code == 0
    for command in (
        "all",
        "e2e_parity",
        "trace_parity",
        "unit_tests_mapping",
        "unit_tests_parity",
        "unit_tests_rust",
    ):
        assert command in captured.out


@pytest.mark.parametrize(
    "argv",
    (
        ("list",),
        ("check",),
        ("run", "--strategy", "unit_tests_parity"),
        ("run", "unit_tests_parity", "--surface", "sdk"),
        ("run", "unit_tests_parity", "--plain"),
        ("run", "unit_tests_parity", "--runner-arg=-x"),
        ("run", "all", "--pytest-arg=-x"),
    ),
)
def test_removed_commands_and_options_are_rejected(argv: tuple[str, ...], capsys: pytest.CaptureFixture[str]) -> None:
    exit_code: Final = main(argv)
    captured: Final = capsys.readouterr()

    assert exit_code == 2
    assert captured.err


def test_strategy_command_forwards_repeated_filters_and_runner_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli: Final = importlib.import_module("tests.rust-python-harness.cli")
    captured: list[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = []

    def capture_run(
        strategies: Sequence[Strategy],
        cases: Sequence[HarnessCase],
        runner_args: Sequence[str] = (),
    ) -> int:
        captured.append(
            (
                tuple(strategy.id for strategy in strategies),
                tuple(case.display_name for case in cases),
                tuple(runner_args),
            )
        )
        return 0

    monkeypatch.setattr(cli, "run_command", capture_run)

    assert (
        main(
            [
                "run",
                "unit_tests_parity",
                "--function",
                "ocr",
                "--function",
                "messages",
                "--pytest-arg=-x",
            ]
        )
        == 0
    )
    assert captured == [
        (("unit_tests_parity",), ("ocr", "messages"), ("-x",)),
    ]


def test_omitted_surface_selects_every_strategy_surface(monkeypatch: pytest.MonkeyPatch) -> None:
    cli: Final = importlib.import_module("tests.rust-python-harness.cli")
    selected: list[str] = []

    def capture_run(
        strategies: Sequence[Strategy],
        cases: Sequence[HarnessCase],
        runner_args: Sequence[str] = (),
    ) -> int:
        del strategies, runner_args
        selected.extend(case.display_name for case in cases)
        return 0

    monkeypatch.setattr(cli, "run_command", capture_run)

    assert main(["run", "e2e_parity", "--function", "ocr"]) == 0
    assert selected == ["sdk/ocr", "gateway/ocr"]


def test_run_all_selects_every_declared_case_once(monkeypatch: pytest.MonkeyPatch) -> None:
    cli: Final = importlib.import_module("tests.rust-python-harness.cli")
    selected: list[HarnessCase] = []

    def capture_run(
        strategies: Sequence[Strategy],
        cases: Sequence[HarnessCase],
        runner_args: Sequence[str] = (),
    ) -> int:
        del strategies, runner_args
        selected.extend(cases)
        return 0

    monkeypatch.setattr(cli, "run_command", capture_run)

    assert main(["run", "all", "--function", "ocr"]) == 0
    assert len(selected) == 7
    assert sum(case.surface is None for case in selected) == 3
    assert sum(case.surface is not None for case in selected) == 4


def test_run_reports_not_implemented_surface_as_not_run(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code: Final = main(["run", "trace_parity", "--surface", "gateway", "--function", "ocr"])
    captured: Final = capsys.readouterr()

    assert exit_code == 0
    assert "Result: NOT RUN" in captured.out
    assert "Harness support: 0/1 cases implemented" in captured.out
    assert "Cases: 1 selected, 1 not implemented, 0 skipped" in captured.out
    assert "Not implemented" in captured.out
    assert "No gateway OCR trace-parity case is registered." in captured.out


def test_keyboard_interrupt_exits_cleanly(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli: Final = importlib.import_module("tests.rust-python-harness.cli")

    def interrupt() -> tuple[object, ...]:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "load_catalog", interrupt)

    exit_code: Final = main(["run", "all"])
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
        run: Final = HarnessRun.from_cases(case for strategy in strategies for case in strategy.cases)
        return 130, run

    monkeypatch.setattr(commands, "run_strategies", interrupt_run)

    exit_code: Final = main(["run", "trace_parity", "--surface", "gateway"])
    captured: Final = capsys.readouterr()

    assert exit_code == 130
    assert "Rust <-> Python parity report" not in captured.out
    assert captured.err == "Interrupted\n"
