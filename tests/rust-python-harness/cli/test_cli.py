from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import pytest

from . import main
from .catalog import STRATEGIES_ROOT, load_catalog
from .commands import _coverage_pytest_args
from .selection import pick_values, select
from ..shared.reporting.models import SDK_FUNCTIONS
from ..shared.reporting.strategy import StrategyDefinition

_SELECTOR_STRATEGY_INIT: Final = (
    "import importlib\n"
    "from pathlib import Path\n"
    "strategy = importlib.import_module('tests.rust-python-harness.shared.reporting.strategy')\n"
    "pytest_runner = importlib.import_module('tests.rust-python-harness.shared.reporting.pytest_runner')\n"
    "STRATEGY = strategy.StrategyDefinition(Path(__file__).parent, strategy.SelectorCaseSpec, pytest_runner.run_pytest)\n"
)
_SUITE_STRATEGY_INIT: Final = (
    "import importlib\n"
    "from pathlib import Path\n"
    "strategy = importlib.import_module('tests.rust-python-harness.shared.reporting.strategy')\n"
    "pytest_runner = importlib.import_module('tests.rust-python-harness.shared.reporting.pytest_runner')\n"
    "STRATEGY = strategy.StrategyDefinition(Path(__file__).parent, strategy.SuiteCaseSpec, pytest_runner.run_pytest)\n"
)


def _manifest(
    *, id: str = "example", drop_function: str | None = None, **cells: object
) -> dict[str, object]:
    functions = {
        function: cells.get(function, {"coverage": "planned"})
        for function in SDK_FUNCTIONS
        if function != drop_function
    }
    return {
        "order": 1,
        "id": id,
        "label": "Example strategy",
        "description": "Example description",
        "functions": functions,
    }


def _write_strategy_folder(
    root: Path,
    name: str = "example",
    *,
    manifest: dict[str, object],
    init_source: str = _SELECTOR_STRATEGY_INIT,
) -> Path:
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "__init__.py").write_text(init_source, encoding="utf-8")
    (folder / "strategy.json").write_text(json.dumps(manifest), encoding="utf-8")
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
        tuple(
            case.sdk_function for case in strategy.cases if case.surface == "sdk"
        )
        == SDK_FUNCTIONS
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
        assert (strategy.directory / "strategy.json").exists()
        assert (strategy.directory / "AGENTS.md").exists()
        for case in strategy.cases:
            assert isinstance(case.spec, definition.case_spec)


def test_should_reject_a_manifest_missing_an_sdk_function(tmp_path: Path) -> None:
    _write_strategy_folder(tmp_path, manifest=_manifest(drop_function="count_tokens"))

    with pytest.raises(ValueError, match="functions must exactly match"):
        load_catalog(tmp_path)


def test_should_reject_a_folder_without_a_strategy_definition(tmp_path: Path) -> None:
    _write_strategy_folder(
        tmp_path, manifest=_manifest(), init_source=""
    )

    with pytest.raises(ValueError, match="STRATEGY"):
        load_catalog(tmp_path)


def test_should_reject_a_manifest_id_that_differs_from_its_folder(tmp_path: Path) -> None:
    _write_strategy_folder(tmp_path, manifest=_manifest(id="other"))

    with pytest.raises(ValueError, match="must match folder name"):
        load_catalog(tmp_path)


def test_should_reject_selectors_in_a_suite_strategy_manifest(tmp_path: Path) -> None:
    _write_strategy_folder(
        tmp_path,
        manifest=_manifest(ocr={"coverage": "partial", "selectors": ["tests/test_api.py"]}),
        init_source=_SUITE_STRATEGY_INIT,
    )

    with pytest.raises(ValueError, match="selectors"):
        load_catalog(tmp_path)


def test_should_reject_a_planned_cell_that_configures_tests(tmp_path: Path) -> None:
    _write_strategy_folder(
        tmp_path,
        manifest=_manifest(ocr={"coverage": "planned", "selectors": ["tests/test_api.py"]}),
    )

    with pytest.raises(ValueError, match="planned case cannot configure tests"):
        load_catalog(tmp_path)


def test_should_filter_the_catalog_by_strategy_and_sdk_function() -> None:
    strategies = load_catalog()

    cases = select(strategies, {"e2e_parity"}, {"messages"})

    assert len(cases) == 1
    assert cases[0].key == "e2e_parity:messages"


def test_should_filter_the_catalog_by_surface() -> None:
    strategies = load_catalog()

    cases = select(strategies, set(), set(), "gateway")

    assert cases == ()
    assert select(strategies, {"e2e_parity"}, set(), "sdk")[0].surface == "sdk"


def test_should_reject_an_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="Unknown strategy"):
        select(load_catalog(), {"not-real"}, set())


def test_should_pick_multiple_interactive_filters() -> None:
    answers = iter(["nope", "1, 3"])

    selected = pick_values(
        "Examples",
        (("one", "One"), ("two", "Two"), ("three", "Three")),
        input_fn=lambda _: next(answers),
    )

    assert selected == {"one", "three"}


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


def test_list_honors_strategy_and_function_filters(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["list", "--strategy", "e2e_parity", "--function", "ocr"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "e2e_parity" in captured.out
    assert "trace_parity" not in captured.out
    assert "messages" not in captured.out


def test_run_reports_planned_cells_as_success(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["run", "--strategy", "trace_parity", "--surface", "gateway", "--plain"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Harness finished with exit code 0" in captured.out


def test_run_rejects_an_unknown_strategy() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["run", "--strategy", "not-real"])

    assert excinfo.value.code == 2


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
