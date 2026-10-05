from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Final
from unittest.mock import Mock, call

from pytest import MonkeyPatch

from ...shared.reporting.models import Coverage, HarnessCase, RunStatus
from ...shared.reporting.strategy import ModuleCaseSpec
from . import runner as e2e_runner
from .runner import E2ECheck, run_e2e_cases


def test_runs_checks_inside_suite_context(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    lifecycle: Final = Mock()

    @contextmanager
    def parity_checks() -> Generator[tuple[E2ECheck, ...]]:
        lifecycle("entered")
        try:
            yield (E2ECheck("check", partial(lifecycle, "checked")),)
        finally:
            lifecycle("exited")

    module: Final = SimpleNamespace(parity_checks=parity_checks)

    def import_module(_name: str, _package: str | None = None) -> SimpleNamespace:
        return module

    monkeypatch.setattr(e2e_runner.importlib, "import_module", import_module)
    case: Final = HarnessCase(
        strategy_id="e2e_parity",
        strategy_label="End-to-end parity",
        sdk_function="ocr",
        spec=ModuleCaseSpec(coverage=Coverage.PARTIAL, module="example"),
        surface="sdk",
    )

    code, run = run_e2e_cases((case,), tmp_path, lambda _: None)

    assert code == 0, run.failures
    assert run.results[case.key].status is RunStatus.PASSED
    assert lifecycle.call_args_list == [call("entered"), call("checked"), call("exited")]
