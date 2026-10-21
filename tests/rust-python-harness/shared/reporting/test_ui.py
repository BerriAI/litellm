from __future__ import annotations

from typing import Final

from .models import Coverage, HarnessCase, HarnessRun, RunStatus
from .strategy import ModuleCaseSpec
from .ui import _format_duration, _summary


def test_should_format_developer_facing_run_context() -> None:
    run: Final = HarnessRun.from_cases(
        (
            HarnessCase(
                strategy_id="example",
                strategy_label="Example",
                sdk_function="messages",
                spec=ModuleCaseSpec(coverage=Coverage.COMPLETE, module="tests.example"),
            ),
        )
    )
    result: Final = next(iter(run.results.values()))
    result.collected.add("tests/test_parity.py::test_one")
    result.record("tests/test_parity.py::test_one", RunStatus.PASSED, 1.25)

    assert _summary(run) == (1, 0, 0, 0)
    assert _format_duration(1.25) == "1.2s"
