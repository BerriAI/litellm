"""Harness tests for the shared HTTP-probe cell body.

No `e2e` marker and no proxy: these drive `_probe_cell` with injected
fakes to pin the three properties the cells depend on - the tiers are
probed concurrently, rows are reported in declared order rather than
completion order, and a probe that raises becomes a value instead of
losing the other tiers' outcomes. Same shape of harness coverage as
`coverage_registry/test_collector.py`.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, List

import pytest
from pydantic import BaseModel

from claude_code._probe_cell import probe_models_parallel, run_probe_cell
from e2e_http import NetworkError, Result, Success


MODELS = ["claude-haiku-4-5", "claude-sonnet-4-5", "claude-opus-4-7"]
BARRIER_TIMEOUT_SECONDS = 10.0


class _Body(BaseModel):
    model: str


def _ok(model: str) -> Result[_Body]:
    return Success(status_code=200, data=_Body(model=model))


def _probed_model(result: Result[_Body]) -> str | None:
    match result:
        case Success(data=data):
            return data.model
        case _:
            return None


def _network_error(result: Result[_Body]) -> str | None:
    match result:
        case NetworkError(message=message):
            return message
        case _:
            return None


class _Recorder:
    """Stand-in for the `compat_result` fixture: keeps the rows a cell reports."""

    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []

    def add(self, row: Dict[str, Any]) -> None:
        self.rows.append(row)


def _staggered_probe(delays: Dict[str, float]) -> Callable[[str], Result[_Body]]:
    """A probe whose completion order is the reverse of the declared order."""

    def probe(model: str) -> Result[_Body]:
        time.sleep(delays[model])
        return _ok(model)

    return probe


def test_tiers_are_probed_concurrently() -> None:
    """Every tier must be in flight at once: the barrier only trips if all
    three probes are running together, so a sequential implementation
    times out on the first wait."""
    barrier = threading.Barrier(len(MODELS), timeout=BARRIER_TIMEOUT_SECONDS)

    def probe(model: str) -> Result[_Body]:
        barrier.wait()
        return _ok(model)

    outcomes = probe_models_parallel(models=MODELS, probe=probe)

    assert [_probed_model(outcomes[model]) for model in MODELS] == MODELS


def test_rows_follow_declared_order_not_completion_order() -> None:
    """The results artifact must be deterministic, so rows are reported in
    the order the cell declares its tiers even when the slowest tier is
    declared first."""
    recorder = _Recorder()
    delays = {
        "claude-haiku-4-5": 0.15,
        "claude-sonnet-4-5": 0.05,
        "claude-opus-4-7": 0.0,
    }

    def check_shape(result: Result[_Body]) -> str | None:
        model = _probed_model(result)
        return None if model == "claude-sonnet-4-5" else f"bad {model}"

    with pytest.raises(pytest.fail.Exception) as excinfo:
        run_probe_cell(
            compat_result=recorder,
            models=MODELS,
            probe=_staggered_probe(delays),
            check_shape=check_shape,
            probe_name="count_tokens",
        )

    assert recorder.rows == [
        {
            "status": "fail",
            "error": (
                "[claude-haiku-4-5] count_tokens probe failed: "
                "bad claude-haiku-4-5"
            ),
        },
        {"status": "pass"},
        {
            "status": "fail",
            "error": (
                "[claude-opus-4-7] count_tokens probe failed: bad claude-opus-4-7"
            ),
        },
    ]
    assert str(excinfo.value).index("claude-haiku-4-5") < str(excinfo.value).index(
        "claude-opus-4-7"
    )


def test_all_tiers_passing_reports_one_pass_row_each() -> None:
    recorder = _Recorder()

    run_probe_cell(
        compat_result=recorder,
        models=MODELS,
        probe=_ok,
        check_shape=lambda _: None,
        probe_name="tool_search",
    )

    assert recorder.rows == [{"status": "pass"}] * len(MODELS)


def test_a_raising_probe_becomes_a_value_and_spares_the_other_tiers() -> None:
    """One tier blowing up (rate-limiter file I/O, an unmappable model id)
    must not discard the tiers that answered."""

    def probe(model: str) -> Result[_Body]:
        if model == "claude-sonnet-4-5":
            raise RuntimeError("boom")
        return _ok(model)

    outcomes = probe_models_parallel(models=MODELS, probe=probe)

    assert _probed_model(outcomes["claude-haiku-4-5"]) == "claude-haiku-4-5"
    assert _probed_model(outcomes["claude-opus-4-7"]) == "claude-opus-4-7"
    message = _network_error(outcomes["claude-sonnet-4-5"])
    assert message is not None
    assert "RuntimeError: boom" in message


def test_no_models_is_a_programming_error() -> None:
    with pytest.raises(ValueError):
        probe_models_parallel(models=[], probe=_ok)
