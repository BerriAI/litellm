from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
from setup.ui_query_isolation import settled_counter, validate_target


@pytest.mark.parametrize("host", ("production.example.com", "db-perf-source.litellm.svc", "localhost"))
def test_query_cancellation_rejects_non_lane_databases(host: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("E2E_UI_PERF_DISPOSABLE", "1")
    monkeypatch.setenv("E2E_RESET_SPEND_LOGS", "0")
    with pytest.raises(AssertionError, match="restricted"):
        validate_target(f"postgresql://user@{host}/litellm")


def test_query_cancellation_requires_explicit_disposable_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("E2E_UI_PERF_DISPOSABLE", raising=False)
    with pytest.raises(AssertionError, match="confirmation"):
        validate_target("postgresql://user@db-perf-perf-test-1.litellm.svc/litellm")


def test_counter_stabilization_waits_for_delayed_failures(tmp_path: Path) -> None:
    readings: Final = iter((0.0, 0.0, 2.0, 2.0, 2.0))
    evidence: Final = tmp_path / "isolation.jsonl"
    assert settled_counter(lambda: next(readings), evidence) == 2
    assert len(evidence.read_text().splitlines()) == 5
