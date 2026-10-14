from __future__ import annotations

import json
from typing import Final

import pytest

from tests.integration._support.budget import (
    IMPORT_SECONDS_VARIABLE,
    READINESS_OVERRIDE_VARIABLE,
    calibration,
    case_timeout_for,
    import_seconds,
    readiness_for,
    readiness_seconds,
)


@pytest.mark.parametrize("workers", (1, 2, 4))
def test_a_box_that_imports_twice_as_slowly_gets_twice_the_readiness_deadline(workers: int) -> None:
    assert readiness_for(14.0, workers) == pytest.approx(2 * readiness_for(7.0, workers))


def test_a_multi_worker_proxy_gets_the_two_imports_its_boot_pays_for() -> None:
    assert readiness_for(7.0, 2) == pytest.approx(2 * readiness_for(7.0, 1))
    assert readiness_for(7.0, 4) == readiness_for(7.0, 2)


@pytest.mark.parametrize("imported", (0.5, 7.0, 40.0))
def test_a_case_timeout_covers_a_previous_teardown_a_two_worker_boot_and_the_call(imported: float) -> None:
    assert case_timeout_for(imported) > readiness_for(imported, 2) + readiness_for(imported, 1)


def test_the_calibration_record_matches_the_budgets_it_reports() -> None:
    record: Final = json.loads(calibration(7.0))
    assert record == {
        "import_seconds": 7.0,
        "readiness_seconds_1_worker": readiness_for(7.0, 1),
        "readiness_seconds_2_workers": readiness_for(7.0, 2),
        "case_timeout_seconds": case_timeout_for(7.0),
    }


def test_the_exported_calibration_is_reused_instead_of_measured_again(monkeypatch: pytest.MonkeyPatch) -> None:
    import_seconds.cache_clear()
    monkeypatch.setenv(IMPORT_SECONDS_VARIABLE, "3.5")
    monkeypatch.delenv(READINESS_OVERRIDE_VARIABLE, raising=False)
    try:
        assert import_seconds() == 3.5
        assert readiness_seconds(2) == readiness_for(3.5, 2)
    finally:
        import_seconds.cache_clear()


def test_an_explicit_readiness_override_wins_over_the_calibration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(READINESS_OVERRIDE_VARIABLE, "42")
    assert readiness_seconds(1) == 42.0
    assert readiness_seconds(2) == 42.0
