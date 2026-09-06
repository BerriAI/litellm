from typing import Final

import pytest
from budgets import Budgets, Ceiling, Measurement, Report, calibrate, violations
from pydantic import ValidationError


def report(value: float = 100, name: str = "sync/empty", runner: str = "codspeed-macro") -> Report:
    return Report(
        environment={"runner": runner},
        revision="abc",
        extension_sha256="123",
        measurements={name: Measurement(iterations=10, samples_ns=(value,) * 30)},
    )


def test_fixed_ceiling_catches_accumulated_regressions() -> None:
    budgets: Final = calibrate((report(),) * 5)
    assert budgets.cases["sync/empty"].ceiling_ns == 120
    assert not violations(report(120), budgets)
    assert any("Over budget" in error for error in violations(report(121), budgets))


def test_missing_unexpected_and_uncalibrated_cases_fail() -> None:
    budgets: Final = calibrate((report(),) * 5)
    errors: Final = violations(report(name="other"), budgets)
    assert any("Missing measurement" in error for error in errors)
    assert any("Unbudgeted measurement" in error for error in errors)
    assert violations(report(), Budgets(environment={}, cases={}))


def test_environment_change_fails() -> None:
    assert "Environment differs from the calibrated baseline" in violations(
        report(runner="local"), calibrate((report(),) * 5)
    )


@pytest.mark.parametrize("value", (float("nan"), float("inf"), -1, 0))
def test_invalid_samples_fail(value: float) -> None:
    with pytest.raises(ValidationError):
        report(value)


def test_incomplete_samples_fail() -> None:
    with pytest.raises(ValidationError):
        Measurement(iterations=1, samples_ns=(100,))


def test_calibration_requires_five_stable_identical_macro_builds() -> None:
    with pytest.raises(ValueError, match="exactly five"):
        calibrate((report(),) * 4)
    with pytest.raises(ValueError, match="Unstable"):
        calibrate((report(),) * 4 + (report(150),))
    with pytest.raises(ValueError, match="codspeed-macro"):
        calibrate((report(runner="local"),) * 5)
    with pytest.raises(ValueError, match="identical"):
        calibrate((report(),) * 4 + (report(name="other"),))
    with pytest.raises(ValueError, match="identical"):
        calibrate((report(),) * 4 + (report().model_copy(update={"revision": "def"}),))


def test_manually_widened_or_wrong_unit_budget_fails() -> None:
    assert any(
        "Invalid budget" in error
        for error in violations(
            report(),
            Budgets(
                environment={"runner": "codspeed-macro"}, cases={"sync/empty": Ceiling(baseline_ns=100, ceiling_ns=200)}
            ),
        )
    )
    assert any(
        "Invalid budget" in error
        for error in violations(
            report(),
            Budgets(
                environment={"runner": "codspeed-macro"},
                cases={"sync/empty": Ceiling(baseline_ns=100, ceiling_ns=120, unit="ms")},
            ),
        )
    )
