from __future__ import annotations

from typing import Final

from phase_budget import Budget, violations


def _budget(*, baseline: float, degraded: float, ceiling: float = 2.0) -> Budget:
    return Budget(name="p99 RSS", baseline=baseline, degraded=degraded, ratio_ceiling=ceiling, unit=" MB", decimals=0)


class TestBudget:
    def test_growth_within_the_ceiling_is_not_a_violation(self) -> None:
        assert _budget(baseline=100, degraded=199).violation() is None

    def test_growth_exactly_at_the_ceiling_is_allowed(self) -> None:
        assert _budget(baseline=100, degraded=200).violation() is None

    def test_growth_past_the_ceiling_reports_both_values_and_the_ratio(self) -> None:
        violation: Final = _budget(baseline=100, degraded=250).violation()

        assert violation is not None
        assert "100 MB" in violation
        assert "250 MB" in violation
        assert "2.5x" in violation
        assert "2.0x allowed" in violation

    def test_shrinking_is_never_a_violation(self) -> None:
        assert _budget(baseline=100, degraded=10).violation() is None

    def test_a_missing_baseline_is_a_violation_rather_than_a_silent_pass(self) -> None:
        # The trap this guards: 0 as a baseline would make every ratio a division by zero, and
        # treating it as "no growth" would pass a run that measured nothing at all.
        violation: Final = _budget(baseline=0, degraded=4000).violation()

        assert violation is not None
        assert "nothing to compare" in violation

    def test_the_unit_and_decimals_carry_into_the_message(self) -> None:
        violation: Final = Budget(
            name="p99 latency", baseline=0.16, degraded=9.5, ratio_ceiling=8.0, unit="s", decimals=3
        ).violation()

        assert violation is not None
        assert "0.160s" in violation
        assert "9.500s" in violation


class TestViolations:
    def test_every_blown_budget_is_reported_not_just_the_first(self) -> None:
        blown: Final = violations(
            (
                _budget(baseline=100, degraded=500),
                _budget(baseline=100, degraded=120),
                Budget(name="CPU per request", baseline=10, degraded=90, ratio_ceiling=6.0, unit=" ms"),
            )
        )

        assert len(blown) == 2
        assert blown[0].startswith("p99 RSS")
        assert blown[1].startswith("CPU per request")

    def test_a_run_inside_every_budget_reports_nothing(self) -> None:
        assert violations((_budget(baseline=100, degraded=150),)) == ()
