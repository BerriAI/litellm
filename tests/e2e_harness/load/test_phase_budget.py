from __future__ import annotations

from typing import Final

from phase_budget import AbsoluteBudget, RatioBudget, violations


def _budget(*, baseline: float, degraded: float, ceiling: float = 2.0) -> RatioBudget:
    return RatioBudget(
        name="p99 RSS", baseline=baseline, degraded=degraded, ratio_ceiling=ceiling, unit=" MB", decimals=0
    )


class TestRatioBudget:
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
        violation: Final = RatioBudget(
            name="p99 latency", baseline=0.16, degraded=9.5, ratio_ceiling=8.0, unit="s", decimals=3
        ).violation()

        assert violation is not None
        assert "0.160s" in violation
        assert "9.500s" in violation


class TestAbsoluteBudget:
    def test_a_value_under_the_ceiling_is_not_a_violation(self) -> None:
        assert AbsoluteBudget(name="p99 latency", measured=1.2, ceiling=5.0, unit="s", decimals=3).violation() is None

    def test_a_value_exactly_at_the_ceiling_is_allowed(self) -> None:
        assert AbsoluteBudget(name="p99 latency", measured=5.0, ceiling=5.0, unit="s", decimals=3).violation() is None

    def test_a_value_past_the_ceiling_reports_the_measurement_and_the_ceiling(self) -> None:
        violation: Final = AbsoluteBudget(
            name="p99 latency", measured=9.5, ceiling=5.0, unit="s", decimals=3
        ).violation()

        assert violation is not None
        assert "9.500s" in violation
        assert "5.000s allowed" in violation

    def test_a_flat_ceiling_fails_a_degraded_phase_that_is_cheaper_than_its_baseline(self) -> None:
        # The whole reason this shape exists: once the breaker opens, requests skip Redis instead
        # of waiting on its socket timeout, so the chaos phase can measure faster than the healthy
        # one. A ratio against that baseline passes; the user still waited 9.5s.
        assert _budget(baseline=20.0, degraded=9.5, ceiling=2.0).violation() is None
        assert AbsoluteBudget(name="p99 latency", measured=9.5, ceiling=5.0, unit="s").violation() is not None

    def test_a_zero_measurement_is_not_a_violation(self) -> None:
        assert AbsoluteBudget(name="log bytes per request", measured=0, ceiling=12_000, unit=" B").violation() is None


class TestViolations:
    def test_every_blown_budget_is_reported_not_just_the_first(self) -> None:
        blown: Final = violations(
            (
                _budget(baseline=100, degraded=500),
                _budget(baseline=100, degraded=120),
                RatioBudget(name="CPU per request", baseline=10, degraded=90, ratio_ceiling=6.0, unit=" ms"),
            )
        )

        assert len(blown) == 2
        assert blown[0].startswith("p99 RSS")
        assert blown[1].startswith("CPU per request")

    def test_both_budget_shapes_report_together(self) -> None:
        blown: Final = violations(
            (
                _budget(baseline=100, degraded=500),
                AbsoluteBudget(name="p99 latency", measured=9.5, ceiling=5.0, unit="s", decimals=3),
            )
        )

        assert len(blown) == 2
        assert blown[0].startswith("p99 RSS")
        assert blown[1].startswith("p99 latency")

    def test_a_run_inside_every_budget_reports_nothing(self) -> None:
        assert violations((_budget(baseline=100, degraded=150),)) == ()
