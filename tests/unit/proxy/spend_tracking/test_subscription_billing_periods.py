from datetime import date

from litellm.proxy.spend_tracking.subscription_billing_periods import (
    BillingPeriodRange,
    billing_periods_starting_within,
)


def test_one_period_per_month_start_inside_the_filter():
    periods = billing_periods_starting_within(date(2026, 8, 5), date(2026, 9, 1), date(2026, 10, 31))

    assert periods == (
        BillingPeriodRange(start=date(2026, 9, 5), end=date(2026, 10, 4)),
        BillingPeriodRange(start=date(2026, 10, 5), end=date(2026, 11, 4)),
    )


def test_filter_boundaries_are_inclusive():
    anchor = date(2026, 10, 1)

    assert billing_periods_starting_within(anchor, date(2026, 10, 1), date(2026, 10, 7)) == (
        BillingPeriodRange(start=date(2026, 10, 1), end=date(2026, 10, 31)),
    )
    assert billing_periods_starting_within(anchor, date(2026, 9, 25), date(2026, 10, 1)) == (
        BillingPeriodRange(start=date(2026, 10, 1), end=date(2026, 10, 31)),
    )


def test_filter_between_two_period_starts_attributes_nothing():
    assert billing_periods_starting_within(date(2026, 10, 15), date(2026, 10, 1), date(2026, 10, 7)) == ()


def test_no_period_before_the_anchor():
    assert billing_periods_starting_within(date(2026, 10, 1), date(2026, 8, 1), date(2026, 9, 30)) == ()
    assert billing_periods_starting_within(date(2026, 10, 1), date(2026, 8, 1), date(2026, 10, 1)) == (
        BillingPeriodRange(start=date(2026, 10, 1), end=date(2026, 10, 31)),
    )


def test_anchor_day_clamps_to_short_months_without_drifting():
    periods = billing_periods_starting_within(date(2026, 1, 31), date(2026, 2, 1), date(2026, 5, 31))

    assert [period.start for period in periods] == [
        date(2026, 2, 28),
        date(2026, 3, 31),
        date(2026, 4, 30),
        date(2026, 5, 31),
    ]
    assert periods[0].end == date(2026, 3, 30)
    assert periods[-1].end == date(2026, 6, 29)


def test_period_ends_the_day_before_the_next_start_across_a_year_boundary():
    periods = billing_periods_starting_within(date(2025, 12, 20), date(2025, 12, 1), date(2026, 1, 31))

    assert periods == (
        BillingPeriodRange(start=date(2025, 12, 20), end=date(2026, 1, 19)),
        BillingPeriodRange(start=date(2026, 1, 20), end=date(2026, 2, 19)),
    )


def test_inverted_filter_is_empty():
    assert billing_periods_starting_within(date(2026, 1, 1), date(2026, 3, 1), date(2026, 2, 1)) == ()


def test_the_calendar_s_last_month_ends_on_its_last_day_instead_of_failing():
    assert billing_periods_starting_within(date(9999, 11, 15), date(9999, 11, 1), date(9999, 12, 31)) == (
        BillingPeriodRange(start=date(9999, 11, 15), end=date(9999, 12, 14)),
        BillingPeriodRange(start=date(9999, 12, 15), end=date(9999, 12, 31)),
    )
    assert billing_periods_starting_within(date(9999, 12, 1), date(9999, 12, 1), date(9999, 12, 31)) == (
        BillingPeriodRange(start=date(9999, 12, 1), end=date(9999, 12, 31)),
    )
