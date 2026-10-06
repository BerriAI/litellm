from calendar import monthrange
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Final


@dataclass(frozen=True, slots=True)
class BillingPeriodRange:
    start: date
    end: date


def _month_index(anchor: date, months_after_anchor: int) -> int:
    return anchor.year * 12 + (anchor.month - 1) + months_after_anchor


_LAST_MONTH_INDEX: Final = _month_index(date.max, 0)


def _month_start(anchor: date, months_after_anchor: int) -> date:
    month_index: Final = _month_index(anchor, months_after_anchor)
    year: Final = month_index // 12
    month: Final = month_index % 12 + 1
    return date(year, month, min(anchor.day, monthrange(year, month)[1]))


def _period_end(anchor: date, months_after_anchor: int) -> date:
    if _month_index(anchor, months_after_anchor + 1) > _LAST_MONTH_INDEX:
        return date.max
    return _month_start(anchor, months_after_anchor + 1) - timedelta(days=1)


def _first_candidate_index(anchor: date, start_date: date) -> int:
    months_apart: Final = (start_date.year - anchor.year) * 12 + (start_date.month - anchor.month)
    return max(0, months_apart - 1)


def billing_periods_starting_within(anchor: date, start_date: date, end_date: date) -> tuple[BillingPeriodRange, ...]:
    """Monthly periods anchored on `anchor`'s day of month whose start lies inside [start_date, end_date]."""
    if end_date < start_date or end_date < anchor:
        return ()
    first_index: Final = _first_candidate_index(anchor, start_date)
    last_index: Final = min(
        first_index + _month_span(start_date, end_date) + 1, _LAST_MONTH_INDEX - _month_index(anchor, 0)
    )
    candidates: Final = (
        BillingPeriodRange(start=_month_start(anchor, index), end=_period_end(anchor, index))
        for index in range(first_index, last_index + 1)
    )
    return tuple(period for period in candidates if start_date <= period.start <= end_date)


def _month_span(start_date: date, end_date: date) -> int:
    return (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
