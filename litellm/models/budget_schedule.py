from collections.abc import Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import date, datetime, time, timedelta, timezone
from itertools import chain, pairwise
from typing import Final, Literal, TypeAlias
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import ConfigDict, field_serializer, field_validator

from litellm.types.llms.base import LiteLLMPydanticObjectBase

Weekday: TypeAlias = Literal["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

WEEKDAYS: Final[tuple[Weekday, ...]] = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
SCHEDULE_PREFIX: Final = "sched:"
_DAY_MINUTES: Final = 24 * 60
_WEEK_MINUTES: Final = 7 * _DAY_MINUTES

_schedule_time: Final[ContextVar[datetime | None]] = ContextVar("budget_schedule_time", default=None)


@contextmanager
def resolve_schedules_at(at: datetime | None) -> Generator[None]:
    """Resolve scheduled windows at the request's start, so a request is charged to the period that admitted it."""
    token: Final = _schedule_time.set(None if at is None else at.astimezone(timezone.utc))
    try:
        yield
    finally:
        _schedule_time.reset(token)


def schedule_clock() -> datetime:
    return _schedule_time.get() or datetime.now(timezone.utc)


class BudgetSchedule(LiteLLMPydanticObjectBase):
    model_config = ConfigDict(frozen=True)

    timezone: str
    days: tuple[Weekday, ...] = WEEKDAYS
    start: time
    end: time

    @field_validator("timezone")
    @classmethod
    def _known_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as error:
            raise ValueError(f"Unknown timezone '{value}'") from error
        return value

    @field_validator("days")
    @classmethod
    def _unique_days(cls, value: tuple[Weekday, ...]) -> tuple[Weekday, ...]:
        if not value or len(set(value)) != len(value):
            raise ValueError("days must list each weekday at most once and at least one weekday")
        return tuple(day for day in WEEKDAYS if day in value)

    @field_validator("start", "end")
    @classmethod
    def _whole_minutes(cls, value: time) -> time:
        if value.second or value.microsecond or value.tzinfo is not None:
            raise ValueError("start and end are local 'HH:MM' times")
        return value

    @field_serializer("start", "end")
    def _as_hh_mm(self, value: time) -> str:
        return f"{value:%H:%M}"

    @classmethod
    def from_label(cls, label: str) -> "BudgetSchedule | None":
        if not label.startswith(SCHEDULE_PREFIX):
            return None
        zone, _, rest = label.removeprefix(SCHEDULE_PREFIX).partition(":")
        days, _, hours = rest.partition(":")
        start, _, end = hours.partition("-")
        try:
            return cls.model_validate({"timezone": zone, "days": tuple(days.split(",")), "start": start, "end": end})
        except ValueError:
            return None

    def label(self) -> str:
        return f"{SCHEDULE_PREFIX}{self.timezone}:{','.join(self.days)}:{self.start:%H:%M}-{self.end:%H:%M}"

    def _length_minutes(self) -> int:
        return (_minutes(self.end) - _minutes(self.start)) % _DAY_MINUTES or _DAY_MINUTES

    def weekly_intervals(self) -> tuple[tuple[int, int], ...]:
        """The [start, end) minutes of the week this schedule covers, split where a period wraps past Sunday."""
        return tuple(chain.from_iterable(map(self._day_intervals, self.days)))

    def _day_intervals(self, day: Weekday) -> tuple[tuple[int, int], ...]:
        start: Final = WEEKDAYS.index(day) * _DAY_MINUTES + _minutes(self.start)
        end: Final = start + self._length_minutes()
        if end <= _WEEK_MINUTES:
            return ((start, end),)
        return ((start, _WEEK_MINUTES), (0, end - _WEEK_MINUTES))

    def _period_from(self, day: date) -> tuple[datetime, datetime]:
        zone: Final = ZoneInfo(self.timezone)
        start: Final = datetime.combine(day, self.start, tzinfo=zone)
        end_day: Final = day if self.end > self.start else day + timedelta(days=1)
        end: Final = datetime.combine(end_day, self.end, tzinfo=zone)
        return start.astimezone(timezone.utc), end.astimezone(timezone.utc)

    def _starts_on(self, day: date) -> bool:
        return WEEKDAYS[day.weekday()] in self.days

    def period_at(self, at: datetime) -> tuple[datetime, datetime] | None:
        """The active period's UTC [start, end), or None when `at` falls outside every period."""
        local_day: Final = at.astimezone(ZoneInfo(self.timezone)).date()
        candidates: Final = (
            self._period_from(day) for day in (local_day, local_day - timedelta(days=1)) if self._starts_on(day)
        )
        return next(((start, end) for start, end in candidates if start <= at < end), None)

    def next_start(self, at: datetime) -> datetime:
        local_day: Final = at.astimezone(ZoneInfo(self.timezone)).date()
        starts: Final = (
            self._period_from(local_day + timedelta(days=offset))[0]
            for offset in range(8)
            if self._starts_on(local_day + timedelta(days=offset))
        )
        return next(start for start in starts if start > at)


def _minutes(value: time) -> int:
    return value.hour * 60 + value.minute


def scheduled_period(budget_duration: str, at: datetime) -> tuple[datetime, datetime] | None:
    schedule: Final = BudgetSchedule.from_label(budget_duration)
    return None if schedule is None else schedule.period_at(at)


def is_scheduled_duration(budget_duration: object) -> bool:
    return isinstance(budget_duration, str) and budget_duration.startswith(SCHEDULE_PREFIX)


def window_counter_name(budget_duration: str, window_start: datetime | None) -> str:
    """Scheduled windows get one counter per period, so a new period never inherits the last one's spend."""
    if window_start is None or not is_scheduled_duration(budget_duration):
        return budget_duration
    return f"{budget_duration}@{window_start.astimezone(timezone.utc).isoformat()}"


def validate_schedules(schedules: Sequence[BudgetSchedule]) -> None:
    if len({schedule.timezone for schedule in schedules}) > 1:
        raise ValueError("All scheduled budget windows must use the same timezone")
    intervals: Final = sorted(chain.from_iterable(map(_indexed_intervals, enumerate(schedules))))
    clash: Final = next((pair for pair in pairwise(intervals) if pair[1][0] < pair[0][1]), None)
    if clash is not None:
        first, second = sorted((clash[0][2], clash[1][2]))
        raise ValueError(
            f"Scheduled budget windows overlap: {schedules[first].label()} and {schedules[second].label()}"
        )


def _indexed_intervals(indexed: tuple[int, BudgetSchedule]) -> tuple[tuple[int, int, int], ...]:
    index, schedule = indexed
    return tuple((start, end, index) for start, end in schedule.weekly_intervals())
