"""Recurring access-schedule evaluation for virtual keys.

A key may carry ``permissions.access_schedule`` describing recurring time
windows (per-weekday, in a named IANA timezone) during which the key is
allowed to make requests. Requests outside every window are denied. Invalid
persisted schedules fail closed (deny).
"""

from __future__ import annotations

from datetime import datetime, time
from typing import Literal, Mapping, Union
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

ACCESS_SCHEDULE_PERMISSION_KEY = "access_schedule"

Weekday = Literal["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

_WEEKDAY_INDEX: Mapping[Weekday, int] = {
    "mon": 0,
    "tue": 1,
    "wed": 2,
    "thu": 3,
    "fri": 4,
    "sat": 5,
    "sun": 6,
}


class AccessWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    days: tuple[Weekday, ...]
    start: time
    end: time

    @field_validator("days")
    @classmethod
    def _days_non_empty(cls, value: tuple[Weekday, ...]) -> tuple[Weekday, ...]:
        if len(value) == 0:
            raise ValueError("'days' must contain at least one weekday")
        return value

    @field_validator("start", "end")
    @classmethod
    def _time_is_naive(cls, value: time) -> time:
        if value.tzinfo is not None:
            raise ValueError("'start'/'end' must be a local wall-clock time without an offset")
        return value

    @model_validator(mode="after")
    def _start_differs_from_end(self) -> AccessWindow:
        if self.start == self.end:
            raise ValueError("'start' and 'end' must differ (use two windows for a full day)")
        return self


class AccessSchedule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timezone: str
    windows: tuple[AccessWindow, ...]

    @field_validator("timezone")
    @classmethod
    def _timezone_is_known(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"unknown timezone '{value}'") from exc
        return value

    @field_validator("windows")
    @classmethod
    def _windows_non_empty(cls, value: tuple[AccessWindow, ...]) -> tuple[AccessWindow, ...]:
        if len(value) == 0:
            raise ValueError("'windows' must contain at least one window")
        return value


class ScheduleAbsent(BaseModel):
    tag: Literal["absent"] = "absent"


class ScheduleValid(BaseModel):
    tag: Literal["valid"] = "valid"
    schedule: AccessSchedule


class ScheduleInvalid(BaseModel):
    tag: Literal["invalid"] = "invalid"
    error: str


ScheduleParseResult = Union[ScheduleAbsent, ScheduleValid, ScheduleInvalid]


def _format_validation_error(exc: ValidationError) -> str:
    first = exc.errors()[0]
    location = ".".join(str(part) for part in first["loc"])
    prefix = f"{location}: " if location else ""
    return f"{prefix}{first['msg']}"


def parse_access_schedule(permissions: Mapping[str, object] | None) -> ScheduleParseResult:
    if not permissions or ACCESS_SCHEDULE_PERMISSION_KEY not in permissions:
        return ScheduleAbsent()
    try:
        schedule = AccessSchedule.model_validate(permissions[ACCESS_SCHEDULE_PERMISSION_KEY])
    except ValidationError as exc:
        return ScheduleInvalid(error=_format_validation_error(exc))
    return ScheduleValid(schedule=schedule)


def _window_is_open(window: AccessWindow, weekday_index: int, wall_clock: time) -> bool:
    day_indexes = frozenset(_WEEKDAY_INDEX[day] for day in window.days)
    if window.start < window.end:
        return weekday_index in day_indexes and window.start <= wall_clock < window.end
    starts_today = weekday_index in day_indexes and wall_clock >= window.start
    spills_from_yesterday = ((weekday_index - 1) % 7) in day_indexes and wall_clock < window.end
    return starts_today or spills_from_yesterday


def is_within_schedule(schedule: AccessSchedule, now: datetime) -> bool:
    local_now = now.astimezone(ZoneInfo(schedule.timezone))
    weekday_index = local_now.weekday()
    wall_clock = local_now.time()
    return any(_window_is_open(window, weekday_index, wall_clock) for window in schedule.windows)


class AccessAllowed(BaseModel):
    tag: Literal["allowed"] = "allowed"


class AccessDenied(BaseModel):
    tag: Literal["denied"] = "denied"
    reason: str


AccessDecision = Union[AccessAllowed, AccessDenied]


def evaluate_access_schedule(permissions: Mapping[str, object] | None, now: datetime) -> AccessDecision:
    parsed = parse_access_schedule(permissions)
    match parsed:
        case ScheduleAbsent():
            return AccessAllowed()
        case ScheduleInvalid(error=error):
            return AccessDenied(reason=f"key has an invalid access_schedule and is denied (fail closed): {error}")
        case ScheduleValid(schedule=schedule):
            if is_within_schedule(schedule, now):
                return AccessAllowed()
            return AccessDenied(
                reason=f"request is outside the key's allowed access_schedule (timezone {schedule.timezone})"
            )
