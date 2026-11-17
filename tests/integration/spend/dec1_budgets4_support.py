"""Shared building blocks for the `test_dec1_budgets4_*` contract files (budgets, round 4).

Round 4 pins what round 3 (`test_dec1_budgets3_*`) listed as not covered: what the scheduled reset job
(`litellm/proxy/common_utils/reset_budget_job.py::ResetBudgetJob.reset_budget`) does to every budgeted level and its
Redis counters, `budget_limits` window resets, what a call in flight across a reset is charged to, the reseed and
floor of Redis spend counters from Postgres, soft budget alerts and two proxies sharing one Redis.

Resets are driven deterministically: an owned proxy runs the reset job every 2 to 3 seconds
(`PROXY_BUDGET_RESCHEDULER_MIN_TIME` / `MAX_TIME`, see `dec1_budgets2_support.FAST_RESET`) and a test makes a row
due by moving its `budget_reset_at` (or a window's `reset_at` inside `budget_limits`) one second into the past with
SQL. The durations themselves stay `1d`, so the reset job always writes the next 00:00 UTC as the new reset time
(`duration_parser.py::get_next_standardized_reset_time`, pinned in `test_dec1_budgets_reset_schedule.py`), and no
test depends on a natural boundary passing.

Price: the round 1 `hold_model` (input free, 0.004 a completion token, 5 completion tokens per served call, so one
served call costs 0.02 and a hold is `max_tokens * 0.004`).
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final, LiteralString

from pydantic import JsonValue

from tests.integration._support.client import Gateway, eventually, list_value, object_value, string_value
from tests.integration._support.database import write_rows
from tests.integration.spend.dec1_budgets2_support import FAST_RESET, owned

RESET_SECONDS: Final = 60

__all__ = [
    "FAST_RESET",
    "RESET_SECONDS",
    "Due",
    "budget_id_of",
    "committed",
    "expire_window",
    "make_budget_row_due",
    "make_due",
    "next_utc_midnight",
    "reset_at_of",
    "reset_proxy",
    "stored_windows",
    "today_utc_midnight",
    "wait_for_row_reset",
    "wait_for_window_reset",
    "window_reset_at",
    "window_spend_row",
]


class Due:
    """The tables whose rows carry their own `budget_reset_at`, and their identity column."""

    KEY: Final = ("LiteLLM_VerificationToken", "token")
    TEAM: Final = ("LiteLLM_TeamTable", "team_id")
    USER: Final = ("LiteLLM_UserTable", "user_id")
    BUDGET: Final = ("LiteLLM_BudgetTable", "budget_id")


_RESET_AT: Final[Mapping[tuple[str, str], LiteralString]] = {
    Due.KEY: "SELECT budget_reset_at AT TIME ZONE 'UTC' AS reset_at FROM \"LiteLLM_VerificationToken\" WHERE token = %s",
    Due.TEAM: "SELECT budget_reset_at AT TIME ZONE 'UTC' AS reset_at FROM \"LiteLLM_TeamTable\" WHERE team_id = %s",
    Due.USER: "SELECT budget_reset_at AT TIME ZONE 'UTC' AS reset_at FROM \"LiteLLM_UserTable\" WHERE user_id = %s",
    Due.BUDGET: (
        "SELECT budget_reset_at AT TIME ZONE 'UTC' AS reset_at FROM \"LiteLLM_BudgetTable\" WHERE budget_id = %s"
    ),
}

_MAKE_DUE: Final[Mapping[tuple[str, str], LiteralString]] = {
    Due.KEY: (
        "UPDATE \"LiteLLM_VerificationToken\" SET budget_reset_at = now() - interval '1 second' WHERE token = %s"
    ),
    Due.TEAM: "UPDATE \"LiteLLM_TeamTable\" SET budget_reset_at = now() - interval '1 second' WHERE team_id = %s",
    Due.USER: "UPDATE \"LiteLLM_UserTable\" SET budget_reset_at = now() - interval '1 second' WHERE user_id = %s",
    Due.BUDGET: (
        "UPDATE \"LiteLLM_BudgetTable\" SET budget_reset_at = now() - interval '1 second' WHERE budget_id = %s"
    ),
}

_BUDGET_ID: Final[Mapping[str, LiteralString]] = {
    "org": 'SELECT budget_id FROM "LiteLLM_OrganizationTable" WHERE organization_id = %s',
    "tag": 'SELECT budget_id FROM "LiteLLM_TagTable" WHERE tag_name = %s',
    "project": 'SELECT budget_id FROM "LiteLLM_ProjectTable" WHERE project_id = %s',
    "end_user": 'SELECT budget_id FROM "LiteLLM_EndUserTable" WHERE user_id = %s',
    "team_default_member": "SELECT metadata->>'team_member_budget_id' AS budget_id FROM \"LiteLLM_TeamTable\" "
    "WHERE team_id = %s",
    "membership": 'SELECT budget_id FROM "LiteLLM_TeamMembership" WHERE user_id = %s AND team_id = %s',
}


def _as_utc(value: JsonValue) -> datetime:
    parsed: Final = datetime.fromisoformat(string_value(value) if isinstance(value, str) else str(value))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def today_utc_midnight() -> datetime:
    return datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)


def next_utc_midnight() -> datetime:
    return today_utc_midnight() + timedelta(days=1)


@contextmanager
def reset_proxy(
    gateway: Gateway,
    directory: Path,
    *,
    general_settings: Mapping[str, JsonValue] | None = None,
    litellm_settings: Mapping[str, JsonValue] | None = None,
    environment: Mapping[str, str] | None = None,
) -> Iterator[Gateway]:
    """An owned proxy on the suite's database and Redis whose reset job ticks every 2 to 3 seconds."""
    with owned(
        gateway,
        directory,
        general_settings=general_settings,
        litellm_settings=litellm_settings,
        environment={**FAST_RESET, **(environment or {})},
    ) as booted:
        yield booted


def reset_at_of(gateway: Gateway, due: tuple[str, str], identity: str) -> datetime | None:
    rows: Final = gateway.rows(_RESET_AT[due], (identity,))
    value: Final = rows[0]["reset_at"] if rows else None
    return None if value is None else _as_utc(value)


def make_due(due: tuple[str, str], identity: str) -> None:
    """Move one row's `budget_reset_at` one second into the past, so the next tick of the reset job resets it."""
    write_rows(_MAKE_DUE[due], (identity,))


def make_budget_row_due(budget_id: str) -> None:
    make_due(Due.BUDGET, budget_id)


def budget_id_of(gateway: Gateway, entity: str, *identity: str) -> str:
    rows: Final = gateway.rows(_BUDGET_ID[entity], tuple(identity))
    assert rows and rows[0]["budget_id"] is not None, (entity, identity, rows)
    return string_value(rows[0]["budget_id"])


def wait_for_row_reset(gateway: Gateway, due: tuple[str, str], identity: str) -> datetime:
    """Wait until the reset job moved the row's `budget_reset_at` into the future again, and return it."""
    now: Final = datetime.now(timezone.utc)
    moved: Final = eventually(
        lambda: reset_at_of(gateway, due, identity),
        lambda value: value is not None and value > now,
        seconds=RESET_SECONDS,
    )
    assert moved is not None
    return moved


_WINDOWS: Final[Mapping[str, LiteralString]] = {
    "key": 'SELECT budget_limits::text AS windows FROM "LiteLLM_VerificationToken" WHERE token = %s',
    "team": 'SELECT budget_limits::text AS windows FROM "LiteLLM_TeamTable" WHERE team_id = %s',
    "user": 'SELECT budget_limits::text AS windows FROM "LiteLLM_UserTable" WHERE user_id = %s',
}

_EXPIRE_WINDOW: Final[Mapping[str, LiteralString]] = {
    entity: (
        f'UPDATE "{table}" SET budget_limits = ('
        "SELECT jsonb_agg(CASE WHEN w->>'budget_duration' = %s THEN jsonb_set(w, '{reset_at}', "
        "to_jsonb(to_char((now() - interval '1 second') AT TIME ZONE 'UTC', "
        '\'YYYY-MM-DD"T"HH24:MI:SS"+00:00"\'))) ELSE w END) '
        f"FROM jsonb_array_elements(budget_limits::jsonb) AS w) WHERE {column} = %s"
    )
    for entity, table, column in (
        ("key", "LiteLLM_VerificationToken", "token"),
        ("team", "LiteLLM_TeamTable", "team_id"),
        ("user", "LiteLLM_UserTable", "user_id"),
    )
}

_AGE_WINDOW_ROW: Final = (
    "UPDATE \"LiteLLM_BudgetWindowSpend\" SET window_start = window_start - interval '1 day' "
    "WHERE entity_type = %s AND entity_id = %s AND window_duration = %s"
)


def stored_windows(gateway: Gateway, entity: str, identity: str) -> list[dict[str, JsonValue]]:
    """The entity's `budget_limits` as stored, one mapping per window."""
    rows: Final = gateway.rows(_WINDOWS[entity], (identity,))
    raw: Final = rows[0]["windows"]
    assert isinstance(raw, str), rows
    return [object_value(window) for window in list_value(json.loads(raw))]


def window_reset_at(gateway: Gateway, entity: str, identity: str, duration: str) -> datetime:
    for window in stored_windows(gateway, entity, identity):
        if window.get("budget_duration") == duration:
            return _as_utc(window["reset_at"])
    raise AssertionError(f"no {duration} window on {entity} {identity}")


def expire_window(entity: str, identity: str, duration: str, *, age_spend_row: bool = True) -> None:
    """Make one `budget_limits` window due: its stored `reset_at` moves one second into the past.

    With `age_spend_row` the window's `LiteLLM_BudgetWindowSpend` row is also moved one day back, as it would be
    when the boundary had really passed, so the reset job's roll (`budget_window_spend_writer.py:467-488`, which
    only moves a row whose `window_start` is behind the new one) applies. `entity_id` of a key row is the hashed
    token, and `entity_type` is `Litellm_EntityType.value` (`key`, `team`, `user`)."""
    write_rows(_EXPIRE_WINDOW[entity], (duration, identity))
    if age_spend_row:
        write_rows(_AGE_WINDOW_ROW, (entity, identity, duration))


def window_spend_row(gateway: Gateway, entity: str, identity: str, duration: str) -> dict[str, JsonValue] | None:
    rows: Final = gateway.rows(
        "SELECT spend, window_start AT TIME ZONE 'UTC' AS window_start FROM \"LiteLLM_BudgetWindowSpend\" "
        "WHERE entity_type = %s AND entity_id = %s AND window_duration = %s",
        (entity, identity, duration),
    )
    return rows[0] if rows else None


def wait_for_window_reset(gateway: Gateway, entity: str, identity: str, duration: str) -> datetime:
    """Wait until the reset job wrote a future `reset_at` into the window again, and return it."""
    now: Final = datetime.now(timezone.utc)
    moved: Final = eventually(
        lambda: window_reset_at(gateway, entity, identity, duration),
        lambda value: value > now,
        seconds=RESET_SECONDS,
    )
    return moved


def committed(counter_key: str) -> str:
    """The charges-only twin of a counter (`committed_spend.py::committed_spend_key`): `spend:committed:` plus the
    counter key without its `spend:` prefix."""
    assert counter_key.startswith("spend:"), counter_key
    return f"spend:committed:{counter_key[len('spend:') :]}"
