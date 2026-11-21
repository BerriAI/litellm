"""Contract: resets of `budget_limits` windows on keys, teams and users, scheduled and manual.

A window is one entry of an entity's `budget_limits` JSON (`{"budget_duration", "max_budget", "reset_at"}`) with its
own Redis counter (`spend:key:<hash>:window:<duration>`, `spend:team:<id>:window:<duration>`,
`spend:user_window:<id>:<duration>`). The scheduled job (`ResetBudgetJob.reset_budget_windows`,
`litellm/proxy/common_utils/reset_budget_job.py:1983-2075`) pages through every row with `budget_limits` and, for
each window whose `reset_at` has passed (`_reset_expired_window`, `:1878-1917`):

    SETs the window counter to 0 (or the overage with `budget_rollover`), it does not delete it
    writes the next calendar boundary (`compute_budget_reset_at`) into that window's `reset_at`
    moves the window's `LiteLLM_BudgetWindowSpend` row onto the new window at spend 0 (`roll_window_spend_row`)

Only expired windows change; other windows of the same row and the entity's lifetime `spend` are untouched.

A manual `/key/{key}/reset_spend` also resets every window of the key (`_reset_key_budget_windows`,
`key_management_endpoints.py:7045-7085`), but differently: the new `reset_at` is "now plus the duration"
(`_advance_one_key_budget_window`, `:7017-7042`), not the calendar boundary, and each window counter is set to 0 with
a 60 second TTL and a floor marker (`set_spend_counter_with_floor_and_broadcast`, `proxy_server.py:4693-4719`).

Tests make a window due by moving its `reset_at` one second into the past and its window spend row one day back
(`dec1_budgets4_support.expire_window`). Price: one served call 0.02 (`hold_model`).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final

import pytest

from tests.integration._support.client import Gateway, eventually
from tests.integration.spend.dec1_budgets2_support import ttl_of, user_window_counter
from tests.integration.spend.dec1_budgets3_support import key_window_counter, team_window_counter
from tests.integration.spend.dec1_budgets4_support import (
    RESET_SECONDS,
    expire_window,
    next_utc_midnight,
    reset_proxy,
    today_utc_midnight,
    wait_for_window_reset,
    window_reset_at,
    window_spend_row,
)
from tests.integration.spend.dec1_budgets_support import (
    CHAT_COST,
    call,
    counter,
    counter_settles,
    hold_model,
    key_counter,
    redis_client,
    refused,
)
from tests.integration.spend.dec1_spend_support import (
    hashed,
    scripted_upstream,
    served,
    spend_of,
    spend_reaches,
)

pytestmark: Final = pytest.mark.shares_worker_state

DAY: Final = [{"budget_duration": "1d", "max_budget": 1.0}]


def _counter_is(cache_value: float | None, expected: float) -> bool:
    return cache_value is not None and abs(cache_value - expected) < 1e-9


@pytest.mark.timeout(300)
def test_a_due_key_window_counter_is_set_to_zero_not_deleted(gateway: Gateway, tmp_path: Path) -> None:
    """A key with one `1d` window of 1.0 is served once (`spend:key:<hash>:window:1d` 0.02). When the window is due,
    the next tick SETs the counter to exactly 0.0: the key still exists in Redis, unlike lifetime counters which the
    job deletes (`reset_budget_job.py:1897-1901` against `:865-882`). A port that deletes window counters sends the
    next request to the window reseed path, which re-sums the spend logs."""
    with (
        scripted_upstream() as wire,
        reset_proxy(gateway, tmp_path) as owned,
        owned.scenario() as scenario,
        redis_client() as cache,
    ):
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model], budget_limits=DAY)
        window: Final = key_window_counter(key, "1d")
        served(call(owned, model, key, max_tokens=5))
        counter_settles(cache, window, CHAT_COST)
        expire_window("key", hashed(key), "1d")
        wait_for_window_reset(owned, "key", hashed(key), "1d")
        assert (
            eventually(lambda: counter(cache, window), lambda value: _counter_is(value, 0.0), seconds=RESET_SECONDS)
            == 0.0
        )


@pytest.mark.timeout(300)
def test_a_window_reset_writes_the_next_utc_midnight_into_that_window(gateway: Gateway, tmp_path: Path) -> None:
    """After the reset of a due `1d` key window, that window's stored `reset_at` is the next 00:00 UTC
    (`compute_budget_reset_at`, `reset_budget_job.py:1902-1905`) and is written back into `budget_limits` with the
    rest of the window unchanged (`max_budget` still 1.0)."""
    with reset_proxy(gateway, tmp_path) as owned, owned.scenario() as scenario:
        key: Final = scenario.key(budget_limits=DAY)
        expire_window("key", hashed(key), "1d", age_spend_row=False)
        assert wait_for_window_reset(owned, "key", hashed(key), "1d") == next_utc_midnight()


@pytest.mark.timeout(300)
def test_a_window_reset_leaves_the_key_lifetime_spend_and_counter_alone(gateway: Gateway, tmp_path: Path) -> None:
    """A key with lifetime `max_budget` 1.0 and a `1d` window is served once. After the window reset, the window
    counter reads 0 while `spend:key:<hash>` and the stored key spend still read 0.02: windows reset independently of
    the lifetime budget (`reset_budget_windows` writes only `budget_limits`, `reset_budget_job.py:619-624`)."""
    with (
        scripted_upstream() as wire,
        reset_proxy(gateway, tmp_path) as owned,
        owned.scenario() as scenario,
        redis_client() as cache,
    ):
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=1.0, budget_limits=DAY)
        served(call(owned, model, key, max_tokens=5))
        spend_reaches(owned, "key", hashed(key), CHAT_COST)
        counter_settles(cache, key_counter(key), CHAT_COST)
        expire_window("key", hashed(key), "1d")
        wait_for_window_reset(owned, "key", hashed(key), "1d")
        eventually(
            lambda: counter(cache, key_window_counter(key, "1d")),
            lambda value: _counter_is(value, 0.0),
            seconds=RESET_SECONDS,
        )
        assert counter(cache, key_counter(key)) == pytest.approx(CHAT_COST)
        assert spend_of(owned, "key", hashed(key)) == pytest.approx(CHAT_COST)


@pytest.mark.timeout(300)
def test_only_the_expired_window_of_two_is_reset(gateway: Gateway, tmp_path: Path) -> None:
    """A key with a `1d` and a `7d` window, both 1.0, is served once (both window counters 0.02). Only the `1d`
    window is made due. After the tick the `1d` counter reads 0, the `7d` counter still reads 0.02 and the `7d`
    window's stored `reset_at` is unchanged (`reset_budget_job.py:2049-2061` loops windows one by one)."""
    with (
        scripted_upstream() as wire,
        reset_proxy(gateway, tmp_path) as owned,
        owned.scenario() as scenario,
        redis_client() as cache,
    ):
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(
            models=[model],
            budget_limits=[{"budget_duration": "1d", "max_budget": 1.0}, {"budget_duration": "7d", "max_budget": 1.0}],
        )
        served(call(owned, model, key, max_tokens=5))
        counter_settles(cache, key_window_counter(key, "7d"), CHAT_COST)
        weekly: Final = window_reset_at(owned, "key", hashed(key), "7d")
        expire_window("key", hashed(key), "1d")
        wait_for_window_reset(owned, "key", hashed(key), "1d")
        eventually(
            lambda: counter(cache, key_window_counter(key, "1d")),
            lambda value: _counter_is(value, 0.0),
            seconds=RESET_SECONDS,
        )
        assert counter(cache, key_window_counter(key, "7d")) == pytest.approx(CHAT_COST)
        assert window_reset_at(owned, "key", hashed(key), "7d") == weekly


@pytest.mark.timeout(300)
def test_a_window_reset_rolls_the_window_spend_row_onto_the_new_window_at_zero(
    gateway: Gateway, tmp_path: Path
) -> None:
    """After a served call, the key's `LiteLLM_BudgetWindowSpend` row for `1d` holds 0.02. The test ages that row by
    one day (as if the boundary had passed) and makes the window due. After the reset the row's spend is 0 and its
    `window_start` is today's 00:00 UTC, the new `reset_at` minus one day (`_roll_window_spend_row`,
    `reset_budget_job.py:1919-1961`). This row is what a lost window counter is reseeded from
    (`spend_counter_reseed.py::window_from_table`), so a reset that skipped it would bring the old spend back.
    UNVERIFIED: that the served call writes the row (the write is skipped when windows are read from the ledger)."""
    with (
        scripted_upstream() as wire,
        reset_proxy(gateway, tmp_path) as owned,
        owned.scenario() as scenario,
    ):
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model], budget_limits=DAY)
        served(call(owned, model, key, max_tokens=5))
        eventually(
            lambda: window_spend_row(owned, "key", hashed(key), "1d"),
            lambda row: row is not None and float(str(row["spend"])) >= CHAT_COST - 1e-9,
            seconds=RESET_SECONDS,
        )
        expire_window("key", hashed(key), "1d")
        wait_for_window_reset(owned, "key", hashed(key), "1d")
        rolled: Final = eventually(
            lambda: window_spend_row(owned, "key", hashed(key), "1d"),
            lambda row: row is not None and float(str(row["spend"])) == 0.0,
            seconds=RESET_SECONDS,
        )
        assert rolled is not None
        start: Final = datetime.fromisoformat(str(rolled["window_start"])).replace(tzinfo=timezone.utc)
        assert start == today_utc_midnight(), rolled


@pytest.mark.timeout(300)
def test_a_window_reset_readmits_a_key_refused_by_that_window(gateway: Gateway, tmp_path: Path) -> None:
    """A key whose only budget is a `1d` window of 0.02 is served once and then refused with HTTP 422 naming the
    key window. After the window reset (counter 0, spend row rolled to 0) the same key is served again, although the
    reset job never evicts the cached key object for a window reset (`_write_key_windows`,
    `reset_budget_job.py:619-624`, only writes the row): the window's floor check reads the rolled spend row
    (`proxy_server.py::get_current_spend`, windows always re-check)."""
    with (
        scripted_upstream() as wire,
        reset_proxy(gateway, tmp_path) as owned,
        owned.scenario() as scenario,
    ):
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model], budget_limits=[{"budget_duration": "1d", "max_budget": CHAT_COST}])
        served(call(owned, model, key, max_tokens=5))
        eventually(
            lambda: call(owned, model, key, max_tokens=5),
            lambda response: response.status_code == 422,
            seconds=RESET_SECONDS,
        )
        expire_window("key", hashed(key), "1d")
        wait_for_window_reset(owned, "key", hashed(key), "1d")
        readmitted: Final = eventually(
            lambda: call(owned, model, key, max_tokens=5),
            lambda response: response.status_code == 200,
            seconds=RESET_SECONDS,
        )
        assert readmitted.status_code == 200, readmitted.text


@pytest.mark.timeout(300)
def test_a_due_team_window_counter_is_set_to_zero(gateway: Gateway, tmp_path: Path) -> None:
    """A team with one `1d` window of 1.0 is charged 0.02 by its key (`spend:team:<id>:window:1d`). When the team
    window is due, the counter is SET to 0.0 and the team's lifetime spend stays 0.02 (`_WINDOW_SOURCES`,
    `reset_budget_job.py:670-698`, team entry)."""
    with (
        scripted_upstream() as wire,
        reset_proxy(gateway, tmp_path) as owned,
        owned.scenario() as scenario,
        redis_client() as cache,
    ):
        model: Final = hold_model(scenario, wire)
        team_id: Final = scenario.team(models=[model], budget_limits=DAY)
        key: Final = scenario.key(team_id=team_id, models=[model])
        served(call(owned, model, key, max_tokens=5))
        counter_settles(cache, team_window_counter(team_id, "1d"), CHAT_COST)
        spend_reaches(owned, "team", team_id, CHAT_COST)
        expire_window("team", team_id, "1d")
        wait_for_window_reset(owned, "team", team_id, "1d")
        eventually(
            lambda: counter(cache, team_window_counter(team_id, "1d")),
            lambda value: _counter_is(value, 0.0),
            seconds=RESET_SECONDS,
        )
        assert spend_of(owned, "team", team_id) == pytest.approx(CHAT_COST)


@pytest.mark.timeout(300)
def test_a_due_user_window_counter_is_set_to_zero(gateway: Gateway, tmp_path: Path) -> None:
    """An internal user with one `1d` window of 1.0 is charged 0.02 by a personal key
    (`spend:user_window:<user_id>:1d`, #51763). When the user window is due, that counter is SET to 0.0
    (`_WINDOW_SOURCES`, `reset_budget_job.py:670-678`, user entry, `user_window_counter`)."""
    with (
        scripted_upstream() as wire,
        reset_proxy(gateway, tmp_path) as owned,
        owned.scenario() as scenario,
        redis_client() as cache,
    ):
        model: Final = hold_model(scenario, wire)
        user_id: Final = scenario.user(user_role="internal_user", budget_limits=DAY)
        key: Final = scenario.key(user_id=user_id, models=[model])
        served(call(owned, model, key, max_tokens=5))
        counter_settles(cache, user_window_counter(user_id, "1d"), CHAT_COST)
        expire_window("user", user_id, "1d")
        wait_for_window_reset(owned, "user", user_id, "1d")
        assert eventually(
            lambda: counter(cache, user_window_counter(user_id, "1d")),
            lambda value: _counter_is(value, 0.0),
            seconds=RESET_SECONDS,
        ) == pytest.approx(0.0)


@pytest.mark.timeout(300)
def test_with_budget_rollover_a_window_carries_its_overage_into_the_new_window(
    gateway: Gateway, tmp_path: Path
) -> None:
    """With `litellm_settings.budget_rollover: true`, a key window of `1d` with `max_budget` 0.01 is served one call
    that costs 0.02 (hold shrunk to 0.01, settled at 0.02), so the window counter reads 0.02. After the window reset
    the counter reads 0.01: the overage above the window cap, read from the counter itself before it is overwritten
    (`_window_carried_spend`, `reset_budget_job.py:1963-1981`)."""
    with (
        scripted_upstream() as wire,
        reset_proxy(gateway, tmp_path, litellm_settings={"budget_rollover": True}) as owned,
        owned.scenario() as scenario,
        redis_client() as cache,
    ):
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model], budget_limits=[{"budget_duration": "1d", "max_budget": 0.01}])
        window: Final = key_window_counter(key, "1d")
        served(call(owned, model, key, max_tokens=5))
        counter_settles(cache, window, CHAT_COST)
        expire_window("key", hashed(key), "1d")
        wait_for_window_reset(owned, "key", hashed(key), "1d")
        assert eventually(
            lambda: counter(cache, window), lambda value: _counter_is(value, 0.01), seconds=RESET_SECONDS
        ) == pytest.approx(0.01)


def test_manual_reset_spend_zeroes_every_key_window_counter_with_a_sixty_second_ttl(gateway: Gateway) -> None:
    """A key with lifetime `max_budget` 1.0 and a `1d` window is served once. `POST /key/<key>/reset_spend` with
    `reset_to: 0` answers 200 and leaves `spend:key:<hash>:window:1d` at 0.0 with a TTL of at most 60 seconds
    (`set_spend_counter_with_floor_and_broadcast` writes `ttl=60`, `proxy_server.py:4708`), unlike the scheduled
    job, which SETs window counters without a TTL of its own. Round 3 pinned only the lifetime counter of this
    route."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=1.0, budget_limits=DAY)
        window: Final = key_window_counter(key, "1d")
        served(call(gateway, model, key, max_tokens=5))
        counter_settles(cache, window, CHAT_COST)
        spend_reaches(gateway, "key", hashed(key), CHAT_COST)
        response: Final = gateway.request("POST", f"/key/{key}/reset_spend", {"reset_to": 0})
        assert response.status_code == 200, response.text
        assert counter(cache, window) == pytest.approx(0.0)
        assert 0 < ttl_of(cache, window) <= 60


def test_manual_reset_spend_floats_the_window_reset_to_now_plus_the_duration(gateway: Gateway) -> None:
    """After `POST /key/<key>/reset_spend` on a key with a `1d` window, the window's stored `reset_at` is about 24
    hours from the call (within a minute), not the next 00:00 UTC the scheduled job would write: a manual reset
    starts a floating window at this moment so `window_start` excludes the spend that triggered it
    (`_advance_one_key_budget_window`, `key_management_endpoints.py:7017-7042`)."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=1.0, budget_limits=DAY)
        served(call(gateway, model, key, max_tokens=5))
        spend_reaches(gateway, "key", hashed(key), CHAT_COST)
        before: Final = datetime.now(timezone.utc)
        response: Final = gateway.request("POST", f"/key/{key}/reset_spend", {"reset_to": 0})
        assert response.status_code == 200, response.text
        floated: Final = window_reset_at(gateway, "key", hashed(key), "1d")
        assert before + timedelta(days=1) - timedelta(seconds=5) <= floated <= before + timedelta(days=1, minutes=1)


def test_manual_reset_spend_readmits_a_key_refused_by_its_window(gateway: Gateway) -> None:
    """A key whose only budget is a `1d` window of 0.02 and no lifetime budget is served once and refused. It has a
    lifetime spend of 0.02, so `reset_to: 0` is accepted; afterwards the key is served at once, because the window
    counter, its floor marker and the window start were all moved (`key_management_endpoints.py:7155-7166`).
    A port that resets only the lifetime counter keeps refusing the key until midnight."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model], budget_limits=[{"budget_duration": "1d", "max_budget": CHAT_COST}])
        served(call(gateway, model, key, max_tokens=5))
        spend_reaches(gateway, "key", hashed(key), CHAT_COST)
        refused(
            eventually(
                lambda: call(gateway, model, key, max_tokens=5),
                lambda response: response.status_code == 422,
                seconds=RESET_SECONDS,
            )
        )
        response: Final = gateway.request("POST", f"/key/{key}/reset_spend", {"reset_to": 0})
        assert response.status_code == 200, response.text
        served(call(gateway, model, key, max_tokens=5))
