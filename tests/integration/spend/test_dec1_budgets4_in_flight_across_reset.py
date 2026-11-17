"""Contract: what a call that is in flight while its budget resets is charged to.

The hold of an admitted call lives in the Redis counter of every budgeted level. When the reset job runs during the
call, it deletes lifetime counters (`reset_budget_job.py:865-882`) or SETs window counters to 0 (`:1897-1901`). At
settlement the Rust engine (`litellm/rust/crates/callbacks-budgets/src/engine.rs:482-560`) moves each entry from
its hold to the real cost, but only on counters that still exist and would not go negative (`applies_to`,
`engine.rs:144-149`). Every other entry is handed back for a reseed (`reseed_reserved_counter`,
`litellm/proxy/spend_tracking/budget_reservation.py:1028-1033`): the counter is raised to the committed table
spend (`reseed_spend_counter_from_db`, `proxy_server.py:3611-3630`) and, only when a table row was found, the real
cost is added on top.

What this means for each kind of counter:

    key, team, user, team member, organization, project lifetime counters  the call is charged to the new period
    key, team and user `budget_limits` window counters                       no table row, so nothing is added:
                                                                             the new window does not see the call
    customer hierarchy window counters (`window_reset_at` on the entry)     settled at 0 (`engine.rs:451-454`)

Tests keep one call in flight at the provider (`HeldProvider`), force the reset with SQL on an owned proxy whose
job ticks every 2 to 3 seconds, wait until the counter was deleted or zeroed, then release the call. Price: hold is
`max_tokens * 0.004`, real cost 0.02 (`hold_model`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest

from tests.integration._support.client import Gateway, eventually
from tests.integration.spend.dec1_budgets3_support import key_window_counter, team_counter, user_counter
from tests.integration.spend.dec1_budgets4_support import (
    RESET_SECONDS,
    Due,
    expire_window,
    make_due,
    reset_proxy,
    wait_for_row_reset,
    wait_for_window_reset,
)
from tests.integration.spend.dec1_budgets_support import (
    CHAT_COST,
    background,
    call,
    counter,
    counter_settles,
    held_provider,
    hold_model,
    key_counter,
    redis_client,
)
from tests.integration.spend.dec1_spend_support import hashed, served, spend_of, spend_reaches

pytestmark: Final = pytest.mark.shares_worker_state

HOLD: Final = 0.04


@pytest.mark.timeout(300)
def test_a_key_call_in_flight_across_its_reset_is_charged_to_the_new_period(gateway: Gateway, tmp_path: Path) -> None:
    """A `1d` key with `max_budget` 1.0 holds 0.04 for a call in flight. The key is reset while the call waits
    (stored spend 0, `spend:key:<hash>` deleted). When the call finishes, the counter is recreated from the reset
    row (0) plus the real cost and reads exactly 0.02, and the key's stored spend reaches 0.02: the call counts in
    the new period, not the closed one (`engine.rs:544-549` reseed, `budget_reservation.py:1028-1033`)."""
    with (
        held_provider() as provider,
        reset_proxy(gateway, tmp_path) as owned,
        owned.scenario() as scenario,
        redis_client() as cache,
        background() as run,
    ):
        model: Final = hold_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=1.0, budget_duration="1d")
        in_flight: Final = run(lambda: call(owned, model, key, hold=True, max_tokens=10))
        provider.wait_for(1)
        assert counter(cache, key_counter(key)) == pytest.approx(HOLD)
        make_due(Due.KEY, hashed(key))
        wait_for_row_reset(owned, Due.KEY, hashed(key))
        eventually(lambda: cache.exists(key_counter(key)), lambda found: found == 0, seconds=RESET_SECONDS)
        provider.release()
        served(in_flight.result(timeout=60))
        assert counter_settles(cache, key_counter(key), CHAT_COST) == pytest.approx(CHAT_COST)
        assert spend_reaches(owned, "key", hashed(key), CHAT_COST) == pytest.approx(CHAT_COST)


@pytest.mark.timeout(300)
def test_a_team_call_in_flight_across_the_team_reset_is_charged_to_the_new_period(
    gateway: Gateway, tmp_path: Path
) -> None:
    """The same for a `1d` team with `max_budget` 1.0: a call held at 0.04 on `spend:team:<id>` finishes after the
    team reset deleted that counter, and the counter then reads exactly 0.02 (reseeded from the team row at 0, plus
    the real cost) while the team's stored spend reaches 0.02."""
    with (
        held_provider() as provider,
        reset_proxy(gateway, tmp_path) as owned,
        owned.scenario() as scenario,
        redis_client() as cache,
        background() as run,
    ):
        model: Final = hold_model(scenario, provider)
        team_id: Final = scenario.team(models=[model], max_budget=1.0, budget_duration="1d")
        key: Final = scenario.key(team_id=team_id, models=[model])
        in_flight: Final = run(lambda: call(owned, model, key, hold=True, max_tokens=10))
        provider.wait_for(1)
        make_due(Due.TEAM, team_id)
        wait_for_row_reset(owned, Due.TEAM, team_id)
        eventually(lambda: cache.exists(team_counter(team_id)), lambda found: found == 0, seconds=RESET_SECONDS)
        provider.release()
        served(in_flight.result(timeout=60))
        assert counter_settles(cache, team_counter(team_id), CHAT_COST) == pytest.approx(CHAT_COST)
        assert spend_reaches(owned, "team", team_id, CHAT_COST) == pytest.approx(CHAT_COST)


@pytest.mark.timeout(300)
def test_a_user_call_in_flight_across_the_user_reset_is_charged_to_the_new_period(
    gateway: Gateway, tmp_path: Path
) -> None:
    """The same for a `1d` internal user with `max_budget` 1.0 calling through a personal key: after the reset and
    the release, `spend:user:<user_id>` reads exactly 0.02 (the user row is a reseed source,
    `spend_counter_reseed.py:225-228`)."""
    with (
        held_provider() as provider,
        reset_proxy(gateway, tmp_path) as owned,
        owned.scenario() as scenario,
        redis_client() as cache,
        background() as run,
    ):
        model: Final = hold_model(scenario, provider)
        user_id: Final = scenario.user(user_role="internal_user", max_budget=1.0, budget_duration="1d")
        key: Final = scenario.key(user_id=user_id, models=[model])
        in_flight: Final = run(lambda: call(owned, model, key, hold=True, max_tokens=10))
        provider.wait_for(1)
        make_due(Due.USER, user_id)
        wait_for_row_reset(owned, Due.USER, user_id)
        eventually(lambda: cache.exists(user_counter(user_id)), lambda found: found == 0, seconds=RESET_SECONDS)
        provider.release()
        served(in_flight.result(timeout=60))
        assert counter_settles(cache, user_counter(user_id), CHAT_COST) == pytest.approx(CHAT_COST)


@pytest.mark.timeout(300)
def test_a_call_that_fails_after_a_reset_leaves_the_counter_at_the_reset_value_not_negative(
    gateway: Gateway, tmp_path: Path
) -> None:
    """A `1d` key with `max_budget` 1.0 holds 0.04 for a call whose provider will answer 500. The key is reset while
    the call waits, then the call fails. The release would take 0.04 off a counter that no longer holds it, so it is
    not applied (`applies_to`, `engine.rs:144-149`) and the counter is at most reseeded from the reset row: it reads
    0 or is absent, never -0.04, and the stored spend stays 0. A negative counter would hand the key 0.04 of extra
    budget in the new period."""
    with (
        held_provider(failing_status=500) as provider,
        reset_proxy(gateway, tmp_path) as owned,
        owned.scenario() as scenario,
        redis_client() as cache,
        background() as run,
    ):
        model: Final = hold_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=1.0, budget_duration="1d")
        in_flight: Final = run(lambda: call(owned, model, key, hold=True, max_tokens=10))
        provider.wait_for(1)
        make_due(Due.KEY, hashed(key))
        wait_for_row_reset(owned, Due.KEY, hashed(key))
        eventually(lambda: cache.exists(key_counter(key)), lambda found: found == 0, seconds=RESET_SECONDS)
        provider.release()
        assert in_flight.result(timeout=60).status_code >= 500
        settled: Final = counter(cache, key_counter(key))
        assert settled is None or settled == pytest.approx(0.0), settled
        assert spend_of(owned, "key", hashed(key)) == pytest.approx(0.0)


@pytest.mark.timeout(300)
def test_a_key_window_call_in_flight_across_the_window_reset_is_not_charged_to_the_new_window(
    gateway: Gateway, tmp_path: Path
) -> None:
    """A key whose only budget is a `1d` window of 1.0 holds 0.04 on `spend:key:<hash>:window:1d` for a call in
    flight. The window resets while the call waits (counter SET to 0). Settling would move the counter by
    0.02 - 0.04 = -0.02 below zero, so the entry is handed to the reseed, and window counters have no table row
    (`_is_key_or_team_window_counter`, `spend_counter_reseed.py:178-183`, `:245-258`), so nothing is added: the
    window counter still reads 0 after the call. The call is billed (lifetime spend 0.02) but the new window never
    sees it, unlike the lifetime counters in the tests above."""
    with (
        held_provider() as provider,
        reset_proxy(gateway, tmp_path) as owned,
        owned.scenario() as scenario,
        redis_client() as cache,
        background() as run,
    ):
        model: Final = hold_model(scenario, provider)
        key: Final = scenario.key(models=[model], budget_limits=[{"budget_duration": "1d", "max_budget": 1.0}])
        window: Final = key_window_counter(key, "1d")
        in_flight: Final = run(lambda: call(owned, model, key, hold=True, max_tokens=10))
        provider.wait_for(1)
        assert counter(cache, window) == pytest.approx(HOLD)
        expire_window("key", hashed(key), "1d", age_spend_row=False)
        wait_for_window_reset(owned, "key", hashed(key), "1d")
        eventually(lambda: counter(cache, window), lambda value: value == pytest.approx(0.0), seconds=RESET_SECONDS)
        provider.release()
        served(in_flight.result(timeout=60))
        spend_reaches(owned, "key", hashed(key), CHAT_COST)
        assert counter(cache, window) == pytest.approx(0.0)


@pytest.mark.timeout(300)
def test_a_window_call_whose_hold_equals_its_cost_leaves_the_reset_window_at_zero(
    gateway: Gateway, tmp_path: Path
) -> None:
    """The same with `max_tokens: 5`, so the hold (0.02) equals the real cost and the settlement has nothing to
    move (`adjustment == 0.0`, `engine.rs:465-467`). The reset window counter still reads 0 after the call: whether
    the hold was over or exactly at the cost, a call in flight across a window boundary stays in the closed
    window."""
    with (
        held_provider() as provider,
        reset_proxy(gateway, tmp_path) as owned,
        owned.scenario() as scenario,
        redis_client() as cache,
        background() as run,
    ):
        model: Final = hold_model(scenario, provider)
        key: Final = scenario.key(models=[model], budget_limits=[{"budget_duration": "1d", "max_budget": 1.0}])
        window: Final = key_window_counter(key, "1d")
        in_flight: Final = run(lambda: call(owned, model, key, hold=True, max_tokens=5))
        provider.wait_for(1)
        expire_window("key", hashed(key), "1d", age_spend_row=False)
        wait_for_window_reset(owned, "key", hashed(key), "1d")
        eventually(lambda: counter(cache, window), lambda value: value == pytest.approx(0.0), seconds=RESET_SECONDS)
        provider.release()
        served(in_flight.result(timeout=60))
        spend_reaches(owned, "key", hashed(key), CHAT_COST)
        assert counter(cache, window) == pytest.approx(0.0)


@pytest.mark.timeout(300)
def test_a_lifetime_and_window_key_split_the_in_flight_call_between_periods(gateway: Gateway, tmp_path: Path) -> None:
    """A `1d` key with lifetime `max_budget` 1.0 and a `1d` window of 1.0 holds 0.04 on both counters. Both the key
    row and the window are made due while the call waits. After the call, `spend:key:<hash>` reads 0.02 (new period)
    while `spend:key:<hash>:window:1d` reads 0 (closed period). Pinned so a port that unifies the two settle paths
    does it on purpose; the asymmetry is listed under suspected bugs in the index."""
    with (
        held_provider() as provider,
        reset_proxy(gateway, tmp_path) as owned,
        owned.scenario() as scenario,
        redis_client() as cache,
        background() as run,
    ):
        model: Final = hold_model(scenario, provider)
        key: Final = scenario.key(
            models=[model],
            max_budget=1.0,
            budget_duration="1d",
            budget_limits=[{"budget_duration": "1d", "max_budget": 1.0}],
        )
        window: Final = key_window_counter(key, "1d")
        in_flight: Final = run(lambda: call(owned, model, key, hold=True, max_tokens=10))
        provider.wait_for(1)
        make_due(Due.KEY, hashed(key))
        expire_window("key", hashed(key), "1d", age_spend_row=False)
        wait_for_row_reset(owned, Due.KEY, hashed(key))
        wait_for_window_reset(owned, "key", hashed(key), "1d")
        eventually(lambda: counter(cache, window), lambda value: value == pytest.approx(0.0), seconds=RESET_SECONDS)
        provider.release()
        served(in_flight.result(timeout=60))
        assert counter_settles(cache, key_counter(key), CHAT_COST) == pytest.approx(CHAT_COST)
        assert counter(cache, window) == pytest.approx(0.0)
