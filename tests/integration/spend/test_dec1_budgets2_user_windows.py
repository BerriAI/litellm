"""Contract: internal user budget windows (`budget_limits` on a user, #51763) and `apply_user_budget_to_team_keys`.

A user's `budget_limits` windows get their own Redis counters `spend:user_window:<user_id>:<duration>`, a prefix of
their own since #51763 so that a user id ending in `:window:<duration>` cannot fill another user's window
(`types.py::user_window_counter`). Auth refuses at or over a window's limit with HTTP 422 and

    ExceededBudget: User=<user_id> over <duration> budget. Spend=$<spend, 4 decimals>, Limit=$<limit, 2 decimals>

(`callbacks-budgets/src/checks.rs::UserWindow`). The hold is added to the window counters too
(`budget_reservation.py::_get_budget_counters`).

Personal budgets (the user's `max_budget` and windows) apply to a key that has no team. A team key is limited by
its team, and its owner's personal budgets are skipped, unless `general_settings.apply_user_budget_to_team_keys` is
true; then both the auth check and the hold include the user's windows.

Price: input free, output 0.004 a token, one served call 0.02 (`hold_model`).
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import Gateway, Scenario, gateway_from_environment
from tests.integration.spend.dec1_budgets2_support import owned, user_window_counter
from tests.integration.spend.dec1_budgets_support import (
    CHAT_COST,
    background,
    call,
    counter,
    counter_settles,
    held_provider,
    hold_model,
    redis_client,
    refused,
)
from tests.integration.spend.dec1_spend_support import scripted_upstream, served

pytestmark: Final = pytest.mark.shares_worker_state


@pytest.fixture(scope="module")
def applying_proxy(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    """An owned proxy with `general_settings.apply_user_budget_to_team_keys: true`."""
    with (
        gateway_from_environment() as shared,
        owned(
            shared,
            tmp_path_factory.mktemp("dec1-budgets2-user-windows"),
            general_settings={"apply_user_budget_to_team_keys": True},
        ) as proxy,
    ):
        yield proxy


def _windowed_user(scenario: Scenario, limit: float, duration: str = "1d") -> str:
    windows: Final[list[JsonValue]] = [{"budget_duration": duration, "max_budget": limit}]
    return scenario.user(user_role="internal_user", budget_limits=windows)


def _team_key_of(scenario: Scenario, model: str, user_id: str) -> str:
    team_id: Final = scenario.team(models=[model])
    scenario.gateway.post("/team/member_add", {"team_id": team_id, "member": {"role": "user", "user_id": user_id}})
    return scenario.key(team_id=team_id, user_id=user_id, models=[model])


def _window_refusal(user_id: str, spend: str, limit: str, duration: str = "1d") -> str:
    return f"ExceededBudget: User={user_id} over {duration} budget. Spend=${spend}, Limit=${limit}"


@pytest.mark.timeout(240)
def test_with_the_flag_on_a_team_key_is_refused_by_its_owners_spent_window(applying_proxy: Gateway) -> None:
    """On a proxy with `apply_user_budget_to_team_keys: true`, a user with a `1d` window of 0.02 owns a key of a team
    without a budget. The first call (0.02) is served; the second is refused with exactly `ExceededBudget: User=<id>
    over 1d budget. Spend=$0.0200, Limit=$0.02`."""
    proxy: Final = applying_proxy
    with scripted_upstream() as wire, proxy.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        user_id: Final = _windowed_user(scenario, CHAT_COST)
        key: Final = _team_key_of(scenario, model, user_id)
        served(call(proxy, model, key, max_tokens=5))
        counter_settles(cache, user_window_counter(user_id, "1d"), CHAT_COST)
        assert refused(call(proxy, model, key, max_tokens=5)) == _window_refusal(user_id, "0.0200", "0.02")


@pytest.mark.timeout(240)
def test_with_the_flag_off_a_team_key_ignores_its_owners_spent_window(gateway: Gateway) -> None:
    """On the suite proxy (flag off), the same setup: a user with a `1d` window of 0.02 owns a team key. Three calls
    (0.06) are all served, because a team key is limited by its team only. A port that applies personal windows to
    every key would lock team keys of users who spent their personal budget."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = hold_model(scenario, wire)
        user_id: Final = _windowed_user(scenario, CHAT_COST)
        key: Final = _team_key_of(scenario, model, user_id)
        statuses: Final = [call(gateway, model, key, max_tokens=5).status_code for _ in range(3)]
        assert statuses == [200, 200, 200]


@pytest.mark.timeout(240)
def test_with_the_flag_off_a_personal_key_is_still_refused_by_the_users_window(gateway: Gateway) -> None:
    """On the suite proxy, a user with a `1d` window of 0.02 owns a key with no team. The first call is served and
    the second is refused with `ExceededBudget: User=<id> over 1d budget. Spend=$0.0200, Limit=$0.02`: the flag
    only decides team keys."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        user_id: Final = _windowed_user(scenario, CHAT_COST)
        key: Final = scenario.key(user_id=user_id, models=[model])
        served(call(gateway, model, key, max_tokens=5))
        counter_settles(cache, user_window_counter(user_id, "1d"), CHAT_COST)
        assert refused(call(gateway, model, key, max_tokens=5)) == _window_refusal(user_id, "0.0200", "0.02")


@pytest.mark.timeout(240)
def test_with_the_flag_on_a_team_key_hold_lands_on_the_users_window_counter(applying_proxy: Gateway) -> None:
    """With the flag on, a team key whose owner has a `1d` window of 1.0 sends a held call with `max_tokens: 10`.
    While it is in flight `spend:user_window:<user_id>:1d` reads exactly 0.04; it settles at 0.02 after release."""
    proxy: Final = applying_proxy
    with held_provider() as provider, proxy.scenario() as scenario, redis_client() as cache, background() as run:
        model: Final = hold_model(scenario, provider)
        user_id: Final = _windowed_user(scenario, 1.0)
        key: Final = _team_key_of(scenario, model, user_id)
        in_flight: Final = run(lambda: call(proxy, model, key, hold=True, max_tokens=10))
        provider.wait_for(1)
        assert counter(cache, user_window_counter(user_id, "1d")) == pytest.approx(0.04)
        provider.release()
        served(in_flight.result(timeout=60))
        assert counter_settles(cache, user_window_counter(user_id, "1d"), CHAT_COST) == pytest.approx(CHAT_COST)


@pytest.mark.timeout(240)
def test_with_the_flag_on_a_concurrent_team_key_call_is_refused_while_a_hold_fills_the_window(
    applying_proxy: Gateway,
) -> None:
    """With the flag on, the owner's `1d` window is 1.0 and a held team key call without `max_tokens` shrinks its
    hold to fill it. A second call from another team key of the same user is refused with `ExceededBudget:
    User=<id> over 1d budget. Spend=$1.0000, Limit=$1.00`: the window is shared by all of the user's keys."""
    proxy: Final = applying_proxy
    with held_provider() as provider, proxy.scenario() as scenario, background() as run:
        model: Final = hold_model(scenario, provider)
        user_id: Final = _windowed_user(scenario, 1.0)
        first_key: Final = _team_key_of(scenario, model, user_id)
        second_key: Final = _team_key_of(scenario, model, user_id)
        in_flight: Final = run(lambda: call(proxy, model, first_key, hold=True))
        provider.wait_for(1)
        try:
            second: Final = call(proxy, model, second_key, max_tokens=5)
        finally:
            provider.release()
        assert refused(second) == _window_refusal(user_id, "1.0000", "1.00")
        served(in_flight.result(timeout=60))


@pytest.mark.timeout(240)
def test_a_user_window_counter_uses_its_own_prefix(gateway: Gateway) -> None:
    """A user with a `1d` window of 1.0 makes one call with a personal key. The window counter is
    `spend:user_window:<user_id>:1d` (0.02) and nothing is written to the pre-#51763 name
    `spend:user:<user_id>:window:1d`, which a user named `<other>:window:1d` could have collided with."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        user_id: Final = _windowed_user(scenario, 1.0)
        key: Final = scenario.key(user_id=user_id, models=[model])
        served(call(gateway, model, key, max_tokens=5))
        counter_settles(cache, user_window_counter(user_id, "1d"), CHAT_COST)
        assert counter(cache, f"spend:user:{user_id}:window:1d") is None


@pytest.mark.timeout(240)
def test_a_user_whose_id_looks_like_another_users_window_has_a_separate_balance(gateway: Gateway) -> None:
    """Two users: `<base>` with a `1d` window of 1.0, and `<base>:window:1d` with a `1d` window of 0.02. The second
    spends its window and is refused; the first is still served. Since #51763 the two never share a counter."""
    base: Final = f"dec1-b2-user-{uuid.uuid4().hex[:10]}"
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        scenario.user(
            user_id=base, user_role="internal_user", budget_limits=[{"budget_duration": "1d", "max_budget": 1.0}]
        )
        tricky: Final = f"{base}:window:1d"
        scenario.user(
            user_id=tricky,
            user_role="internal_user",
            budget_limits=[{"budget_duration": "1d", "max_budget": CHAT_COST}],
        )
        base_key: Final = scenario.key(user_id=base, models=[model])
        tricky_key: Final = scenario.key(user_id=tricky, models=[model])
        served(call(gateway, model, tricky_key, max_tokens=5))
        counter_settles(cache, user_window_counter(tricky, "1d"), CHAT_COST)
        refusal: Final = refused(call(gateway, model, tricky_key, max_tokens=5))
        assert refusal == _window_refusal(tricky, "0.0200", "0.02")
        assert served(call(gateway, model, base_key, max_tokens=5))["usage"] is not None
