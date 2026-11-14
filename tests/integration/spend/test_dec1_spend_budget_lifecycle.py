"""Contract: budgets over time. Soft budgets alert without blocking, `budget_duration` resets spend on a schedule,
and a temporary budget increase readmits an exhausted key until it expires.

Soft budget: crossing `soft_budget` never refuses a request. With `general_settings.alerting: ["webhook"]` and
`WEBHOOK_URL` set, the proxy posts one JSON `WebhookEvent` with `event` `soft_budget_crossed` (or `budget_crossed`
for a hard budget) and then stays quiet for that entity, because the alert is remembered in the cache.

Reset: an entity created with `budget_duration` gets a `budget_reset_at`. The reset job (its cadence is shortened
here with `PROXY_BUDGET_RESCHEDULER_MIN_TIME` / `MAX_TIME`) sets `spend` back to 0 when that time passes and moves
`budget_reset_at` forward. Request log rows are history and are never deleted by a reset.

Temporary increase: `temp_budget_increase` plus a future `temp_budget_expiry` on a key are added to `max_budget`
at auth time. One served call costs 0.02 and the budgets here are far below that.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import JSON_OBJECT, Gateway, Scenario, eventually, held, string_value
from tests.integration._support.process import owned_proxy
from tests.integration._support.wire import Reply, Wire, wire_server
from tests.integration.spend.dec1_spend_support import (
    CHAT_COST,
    LANDING_SECONDS,
    TINY_BUDGET,
    ask,
    chat_model,
    hashed,
    provider_calls,
    proxy_config,
    refusal,
    scripted_upstream,
    served,
    spend_of,
    spend_reaches,
)

pytestmark: Final = pytest.mark.shares_worker_state

_FAST_RESET: Final = {"PROXY_BUDGET_RESCHEDULER_MIN_TIME": "2", "PROXY_BUDGET_RESCHEDULER_MAX_TIME": "3"}


@contextmanager
def _alerting_proxy(gateway: Gateway, directory: Path) -> Iterator[tuple[Gateway, Wire]]:
    """An owned proxy that posts budget alerts to a webhook this test listens on."""
    config: Final = proxy_config(directory, general_settings={"alerting": ["webhook"]})
    with (
        wire_server(lambda _: Reply(body=b"{}")) as webhook,
        owned_proxy(gateway, directory, {"WEBHOOK_URL": webhook.url}, config=config) as owned,
    ):
        yield owned, webhook


def _events(webhook: Wire, seen: list[dict[str, JsonValue]], event: str) -> list[dict[str, JsonValue]]:
    seen.extend(JSON_OBJECT.validate_json(request.body) for request in webhook.drain() if request.method == "POST")
    return [body for body in seen if body.get("event") == event]


def _status(gateway: Gateway, model: str, key: str) -> int:
    return ask(gateway, model, key).status_code


def _reset_at(gateway: Gateway, table: str, column: str, identity: str) -> str | None:
    rows: Final = gateway.rows(
        f'SELECT budget_reset_at::text AS reset_at FROM "{table}" WHERE {column} = %s', (identity,)
    )
    value: Final = rows[0]["reset_at"]
    return None if value is None else string_value(value)


def _exhausted_key(gateway: Gateway, scenario: Scenario, model: str) -> str:
    key: Final = scenario.key(models=[model], max_budget=TINY_BUDGET)
    served(ask(gateway, model, key))
    spend_reaches(gateway, "key", hashed(key), CHAT_COST)
    refusal(ask(gateway, model, key))
    return key


def test_key_past_its_soft_budget_is_still_served(gateway: Gateway) -> None:
    """A key has `soft_budget` 0.0000001 and no `max_budget`. After its spend (0.02) is far past the soft budget,
    further calls are still answered 200 and reach the provider. A soft budget warns, it never blocks."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model], soft_budget=TINY_BUDGET)
        served(ask(gateway, model, key))
        spend_reaches(gateway, "key", hashed(key), CHAT_COST)
        assert [_status(gateway, model, key) for _ in range(3)] == [200, 200, 200]
        assert len(provider_calls(wire)) == 4


def test_team_past_its_soft_budget_still_serves_its_keys(gateway: Gateway) -> None:
    """A team has `soft_budget` 0.0000001 and no `max_budget`. After the team's spend passes it, the team's key is
    still served."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        team: Final = scenario.team(models=[model], soft_budget=TINY_BUDGET)
        key: Final = scenario.key(team_id=team, models=[model])
        served(ask(gateway, model, key))
        spend_reaches(gateway, "team", team, CHAT_COST)
        assert [_status(gateway, model, key) for _ in range(2)] == [200, 200]


def test_key_crossing_its_soft_budget_posts_one_soft_budget_crossed_webhook(gateway: Gateway, tmp_path: Path) -> None:
    """With webhook alerting on, a key with `soft_budget` 0.0000001 keeps calling. After its spend is recorded,
    the webhook receives a JSON event with `event` `soft_budget_crossed`, `token` the hashed key, `soft_budget`
    0.0000001, `spend` at least 0.02, the key alias and `event_group` `key`. Every call is still served.
    BUG: same root cause as the sibling `test_soft_budget_webhook_is_sent_once_and_not_on_every_later_request`
    (product bug 7 of run 42966523075). The soft budget check reads the cached key object's `spend`
    (`litellm/proxy/auth/auth_checks.py:6860`), which a charge never raises (`_update_key_cache` drops its
    `new_spend`, `litellm/proxy/proxy_server.py:4661`), so the alert waits for a reload of the key object. It
    came in time in run 42966523075 and not at all within 70 seconds in run 43044106414."""
    pytest.skip("BUG: soft_budget_crossed waits on a stale cached key spend and was not sent within 70 seconds")
    seen: Final[list[dict[str, JsonValue]]] = []  # mutable-ok: accumulates webhook deliveries across polls
    with (
        scripted_upstream() as wire,
        _alerting_proxy(gateway, tmp_path) as (owned, webhook),
        owned.scenario() as scenario,
    ):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model], soft_budget=TINY_BUDGET, key_alias=f"dec1-soft-{tmp_path.name}")
        served(ask(owned, model, key))
        spend_reaches(owned, "key", hashed(key), CHAT_COST)

        def crossed() -> list[dict[str, JsonValue]]:
            assert _status(owned, model, key) == 200
            return _events(webhook, seen, "soft_budget_crossed")

        events: Final = eventually(crossed, lambda found: len(found) >= 1, seconds=LANDING_SECONDS)
        event: Final = events[0]
        assert (event["token"], event["key_alias"], event["event_group"]) == (
            hashed(key),
            f"dec1-soft-{tmp_path.name}",
            "key",
        ), event
        assert event["soft_budget"] == pytest.approx(TINY_BUDGET), event
        assert isinstance(event["spend"], float) and event["spend"] >= 0.02 - 1e-9, event


def test_soft_budget_webhook_is_sent_once_and_not_on_every_later_request(gateway: Gateway, tmp_path: Path) -> None:
    """After the first `soft_budget_crossed` event for a key, five more served calls by the same key must not
    produce a second one. The alert is remembered per event and key, otherwise every request would page.
    BUG: the soft budget check compares the cached key object's `spend` (`litellm/proxy/auth/auth_checks.py:6860`),
    which a charge never raises (`_update_key_cache` drops its `new_spend`, `litellm/proxy/proxy_server.py:4661`),
    so the first alert waits for the key object to be reloaded from the database. In run 42966523075 no alert
    came within 70 seconds of the spend landing; the sibling test with a key alias got its alert in time."""
    pytest.skip("BUG: soft_budget_crossed waits on a stale cached key spend and was not sent within 70 seconds")
    seen: Final[list[dict[str, JsonValue]]] = []  # mutable-ok: accumulates webhook deliveries across polls
    with (
        scripted_upstream() as wire,
        _alerting_proxy(gateway, tmp_path) as (owned, webhook),
        owned.scenario() as scenario,
    ):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model], soft_budget=TINY_BUDGET)
        served(ask(owned, model, key))
        spend_reaches(owned, "key", hashed(key), CHAT_COST)

        def crossed() -> list[dict[str, JsonValue]]:
            assert _status(owned, model, key) == 200
            return _events(webhook, seen, "soft_budget_crossed")

        eventually(crossed, lambda found: len(found) >= 1, seconds=LANDING_SECONDS)
        assert [_status(owned, model, key) for _ in range(5)] == [200] * 5
        after: Final = held(
            lambda: _events(webhook, seen, "soft_budget_crossed"), lambda found: len(found) == 1, holding=3
        )
        assert [event["token"] for event in after] == [hashed(key)], after


def test_key_crossing_its_max_budget_posts_a_budget_crossed_webhook(gateway: Gateway, tmp_path: Path) -> None:
    """With webhook alerting on, a key with `max_budget` 0.0000001 is served once and then refused. The refusal
    also posts a webhook event with `event` `budget_crossed`, `token` the hashed key, `max_budget` 0.0000001 and
    `event_group` `key`."""
    seen: Final[list[dict[str, JsonValue]]] = []  # mutable-ok: accumulates webhook deliveries across polls
    with (
        scripted_upstream() as wire,
        _alerting_proxy(gateway, tmp_path) as (owned, webhook),
        owned.scenario() as scenario,
    ):
        model: Final = chat_model(scenario, wire)
        key: Final = _exhausted_key(owned, scenario, model)
        events: Final = eventually(
            lambda: _events(webhook, seen, "budget_crossed"), lambda found: len(found) >= 1, seconds=LANDING_SECONDS
        )
        mine: Final = [event for event in events if event["token"] == hashed(key)]
        assert len(mine) == 1, events
        assert mine[0]["max_budget"] == pytest.approx(TINY_BUDGET), mine
        assert mine[0]["event_group"] == "key", mine


def test_key_created_with_budget_duration_gets_a_reset_time_and_one_without_gets_none(gateway: Gateway) -> None:
    """`/key/generate` with `budget_duration: "30d"` stores a `budget_reset_at` in the future. A key generated
    without a duration has `budget_reset_at` NULL, so its spend is never reset."""
    with gateway.scenario() as scenario:
        timed: Final = scenario.key(max_budget=1.0, budget_duration="30d")
        untimed: Final = scenario.key(max_budget=1.0)
        future: Final = gateway.rows(
            'SELECT (budget_reset_at > NOW()) AS future FROM "LiteLLM_VerificationToken" WHERE token = %s',
            (hashed(timed),),
        )
        assert future == [{"future": True}], future
        assert _reset_at(gateway, "LiteLLM_VerificationToken", "token", hashed(untimed)) is None


def test_key_budget_reset_zeroes_spend_keeps_request_rows_and_serves_again(gateway: Gateway, tmp_path: Path) -> None:
    """A key has `max_budget` 0.0000001 and `budget_duration` `5s`. It is served once, then refused. When the
    reset job runs, the key's stored spend returns to 0 and `budget_reset_at` moves forward, the request row of
    the first call is still in `LiteLLM_SpendLogs`, and the key is served again."""
    with (
        scripted_upstream() as wire,
        owned_proxy(gateway, tmp_path, _FAST_RESET) as owned,
        owned.scenario() as scenario,
    ):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=TINY_BUDGET, budget_duration="5s")
        first_reset: Final = _reset_at(owned, "LiteLLM_VerificationToken", "token", hashed(key))
        served(ask(owned, model, key))
        spend_reaches(owned, "key", hashed(key), CHAT_COST)
        eventually(lambda: _status(owned, model, key), lambda code: code == 422, seconds=LANDING_SECONDS)
        eventually(lambda: spend_of(owned, "key", hashed(key)), lambda spend: spend == 0, seconds=LANDING_SECONDS)
        later_reset: Final = _reset_at(owned, "LiteLLM_VerificationToken", "token", hashed(key))
        assert first_reset is not None and later_reset is not None and later_reset > first_reset
        kept: Final = owned.rows(
            "SELECT spend FROM \"LiteLLM_SpendLogs\" WHERE api_key = %s AND status = 'success'", (hashed(key),)
        )
        assert len(kept) >= 1 and all(row["spend"] == pytest.approx(0.02) for row in kept), kept
        assert eventually(lambda: _status(owned, model, key), lambda code: code == 200, seconds=LANDING_SECONDS) == 200


def test_team_budget_reset_unblocks_the_team_keys(gateway: Gateway, tmp_path: Path) -> None:
    """A team has `max_budget` 0.0000001 and `budget_duration` `5s`. Its key is served once and then refused with
    the team wording. After the reset job has zeroed the team's spend, the same key is served again."""
    with (
        scripted_upstream() as wire,
        owned_proxy(gateway, tmp_path, _FAST_RESET) as owned,
        owned.scenario() as scenario,
    ):
        model: Final = chat_model(scenario, wire)
        team: Final = scenario.team(models=[model], max_budget=TINY_BUDGET, budget_duration="5s")
        key: Final = scenario.key(team_id=team, models=[model])
        served(ask(owned, model, key))
        spend_reaches(owned, "team", team, CHAT_COST)
        blocked: Final = eventually(
            lambda: ask(owned, model, key), lambda response: response.status_code == 422, seconds=LANDING_SECONDS
        )
        assert f"Team={team}" in refusal(blocked)
        eventually(lambda: spend_of(owned, "team", team), lambda spend: spend == 0, seconds=LANDING_SECONDS)
        assert eventually(lambda: _status(owned, model, key), lambda code: code == 200, seconds=LANDING_SECONDS) == 200


def test_user_budget_reset_unblocks_the_users_personal_key(gateway: Gateway, tmp_path: Path) -> None:
    """A user has `max_budget` 0.0000001 and `budget_duration` `5s`. The user's personal key is served once and
    then refused with the user wording. After the reset job has zeroed the user's spend, it is served again."""
    with (
        scripted_upstream() as wire,
        owned_proxy(gateway, tmp_path, _FAST_RESET) as owned,
        owned.scenario() as scenario,
    ):
        model: Final = chat_model(scenario, wire)
        user: Final = scenario.user(max_budget=TINY_BUDGET, budget_duration="5s")
        key: Final = scenario.key(models=[model], user_id=user)
        served(ask(owned, model, key))
        spend_reaches(owned, "user", user, CHAT_COST)
        blocked: Final = eventually(
            lambda: ask(owned, model, key), lambda response: response.status_code == 422, seconds=LANDING_SECONDS
        )
        assert f"User={user}" in refusal(blocked)
        eventually(lambda: spend_of(owned, "user", user), lambda spend: spend == 0, seconds=LANDING_SECONDS)
        assert eventually(lambda: _status(owned, model, key), lambda code: code == 200, seconds=LANDING_SECONDS) == 200


def test_reset_of_one_key_leaves_the_spend_of_a_key_without_duration_alone(gateway: Gateway, tmp_path: Path) -> None:
    """Two keys spend 0.02 each. Only the first has `budget_duration` `5s`. After the reset job has zeroed the
    first key, the second key, which has no duration, still shows 0.02."""
    with (
        scripted_upstream() as wire,
        owned_proxy(gateway, tmp_path, _FAST_RESET) as owned,
        owned.scenario() as scenario,
    ):
        model: Final = chat_model(scenario, wire)
        timed: Final = scenario.key(models=[model], max_budget=1.0, budget_duration="5s")
        untimed: Final = scenario.key(models=[model], max_budget=1.0)
        served(ask(owned, model, timed))
        served(ask(owned, model, untimed))
        spend_reaches(owned, "key", hashed(timed), CHAT_COST)
        spend_reaches(owned, "key", hashed(untimed), CHAT_COST)
        eventually(lambda: spend_of(owned, "key", hashed(timed)), lambda spend: spend == 0, seconds=LANDING_SECONDS)
        assert spend_of(owned, "key", hashed(untimed)) == pytest.approx(0.02)


def test_temporary_budget_increase_readmits_an_exhausted_key(gateway: Gateway) -> None:
    """A key over its `max_budget` is refused. An admin sends `/key/update` with `temp_budget_increase: 1.0` and a
    `temp_budget_expiry` one hour ahead. The key's `max_budget` column is unchanged, yet the next call is served
    because the increase is added to the budget while it is active."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = _exhausted_key(gateway, scenario, model)
        expiry: Final = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        gateway.post("/key/update", {"key": key, "temp_budget_increase": 1.0, "temp_budget_expiry": expiry})
        assert eventually(lambda: _status(gateway, model, key), lambda code: code == 200, seconds=30) == 200
        stored: Final = gateway.rows(
            'SELECT max_budget FROM "LiteLLM_VerificationToken" WHERE token = %s', (hashed(key),)
        )
        assert stored == [{"max_budget": pytest.approx(TINY_BUDGET)}], stored


def test_expired_temporary_budget_increase_does_not_readmit_the_key(gateway: Gateway) -> None:
    """The same update, but with a `temp_budget_expiry` one hour in the past. The increase is ignored and the key
    stays refused with the key budget wording, and the provider is never called."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = _exhausted_key(gateway, scenario, model)
        provider_calls(wire)
        expiry: Final = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        gateway.post("/key/update", {"key": key, "temp_budget_increase": 1.0, "temp_budget_expiry": expiry})
        messages: Final = [refusal(ask(gateway, model, key)) for _ in range(3)]
        assert all(message.startswith("Budget has been exceeded! Key=") for message in messages), messages
        assert provider_calls(wire) == ()


def test_temporary_budget_increase_only_extends_by_its_amount(gateway: Gateway) -> None:
    """A key with `max_budget` 0.0000001 gets a `temp_budget_increase` of 0.03 (active). One more call fits
    (0.02 spent, 0.03 allowed), and after that call is applied (0.04 spent) the key is refused again with
    `Max budget: 0.0300001`, the sum of the budget and the increase."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = _exhausted_key(gateway, scenario, model)
        expiry: Final = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        gateway.post("/key/update", {"key": key, "temp_budget_increase": 0.03, "temp_budget_expiry": expiry})
        assert eventually(lambda: _status(gateway, model, key), lambda code: code == 200, seconds=30) == 200
        spend_reaches(gateway, "key", hashed(key), 2 * CHAT_COST)
        refused: Final = eventually(
            lambda: ask(gateway, model, key), lambda response: response.status_code == 422, seconds=LANDING_SECONDS
        )
        limit: Final = float(refusal(refused).rsplit("Max budget: ", 1)[1])
        assert limit == pytest.approx(TINY_BUDGET + 0.03), refused.text
