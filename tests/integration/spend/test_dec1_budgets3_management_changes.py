"""Contract: budget and spend changes made through the management API take effect on the very next request.

Since #51535 the auth checks read Redis spend counters first (`get_current_spend`, `proxy_server.py:3489-3575`),
so a management write that only touches Postgres is not enough: whatever the counter says wins while it is at or
above the budget (`if max_budget is None or current >= max_budget: return current`). Each endpoint therefore has
its own counter handling, and these tests pin what a caller sees after each one:

- `/key/update` with `spend` sets `spend:key:<hash>` to the new value (`key_management_endpoints.py:4408-4421`),
- `/key/{key}/reset_spend` sets the counter with a floor and broadcast (`key_management_endpoints.py:7155-7166`),
- `/user/update` with `spend` invalidates `spend:user:<id>` (`internal_user_endpoints.py:1545-1562`),
- `/customer/delete` zeroes `spend:end_user:<id>` (`customer_endpoints.py:850-856`),
- `/tag/new` invalidates `spend:tag:<name>`, so a recreated tag starts fresh (`tag_management_endpoints.py:326`),
- `/organization/update` with `spend` only drops the committed twin, not `spend:org:<id>`
  (`organization_endpoints.py:846-849`), which this file reports as a suspected bug.

Raising or removing a `max_budget` takes effect through the cached auth object, not the counter. Price as in
round 1: one call costs 0.02, budgets of 0.01 are spent by one call.
"""

from __future__ import annotations

from typing import Final

import pytest

from tests.integration._support.client import Gateway, eventually
from tests.integration.spend.dec1_budgets3_support import (
    COST,
    TOKENS,
    end_user_counter,
    org_counter,
    spend_once,
    tag_counter,
    team_counter,
    user_counter,
)
from tests.integration.spend.dec1_budgets_support import (
    background,
    call,
    counter,
    counter_settles,
    held_provider,
    hold_model,
    key_counter,
    redis_client,
    refused,
)
from tests.integration.spend.dec1_spend_support import new_customer, new_tag, scripted_upstream, served

pytestmark: Final = pytest.mark.shares_worker_state

SMALL: Final = 0.01


def test_setting_key_spend_to_zero_readmits_an_exhausted_key_at_once(gateway: Gateway) -> None:
    """A key with `max_budget` 0.01 spends 0.02 and is refused. `/key/update` with `spend: 0` writes 0 into
    `spend:key:<hash>` (TTL 60s) as well as the table, so the very next call is served without waiting for any
    cache to expire."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=SMALL)
        spend_once(gateway, model, key)
        counter_settles(cache, key_counter(key), COST)
        assert refused(call(gateway, model, key, max_tokens=TOKENS))
        gateway.post("/key/update", {"key": key, "spend": 0})
        assert counter(cache, key_counter(key)) == pytest.approx(0.0)
        assert served(call(gateway, model, key, max_tokens=1))


def test_reset_spend_readmits_an_exhausted_key_and_sets_the_counter_to_the_reset_value(gateway: Gateway) -> None:
    """A key with `max_budget` 0.03 spends 0.04 over two calls and is refused. `POST /key/<key>/reset_spend` with
    `reset_to: 0.01` leaves `spend:key:<hash>` at exactly 0.01 (a partial reset, not 0), and the next call is
    served."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=0.03)
        spend_once(gateway, model, key)
        spend_once(gateway, model, key)
        counter_settles(cache, key_counter(key), 0.04)
        assert refused(call(gateway, model, key, max_tokens=TOKENS))
        gateway.post(f"/key/{key}/reset_spend", {"reset_to": 0.01})
        counter_settles(cache, key_counter(key), 0.01)
        assert served(call(gateway, model, key, max_tokens=1))


def test_reset_spend_above_the_current_spend_is_refused(gateway: Gateway) -> None:
    """`POST /key/<key>/reset_spend` with `reset_to` larger than the key's spend is refused with 400 by
    `_validate_reset_spend_value` (`key_management_endpoints.py:6943`), and the counter keeps its value: a reset
    can only lower spend. UNVERIFIED: exact error text not pinned."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=1.0)
        spend_once(gateway, model, key)
        counter_settles(cache, key_counter(key), COST)
        response: Final = gateway.request("POST", f"/key/{key}/reset_spend", {"reset_to": 0.5})
        assert response.status_code == 400, response.text
        assert counter(cache, key_counter(key)) == pytest.approx(COST)


def test_removing_a_key_budget_readmits_it_and_takes_no_more_holds(gateway: Gateway) -> None:
    """An exhausted key (`max_budget` 0.01, counter 0.02) is updated with `max_budget: null`. The next call is
    served, and while a later call is held the key counter stays at the settled value instead of growing by a
    hold: without a budget no key counter is planned (`budget_reservation.py:522`).
    UNVERIFIED: that `/key/update` accepts an explicit null to clear the budget."""
    with held_provider() as provider, gateway.scenario() as scenario, redis_client() as cache, background() as run:
        model: Final = hold_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=SMALL)
        spend_once(gateway, model, key)
        counter_settles(cache, key_counter(key), COST)
        gateway.post("/key/update", {"key": key, "max_budget": None})
        before: Final = counter(cache, key_counter(key))
        pending: Final = run(lambda: call(gateway, model, key, hold=True, max_tokens=TOKENS))
        provider.wait_for(1)
        during: Final = counter(cache, key_counter(key))
        provider.release()
        served(pending.result(timeout=60))
        assert during == before


def test_raising_a_team_budget_readmits_its_keys(gateway: Gateway) -> None:
    """A team with `max_budget` 0.01 is spent (counter 0.02) and its key is refused with the team wording.
    `/team/update` with `max_budget: 1.0` makes the next call of the same key served: the team object cache is
    refreshed by the update."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        team: Final = scenario.team(models=[model], max_budget=SMALL)
        key: Final = scenario.key(team_id=team, models=[model])
        spend_once(gateway, model, key)
        counter_settles(cache, team_counter(team), COST)
        assert refused(call(gateway, model, key, max_tokens=TOKENS)).startswith(
            f"Budget has been exceeded! Team={team}"
        )
        gateway.post("/team/update", {"team_id": team, "max_budget": 1.0})
        assert served(call(gateway, model, key, max_tokens=TOKENS))


def test_raising_an_organization_budget_readmits_its_team_keys(gateway: Gateway) -> None:
    """An organization with `max_budget` 0.01 is spent by one team key call. `/organization/update` with
    `max_budget: 1.0` makes the next call served. UNVERIFIED: assumes the update evicts the
    `org_id:<id>:with_budget` cache entry that the organization check and the reservation read."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        organization: Final = scenario.organization(models=[model], max_budget=SMALL)
        team: Final = scenario.team(organization_id=organization, models=[model])
        key: Final = scenario.key(team_id=team, models=[model])
        spend_once(gateway, model, key)
        counter_settles(cache, org_counter(organization), COST)
        assert refused(call(gateway, model, key, max_tokens=TOKENS))
        gateway.request("PATCH", "/organization/update", {"organization_id": organization, "max_budget": 1.0})
        assert served(call(gateway, model, key, max_tokens=TOKENS))


def test_setting_organization_spend_to_zero_readmits_its_keys(gateway: Gateway) -> None:
    """An organization with `max_budget` 0.01 is spent (counter 0.02). `/organization/update` with `spend: 0`
    resets its stored spend; the next team key call should be served."""
    pytest.skip(
        "BUG: /organization/update with spend only drops spend:committed:org:<id> (organization_endpoints.py:846-849) "
        "and leaves spend:org:<id> at 0.02, which get_current_spend returns as-is because it is above max_budget, "
        "so the organization stays refused until the counter expires"
    )
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        organization: Final = scenario.organization(models=[model], max_budget=SMALL)
        team: Final = scenario.team(organization_id=organization, models=[model])
        key: Final = scenario.key(team_id=team, models=[model])
        spend_once(gateway, model, key)
        counter_settles(cache, org_counter(organization), COST)
        gateway.request("PATCH", "/organization/update", {"organization_id": organization, "spend": 0})
        assert served(call(gateway, model, key, max_tokens=1))


def test_setting_user_spend_to_zero_readmits_the_users_personal_key(gateway: Gateway) -> None:
    """A user with `max_budget` 0.01 spends 0.02 through a personal key and is refused with the user wording.
    `/user/update` with `spend: 0` invalidates `spend:user:<id>` and its committed twin, so the next call is
    served (the counter is reseeded from the table, now 0)."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        user: Final = scenario.user(max_budget=SMALL)
        key: Final = scenario.key(user_id=user, models=[model])
        spend_once(gateway, model, key)
        counter_settles(cache, user_counter(user), COST)
        assert refused(call(gateway, model, key, max_tokens=TOKENS))
        gateway.post("/user/update", {"user_id": user, "spend": 0})
        assert served(call(gateway, model, key, max_tokens=1))


def test_raising_a_customer_budget_readmits_the_customer(gateway: Gateway) -> None:
    """A customer with `max_budget` 0.01 spends 0.02 and its next call is refused. `/customer/update` with
    `max_budget: 1.0` makes the next call for that customer served. UNVERIFIED: assumes the update evicts the
    end user cache entry (`end_user_cache_key`)."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        customer: Final = new_customer(gateway, scenario, max_budget=SMALL)
        spend_once(gateway, model, key, user=customer)
        counter_settles(cache, end_user_counter(customer), COST)
        assert refused(call(gateway, model, key, max_tokens=TOKENS, user=customer))
        gateway.post("/customer/update", {"user_id": customer, "max_budget": 1.0})
        assert served(call(gateway, model, key, max_tokens=TOKENS, user=customer))


def test_a_deleted_and_recreated_customer_starts_with_a_fresh_budget(gateway: Gateway) -> None:
    """A customer with `max_budget` 0.01 spends 0.02. After `/customer/delete` the counter `spend:end_user:<id>`
    reads 0 (`set_spend_counter_with_floor_and_broadcast(..., 0.0)`), and a customer created again with the same id
    and `max_budget` 0.01 is served."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        customer: Final = new_customer(gateway, scenario, max_budget=SMALL)
        spend_once(gateway, model, key, user=customer)
        counter_settles(cache, end_user_counter(customer), COST)
        gateway.post("/customer/delete", {"user_ids": [customer]})
        counter_settles(cache, end_user_counter(customer), 0.0)
        gateway.post("/customer/new", {"user_id": customer, "max_budget": SMALL})
        assert served(call(gateway, model, key, max_tokens=1, user=customer))


def test_a_deleted_and_recreated_tag_starts_with_a_fresh_budget(gateway: Gateway) -> None:
    """A tag with `max_budget` 0.01 is spent (counter 0.02). The tag is deleted and created again under the same
    name with `max_budget` 0.01; `/tag/new` invalidates `spend:tag:<name>`, so the next tagged call is served
    instead of inheriting the old tag's spend."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        tag: Final = new_tag(gateway, scenario, max_budget=SMALL)
        spend_once(gateway, model, key, metadata={"tags": [tag]})
        counter_settles(cache, tag_counter(tag), COST)
        gateway.post("/tag/delete", {"name": tag})
        gateway.post("/tag/new", {"name": tag, "max_budget": SMALL})
        assert counter(cache, tag_counter(tag)) is None
        assert served(call(gateway, model, key, max_tokens=1, metadata={"tags": [tag]}))


def test_raising_a_tag_budget_readmits_tagged_calls(gateway: Gateway) -> None:
    """A tag with `max_budget` 0.01 is spent and a tagged call is refused with `Tag=<tag>`. `/tag/update` with
    `max_budget: 1.0` makes the next tagged call served. UNVERIFIED: assumes `/tag/update` evicts
    `tag_cache_key(<tag>)` like `/tag/new` does."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        tag: Final = new_tag(gateway, scenario, max_budget=SMALL)
        spend_once(gateway, model, key, metadata={"tags": [tag]})
        counter_settles(cache, tag_counter(tag), COST)
        assert refused(call(gateway, model, key, max_tokens=TOKENS, metadata={"tags": [tag]}))
        gateway.post("/tag/update", {"name": tag, "max_budget": 1.0})
        assert served(call(gateway, model, key, max_tokens=TOKENS, metadata={"tags": [tag]}))


def test_raising_a_members_budget_in_the_team_readmits_the_member(gateway: Gateway) -> None:
    """A team's default member budget is 0.01 and a member spends 0.02 and is refused. `/team/member_update` with
    `max_budget_in_team: 1.0` gives that member a budget of their own and the next call is served. UNVERIFIED:
    the membership cache (`team_membership_auth_cache_key`, TTL 5s, `user_api_key_auth.py:3303-3325`) may serve
    the old budget for up to 5 seconds, so the test polls."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = hold_model(scenario, wire)
        team: Final = scenario.team(models=[model], team_member_budget=SMALL)
        member: Final = scenario.member(team)
        key: Final = scenario.key(user_id=member, team_id=team, models=[model])
        spend_once(gateway, model, key)
        eventually(
            lambda: call(gateway, model, key, max_tokens=TOKENS).status_code,
            lambda status: status == 422,
            seconds=70,
        )
        gateway.post("/team/member_update", {"team_id": team, "user_id": member, "max_budget_in_team": 1.0})
        readmitted: Final = eventually(
            lambda: call(gateway, model, key, max_tokens=1).status_code,
            lambda status: status == 200,
            seconds=30,
        )
        assert readmitted == 200
