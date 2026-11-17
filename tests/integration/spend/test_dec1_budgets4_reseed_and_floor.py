"""Contract: Redis spend counters are rebuilt and corrected from Postgres, and admin writes set them with a floor.

Reseed. A counter that Redis lost is rebuilt before use from the table that owns it
(`litellm/proxy/db/spend_counter_reseed.py:113-131`, `_counter_row`, `:197-238`): key, team, team member, user,
organization and project counters from their rows, customer windows from their window rows. Tag and end user
counters are never reseeded from the table; the caller's freshly loaded object spend is their fallback (`:125-131`).
Round 2 pinned key, team, user and key window reseeds; this file adds organization, project and team member.

Floor. A counter that exists but reads below the committed spend (a Redis restart that loaded an older snapshot)
is a hit, not a miss, so it is never reseeded. `get_current_spend` (`litellm/proxy/proxy_server.py:3487-3577`)
re-checks it against the table whenever the caller's own spend is higher (primary counters) or always (window
counters), and raises it with a monotonic set-max (`_repair_stale_spend_counter`, `:3580-3608`). The value read is
kept for 5 seconds per worker in an in-memory `spend_db_floor:<counter>` marker (`_authoritative_floor_spend`,
`:3656-3690`, `SPEND_DB_FLOOR_CACHE_TTL_SECONDS`).

Admin writes. `/key/<key>/reset_spend`, `/tag/update` with `spend` and `/customer/delete` write the counter with
`set_spend_counter_with_floor_and_broadcast` (`proxy_server.py:4693-4719`): a SET with a 60 second TTL, plus the floor
marker, broadcast to every worker, so no worker raises the counter back from a pre-write floor.

Price: one served call 0.02, a held call with `max_tokens: 10` holds 0.04 (`hold_model`).
"""

from __future__ import annotations

from typing import Final

import pytest

from tests.integration._support.client import Gateway, eventually
from tests.integration._support.database import write_rows
from tests.integration.spend.dec1_budgets2_support import ttl_of
from tests.integration.spend.dec1_budgets3_support import (
    end_user_counter,
    key_window_counter,
    org_counter,
    project_counter,
    tag_counter,
    team_member_counter,
)
from tests.integration.spend.dec1_budgets_support import (
    CHAT_COST,
    LANDING_SECONDS,
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
from tests.integration.spend.dec1_spend_support import (
    hashed,
    new_customer,
    new_tag,
    scripted_upstream,
    served,
    spend_reaches,
)

pytestmark: Final = pytest.mark.shares_worker_state


def test_a_stale_low_key_counter_is_raised_to_the_stored_spend_and_refuses(gateway: Gateway) -> None:
    """A key with `max_budget` 1.0 has its stored spend set to 1.0 before its first request, and its counter
    `spend:key:<hash>` written as 0.0 (a stale snapshot, so a hit, not a miss). The first call loads the key with
    spend 1.0, sees the counter below it, reads the table and raises the counter (`proxy_server.py:3545-3561`). The
    call is refused with HTTP 422 `budget_exceeded` and the counter afterwards reads 1.0. A port that trusts any
    existing counter lets the key spend its whole budget again after every Redis restore."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=1.0)
        write_rows('UPDATE "LiteLLM_VerificationToken" SET spend = 1.0 WHERE token = %s', (hashed(key),))
        cache.set(key_counter(key), "0.0")
        refused(call(gateway, model, key, max_tokens=5))
        assert counter(cache, key_counter(key)) == pytest.approx(1.0)


def test_a_stale_low_key_counter_below_the_budget_is_raised_before_the_hold(gateway: Gateway) -> None:
    """A key with `max_budget` 1.0 and stored spend 0.5 (set before its first request) has its counter written as
    0.0. A held call with `max_tokens: 10` is admitted and, while it waits, the counter reads exactly 0.54: the
    floor raised it to 0.5 before the 0.04 hold was added, so the stale snapshot never becomes free budget."""
    with (
        held_provider() as provider,
        gateway.scenario() as scenario,
        redis_client() as cache,
        background() as run,
    ):
        model: Final = hold_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=1.0)
        write_rows('UPDATE "LiteLLM_VerificationToken" SET spend = 0.5 WHERE token = %s', (hashed(key),))
        cache.set(key_counter(key), "0.0")
        in_flight: Final = run(lambda: call(gateway, model, key, hold=True, max_tokens=10))
        provider.wait_for(1)
        assert counter(cache, key_counter(key)) == pytest.approx(0.54)
        provider.release()
        served(in_flight.result(timeout=60))


def test_a_stale_low_key_window_counter_is_raised_to_the_window_spend_row(gateway: Gateway) -> None:
    """A key whose only budget is a `1d` window of 0.02 is served once, so its `LiteLLM_BudgetWindowSpend` row holds
    0.02. The test writes `spend:key:<hash>:window:1d` as 0.0. The next call is refused with HTTP 422: window counters
    are always re-checked against their row (`is_window`, `proxy_server.py:3545`), there is no caller spend to
    compare with. Afterwards the window counter reads 0.02. UNVERIFIED: that the row is written for this key
    (see `test_a_window_reset_rolls_the_window_spend_row_onto_the_new_window_at_zero`)."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model], budget_limits=[{"budget_duration": "1d", "max_budget": CHAT_COST}])
        window: Final = key_window_counter(key, "1d")
        served(call(gateway, model, key, max_tokens=5))
        counter_settles(cache, window, CHAT_COST)
        eventually(
            lambda: gateway.rows(
                'SELECT spend FROM "LiteLLM_BudgetWindowSpend" WHERE entity_type = %s AND entity_id = %s',
                ("key", hashed(key)),
            ),
            lambda rows: len(rows) == 1 and float(str(rows[0]["spend"])) >= CHAT_COST - 1e-9,
            seconds=LANDING_SECONDS,
        )
        cache.set(window, "0.0")
        refused(call(gateway, model, key, max_tokens=5))
        assert counter(cache, window) == pytest.approx(CHAT_COST)


def test_a_floor_repair_never_lowers_a_counter_above_the_stored_spend(gateway: Gateway) -> None:
    """A key with `max_budget` 1.0 is served once (stored spend 0.02); the test then writes its counter as 0.9,
    above the table. The next call is still admitted (0.9 plus a 0.02 hold fits) and the counter ends at 0.92: the
    repair is a set-max that only raises (`spend_counter_store.set_max`, `proxy_server.py:3597-3606`), and a counter
    at or above the caller's spend never reads the table."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=1.0)
        served(call(gateway, model, key, max_tokens=5))
        spend_reaches(gateway, "key", hashed(key), CHAT_COST)
        counter_settles(cache, key_counter(key), CHAT_COST)
        cache.set(key_counter(key), "0.9")
        served(call(gateway, model, key, max_tokens=5))
        assert counter_settles(cache, key_counter(key), 0.92) == pytest.approx(0.92)


def test_a_lost_organization_counter_is_reseeded_from_the_organization_table(gateway: Gateway) -> None:
    """An organization with `max_budget` 1.0 has its stored spend set to 0.5 and `spend:org:<id>` deleted. A held
    call from a key of one of its teams reads exactly 0.54 on the recreated counter (`_counter_row`,
    `spend_counter_reseed.py:229-232`, then the 0.04 hold). A port that restarts a lost organization counter at 0
    reads 0.04."""
    with (
        held_provider() as provider,
        gateway.scenario() as scenario,
        redis_client() as cache,
        background() as run,
    ):
        model: Final = hold_model(scenario, provider)
        org_id: Final = scenario.organization(models=[model], max_budget=1.0)
        team_id: Final = scenario.team(organization_id=org_id, models=[model])
        key: Final = scenario.key(team_id=team_id, models=[model])
        write_rows('UPDATE "LiteLLM_OrganizationTable" SET spend = 0.5 WHERE organization_id = %s', (org_id,))
        cache.delete(org_counter(org_id))
        in_flight: Final = run(lambda: call(gateway, model, key, hold=True, max_tokens=10))
        provider.wait_for(1)
        assert counter(cache, org_counter(org_id)) == pytest.approx(0.54)
        provider.release()
        served(in_flight.result(timeout=60))


def test_a_lost_project_counter_is_reseeded_from_the_project_table(gateway: Gateway) -> None:
    """A project with `max_budget` 1.0 has its stored spend set to 0.5 and `spend:project:<id>` deleted. A held call
    from the project's key reads exactly 0.54 on the recreated counter (`spend_counter_reseed.py:233-236`)."""
    with (
        held_provider() as provider,
        gateway.scenario() as scenario,
        redis_client() as cache,
        background() as run,
    ):
        model: Final = hold_model(scenario, provider)
        team_id: Final = scenario.team(models=[model])
        project_id: Final = scenario.project(team_id, models=[model], max_budget=1.0)
        key: Final = scenario.key(team_id=team_id, project_id=project_id, models=[model])
        write_rows('UPDATE "LiteLLM_ProjectTable" SET spend = 0.5 WHERE project_id = %s', (project_id,))
        cache.delete(project_counter(project_id))
        in_flight: Final = run(lambda: call(gateway, model, key, hold=True, max_tokens=10))
        provider.wait_for(1)
        assert counter(cache, project_counter(project_id)) == pytest.approx(0.54)
        provider.release()
        served(in_flight.result(timeout=60))


def test_a_lost_team_member_counter_is_reseeded_from_the_membership_row(gateway: Gateway) -> None:
    """A team with `team_member_budget` 1.0 has a member whose `LiteLLM_TeamMembership.spend` is set to 0.5 and whose
    `spend:team_member:<user>:<team>` is deleted. A held call from the member's key reads exactly 0.54 on the
    recreated counter (`spend_counter_reseed.py:213-220`, keyed by the last `:` so user ids with colons work).
    UNVERIFIED: the member counter needs the reservation membership cache (see the round 3 index)."""
    with (
        held_provider() as provider,
        gateway.scenario() as scenario,
        redis_client() as cache,
        background() as run,
    ):
        model: Final = hold_model(scenario, provider)
        team_id: Final = scenario.team(models=[model], team_member_budget=1.0)
        user_id: Final = scenario.member(team_id)
        key: Final = scenario.key(team_id=team_id, user_id=user_id, models=[model])
        write_rows(
            'UPDATE "LiteLLM_TeamMembership" SET spend = 0.5 WHERE user_id = %s AND team_id = %s', (user_id, team_id)
        )
        cache.delete(team_member_counter(user_id, team_id))
        in_flight: Final = run(lambda: call(gateway, model, key, hold=True, max_tokens=10))
        provider.wait_for(1)
        assert counter(cache, team_member_counter(user_id, team_id)) == pytest.approx(0.54)
        provider.release()
        served(in_flight.result(timeout=60))


def test_a_lost_tag_counter_is_rebuilt_from_the_tag_object_not_the_tag_table(gateway: Gateway) -> None:
    """A tag with `max_budget` 1.0 is charged 0.02 by one tagged call. The test then writes 0.5 into
    `LiteLLM_TagTable.spend` and deletes `spend:tag:<name>`. A held tagged call with `max_tokens: 10` reads 0.06 on the
    rebuilt counter (the 0.02 the tag object the proxy already loaded carries, plus the hold), not 0.54: tag counters
    have no reseed from the table (`spend_counter_reseed.py:125-131`, `_counter_row` returns None for `spend:tag:`).
    UNVERIFIED: whether the tag object is cached or reloaded at this call; a reload would read 0.54."""
    with (
        held_provider() as provider,
        gateway.scenario() as scenario,
        redis_client() as cache,
        background() as run,
    ):
        model: Final = hold_model(scenario, provider)
        tag: Final = new_tag(gateway, scenario, max_budget=1.0)
        key: Final = scenario.key(models=[model])
        served(call(gateway, model, key, max_tokens=5, metadata={"tags": [tag]}))
        counter_settles(cache, tag_counter(tag), CHAT_COST)
        spend_reaches(gateway, "tag", tag, CHAT_COST)
        write_rows('UPDATE "LiteLLM_TagTable" SET spend = 0.5 WHERE tag_name = %s', (tag,))
        cache.delete(tag_counter(tag))
        in_flight: Final = run(lambda: call(gateway, model, key, hold=True, max_tokens=10, metadata={"tags": [tag]}))
        provider.wait_for(1)
        assert counter(cache, tag_counter(tag)) == pytest.approx(0.06)
        provider.release()
        served(in_flight.result(timeout=60))


def test_a_stale_low_end_user_counter_is_raised_to_the_end_user_row(gateway: Gateway) -> None:
    """A customer with `max_budget` 1.0 has `LiteLLM_EndUserTable.spend` set to 1.0 before its first request and
    `spend:end_user:<id>` written as 0.0. A call naming it as `user` is refused with HTTP 422 and the counter then
    reads 1.0: end user counters take their floor from the end user row, the row a budget reset zeroes
    (`_floor_spend_from_db`, `proxy_server.py:3640-3641`, `end_user_from_db`, `spend_counter_reseed.py:240-256`)."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        customer: Final = new_customer(gateway, scenario, max_budget=1.0)
        key: Final = scenario.key(models=[model])
        write_rows('UPDATE "LiteLLM_EndUserTable" SET spend = 1.0 WHERE user_id = %s', (customer,))
        cache.set(end_user_counter(customer), "0.0")
        refused(call(gateway, model, key, max_tokens=5, user=customer))
        assert counter(cache, end_user_counter(customer)) == pytest.approx(1.0)


def test_reset_spend_writes_the_key_counter_with_a_sixty_second_ttl(gateway: Gateway) -> None:
    """After `POST /key/<key>/reset_spend` with `reset_to: 0.01` on a key that spent 0.02, `spend:key:<hash>` reads
    0.01 and has a TTL of at most 60 seconds (`set_spend_counter_with_floor_and_broadcast`,
    `proxy_server.py:4706-4708`). The TTL is what lets the counter fall back to the committed row once the
    broadcast floor expired; a port that writes it without one pins the reset value forever."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=1.0)
        served(call(gateway, model, key, max_tokens=5))
        spend_reaches(gateway, "key", hashed(key), CHAT_COST)
        counter_settles(cache, key_counter(key), CHAT_COST)
        response: Final = gateway.request("POST", f"/key/{key}/reset_spend", {"reset_to": 0.01})
        assert response.status_code == 200, response.text
        assert counter(cache, key_counter(key)) == pytest.approx(0.01)
        assert 0 < ttl_of(cache, key_counter(key)) <= 60


def test_tag_update_with_spend_sets_the_tag_counter_to_that_spend(gateway: Gateway) -> None:
    """A tag with `max_budget` 1.0 is charged 0.02. `POST /tag/update` with `{"name": <tag>, "spend": 0.005}` answers
    200, stores 0.005 and sets `spend:tag:<name>` to exactly 0.005 with a TTL of at most 60 seconds
    (`tag_management_endpoints.py:478-479`). Tag counters have no table reseed, so this write is the only way an admin
    correction reaches enforcement before the counter expires."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        tag: Final = new_tag(gateway, scenario, max_budget=1.0)
        key: Final = scenario.key(models=[model])
        served(call(gateway, model, key, max_tokens=5, metadata={"tags": [tag]}))
        counter_settles(cache, tag_counter(tag), CHAT_COST)
        response: Final = gateway.request("POST", "/tag/update", {"name": tag, "spend": 0.005})
        assert response.status_code == 200, response.text
        assert counter(cache, tag_counter(tag)) == pytest.approx(0.005)
        assert 0 < ttl_of(cache, tag_counter(tag)) <= 60


def test_tag_update_with_spend_zero_readmits_a_spent_tag(gateway: Gateway) -> None:
    """A tag with `max_budget` 0.02 is spent by one tagged call and the next tagged call is refused with HTTP 422
    naming the tag. After `POST /tag/update` with `spend: 0`, the next tagged call on the same proxy is served: the
    counter, the floor marker and the cached tag object were all replaced (`tag_management_endpoints.py:474-479`)."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = hold_model(scenario, wire)
        tag: Final = new_tag(gateway, scenario, max_budget=CHAT_COST)
        key: Final = scenario.key(models=[model])
        served(call(gateway, model, key, max_tokens=5, metadata={"tags": [tag]}))
        spend_reaches(gateway, "tag", tag, CHAT_COST)
        assert f"Tag={tag}" in refused(
            eventually(
                lambda: call(gateway, model, key, max_tokens=5, metadata={"tags": [tag]}),
                lambda response: response.status_code == 422,
                seconds=LANDING_SECONDS,
            )
        )
        response: Final = gateway.request("POST", "/tag/update", {"name": tag, "spend": 0})
        assert response.status_code == 200, response.text
        served(call(gateway, model, key, max_tokens=5, metadata={"tags": [tag]}))


def test_customer_delete_leaves_a_zero_end_user_counter_that_expires(gateway: Gateway) -> None:
    """A customer with `max_budget` 1.0 spends 0.02 (`spend:end_user:<id>` 0.02). `POST /customer/delete` deletes the
    counter and then writes it back as 0.0 with a TTL of at most 60 seconds (`customer_endpoints.py:853-856`), so a
    worker holding the pre-delete floor cannot raise it back. A port that only deletes it leaves that window open."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        customer: Final = new_customer(gateway, scenario, max_budget=1.0)
        key: Final = scenario.key(models=[model])
        served(call(gateway, model, key, max_tokens=5, user=customer))
        counter_settles(cache, end_user_counter(customer), CHAT_COST)
        response: Final = gateway.request("POST", "/customer/delete", {"user_ids": [customer]})
        assert response.status_code == 200, response.text
        assert counter(cache, end_user_counter(customer)) == pytest.approx(0.0)
        assert 0 < ttl_of(cache, end_user_counter(customer)) <= 60
