"""Contract: when several budgets on one request are spent at once, which one the caller is told about.

Auth refuses on the first spent budget it checks, and the order is fixed by code, not by how far over each
budget is:

1. the team member check in the key builder (`user_api_key_auth.py:3300-3364`, `team_member_positive`),
2. the key's own `max_budget` (`user_api_key_auth.py:3398-3405`, `virtual_key_max_budget_check`), both before
   `common_checks` runs,
3. the project checks (`auth_checks.py`, `_run_project_checks`, called before the budget gather),
4. the gathered checks, whose first error wins "in scope-priority order" (`auth_checks.py:1505-1555`): team
   `max_budget`, team windows, key windows, organization, tag, model access group, user, team member, end user,
5. the proxy-wide `litellm.max_budget` (`_global_proxy_budget_check`, after the gather).

A Rust port that checks the levels concurrently and reports whichever answers first, or that reorders them,
changes which entity a client is told to top up. Every test spends both budgets with one served call: the key's
first call costs 0.02 (input free, 0.004 per output token, 5 tokens) and each budget is 0.01, so both counters
read 0.02 and both entities are over. The wording of each check is in
`litellm/rust/crates/callbacks-budgets/src/checks.rs:150-470`.
"""

from __future__ import annotations

from typing import Final

import pytest

from tests.integration._support.client import Gateway
from tests.integration.spend.dec1_budgets3_support import (
    COST,
    TOKENS,
    end_user_counter,
    key_window_counter,
    org_counter,
    project_counter,
    spend_once,
    tag_counter,
    team_counter,
    team_member_counter,
    team_window_counter,
    user_counter,
)
from tests.integration.spend.dec1_budgets_support import (
    budget_message,
    call,
    counter_settles,
    hold_model,
    key_counter,
    masked,
    redis_client,
    refused,
)
from tests.integration.spend.dec1_spend_support import new_customer, new_tag, provider_calls, scripted_upstream

pytestmark: Final = pytest.mark.shares_worker_state

SMALL: Final = 0.01


def test_a_spent_key_is_reported_before_its_spent_team(gateway: Gateway) -> None:
    """A team and its key both have `max_budget` 0.01 and one call spends 0.02 on both counters. The next call is
    refused with the key wording `Budget has been exceeded! Key=key (sk-...<last 4>) Current cost: 0.02, Max
    budget: 0.01`, not the team wording, because the key check runs in the builder before `common_checks`
    (`user_api_key_auth.py:3398-3405`)."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        team: Final = scenario.team(models=[model], max_budget=SMALL)
        key: Final = scenario.key(team_id=team, models=[model], max_budget=SMALL)
        spend_once(gateway, model, key)
        counter_settles(cache, key_counter(key), COST)
        counter_settles(cache, team_counter(team), COST)
        assert refused(call(gateway, model, key, max_tokens=TOKENS)) == budget_message(
            f"Key=key ({masked(key)})", "0.02", "0.01"
        )


def test_a_spent_key_is_reported_before_its_own_spent_window(gateway: Gateway) -> None:
    """A key has `max_budget` 0.01 and a `1h` window of 0.01. After one call both are spent. The refusal is the
    lifetime key wording (`Key=key (...)`), not `ExceededBudget: Key over 1h budget`, because key windows are only
    checked inside `common_checks` (`_virtual_key_multi_budget_check`, `auth_checks.py:7152`)."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(
            models=[model], max_budget=SMALL, budget_limits=[{"budget_duration": "1h", "max_budget": SMALL}]
        )
        spend_once(gateway, model, key)
        counter_settles(cache, key_window_counter(key, "1h"), COST)
        assert refused(call(gateway, model, key, max_tokens=TOKENS)).startswith("Budget has been exceeded! Key=key ")


def test_a_spent_project_is_reported_before_its_spent_team(gateway: Gateway) -> None:
    """A project and its team both have `max_budget` 0.01 and a key of the project spends 0.02. The refusal names
    the project: `Budget has been exceeded! Project=<project_id> Current cost: 0.02, Max budget: 0.01`, because
    `_run_project_checks` runs before the gathered team check (`auth_checks.py`, common_checks step 1.1-3.0.3).
    UNVERIFIED: the project check reads `spend:project:<id>`; that the key needs no budget of its own for the
    project to be charged is taken from round 1's project tests."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        team: Final = scenario.team(models=[model], max_budget=SMALL)
        project: Final = scenario.project(team, models=[model], max_budget=SMALL)
        key: Final = scenario.key(team_id=team, project_id=project, models=[model])
        spend_once(gateway, model, key)
        counter_settles(cache, project_counter(project), COST)
        counter_settles(cache, team_counter(team), COST)
        assert refused(call(gateway, model, key, max_tokens=TOKENS)) == budget_message(
            f"Project={project}", "0.02", "0.01"
        )


def test_a_spent_team_is_reported_before_its_spent_organization(gateway: Gateway) -> None:
    """An organization and its team both have `max_budget` 0.01 and a team key spends 0.02. The refusal is the
    team wording `Budget has been exceeded! Team=<team_id> Current cost: 0.02, Max budget: 0.01`: the team check
    is first in the gathered tuple (`auth_checks.py:1505`) and the first error in that order is raised."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        organization: Final = scenario.organization(models=[model], max_budget=SMALL)
        team: Final = scenario.team(organization_id=organization, models=[model], max_budget=SMALL)
        key: Final = scenario.key(team_id=team, models=[model])
        spend_once(gateway, model, key)
        counter_settles(cache, org_counter(organization), COST)
        counter_settles(cache, team_counter(team), COST)
        assert refused(call(gateway, model, key, max_tokens=TOKENS)) == budget_message(f"Team={team}", "0.02", "0.01")


def test_a_spent_team_window_is_reported_before_a_spent_key_window(gateway: Gateway) -> None:
    """A team has a `1h` window of 0.01 and its key a `1h` window of 0.01; one call spends 0.02 on both. The
    refusal is `ExceededBudget: Team=<team_id> over 1h budget. Spend=$0.0200, Limit=$0.01`: team windows come
    before key windows in the gather (`auth_checks.py:1515-1516`)."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        window: Final = [{"budget_duration": "1h", "max_budget": SMALL}]
        team: Final = scenario.team(models=[model], budget_limits=window)
        key: Final = scenario.key(team_id=team, models=[model], budget_limits=window)
        spend_once(gateway, model, key)
        counter_settles(cache, team_window_counter(team, "1h"), COST)
        counter_settles(cache, key_window_counter(key, "1h"), COST)
        assert (
            refused(call(gateway, model, key, max_tokens=TOKENS))
            == f"ExceededBudget: Team={team} over 1h budget. Spend=$0.0200, Limit=$0.01"
        )


def test_a_spent_key_window_is_reported_before_a_spent_organization(gateway: Gateway) -> None:
    """A key has a `1h` window of 0.01 and its team's organization `max_budget` 0.01; one call spends both. The
    refusal is `ExceededBudget: Key over 1h budget. Spend=$0.0200, Limit=$0.01` (key windows precede the
    organization in the gather)."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        organization: Final = scenario.organization(models=[model], max_budget=SMALL)
        team: Final = scenario.team(organization_id=organization, models=[model])
        key: Final = scenario.key(
            team_id=team, models=[model], budget_limits=[{"budget_duration": "1h", "max_budget": SMALL}]
        )
        spend_once(gateway, model, key)
        counter_settles(cache, org_counter(organization), COST)
        assert (
            refused(call(gateway, model, key, max_tokens=TOKENS))
            == "ExceededBudget: Key over 1h budget. Spend=$0.0200, Limit=$0.01"
        )


def test_a_spent_organization_is_reported_before_a_spent_tag(gateway: Gateway) -> None:
    """An organization (via the key's team) and a tag sent in `metadata.tags` both have `max_budget` 0.01 and one
    call spends both. The next tagged call is refused with `Budget has been exceeded! Organization=<org_id>
    Current cost: 0.02, Max budget: 0.01`, never the `Tag=` wording."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        organization: Final = scenario.organization(models=[model], max_budget=SMALL)
        team: Final = scenario.team(organization_id=organization, models=[model])
        key: Final = scenario.key(team_id=team, models=[model])
        tag: Final = new_tag(gateway, scenario, max_budget=SMALL)
        spend_once(gateway, model, key, metadata={"tags": [tag]})
        counter_settles(cache, tag_counter(tag), COST)
        counter_settles(cache, org_counter(organization), COST)
        assert refused(call(gateway, model, key, max_tokens=TOKENS, metadata={"tags": [tag]})) == budget_message(
            f"Organization={organization}", "0.02", "0.01"
        )


def test_a_spent_tag_is_reported_before_a_spent_user(gateway: Gateway) -> None:
    """A user with `max_budget` 0.01 owns a personal key, and a tag with `max_budget` 0.01 is sent on the call;
    one call spends both. The refusal is `Budget has been exceeded! Tag=<tag> Current cost: 0.02, Max budget:
    0.01`: the tag check precedes the user check in the gather (`auth_checks.py:1525-1539`)."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        user: Final = scenario.user(max_budget=SMALL)
        key: Final = scenario.key(user_id=user, models=[model])
        tag: Final = new_tag(gateway, scenario, max_budget=SMALL)
        spend_once(gateway, model, key, metadata={"tags": [tag]})
        counter_settles(cache, user_counter(user), COST)
        counter_settles(cache, tag_counter(tag), COST)
        assert refused(call(gateway, model, key, max_tokens=TOKENS, metadata={"tags": [tag]})) == budget_message(
            f"Tag={tag}", "0.02", "0.01"
        )


def test_a_spent_user_is_reported_before_a_spent_end_user(gateway: Gateway) -> None:
    """A user with `max_budget` 0.01 owns a personal key and calls for a customer with `max_budget` 0.01; one call
    spends both. The refusal is the user wording `ExceededBudget: User=<user_id> over budget. Spend=0.02,
    Budget=0.01`, since the end user check is the last of the gather."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        user: Final = scenario.user(max_budget=SMALL)
        key: Final = scenario.key(user_id=user, models=[model])
        customer: Final = new_customer(gateway, scenario, max_budget=SMALL)
        spend_once(gateway, model, key, user=customer)
        counter_settles(cache, user_counter(user), COST)
        counter_settles(cache, end_user_counter(customer), COST)
        assert (
            refused(call(gateway, model, key, max_tokens=TOKENS, user=customer))
            == f"ExceededBudget: User={user} over budget. Spend=0.02, Budget=0.01"
        )


def test_a_spent_organization_is_reported_before_a_spent_end_user(gateway: Gateway) -> None:
    """A team key in an organization with `max_budget` 0.01 calls for a customer with `max_budget` 0.01; one call
    spends both. The refusal names the organization, so a customer at its cap is never blamed for an
    organization that is out of money."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        organization: Final = scenario.organization(models=[model], max_budget=SMALL)
        team: Final = scenario.team(organization_id=organization, models=[model])
        key: Final = scenario.key(team_id=team, models=[model])
        customer: Final = new_customer(gateway, scenario, max_budget=SMALL)
        spend_once(gateway, model, key, user=customer)
        counter_settles(cache, org_counter(organization), COST)
        counter_settles(cache, end_user_counter(customer), COST)
        assert refused(call(gateway, model, key, max_tokens=TOKENS, user=customer)) == budget_message(
            f"Organization={organization}", "0.02", "0.01"
        )


def test_a_spent_team_member_is_reported_before_the_members_spent_key(gateway: Gateway) -> None:
    """A team's default `team_member_budget` is 0.01 and the member's key also has `max_budget` 0.01; one call
    spends both. The refusal is the builder's team member wording `Budget has been exceeded!
    TeamMember=<user_id>:<team_id> Current cost: 0.02, Max budget: 0.01`, because that check
    (`user_api_key_auth.py:3300-3364`) runs before the key check at line 3401.
    UNVERIFIED: it only runs when the cached key carries `team_member_spend`, which the joint key view fills for a
    member key; if it is missing, the key wording is reported instead."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        team: Final = scenario.team(models=[model], team_member_budget=SMALL)
        member: Final = scenario.member(team)
        key: Final = scenario.key(user_id=member, team_id=team, models=[model], max_budget=SMALL)
        spend_once(gateway, model, key)
        counter_settles(cache, team_member_counter(member, team), COST)
        assert refused(call(gateway, model, key, max_tokens=TOKENS)) == budget_message(
            f"TeamMember={member}:{team}", "0.02", "0.01"
        )


def test_a_level_refusal_never_reaches_the_provider_whichever_level_refuses(gateway: Gateway) -> None:
    """After a team and organization have both been spent by one call, three more calls of that key are refused
    and the provider sees none of them: refusing on the first level must not let the request run on to a level
    that would have admitted it."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        organization: Final = scenario.organization(models=[model], max_budget=SMALL)
        team: Final = scenario.team(organization_id=organization, models=[model], max_budget=SMALL)
        key: Final = scenario.key(team_id=team, models=[model])
        spend_once(gateway, model, key)
        counter_settles(cache, org_counter(organization), COST)
        provider_calls(wire)
        for _ in range(3):
            refused(call(gateway, model, key, max_tokens=TOKENS))
        assert provider_calls(wire) == ()
        assert float(cache.get(team_counter(team)) or 0) == pytest.approx(COST)
