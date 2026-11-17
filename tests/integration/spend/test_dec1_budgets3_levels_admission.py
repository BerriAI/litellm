"""Contract: one request holds its estimate on every budgeted level at once, and the tightest level sizes it.

After the auth checks pass, the admission engine (`litellm/rust/crates/callbacks-budgets/src/engine.rs:276-404`)
charges the estimate to every counter the request carries, in the order `_get_budget_counters` builds them
(`litellm/proxy/spend_tracking/budget_reservation.py:494-636`): customer windows, key, key windows, team, team
windows, user windows, user, end user, tags, model access groups, team member, organization, project.

When every counter has room the whole estimate is charged to all of them in one batch. Otherwise each counter is
charged in that order; the first one that overflows either shrinks the hold of every counter charged so far to
its own remaining room (`over_budget`, `apply_actual(..., remaining)`), or, when it has no room at all, refuses
with its own label and releases what was charged (`release_best_effort`). So the levels share one hold size and
the first level in plan order with no room is the one named. Earlier rounds pinned the hold on one level at a
time; a port that sizes each counter independently, or reorders the plan, breaks these.

Price as in round 1: input free, 0.004 per output token, `max_tokens: 10` holds 0.04 and a served call costs 0.02.
"""

from __future__ import annotations

from typing import Final

import pytest

from tests.integration._support.client import Gateway
from tests.integration.spend.dec1_budgets3_support import (
    COST,
    HOLD,
    TOKENS,
    end_user_counter,
    levels,
    org_counter,
    project_counter,
    spend_once,
    tag_counter,
    team_counter,
)
from tests.integration.spend.dec1_budgets_support import (
    background,
    budget_message,
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

ROOMY: Final = 5.0


def test_one_held_call_holds_its_estimate_on_every_budgeted_level(gateway: Gateway) -> None:
    """A key sits under a budgeted organization, team, team member default budget, project, tag and customer
    (all 5.0, the key too). While one call with `max_tokens: 10` is held at the provider, each of the seven
    counters (`spend:key`, `spend:team`, `spend:team_member`, `spend:org`, `spend:project`, `spend:tag`,
    `spend:end_user`) reads exactly 0.04: the engine charges every planned counter in one batch when all fit.
    UNVERIFIED: the team member counter needs the membership in the reservation cache
    (`team_membership_reservation_cache_key`); if the default member budget is not loaded there, that one counter
    is absent."""
    with held_provider() as provider, gateway.scenario() as scenario, redis_client() as cache, background() as run:
        model: Final = hold_model(scenario, provider)
        tree: Final = levels(
            gateway, scenario, model, key_budget=ROOMY, team_budget=ROOMY, member_budget=ROOMY,
            org_budget=ROOMY, project_budget=ROOMY, tag_budget=ROOMY, customer_budget=ROOMY,
        )  # fmt: skip
        pending: Final = run(lambda: tree.call(hold=True))
        provider.wait_for(1)
        held: Final = {level: counter(cache, name) for level, name in tree.counters().items()}
        provider.release()
        served(pending.result(timeout=60))
        assert held == {level: pytest.approx(HOLD) for level in held}, held


def test_every_level_settles_to_the_same_real_cost(gateway: Gateway) -> None:
    """The same tree after the held call is answered: every one of the seven counters settles to 0.02, the real
    cost, because settlement applies `actual - reserved` to each entry of the one reservation
    (`engine.rs::settle`)."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        tree: Final = levels(
            gateway, scenario, model, key_budget=ROOMY, team_budget=ROOMY, member_budget=ROOMY,
            org_budget=ROOMY, project_budget=ROOMY, tag_budget=ROOMY, customer_budget=ROOMY,
        )  # fmt: skip
        served(tree.call())
        for name in tree.counters().values():
            assert counter_settles(cache, name, COST) == pytest.approx(COST)


def test_levels_without_a_budget_get_no_counter(gateway: Gateway) -> None:
    """In the same tree only the key has a budget (5.0). After one served call `spend:key:<hash>` reads 0.02 and
    none of `spend:team`, `spend:org`, `spend:project`, `spend:tag` or `spend:end_user` exists: a level is planned
    only when its `max_budget` is above 0 (`budget_reservation.py:522-636`)."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        tree: Final = levels(gateway, scenario, model, key_budget=ROOMY)
        served(tree.call())
        counter_settles(cache, key_counter(tree.key), COST)
        names: Final = tree.counters()
        assert {level: counter(cache, names[level]) for level in ("team", "org", "project", "tag", "end_user")} == {
            "team": None, "org": None, "project": None, "tag": None, "end_user": None,
        }  # fmt: skip


def test_the_tightest_level_shrinks_the_hold_on_every_level(gateway: Gateway) -> None:
    """Key, team and organization have 5.0 each, the tag only 0.03. A held call with `max_tokens: 10` (0.04) does
    not fit the tag, so the engine charges level by level and, at the tag, shrinks every entry charged so far to
    the tag's room: key, team, organization and tag counters all read 0.03 while the call is held, not 0.04 on the
    roomy levels (`engine.rs::over_budget`, `apply_actual(..., remaining)`).
    UNVERIFIED: the tag is planned before the organization (`budget_reservation.py:600-633`), so the organization
    is charged the already shrunk 0.03 after the tag; either way all four read 0.03."""
    with held_provider() as provider, gateway.scenario() as scenario, redis_client() as cache, background() as run:
        model: Final = hold_model(scenario, provider)
        tree: Final = levels(
            gateway, scenario, model, key_budget=ROOMY, team_budget=ROOMY, org_budget=ROOMY, tag_budget=0.03
        )
        pending: Final = run(lambda: tree.call(hold=True))
        provider.wait_for(1)
        names: Final = tree.counters()
        held: Final = {level: counter(cache, names[level]) for level in ("key", "team", "tag", "org")}
        provider.release()
        served(pending.result(timeout=60))
        assert held == {level: pytest.approx(0.03) for level in held}, held


def test_a_shrunk_hold_still_settles_every_level_to_the_full_real_cost(gateway: Gateway) -> None:
    """Key and team have 5.0 and the tag 0.01. One call is admitted with its hold shrunk to 0.01 and costs 0.02.
    After it, key, team and tag counters all read 0.02: shrinking only bounds the hold, the real cost is charged
    in full, so the tag ends 0.01 over its budget."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        tree: Final = levels(gateway, scenario, model, key_budget=ROOMY, team_budget=ROOMY, tag_budget=0.01)
        served(tree.call())
        names: Final = tree.counters()
        for level in ("key", "team", "tag"):
            assert counter_settles(cache, names[level], COST) == pytest.approx(COST)


def test_a_team_at_its_budget_is_named_before_a_tag_at_its_budget(gateway: Gateway) -> None:
    """A team and a tag each have `max_budget` 0.02 and one tagged call spends exactly 0.02 on both. Neither auth
    check refuses at equality (both compare `>`), so the next tagged call reaches admission, where the team is
    planned before the tag and is named: `Budget has been exceeded! Team=<team_id> Current cost: 0.06, Max
    budget: 0.02`."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        team: Final = scenario.team(models=[model], max_budget=COST)
        key: Final = scenario.key(team_id=team, models=[model])
        tag: Final = new_tag(gateway, scenario, max_budget=COST)
        spend_once(gateway, model, key, metadata={"tags": [tag]})
        counter_settles(cache, tag_counter(tag), COST)
        counter_settles(cache, team_counter(team), COST)
        assert refused(call(gateway, model, key, max_tokens=TOKENS, metadata={"tags": [tag]})) == budget_message(
            f"Team={team}", "0.06", "0.02"
        )


def test_an_end_user_at_its_budget_is_named_before_a_tag_at_its_budget(gateway: Gateway) -> None:
    """A customer and a tag each have `max_budget` 0.02 and one call spends 0.02 on both. The next call is refused
    at admission naming the end user, which is planned before tags: `Budget has been exceeded!
    EndUser=<customer> Current cost: 0.06, Max budget: 0.02`."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        customer: Final = new_customer(gateway, scenario, max_budget=COST)
        tag: Final = new_tag(gateway, scenario, max_budget=COST)
        spend_once(gateway, model, key, user=customer, metadata={"tags": [tag]})
        counter_settles(cache, end_user_counter(customer), COST)
        counter_settles(cache, tag_counter(tag), COST)
        assert refused(
            call(gateway, model, key, max_tokens=TOKENS, user=customer, metadata={"tags": [tag]})
        ) == budget_message(f"EndUser={customer}", "0.06", "0.02")


def test_an_admission_refusal_at_a_later_level_gives_back_the_hold_on_earlier_levels(gateway: Gateway) -> None:
    """A key with `max_budget` 5.0 belongs to a team with `max_budget` 0.02. After one call (both counters 0.02)
    the next call is charged to the key first and then refused at the team. Afterwards the key counter is back at
    exactly 0.02: the 0.04 already charged to the key is released by `release_best_effort` (`engine.rs:383-386`),
    so a refusal never leaves a phantom hold on a level that had room."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        team: Final = scenario.team(models=[model], max_budget=COST)
        key: Final = scenario.key(team_id=team, models=[model], max_budget=ROOMY)
        spend_once(gateway, model, key)
        counter_settles(cache, team_counter(team), COST)
        counter_settles(cache, key_counter(key), COST)
        assert refused(call(gateway, model, key, max_tokens=TOKENS))
        assert counter(cache, key_counter(key)) == pytest.approx(COST)


def test_a_hold_on_the_organization_refuses_a_key_of_another_team_in_that_organization(gateway: Gateway) -> None:
    """Two teams share an organization with `max_budget` 0.04. A key of team A holds a call with `max_tokens: 10`
    (0.04, the whole organization budget). A key of team B, whose team has no budget, is refused meanwhile by the
    organization check, which reads the counter including the hold: `Budget has been exceeded!
    Organization=<org_id> Current cost: 0.04, Max budget: 0.04` (`checks.rs` organization compares `>=`)."""
    with held_provider() as provider, gateway.scenario() as scenario, redis_client() as cache, background() as run:
        model: Final = hold_model(scenario, provider)
        organization: Final = scenario.organization(models=[model], max_budget=HOLD)
        first: Final = scenario.key(team_id=scenario.team(organization_id=organization, models=[model]), models=[model])
        second: Final = scenario.key(
            team_id=scenario.team(organization_id=organization, models=[model]), models=[model]
        )
        pending: Final = run(lambda: call(gateway, model, first, hold=True, max_tokens=TOKENS))
        provider.wait_for(1)
        counter_settles(cache, org_counter(organization), HOLD)
        message: Final = refused(call(gateway, model, second, max_tokens=TOKENS))
        provider.release()
        served(pending.result(timeout=60))
        assert message == budget_message(f"Organization={organization}", "0.04", "0.04")


def test_a_hold_on_the_project_refuses_another_key_of_the_project(gateway: Gateway) -> None:
    """Two keys share a project with `max_budget` 0.04. While the first key holds a 0.04 call, the second key is
    refused with `Budget has been exceeded! Project=<project_id> Current cost: 0.04, Max budget: 0.04`: the
    project check refuses at `>=` on the counter that already carries the hold."""
    with held_provider() as provider, gateway.scenario() as scenario, redis_client() as cache, background() as run:
        model: Final = hold_model(scenario, provider)
        team: Final = scenario.team(models=[model])
        project: Final = scenario.project(team, models=[model], max_budget=HOLD)
        first: Final = scenario.key(team_id=team, project_id=project, models=[model])
        second: Final = scenario.key(team_id=team, project_id=project, models=[model])
        pending: Final = run(lambda: call(gateway, model, first, hold=True, max_tokens=TOKENS))
        provider.wait_for(1)
        counter_settles(cache, project_counter(project), HOLD)
        message: Final = refused(call(gateway, model, second, max_tokens=TOKENS))
        provider.release()
        served(pending.result(timeout=60))
        assert message == budget_message(f"Project={project}", "0.04", "0.04")


def test_a_hold_on_a_tag_refuses_another_teams_call_with_that_tag_at_admission(gateway: Gateway) -> None:
    """A tag has `max_budget` 0.04 and keys of two unrelated teams send it. While the first holds 0.04, the second
    passes the tag auth check (`spend <= max_budget`) and is refused at admission with `Budget has been exceeded!
    Tag=<tag> Current cost: 0.08, Max budget: 0.04`: a tag budget is shared across teams and keys."""
    with held_provider() as provider, gateway.scenario() as scenario, redis_client() as cache, background() as run:
        model: Final = hold_model(scenario, provider)
        tag: Final = new_tag(gateway, scenario, max_budget=HOLD)
        first: Final = scenario.key(team_id=scenario.team(models=[model]), models=[model])
        second: Final = scenario.key(team_id=scenario.team(models=[model]), models=[model])
        tags: Final = {"tags": [tag]}
        pending: Final = run(lambda: call(gateway, model, first, hold=True, max_tokens=TOKENS, metadata=tags))
        provider.wait_for(1)
        counter_settles(cache, tag_counter(tag), HOLD)
        message: Final = refused(call(gateway, model, second, max_tokens=TOKENS, metadata=tags))
        provider.release()
        served(pending.result(timeout=60))
        assert message == budget_message(f"Tag={tag}", "0.08", "0.04")


def test_a_hold_for_an_end_user_refuses_the_same_end_user_on_another_key(gateway: Gateway) -> None:
    """A customer has `max_budget` 0.04. Key A holds a 0.04 call for it; key B calls for the same customer and is
    refused at admission with `Budget has been exceeded! EndUser=<customer> Current cost: 0.08, Max budget: 0.04`.
    UNVERIFIED: assumes the auth end user check (`>` on the cached end user object, `auth_checks.py:2044`) still
    reads 0 while the hold is in flight, so the admission wording is reported rather than the auth one."""
    with held_provider() as provider, gateway.scenario() as scenario, redis_client() as cache, background() as run:
        model: Final = hold_model(scenario, provider)
        customer: Final = new_customer(gateway, scenario, max_budget=HOLD)
        first: Final = scenario.key(models=[model])
        second: Final = scenario.key(models=[model])
        pending: Final = run(lambda: call(gateway, model, first, hold=True, max_tokens=TOKENS, user=customer))
        provider.wait_for(1)
        counter_settles(cache, end_user_counter(customer), HOLD)
        message: Final = refused(call(gateway, model, second, max_tokens=TOKENS, user=customer))
        provider.release()
        served(pending.result(timeout=60))
        assert message == budget_message(f"EndUser={customer}", "0.08", "0.04")
