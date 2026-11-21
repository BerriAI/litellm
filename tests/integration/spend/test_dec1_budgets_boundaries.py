"""Contract: what happens when an entity's spend lands exactly on its budget, and the refusal wording of the
entities and windows the other contract files do not pin.

The auth budget checks now decide in Rust (`litellm/rust/crates/callbacks-budgets/src/checks.rs::decide`). They
do not all compare the same way, and the difference is visible when spend equals the budget:

    refused at spend >= budget  key, key window, team window, user window, user, organization, project,
                                team member, model access group
    refused at spend >  budget  team, tag, end user, the proxy-wide budget

An entity of the second group with spend exactly at its budget passes auth, but the admission hold that follows
(`engine.rs::reserve`) finds no room left and refuses with its own wording: the current cost it reports is the
counter plus this request's estimate. Both refusals are HTTP 422 `budget_exceeded`, only the message differs.
A Rust port that unifies the comparisons, or that drops one of the two stages, changes these messages.

Every call here uses the `hold_model` price (input free, output 0.004 a token) with `max_tokens: 10`, so each
served call costs 0.02 and each estimate is exactly 0.04. A budget of 0.02 is used up by exactly one call.
"""

from __future__ import annotations

import uuid
from typing import Final

import pytest

from tests.integration._support.client import Gateway, Scenario
from tests.integration._support.wire import Wire
from tests.integration.spend.dec1_budgets_support import (
    CHAT_COST,
    budget_message,
    call,
    counter,
    counter_settles,
    hashed,
    hold_model,
    masked,
    redis_client,
    refused,
)
from tests.integration.spend.dec1_spend_support import (
    new_customer,
    new_tag,
    provider_calls,
    scripted_upstream,
    served,
    spend_reaches,
)

pytestmark: Final = pytest.mark.shares_worker_state

EXACT: Final = 0.02
TOKENS: Final = 10


def _spend_once(gateway: Gateway, model: str, key: str, **extra: object) -> None:
    served(call(gateway, model, key, max_tokens=TOKENS, **extra))  # type: ignore[arg-type]


def _model(scenario: Scenario, wire: Wire) -> str:
    return hold_model(scenario, wire)


def test_a_key_whose_spend_equals_its_budget_is_refused_by_auth_naming_key_and_masked_secret(
    gateway: Gateway,
) -> None:
    """A key without an alias has `max_budget` 0.02. One call spends exactly 0.02. The next call is refused by
    the key check (`spend >= max_budget`) with exactly `Budget has been exceeded! Key=key (sk-...<last 4>)
    Current cost: 0.02, Max budget: 0.02`: with no alias the label is the literal word `key`, followed by the
    stored masked key name, and never the secret or its hash."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = _model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=EXACT)
        _spend_once(gateway, model, key)
        counter_settles(cache, f"spend:key:{hashed(key)}", EXACT)
        message: Final = refused(call(gateway, model, key, max_tokens=TOKENS))
        assert message == budget_message(f"Key=key ({masked(key)})", "0.02", "0.02")
        assert key not in message and hashed(key) not in message


def test_a_key_refusal_uses_the_alias_when_the_alias_is_not_empty(gateway: Gateway) -> None:
    """The same exhausted key but created with `key_alias`. The label becomes `Key=<alias> (sk-...<last 4>)`.
    An empty string alias would fall back to `key` (`checks.rs`, `filter(|alias| !alias.is_empty())`)."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = _model(scenario, wire)
        alias: Final = f"dec1-budgets-alias-{uuid.uuid4().hex}"
        key: Final = scenario.key(models=[model], max_budget=EXACT, key_alias=alias)
        _spend_once(gateway, model, key)
        counter_settles(cache, f"spend:key:{hashed(key)}", EXACT)
        assert refused(call(gateway, model, key, max_tokens=TOKENS)) == budget_message(
            f"Key={alias} ({masked(key)})", "0.02", "0.02"
        )


def test_a_team_at_exactly_its_budget_passes_auth_and_is_refused_by_the_admission_hold(gateway: Gateway) -> None:
    """A team has `max_budget` 0.02 and a key with no budget of its own. After one call the team counter reads
    exactly 0.02. The team check only refuses above the budget, so the next call reaches admission, where the
    0.04 estimate finds no room and is refused with exactly `Budget has been exceeded! Team=<team_id> Current
    cost: 0.06, Max budget: 0.02`. Afterwards the team counter is back at 0.02 (the refused hold is released) and
    the provider saw only the first call."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = _model(scenario, wire)
        team: Final = scenario.team(max_budget=EXACT, models=[model])
        key: Final = scenario.key(team_id=team, models=[model])
        _spend_once(gateway, model, key)
        counter_settles(cache, f"spend:team:{team}", EXACT)
        assert len(provider_calls(wire)) == 1
        message: Final = refused(call(gateway, model, key, max_tokens=TOKENS))
        assert message == budget_message(f"Team={team}", "0.06", "0.02")
        assert counter(cache, f"spend:team:{team}") == pytest.approx(EXACT)
        assert provider_calls(wire) == ()


def test_a_team_above_its_budget_is_refused_by_auth_with_its_stored_spend(gateway: Gateway) -> None:
    """A team has `max_budget` 0.01 and spends 0.02 in one call (the hold was shrunk to 0.01, the real cost is
    still charged). The next call is refused by the team check itself, which reports the counter without any
    estimate: exactly `Budget has been exceeded! Team=<team_id> Current cost: 0.02, Max budget: 0.01`."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = _model(scenario, wire)
        team: Final = scenario.team(max_budget=0.01, models=[model])
        key: Final = scenario.key(team_id=team, models=[model])
        _spend_once(gateway, model, key)
        counter_settles(cache, f"spend:team:{team}", CHAT_COST)
        assert refused(call(gateway, model, key, max_tokens=TOKENS)) == budget_message(f"Team={team}", "0.02", "0.01")


def test_a_tag_at_exactly_its_budget_is_refused_by_the_admission_hold(gateway: Gateway) -> None:
    """A tag has `max_budget` 0.02. One call carrying `metadata.tags: [<tag>]` spends 0.02. The tag check only
    refuses above the budget, so the next tagged call is refused at admission with exactly `Budget has been
    exceeded! Tag=<tag> Current cost: 0.06, Max budget: 0.02`. An untagged call by the same key is still served."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = _model(scenario, wire)
        tag: Final = new_tag(gateway, scenario, max_budget=EXACT)
        key: Final = scenario.key(models=[model])
        _spend_once(gateway, model, key, metadata={"tags": [tag]})
        counter_settles(cache, f"spend:tag:{tag}", EXACT)
        message: Final = refused(call(gateway, model, key, max_tokens=TOKENS, metadata={"tags": [tag]}))
        assert message == budget_message(f"Tag={tag}", "0.06", "0.02")
        served(call(gateway, model, key, max_tokens=TOKENS))


def test_an_end_user_at_exactly_its_budget_is_refused_by_the_admission_hold(gateway: Gateway) -> None:
    """A customer has `max_budget` 0.02. One call naming it in the body `user` field spends 0.02. The end user
    check only refuses above the budget, so the next call for that customer is refused at admission with exactly
    `Budget has been exceeded! EndUser=<customer id> Current cost: 0.06, Max budget: 0.02`, a different family
    than the `ExceededBudget: End User=...` wording the auth check uses once spend is above the budget."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = _model(scenario, wire)
        customer: Final = new_customer(gateway, scenario, max_budget=EXACT)
        key: Final = scenario.key(models=[model])
        _spend_once(gateway, model, key, user=customer)
        counter_settles(cache, f"spend:end_user:{customer}", EXACT)
        message: Final = refused(call(gateway, model, key, max_tokens=TOKENS, user=customer))
        assert message == budget_message(f"EndUser={customer}", "0.06", "0.02")


def test_an_end_user_above_its_budget_is_refused_by_auth_with_the_end_user_wording(gateway: Gateway) -> None:
    """A customer has `max_budget` 0.01 and spends 0.02 in one call. The next call for it is refused by the auth
    check with exactly `ExceededBudget: End User=<customer id> over budget. Spend=0.02, Budget=0.01`.
    UNVERIFIED: assumes the end user object auth reads carries the counter value 0.02 rather than a stale 0.0;
    `auth_checks.py:2044` passes the object's spend and the counter path may differ."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = _model(scenario, wire)
        customer: Final = new_customer(gateway, scenario, max_budget=0.01)
        key: Final = scenario.key(models=[model])
        _spend_once(gateway, model, key, user=customer)
        spend_reaches(gateway, "end_user", customer, CHAT_COST)
        assert (
            refused(call(gateway, model, key, max_tokens=TOKENS, user=customer))
            == f"ExceededBudget: End User={customer} over budget. Spend=0.02, Budget=0.01"
        )


def test_an_organization_at_exactly_its_budget_is_refused_by_auth(gateway: Gateway) -> None:
    """An organization has `max_budget` 0.02 and one team key spends 0.02. The organization check refuses at
    equality (`spend >= max_budget`), so the next call is refused by auth with exactly `Budget has been exceeded!
    Organization=<org_id> Current cost: 0.02, Max budget: 0.02`, not by the admission hold."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = _model(scenario, wire)
        organization: Final = scenario.organization(max_budget=EXACT, models=[model])
        team: Final = scenario.team(organization_id=organization, models=[model])
        key: Final = scenario.key(team_id=team, models=[model])
        _spend_once(gateway, model, key)
        counter_settles(cache, f"spend:org:{organization}", EXACT)
        assert refused(call(gateway, model, key, max_tokens=TOKENS)) == budget_message(
            f"Organization={organization}", "0.02", "0.02"
        )


def test_a_user_at_exactly_its_budget_is_refused_with_the_user_wording(gateway: Gateway) -> None:
    """An internal user has `max_budget` 0.02 and a personal key (no team). After one call the user's next call
    is refused at equality with exactly `ExceededBudget: User=<user_id> over budget. Spend=0.02, Budget=0.02`."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = _model(scenario, wire)
        user: Final = scenario.user(max_budget=EXACT)
        key: Final = scenario.key(user_id=user, models=[model])
        _spend_once(gateway, model, key)
        counter_settles(cache, f"spend:user:{user}", EXACT)
        assert (
            refused(call(gateway, model, key, max_tokens=TOKENS))
            == f"ExceededBudget: User={user} over budget. Spend=0.02, Budget=0.02"
        )


def test_a_project_over_its_budget_refuses_its_keys_with_the_project_wording(gateway: Gateway) -> None:
    """A project (inside a team) has `max_budget` 0.02 and a key bound to it with `project_id`. After one call the
    project counter `spend:project:<project_id>` reads 0.02 and the next call is refused with exactly `Budget has
    been exceeded! Project=<project_id> Current cost: 0.02, Max budget: 0.02`. Projects are the newest budgeted
    entity and have no other refusal test.
    UNVERIFIED: assumes `/key/generate` accepts `project_id` for a key of the project's team and that
    `/project/new` stores `max_budget` in the project's budget row (`NewProjectRequest` extends the budget table)."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = _model(scenario, wire)
        team: Final = scenario.team(models=[model])
        project: Final = scenario.project(team, max_budget=EXACT, models=[model])
        key: Final = scenario.key(team_id=team, project_id=project, models=[model])
        _spend_once(gateway, model, key)
        counter_settles(cache, f"spend:project:{project}", EXACT)
        assert refused(call(gateway, model, key, max_tokens=TOKENS)) == budget_message(
            f"Project={project}", "0.02", "0.02"
        )


def test_a_project_budget_is_shared_by_every_key_of_the_project(gateway: Gateway) -> None:
    """Two keys are bound to the same project with `max_budget` 0.02. The first key spends 0.02, then the second
    key, which never made a call, is refused with the project wording. UNVERIFIED: same assumptions as the test
    above."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = _model(scenario, wire)
        team: Final = scenario.team(models=[model])
        project: Final = scenario.project(team, max_budget=EXACT, models=[model])
        first: Final = scenario.key(team_id=team, project_id=project, models=[model])
        second: Final = scenario.key(team_id=team, project_id=project, models=[model])
        _spend_once(gateway, model, first)
        counter_settles(cache, f"spend:project:{project}", EXACT)
        assert refused(call(gateway, model, second, max_tokens=TOKENS)).startswith(
            f"Budget has been exceeded! Project={project} "
        )


def test_a_team_budget_window_refuses_with_the_team_window_wording(gateway: Gateway) -> None:
    """A team has no `max_budget` but one window, `budget_limits: [{budget_duration: "1d", max_budget: 0.01}]`.
    After one call of 0.02 the team key is refused with exactly `ExceededBudget: Team=<team_id> over 1d budget.
    Spend=$0.0200, Limit=$0.01` (spend with 4 decimals, limit with 2). The window counter is
    `spend:team:<team_id>:window:1d`."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = _model(scenario, wire)
        team: Final = scenario.team(models=[model], budget_limits=[{"budget_duration": "1d", "max_budget": 0.01}])
        key: Final = scenario.key(team_id=team, models=[model])
        _spend_once(gateway, model, key)
        counter_settles(cache, f"spend:team:{team}:window:1d", CHAT_COST)
        assert (
            refused(call(gateway, model, key, max_tokens=TOKENS))
            == f"ExceededBudget: Team={team} over 1d budget. Spend=$0.0200, Limit=$0.01"
        )


def test_a_user_budget_window_refuses_with_the_user_window_wording(gateway: Gateway) -> None:
    """Since #51763 internal users carry `budget_limits` too. A user with `budget_limits: [{budget_duration:
    "1h", max_budget: 0.01}]` spends 0.02 with a personal key; the next call is refused with exactly
    `ExceededBudget: User=<user_id> over 1h budget. Spend=$0.0200, Limit=$0.01`, read from the counter
    `spend:user_window:<user_id>:1h`."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = _model(scenario, wire)
        user: Final = scenario.user(budget_limits=[{"budget_duration": "1h", "max_budget": 0.01}])
        key: Final = scenario.key(user_id=user, models=[model])
        _spend_once(gateway, model, key)
        counter_settles(cache, f"spend:user_window:{user}:1h", CHAT_COST)
        assert (
            refused(call(gateway, model, key, max_tokens=TOKENS))
            == f"ExceededBudget: User={user} over 1h budget. Spend=$0.0200, Limit=$0.01"
        )


def test_a_user_window_does_not_apply_to_the_users_team_key_by_default(gateway: Gateway) -> None:
    """The same user window exhausted by a personal key does not stop the user's team key, because user budgets
    (and, since #51763, user windows) only apply to team keys when `apply_user_budget_to_team_keys` is set
    (`budget_reservation.py`, `not is_team_key or apply_user_budget_to_team_keys`)."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = _model(scenario, wire)
        user: Final = scenario.user(budget_limits=[{"budget_duration": "1h", "max_budget": 0.01}])
        team: Final = scenario.team(models=[model])
        gateway.post("/team/member_add", {"team_id": team, "member": {"role": "user", "user_id": user}})
        personal: Final = scenario.key(user_id=user, models=[model])
        team_key: Final = scenario.key(user_id=user, team_id=team, models=[model])
        _spend_once(gateway, model, personal)
        counter_settles(cache, f"spend:user_window:{user}:1h", CHAT_COST)
        assert refused(call(gateway, model, personal, max_tokens=TOKENS)).startswith(f"ExceededBudget: User={user} ")
        served(call(gateway, model, team_key, max_tokens=TOKENS))


def test_the_tightest_of_two_key_windows_refuses_first(gateway: Gateway) -> None:
    """A key carries two windows, `1h` with 0.01 and `30d` with 1.0. After one call of 0.02 the hour window is
    used up and the refusal names it: `ExceededBudget: Key over 1h budget. Spend=$0.0200, Limit=$0.01`. Each
    window has its own counter (`spend:key:<hash>:window:1h` and `...:window:30d`), both at 0.02."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = _model(scenario, wire)
        key: Final = scenario.key(
            models=[model],
            budget_limits=[
                {"budget_duration": "1h", "max_budget": 0.01},
                {"budget_duration": "30d", "max_budget": 1.0},
            ],
        )
        _spend_once(gateway, model, key)
        counter_settles(cache, f"spend:key:{hashed(key)}:window:1h", CHAT_COST)
        counter_settles(cache, f"spend:key:{hashed(key)}:window:30d", CHAT_COST)
        assert (
            refused(call(gateway, model, key, max_tokens=TOKENS))
            == "ExceededBudget: Key over 1h budget. Spend=$0.0200, Limit=$0.01"
        )


def test_a_team_budget_refuses_before_a_key_budget_that_still_has_room(gateway: Gateway) -> None:
    """A key with `max_budget` 1.0 belongs to a team with `max_budget` 0.01. After one call of 0.02 the refusal
    names the team (`Team=<team_id>`), not the key: the key's own room never overrides a smaller team budget."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = _model(scenario, wire)
        team: Final = scenario.team(max_budget=0.01, models=[model])
        key: Final = scenario.key(team_id=team, models=[model], max_budget=1.0)
        _spend_once(gateway, model, key)
        counter_settles(cache, f"spend:team:{team}", CHAT_COST)
        assert refused(call(gateway, model, key, max_tokens=TOKENS)).startswith(
            f"Budget has been exceeded! Team={team} "
        )


def test_a_budget_refusal_does_not_change_any_counter(gateway: Gateway) -> None:
    """A key with `max_budget` 0.02 inside a team with `max_budget` 1.0 spends 0.02. The next call is refused by
    the key check. Neither `spend:key:<hash>` nor `spend:team:<team_id>` moves (both stay 0.02), so a stream of
    refused requests can never eat into the team's budget."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = _model(scenario, wire)
        team: Final = scenario.team(max_budget=1.0, models=[model])
        key: Final = scenario.key(team_id=team, models=[model], max_budget=EXACT)
        _spend_once(gateway, model, key)
        counter_settles(cache, f"spend:team:{team}", EXACT)
        counter_settles(cache, f"spend:key:{hashed(key)}", EXACT)
        for _ in range(3):
            refused(call(gateway, model, key, max_tokens=TOKENS))
        assert counter(cache, f"spend:team:{team}") == pytest.approx(EXACT)
        assert counter(cache, f"spend:key:{hashed(key)}") == pytest.approx(EXACT)


def test_a_default_team_member_budget_refuses_a_member_who_used_it_up(gateway: Gateway) -> None:
    """A team is created with `team_member_budget` 0.01, the default budget every member gets. A member's key
    spends 0.02, then the member's next call is refused with HTTP 422 and a message that names both the user and
    the team. Another member of the same team is still served.
    UNVERIFIED: which of the two team member wordings is used (`User=<u> in Team=<t>` from `team_member`, or
    `TeamMember=<u>:<t>` from `team_member_positive`, `checks.rs`) depends on which auth path loads the
    membership first; the test accepts either."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = _model(scenario, wire)
        team: Final = scenario.team(models=[model], team_member_budget=0.01)
        spender: Final = scenario.member(team)
        other: Final = scenario.member(team)
        spender_key: Final = scenario.key(user_id=spender, team_id=team, models=[model])
        other_key: Final = scenario.key(user_id=other, team_id=team, models=[model])
        _spend_once(gateway, model, spender_key)
        spend_reaches(gateway, "key", hashed(spender_key), CHAT_COST)
        message: Final = refused(call(gateway, model, spender_key, max_tokens=TOKENS))
        assert message in (
            budget_message(f"User={spender} in Team={team}", "0.02", "0.01"),
            budget_message(f"TeamMember={spender}:{team}", "0.02", "0.01"),
        ), message
        served(call(gateway, model, other_key, max_tokens=TOKENS))
