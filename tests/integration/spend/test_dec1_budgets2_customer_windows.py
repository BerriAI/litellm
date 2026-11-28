"""Contract: customer hierarchy budget windows (customer organization, customer team, customer member).

A request names its customer member with `x-litellm-customer-member-id` (or only an organization with
`x-litellm-customer-org-alias`). Auth loads the member row with its team and organization
(`litellm/proxy/auth/customer_hierarchy_auth.py::customer_hierarchy_from_member_row`, #51489) and builds up to three
scopes, organization first: `customer_org`, the member's first team as `customer_team` (or the team named by
`x-litellm-customer-team-id`), and `customer_member` with its effective budget (the organization's template under
any unexpired override, #51582).

Each scope's `budget_limits` (durations `1mo` and `12mo` only, `litellm/models/budget.py::BudgetLimit`) gets its own
Redis counter owned by the entity, not by the shared template row
(`budget_reservation.py::customer_window_counters`):

    spend:<customer_org|customer_team|customer_member>:<quoted id>:window:<duration>@<window start, UTC isoformat>

Before the provider is called the Rust engine adds the request's hold to every one of those counters, and the call
settles them to the real cost like any other counter. Auth refuses a request once a window counter (holds included)
reaches its limit, checking organization, then team, then member, with HTTP 422, `error.type` `budget_exceeded` and

    ExceededBudget: <scope>=<entity id> over <duration> budget.<hint>

where the hint is ` Spend resumes when the budget window resets.` unless the scope's budget has `cap_policy:
pause_allow_increase`, which names `POST /management/v1/customer_budget_increase_requests` instead.

Price: input free, output 0.004 a token, one served call 0.02 (`dec1_budgets_support.hold_model`).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import pytest

from tests.integration._support.client import Gateway, eventually, held
from tests.integration._support.database import write_rows
from tests.integration.spend.dec1_budgets2_support import (
    FAST_RESET,
    INCREASE_HINT,
    ORG_ALIAS_HEADER,
    RESET_HINT,
    TEAM_HEADER,
    customer,
    customer_window_keys,
    error_message,
    owned,
)
from tests.integration.spend.dec1_budgets_support import (
    CHAT_COST,
    LANDING_SECONDS,
    background,
    held_provider,
    hold_model,
    redis_client,
    refused,
)
from tests.integration.spend.dec1_spend_support import scripted_upstream, served

pytestmark: Final = pytest.mark.shares_worker_state


def _window(max_budget: float, duration: str = "1mo", **extra: object) -> dict[str, object]:
    return {"budget_limits": [{"budget_duration": duration, "max_budget": max_budget}], **extra}


def _month_start() -> str:
    now: Final = datetime.now(timezone.utc)
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()


def _settles(cache: object, scope: str, entity_id: str, amount: float) -> dict[str, float]:
    return eventually(
        lambda: customer_window_keys(cache, scope, entity_id),  # type: ignore[arg-type]
        lambda found: len(found) == 1 and abs(next(iter(found.values())) - amount) < 1e-9,
        seconds=LANDING_SECONDS,
    )


@pytest.mark.timeout(180)
def test_a_served_member_call_charges_one_window_counter_per_customer_level(gateway: Gateway) -> None:
    """Organization, team and member each have a `1mo` window of 1.0. One call with `max_tokens: 5` sent with
    `x-litellm-customer-member-id` is served, and afterwards Redis holds exactly one counter
    `spend:customer_org:<org>:window:1mo@<start>`, one `spend:customer_team:...` and one `spend:customer_member:...`,
    each settled at 0.02. A port that charges only the member, or one shared counter, fails here."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        tree: Final = customer(gateway, scenario).create(
            org_budget=_window(1.0), team_budget=_window(1.0), member_template=_window(1.0)
        )
        served(tree.chat(model, key))
        _settles(cache, "customer_org", tree.org_id, CHAT_COST)
        _settles(cache, "customer_team", tree.team_id, CHAT_COST)
        member: Final = _settles(cache, "customer_member", tree.member_id, CHAT_COST)
        assert list(member.values()) == [pytest.approx(CHAT_COST)]


@pytest.mark.timeout(180)
def test_the_window_counter_key_ends_with_the_utc_start_of_the_current_month(gateway: Gateway) -> None:
    """With the default reset settings (UTC, midnight) the member's `1mo` window counter is named
    `spend:customer_member:<id>:window:1mo@<first of this month>T00:00:00+00:00`, the start of the calendar window
    (`types.py::customer_window_counter`). The start is part of the name so a new month starts a fresh counter
    without anyone deleting the old one."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        tree: Final = customer(gateway, scenario).create(
            org_budget=None, team_budget=None, member_template=_window(1.0)
        )
        served(tree.chat(model, key))
        found: Final = _settles(cache, "customer_member", tree.member_id, CHAT_COST)
        assert list(found) == [f"spend:customer_member:{tree.member_id}:window:1mo@{_month_start()}"]


@pytest.mark.timeout(180)
def test_an_annual_window_counter_starts_on_january_first(gateway: Gateway) -> None:
    """A member template with a `12mo` window of 1.0 gets a counter whose name ends with
    `@<this year>-01-01T00:00:00+00:00` (annual calendar resets, #51508). A port that counts twelve months back from
    now would name a different window and lose the year's spend at every restart."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        tree: Final = customer(gateway, scenario).create(
            org_budget=None, team_budget=None, member_template=_window(1.0, "12mo")
        )
        served(tree.chat(model, key))
        found: Final = eventually(
            lambda: customer_window_keys(cache, "customer_member", tree.member_id, "12mo"),
            lambda keys: len(keys) == 1,
            seconds=LANDING_SECONDS,
        )
        year: Final = datetime.now(timezone.utc).year
        assert list(found) == [f"spend:customer_member:{tree.member_id}:window:12mo@{year}-01-01T00:00:00+00:00"]


@pytest.mark.timeout(180)
def test_an_in_flight_member_call_holds_its_estimate_on_every_customer_window(gateway: Gateway) -> None:
    """Organization, team and member each have a `1mo` window of 1.0. While a call with `max_tokens: 10` is held
    at the provider, all three window counters read exactly 0.04 (10 output tokens at 0.004); after release they
    settle at 0.02. The customer windows are part of the same Rust hold as key and team counters."""
    with held_provider() as provider, gateway.scenario() as scenario, redis_client() as cache, background() as run:
        model: Final = hold_model(scenario, provider)
        key: Final = scenario.key(models=[model])
        tree: Final = customer(gateway, scenario).create(
            org_budget=_window(1.0), team_budget=_window(1.0), member_template=_window(1.0)
        )
        in_flight: Final = run(
            lambda: gateway.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": "dec1-budgets-hold x"}], "max_tokens": 10},
                key=key,
                headers={"x-litellm-customer-member-id": tree.member_id},
            )
        )
        provider.wait_for(1)
        for scope, identity in (
            ("customer_org", tree.org_id),
            ("customer_team", tree.team_id),
            ("customer_member", tree.member_id),
        ):
            assert list(customer_window_keys(cache, scope, identity).values()) == [pytest.approx(0.04)], scope
        provider.release()
        served(in_flight.result(timeout=60))
        _settles(cache, "customer_member", tree.member_id, CHAT_COST)


@pytest.mark.timeout(180)
def test_a_member_whose_window_is_spent_is_refused_with_the_reset_hint(gateway: Gateway) -> None:
    """The member template has a `1mo` window of exactly 0.02. The first call (0.02) is served; the second is
    refused with HTTP 422, `error.type` `budget_exceeded` and exactly `ExceededBudget: customer_member=<member id>
    over 1mo budget. Spend resumes when the budget window resets.` (`check_customer_hierarchy_windows`, spend at
    the limit counts as over). The provider sees one call."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        tree: Final = customer(gateway, scenario).create(
            org_budget=None, team_budget=None, member_template=_window(CHAT_COST)
        )
        served(tree.chat(model, key))
        _settles(cache, "customer_member", tree.member_id, CHAT_COST)
        message: Final = refused(tree.chat(model, key))
        assert message == f"ExceededBudget: customer_member={tree.member_id} over 1mo budget.{RESET_HINT}"


@pytest.mark.timeout(180)
def test_a_pause_allow_increase_window_names_the_increase_request_route(gateway: Gateway) -> None:
    """Same as above, but the member template has `cap_policy: pause_allow_increase` (#51582). The refusal message
    ends with ` Request a temporary increase through POST /management/v1/customer_budget_increase_requests.`
    instead of the reset hint (`cap_policy_hint`)."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        tree: Final = customer(gateway, scenario).create(
            org_budget=None,
            team_budget=None,
            member_template=_window(CHAT_COST, cap_policy="pause_allow_increase"),
        )
        served(tree.chat(model, key))
        _settles(cache, "customer_member", tree.member_id, CHAT_COST)
        assert refused(tree.chat(model, key)) == (
            f"ExceededBudget: customer_member={tree.member_id} over 1mo budget.{INCREASE_HINT}"
        )


@pytest.mark.timeout(180)
def test_the_organization_window_is_checked_before_the_member_window(gateway: Gateway) -> None:
    """Organization and member windows are both 0.02. After one call both are spent, and the next call is refused
    naming the organization: `ExceededBudget: customer_org=<org id> over 1mo budget. Spend resumes when the budget
    window resets.` The scopes are walked organization, team, member, and the first spent one wins."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        tree: Final = customer(gateway, scenario).create(
            org_budget=_window(CHAT_COST), team_budget=None, member_template=_window(CHAT_COST)
        )
        served(tree.chat(model, key))
        _settles(cache, "customer_org", tree.org_id, CHAT_COST)
        assert (
            refused(tree.chat(model, key)) == f"ExceededBudget: customer_org={tree.org_id} over 1mo budget.{RESET_HINT}"
        )


@pytest.mark.timeout(180)
def test_two_members_on_one_template_have_independent_balances(gateway: Gateway) -> None:
    """Two members of the same team share the organization's `member` template with a `1mo` window of 0.02. The
    first member spends it and is refused; the second member, on the same template row, is still served. The
    balance belongs to the entity, not to the shared budget row (`customer_window_counters` docstring)."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        tree: Final = customer(gateway, scenario).create(
            org_budget=None, team_budget=None, member_template=_window(CHAT_COST)
        )
        second_member: Final = f"{tree.member_id}-second"
        created: Final = gateway.request(
            "POST",
            "/management/v1/customer_members",
            {"customer_member_id": second_member, "parent_team_ids": [tree.team_id]},
        )
        assert created.status_code == 201, created.text
        served(tree.chat(model, key))
        _settles(cache, "customer_member", tree.member_id, CHAT_COST)
        refused(tree.chat(model, key))
        other: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "second member"}], "max_tokens": 5},
            key=key,
            headers={"x-litellm-customer-member-id": second_member},
        )
        served(other)


@pytest.mark.timeout(180)
def test_a_spent_organization_window_refuses_every_member_of_the_organization(gateway: Gateway) -> None:
    """The organization window is 0.02 and members have a generous window of 1.0. After the first member spends
    0.02, a second member of the same organization is refused with `ExceededBudget: customer_org=<org id> over 1mo
    budget. Spend resumes when the budget window resets.` The organization window is shared by its members."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        tree: Final = customer(gateway, scenario).create(
            org_budget=_window(CHAT_COST), team_budget=None, member_template=_window(1.0)
        )
        second_member: Final = f"{tree.member_id}-second"
        created: Final = gateway.request(
            "POST",
            "/management/v1/customer_members",
            {"customer_member_id": second_member, "parent_team_ids": [tree.team_id]},
        )
        assert created.status_code == 201, created.text
        served(tree.chat(model, key))
        _settles(cache, "customer_org", tree.org_id, CHAT_COST)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "other member"}], "max_tokens": 5},
            key=key,
            headers={"x-litellm-customer-member-id": second_member},
        )
        assert refused(response) == f"ExceededBudget: customer_org={tree.org_id} over 1mo budget.{RESET_HINT}"


@pytest.mark.timeout(180)
def test_a_concurrent_member_call_is_refused_while_a_hold_fills_the_member_window(gateway: Gateway) -> None:
    """The member window is 1.0. A held call without `max_tokens` is shrunk to the whole window (counter 1.0).
    A second call for the same member, sent while the first is in flight, is refused with `ExceededBudget:
    customer_member=<id> over 1mo budget. ...`: the auth window check reads the counter holds included
    (`require_authoritative_window=True`). The first call is then served and the counter settles at 0.02."""
    with held_provider() as provider, gateway.scenario() as scenario, redis_client() as cache, background() as run:
        model: Final = hold_model(scenario, provider)
        key: Final = scenario.key(models=[model])
        tree: Final = customer(gateway, scenario).create(
            org_budget=None, team_budget=None, member_template=_window(1.0)
        )
        in_flight: Final = run(
            lambda: gateway.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": "dec1-budgets-hold y"}]},
                key=key,
                headers={"x-litellm-customer-member-id": tree.member_id},
            )
        )
        provider.wait_for(1)
        try:
            second: Final = tree.chat(model, key)
        finally:
            provider.release()
        assert refused(second) == f"ExceededBudget: customer_member={tree.member_id} over 1mo budget.{RESET_HINT}"
        served(in_flight.result(timeout=60))
        _settles(cache, "customer_member", tree.member_id, CHAT_COST)


@pytest.mark.timeout(180)
def test_naming_another_team_of_the_member_charges_that_teams_window(gateway: Gateway) -> None:
    """The member belongs to two customer teams, each with a `1mo` window of 1.0. A call with
    `x-litellm-customer-team-id: <second team>` charges 0.02 to the second team's window counter and leaves the
    first (default) team without a counter (`select_customer_team`)."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        tree: Final = customer(gateway, scenario).create(org_budget=None, team_budget=_window(1.0), member_template={})
        other_team: Final = f"{tree.team_id}-other"
        created: Final = gateway.request(
            "POST",
            "/management/v1/customer_teams",
            {"customer_team_id": other_team, "parent_org_id": tree.org_id, "budget": _window(1.0)},
        )
        assert created.status_code == 201, created.text
        moved: Final = gateway.request(
            "PATCH",
            f"/management/v1/customer_members/{tree.member_id}",
            {"parent_team_ids": [tree.team_id, other_team]},
        )
        assert moved.status_code == 200, moved.text
        served(tree.chat(model, key, **{TEAM_HEADER: other_team}))
        _settles(cache, "customer_team", other_team, CHAT_COST)
        assert customer_window_keys(cache, "customer_team", tree.team_id) == {}


@pytest.mark.timeout(180)
def test_naming_a_team_the_member_is_not_in_is_refused_403(gateway: Gateway) -> None:
    """`x-litellm-customer-team-id` names a team the member does not belong to. The request is refused with HTTP
    403 and the message `Customer member <member id> is not in customer team <team id>`, and no window counter is
    charged."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        tree: Final = customer(gateway, scenario).create(
            org_budget=None, team_budget=_window(1.0), member_template=_window(1.0)
        )
        stranger: Final = f"{tree.team_id}-stranger"
        response: Final = tree.chat(model, key, **{TEAM_HEADER: stranger})
        assert response.status_code == 403, response.text
        assert error_message(response) == f"Customer member {tree.member_id} is not in customer team {stranger}"
        assert customer_window_keys(cache, "customer_member", tree.member_id) == {}


@pytest.mark.timeout(180)
def test_an_organization_alias_alone_charges_only_the_organization_window(gateway: Gateway) -> None:
    """The organization has alias `<alias>` and a `1mo` window of 1.0; team and member have windows too. A call that
    sends only `x-litellm-customer-org-alias: <alias>` is served and charges 0.02 to the organization window alone:
    no team or member counter exists afterwards (`customer_hierarchy_from_organization_row`)."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        tree: Final = customer(gateway, scenario)
        alias: Final = f"{tree.org_id}-alias"
        tree.create(org_budget=_window(1.0), team_budget=_window(1.0), member_template=_window(1.0), alias=alias)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "org only"}], "max_tokens": 5},
            key=key,
            headers={ORG_ALIAS_HEADER: alias},
        )
        served(response)
        _settles(cache, "customer_org", tree.org_id, CHAT_COST)
        assert customer_window_keys(cache, "customer_team", tree.team_id) == {}
        assert customer_window_keys(cache, "customer_member", tree.member_id) == {}


@pytest.mark.timeout(180)
def test_an_unknown_member_id_runs_untracked_without_window_counters(gateway: Gateway) -> None:
    """A call names `x-litellm-customer-member-id` that no customer row has, and sends no organization alias. It is
    served (#51548: unknown customer ids are attribution, never a refusal) and no `spend:customer_member:<id>:window`
    counter is created for that id."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        unknown: Final = "dec1-b2-member-never-created"
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "unknown member"}], "max_tokens": 5},
            key=key,
            headers={"x-litellm-customer-member-id": unknown},
        )
        served(response)
        assert (
            held(lambda: customer_window_keys(cache, "customer_member", unknown), lambda found: found == {}, holding=3)
            == {}
        )


@pytest.mark.timeout(180)
def test_an_info_route_is_not_refused_for_a_spent_member_window(gateway: Gateway) -> None:
    """After the member's 0.02 window is spent and a chat call is refused, `GET /v1/models` with the same key and
    member header still answers 200: info routes skip the customer budget checks (`RouteChecks.is_info_route` in
    `check_customer_hierarchy_access`)."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        tree: Final = customer(gateway, scenario).create(
            org_budget=None, team_budget=None, member_template=_window(CHAT_COST)
        )
        served(tree.chat(model, key))
        _settles(cache, "customer_member", tree.member_id, CHAT_COST)
        refused(tree.chat(model, key))
        listing: Final = gateway.request(
            "GET", "/v1/models", key=key, headers={"x-litellm-customer-member-id": tree.member_id}
        )
        assert listing.status_code == 200, listing.text


@pytest.mark.timeout(180)
def test_a_failed_provider_call_releases_the_member_window_hold(gateway: Gateway) -> None:
    """The provider answers 500 to a member call with `max_tokens: 10` (hold 0.04). The caller gets an error and
    the member window counter settles back to 0, so a failed call never eats the customer's monthly window."""
    with held_provider(failing_status=500) as provider, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, provider)
        key: Final = scenario.key(models=[model])
        tree: Final = customer(gateway, scenario).create(
            org_budget=None, team_budget=None, member_template=_window(1.0)
        )
        response: Final = tree.chat(model, key, max_tokens=10)
        assert response.status_code >= 500, response.text
        eventually(
            lambda: customer_window_keys(cache, "customer_member", tree.member_id),
            lambda found: all(abs(value) < 1e-9 for value in found.values()),
            seconds=LANDING_SECONDS,
        )


@pytest.mark.timeout(180)
def test_the_settled_window_spend_is_recorded_in_the_budget_window_spend_table(gateway: Gateway) -> None:
    """After one served member call, `LiteLLM_BudgetWindowSpend` has a row with `entity_type` `customer_member`,
    `entity_id` the member id, `window_duration` `1mo` and `spend` 0.02. That row is what a lost Redis window
    counter is reseeded from (`spend_counter_reseed.py::window_from_table`). UNVERIFIED: that the customer settlement
    writes this row (read from the reseed side only); check `settle_customer_spend` in proxy_track_cost_callback.py."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        tree: Final = customer(gateway, scenario).create(
            org_budget=None, team_budget=None, member_template=_window(1.0)
        )
        served(tree.chat(model, key))
        rows: Final = eventually(
            lambda: gateway.rows(
                'SELECT spend FROM "LiteLLM_BudgetWindowSpend" WHERE entity_type = %s AND entity_id = %s '
                "AND window_duration = %s",
                ("customer_member", tree.member_id, "1mo"),
            ),
            lambda found: len(found) == 1 and float(str(found[0]["spend"])) >= CHAT_COST - 1e-9,
            seconds=LANDING_SECONDS,
        )
        assert float(str(rows[0]["spend"])) == pytest.approx(CHAT_COST)


@pytest.mark.timeout(240)
def test_a_lost_member_window_counter_is_reseeded_from_postgres(gateway: Gateway) -> None:
    """After one served call (window spend 0.02 recorded), the test deletes the member's window counter from Redis.
    The next call with `max_tokens: 5` is served and the recreated counter settles at 0.04, not 0.02: the window was
    reseeded from `LiteLLM_BudgetWindowSpend` (`coalesced_window`, strict for customer scopes) before the new hold
    was added. A port that restarts a lost window at zero lets a customer spend its window twice."""
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        tree: Final = customer(gateway, scenario).create(
            org_budget=None, team_budget=None, member_template=_window(1.0)
        )
        served(tree.chat(model, key))
        first: Final = _settles(cache, "customer_member", tree.member_id, CHAT_COST)
        eventually(
            lambda: gateway.rows(
                'SELECT spend FROM "LiteLLM_BudgetWindowSpend" WHERE entity_type = %s AND entity_id = %s',
                ("customer_member", tree.member_id),
            ),
            lambda found: len(found) == 1 and float(str(found[0]["spend"])) >= CHAT_COST - 1e-9,
            seconds=LANDING_SECONDS,
        )
        cache.delete(*first)
        served(tree.chat(model, key))
        rebuilt: Final = _settles(cache, "customer_member", tree.member_id, 2 * CHAT_COST)
        assert list(rebuilt.values()) == [pytest.approx(2 * CHAT_COST)]


@pytest.mark.timeout(300)
def test_an_uncapped_customer_organization_keeps_its_spend_across_a_scheduled_reset(
    gateway: Gateway, tmp_path: Path
) -> None:
    """#51918. On a proxy whose reset job runs every few seconds, a customer organization has a budget with
    `budget_duration: 1d` and no `max_budget`. After one member call its `spend` is 0.02. The test moves the budget
    row's `budget_reset_at` into the past; once the reset job has advanced it, the organization's `spend` still
    reads 0.02. Only capped customer budgets are zeroed by the scheduled reset."""
    with (
        scripted_upstream() as wire,
        owned(gateway, tmp_path, environment=FAST_RESET) as proxy,
        proxy.scenario() as scenario,
    ):
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        tree: Final = customer(proxy, scenario).create(
            org_budget={"budget_duration": "1d"}, team_budget=None, member_template={}
        )
        served(tree.chat(model, key))
        spend_query: Final = 'SELECT spend FROM "LiteLLM_CustomerOrganizationTable" WHERE customer_org_id = %s'
        eventually(
            lambda: proxy.rows(spend_query, (tree.org_id,)),
            lambda rows: float(str(rows[0]["spend"])) >= CHAT_COST - 1e-9,
            seconds=LANDING_SECONDS,
        )
        budget_id: Final = str(tree.read("customer_organizations", tree.org_id)["budget_id"])
        write_rows(
            "UPDATE \"LiteLLM_BudgetTable\" SET budget_reset_at = now() - interval '1 minute' WHERE budget_id = %s",
            (budget_id,),
        )
        eventually(
            lambda: proxy.rows(
                'SELECT budget_reset_at > now() AS advanced FROM "LiteLLM_BudgetTable" WHERE budget_id = %s',
                (budget_id,),
            ),
            lambda rows: rows[0]["advanced"] is True,
            seconds=LANDING_SECONDS,
        )
        assert float(str(proxy.rows(spend_query, (tree.org_id,))[0]["spend"])) == pytest.approx(CHAT_COST)


@pytest.mark.timeout(300)
def test_a_capped_customer_organization_is_zeroed_by_its_scheduled_reset(gateway: Gateway, tmp_path: Path) -> None:
    """The counterpart of the test above: the organization's budget has `budget_duration: 1d` and `max_budget`
    10. After one call and a `budget_reset_at` moved into the past, the reset job sets the organization's `spend`
    back to 0. UNVERIFIED: that the reset job zeroes customer organization spend on the budget row's window (read
    from `reset_budget_job.py::_queue_budget_linked_resets` with `customer_budget_ids`)."""
    with (
        scripted_upstream() as wire,
        owned(gateway, tmp_path, environment=FAST_RESET) as proxy,
        proxy.scenario() as scenario,
    ):
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        tree: Final = customer(proxy, scenario).create(
            org_budget={"budget_duration": "1d", "max_budget": 10.0}, team_budget=None, member_template={}
        )
        served(tree.chat(model, key))
        spend_query: Final = 'SELECT spend FROM "LiteLLM_CustomerOrganizationTable" WHERE customer_org_id = %s'
        eventually(
            lambda: proxy.rows(spend_query, (tree.org_id,)),
            lambda rows: float(str(rows[0]["spend"])) >= CHAT_COST - 1e-9,
            seconds=LANDING_SECONDS,
        )
        budget_id: Final = str(tree.read("customer_organizations", tree.org_id)["budget_id"])
        write_rows(
            "UPDATE \"LiteLLM_BudgetTable\" SET budget_reset_at = now() - interval '1 minute' WHERE budget_id = %s",
            (budget_id,),
        )
        zeroed: Final = eventually(
            lambda: proxy.rows(spend_query, (tree.org_id,)),
            lambda rows: float(str(rows[0]["spend"])) == 0.0,
            seconds=LANDING_SECONDS,
        )
        assert float(str(zeroed[0]["spend"])) == 0.0


@pytest.mark.timeout(180)
def test_a_scalar_member_budget_refuses_with_the_spend_and_budget_in_the_message(gateway: Gateway) -> None:
    """The member template has `max_budget` 0.02 and no windows. After one served call, the next is refused with
    HTTP 422 and exactly `ExceededBudget: customer_member=<id> over budget. Spend=0.02, Budget=0.02. Spend resumes
    when the budget window resets.` (`check_customer_hierarchy_budget`). UNVERIFIED: the float formatting of the
    spend read back from the customer lifetime counter (`current_customer_spend`)."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        tree: Final = customer(gateway, scenario).create(
            org_budget=None, team_budget=None, member_template={"max_budget": CHAT_COST}
        )
        served(tree.chat(model, key))
        eventually(
            lambda: gateway.rows(
                'SELECT spend FROM "LiteLLM_CustomerMemberTable" WHERE customer_member_id = %s', (tree.member_id,)
            ),
            lambda rows: float(str(rows[0]["spend"])) >= CHAT_COST - 1e-9,
            seconds=LANDING_SECONDS,
        )
        assert refused(tree.chat(model, key)) == (
            f"ExceededBudget: customer_member={tree.member_id} over budget. Spend=0.02, Budget=0.02.{RESET_HINT}"
        )
