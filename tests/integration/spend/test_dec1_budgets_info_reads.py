"""Contract: the budget fields and spend that the management read routes report after calls.

The plain `spend` numbers on `/key/info`, `/team/info`, `/user/info` and `/customer/info` are pinned in
`test_dec1_spend_entity_increments.py`. This file pins the budget configuration that comes back next to them
(`max_budget`, `soft_budget`, `budget_duration`, `budget_reset_at`, the budget row of customers, organizations
and tags, the team member's budget and spend) and that a failed call or a refused call shows no spend. These are
what the admin UI's budget gauges read, and the Rust auth object stores must keep them in the same shape.

One served call costs 0.02 (`chat_model` price).
"""

from __future__ import annotations

from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import Gateway, eventually, held, list_value, object_value
from tests.integration.spend.dec1_budgets_support import held_provider, hold_model
from tests.integration.spend.dec1_spend_support import (
    CHAT_COST,
    LANDING_SECONDS,
    TINY_BUDGET,
    ask,
    chat_model,
    hashed,
    new_customer,
    new_tag,
    scripted_upstream,
    served,
    spend_of,
)

pytestmark: Final = pytest.mark.shares_worker_state


def _number(value: JsonValue) -> float:
    assert isinstance(value, (int, float)) and not isinstance(value, bool), value
    return float(value)


def test_key_info_reports_the_budget_settings_and_the_reset_time(gateway: Gateway) -> None:
    """A key is created with `max_budget` 1.0, `soft_budget` 0.5 and `budget_duration` `30d`, and makes one call.
    `GET /key/info` reports `info.max_budget` 1.0, `info.soft_budget` 0.5, `info.budget_duration` `30d`, a
    non-null `info.budget_reset_at` and, once the call is applied, `info.spend` 0.02."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=1.0, soft_budget=0.5, budget_duration="30d")
        served(ask(gateway, model, key))
        info: Final = eventually(
            lambda: object_value(gateway.get("/key/info", {"key": hashed(key)})["info"]),
            lambda found: _number(found["spend"] or 0.0) == pytest.approx(CHAT_COST),
            seconds=LANDING_SECONDS,
        )
        assert (info["max_budget"], info["soft_budget"], info["budget_duration"]) == (1.0, 0.5, "30d"), info
        assert info["budget_reset_at"] is not None, info


def test_team_info_reports_the_members_budget_and_spend(gateway: Gateway) -> None:
    """A member is added to a team with `max_budget_in_team` 0.5 and makes one call with a team key. `GET
    /team/info` lists that member in `team_memberships` with `spend` 0.02 and `litellm_budget_table.max_budget`
    0.5, and `team_info.spend` 0.02. UNVERIFIED: assumes `/team/member_add` accepts `max_budget_in_team` at the
    top level and creates a budget row for the membership."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        team: Final = scenario.team(models=[model])
        user: Final = scenario.user(user_role="internal_user")
        gateway.post(
            "/team/member_add",
            {"team_id": team, "member": {"role": "user", "user_id": user}, "max_budget_in_team": 0.5},
        )
        key: Final = scenario.key(user_id=user, team_id=team, models=[model])
        served(ask(gateway, model, key))

        def membership() -> dict[str, JsonValue]:
            body: Final = gateway.get("/team/info", {"team_id": team})
            rows: Final = [object_value(row) for row in list_value(body["team_memberships"])]
            return next(row for row in rows if row["user_id"] == user)

        row: Final = eventually(
            membership,
            lambda found: _number(found["spend"] or 0.0) == pytest.approx(CHAT_COST),
            seconds=LANDING_SECONDS,
        )
        assert object_value(row["litellm_budget_table"])["max_budget"] == 0.5, row
        team_info: Final = object_value(gateway.get("/team/info", {"team_id": team})["team_info"])
        assert _number(team_info["spend"]) == pytest.approx(CHAT_COST), team_info


def test_customer_info_reports_the_budget_row_and_spend(gateway: Gateway) -> None:
    """A customer created with `max_budget` 1.0 and `budget_duration` `1d` is named in one call. `GET
    /customer/info?end_user_id=<id>` reports `spend` 0.02 and `litellm_budget_table.max_budget` 1.0 and
    `litellm_budget_table.budget_duration` `1d`."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        customer: Final = new_customer(gateway, scenario, max_budget=1.0, budget_duration="1d")
        key: Final = scenario.key(models=[model])
        served(ask(gateway, model, key, user=customer))
        body: Final = eventually(
            lambda: gateway.get("/customer/info", {"end_user_id": customer}),
            lambda found: _number(found["spend"] or 0.0) == pytest.approx(CHAT_COST),
            seconds=LANDING_SECONDS,
        )
        budget: Final = object_value(body["litellm_budget_table"])
        assert (budget["max_budget"], budget["budget_duration"]) == (1.0, "1d"), body


def test_organization_info_reports_the_budget_row_and_spend(gateway: Gateway) -> None:
    """An organization with `max_budget` 1.0 has a team whose key makes one call. `GET
    /organization/info?organization_id=<id>` reports `spend` 0.02 and `litellm_budget_table.max_budget` 1.0."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        organization: Final = scenario.organization(max_budget=1.0, models=[model])
        team: Final = scenario.team(organization_id=organization, models=[model])
        key: Final = scenario.key(team_id=team, models=[model])
        served(ask(gateway, model, key))
        body: Final = eventually(
            lambda: gateway.get("/organization/info", {"organization_id": organization}),
            lambda found: _number(found["spend"] or 0.0) == pytest.approx(CHAT_COST),
            seconds=LANDING_SECONDS,
        )
        assert object_value(body["litellm_budget_table"])["max_budget"] == 1.0, body


def test_tag_info_reports_the_tag_budget(gateway: Gateway) -> None:
    """A tag created with `max_budget` 1.0 is carried by one call. `POST /tag/info` with `names: [<tag>]` returns
    the tag with its budget (`litellm_budget_table.max_budget` 1.0) and, once applied, `spend` 0.02.
    UNVERIFIED: the response is assumed to be a mapping from tag name to the tag object."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        tag: Final = new_tag(gateway, scenario, max_budget=1.0)
        key: Final = scenario.key(models=[model])
        served(ask(gateway, model, key, metadata={"tags": [tag]}))
        body: Final = eventually(
            lambda: object_value(gateway.post("/tag/info", {"names": [tag]})[tag]),
            lambda found: _number(found.get("spend") or 0.0) == pytest.approx(CHAT_COST),
            seconds=LANDING_SECONDS,
        )
        assert object_value(body["litellm_budget_table"])["max_budget"] == 1.0, body


def test_a_refused_call_adds_no_spend_to_the_key_or_its_team(gateway: Gateway) -> None:
    """A key with `max_budget` 0.0000001 in a team spends 0.02 once and is then refused three times. Key and team
    spend stay at exactly 0.02 for the following three seconds: a refusal is never charged."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        team: Final = scenario.team(models=[model])
        key: Final = scenario.key(team_id=team, models=[model], max_budget=TINY_BUDGET)
        served(ask(gateway, model, key))
        eventually(lambda: spend_of(gateway, "key", hashed(key)), lambda s: s == pytest.approx(CHAT_COST), seconds=70)
        assert [ask(gateway, model, key).status_code for _ in range(3)] == [422] * 3
        held(
            lambda: (spend_of(gateway, "key", hashed(key)), spend_of(gateway, "team", team)),
            lambda pair: pair[0] == pytest.approx(CHAT_COST) and pair[1] == pytest.approx(CHAT_COST),
            holding=3,
            seconds=LANDING_SECONDS,
        )


def test_a_failed_provider_call_shows_no_spend_on_key_info(gateway: Gateway) -> None:
    """A budgeted key's call fails at the provider (HTTP 500). `GET /key/info` keeps reporting `info.spend` 0 for
    the next three seconds: neither the admission hold nor the failure is turned into spend."""
    with held_provider(failing_status=500) as provider, gateway.scenario() as scenario:
        model: Final = hold_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=1.0)
        response: Final = ask(gateway, model, key, max_tokens=10)
        assert response.status_code >= 500, response.text
        held(
            lambda: _number(object_value(gateway.get("/key/info", {"key": hashed(key)})["info"])["spend"] or 0.0),
            lambda spend: spend == 0.0,
            holding=3,
        )
