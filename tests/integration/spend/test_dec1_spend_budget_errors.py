"""Contract: which budget stops a request, with which status and which exact message.

Every budget is enforced the same way from the caller's side: the request is refused before any provider is
called with HTTP 422, `error.type` `budget_exceeded`, `error.code` `"422"` and a message that names the entity
and the two numbers. Clients and dashboards parse these messages, and each entity has its own wording, read from
`litellm/proxy/auth/auth_checks.py` and `litellm/proxy/hooks/model_max_budget_limiter.py`:

    key           Budget has been exceeded! Key=<alias or "key"> ... Current cost: <spend>, Max budget: <budget>
    key window    ExceededBudget: Key over <duration> budget. Spend=$<4dp>, Limit=$<2dp>
    user          ExceededBudget: User=<user_id> over budget. Spend=<spend>, Budget=<budget>
    end user      ExceededBudget: End User=<id> over budget. Spend=<spend>, Budget=<budget>
    organization  Budget has been exceeded! Organization=<org_id> Current cost: <spend>, Max budget: <budget>
    team member   Budget has been exceeded! User=<user_id> in Team=<team_id> Current cost: ..., Max budget: ...
    key model     LiteLLM Virtual Key: <hashed key>, key_alias: <alias>, exceeded budget for model=<model>
    team model    LiteLLM Team: <team_id>, exceeded budget for model=<model>
    user model    LiteLLM User: <user_id>, exceeded budget for model=<model>

Key, team, tag and customer refusals already have tests next to this file. This file adds the entities and the
exact wordings those do not pin. One served call costs 0.02 and every budget here is far below that, so the
first call is served and the second is refused once the first has been applied.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Final

import pytest

from tests.integration._support.client import JSON_OBJECT, Gateway, eventually, object_value, string_value
from tests.integration._support.process import OwnedState, owned_proxy
from tests.integration.spend.dec1_spend_support import (
    CHAT,
    CHAT_COST,
    LANDING_SECONDS,
    SURFACES,
    TINY_BUDGET,
    Surface,
    ask,
    chat_model,
    hashed,
    new_customer,
    number,
    provider_calls,
    proxy_config,
    refusal,
    scripted_upstream,
    served,
    spend_reaches,
)

_IDS: Final = tuple(surface.name for surface in SURFACES)
_USER: Final = re.compile(r"^ExceededBudget: User=(\S+) over budget\. Spend=([^,]+), Budget=(\S+)$")
_END_USER: Final = re.compile(r"^ExceededBudget: End User=(\S+) over budget\. Spend=([^,]+), Budget=(\S+)$")
_ORGANIZATION: Final = re.compile(
    r"^Budget has been exceeded! Organization=(\S+) Current cost: ([^,]+), Max budget: (\S+)$"
)
_KEY: Final = re.compile(r"^Budget has been exceeded! Key=(.+) Current cost: ([^,]+), Max budget: (\S+)$")


def _figures(pattern: re.Pattern[str], message: str) -> tuple[str, float, float]:
    found: Final = pattern.match(message)
    assert found is not None, message
    return found[1], float(found[2]), float(found[3])


def test_user_over_personal_budget_is_refused_with_the_user_wording(gateway: Gateway) -> None:
    """A user has `max_budget` 0.0000001 and a personal key. The first call is served and applied (0.02). The
    second call is refused with 422 and exactly
    `ExceededBudget: User=<user_id> over budget. Spend=0.02, Budget=1e-07`, and the provider is not called."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        user: Final = scenario.user(max_budget=TINY_BUDGET)
        key: Final = scenario.key(models=[model], user_id=user)
        served(ask(gateway, model, key))
        spend_reaches(gateway, "user", user, CHAT_COST)
        assert len(provider_calls(wire)) == 1
        named, spent, limit = _figures(_USER, refusal(ask(gateway, model, key)))
        assert (named, spent, limit) == (user, pytest.approx(0.02), pytest.approx(TINY_BUDGET))
        assert provider_calls(wire) == ()


def test_user_personal_budget_does_not_stop_the_users_team_key_by_default(gateway: Gateway) -> None:
    """The same over-budget user also owns a key in a team. Without `apply_user_budget_to_team_keys`, the
    personal key is refused but the team key is still served, because team keys spend the team's money."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        user: Final = scenario.user(max_budget=TINY_BUDGET)
        team: Final = scenario.team(models=[model])
        gateway.post("/team/member_add", {"team_id": team, "member": {"role": "user", "user_id": user}})
        personal: Final = scenario.key(models=[model], user_id=user)
        team_key: Final = scenario.key(team_id=team, user_id=user, models=[model])
        served(ask(gateway, model, personal))
        spend_reaches(gateway, "user", user, CHAT_COST)
        named, _, _ = _figures(_USER, refusal(ask(gateway, model, personal)))
        assert named == user
        assert ask(gateway, model, team_key).status_code == 200


def test_every_key_of_an_over_budget_user_is_refused(gateway: Gateway) -> None:
    """A user budget is shared by all of the user's personal keys. One key spends past the budget, then a second
    personal key that never made a call is refused with the user wording as well."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        user: Final = scenario.user(max_budget=TINY_BUDGET)
        spender: Final = scenario.key(models=[model], user_id=user)
        bystander: Final = scenario.key(models=[model], user_id=user)
        served(ask(gateway, model, spender))
        spend_reaches(gateway, "user", user, CHAT_COST)
        named, _, _ = _figures(_USER, refusal(ask(gateway, model, bystander)))
        assert named == user


def test_organization_over_budget_refuses_its_team_keys_with_the_organization_wording(gateway: Gateway) -> None:
    """An organization has `max_budget` 0.0000001. A key of a team in that organization is served once, then
    refused with exactly `Budget has been exceeded! Organization=<org_id> Current cost: 0.02, Max budget: 1e-07`
    although neither the key nor the team has a budget of its own."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        organization: Final = scenario.organization(models=[model], max_budget=TINY_BUDGET)
        team: Final = scenario.team(organization_id=organization, models=[model])
        key: Final = scenario.key(team_id=team, models=[model])
        served(ask(gateway, model, key))
        spend_reaches(gateway, "org", organization, CHAT_COST)
        provider_calls(wire)
        named, spent, limit = _figures(_ORGANIZATION, refusal(ask(gateway, model, key)))
        assert (named, spent, limit) == (organization, pytest.approx(0.02), pytest.approx(TINY_BUDGET))
        assert provider_calls(wire) == ()


def test_organization_budget_is_shared_by_all_teams_of_the_organization(gateway: Gateway) -> None:
    """Two teams belong to one budgeted organization. The first team's key spends past the organization budget,
    then the second team's key, which never made a call, is refused with the organization wording."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        organization: Final = scenario.organization(models=[model], max_budget=TINY_BUDGET)
        first: Final = scenario.key(team_id=scenario.team(organization_id=organization, models=[model]))
        second: Final = scenario.key(team_id=scenario.team(organization_id=organization, models=[model]))
        served(ask(gateway, model, first))
        spend_reaches(gateway, "org", organization, CHAT_COST)
        named, _, _ = _figures(_ORGANIZATION, refusal(ask(gateway, model, second)))
        assert named == organization


def test_end_user_over_budget_is_refused_with_the_end_user_wording(gateway: Gateway) -> None:
    """A customer has `max_budget` 0.0000001. A call naming it in the body `user` field is served once, then
    refused with exactly `ExceededBudget: End User=<id> over budget. Spend=0.02, Budget=1e-07`. The same key
    naming no customer is still served."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        customer: Final = new_customer(gateway, scenario, max_budget=TINY_BUDGET)
        served(ask(gateway, model, key, user=customer))
        spend_reaches(gateway, "end_user", customer, CHAT_COST)
        named, spent, limit = _figures(_END_USER, refusal(ask(gateway, model, key, user=customer)))
        assert (named, spent, limit) == (customer, pytest.approx(0.02), pytest.approx(TINY_BUDGET))
        served(ask(gateway, model, key))


def test_key_refusal_message_names_the_key_alias_and_never_the_secret(gateway: Gateway) -> None:
    """A key with an alias is exhausted. The refusal reads
    `Budget has been exceeded! Key=<alias> (<masked key name>) Current cost: 0.02, Max budget: 1e-07`. The alias
    is in the message and neither the raw key nor its hash is."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        alias: Final = f"dec1-budget-alias-{uuid.uuid4().hex}"
        key: Final = scenario.key(models=[model], key_alias=alias, max_budget=TINY_BUDGET)
        served(ask(gateway, model, key))
        spend_reaches(gateway, "key", hashed(key), CHAT_COST)
        message: Final = refusal(ask(gateway, model, key))
        named, spent, limit = _figures(_KEY, message)
        assert named.startswith(alias), message
        assert (spent, limit) == (pytest.approx(0.02), pytest.approx(TINY_BUDGET)), message
        assert key not in message and hashed(key) not in message, message


def test_key_budget_window_is_refused_with_the_window_wording(gateway: Gateway) -> None:
    """A key has no `max_budget` but one budget window, `budget_limits: [{budget_duration: "1d", max_budget:
    0.01}]`. After one served call (0.02) the next is refused with exactly
    `ExceededBudget: Key over 1d budget. Spend=$0.0200, Limit=$0.01`."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model], budget_limits=[{"budget_duration": "1d", "max_budget": 0.01}])
        served(ask(gateway, model, key))
        spend_reaches(gateway, "key", hashed(key), CHAT_COST)
        refused: Final = eventually(
            lambda: ask(gateway, model, key),
            lambda response: response.status_code != 200,
            seconds=LANDING_SECONDS,
        )
        assert refusal(refused) == "ExceededBudget: Key over 1d budget. Spend=$0.0200, Limit=$0.01"


def test_team_member_over_budget_refusal_names_the_user_and_the_team(gateway: Gateway) -> None:
    """A team member has `max_budget_in_team` 0.0000001. After one served call the member's team key is refused.
    Two code paths can produce the message, `User=<user> in Team=<team>` or `TeamMember=<user>:<team>`, and
    either way it starts with `Budget has been exceeded!` and names both ids."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        team: Final = scenario.team(models=[model])
        user: Final = scenario.user()
        gateway.post(
            "/team/member_add",
            {"team_id": team, "member": {"role": "user", "user_id": user}, "max_budget_in_team": TINY_BUDGET},
        )
        key: Final = scenario.key(team_id=team, user_id=user, models=[model])
        served(ask(gateway, model, key))
        eventually(
            lambda: gateway.rows(
                'SELECT spend FROM "LiteLLM_TeamMembership" WHERE user_id = %s AND team_id = %s', (user, team)
            ),
            lambda rows: len(rows) == 1 and number(rows[0]["spend"]) >= CHAT_COST - 1e-9,
            seconds=LANDING_SECONDS,
        )
        message: Final = refusal(ask(gateway, model, key))
        assert message.startswith("Budget has been exceeded! "), message
        assert f"User={user} in Team={team} " in message or f"TeamMember={user}:{team} " in message, message


def test_key_model_budget_refuses_that_model_and_still_serves_another(gateway: Gateway) -> None:
    """A key has `model_max_budget` for one of its two models only. After one call to the budgeted model it is
    refused with `LiteLLM Virtual Key: <hashed key>, key_alias: <alias>, exceeded budget for model=<model>`,
    while the other model on the same key is still served."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        capped: Final = chat_model(scenario, wire)
        free: Final = chat_model(scenario, wire)
        alias: Final = f"dec1-model-budget-{uuid.uuid4().hex}"
        key: Final = scenario.key(
            models=[capped, free],
            key_alias=alias,
            model_max_budget={capped: {"max_budget": TINY_BUDGET, "budget_duration": "1d"}},
        )
        served(ask(gateway, capped, key))
        refused: Final = eventually(
            lambda: ask(gateway, capped, key), lambda response: response.status_code != 200, seconds=LANDING_SECONDS
        )
        assert refusal(refused) == (
            f"LiteLLM Virtual Key: {hashed(key)}, key_alias: {alias}, exceeded budget for model={capped}"
        )
        served(ask(gateway, free, key))


def test_team_model_budget_refuses_every_key_of_the_team_for_that_model(gateway: Gateway) -> None:
    """A team has `model_max_budget` for one model. One team key spends on it, then another key of the same team
    is refused with `LiteLLM Team: <team_id>, exceeded budget for model=<model>`.
    UNVERIFIED: assumes `/team/new` accepts `model_max_budget` in the same shape as keys."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        team: Final = scenario.team(
            models=[model], model_max_budget={model: {"max_budget": TINY_BUDGET, "budget_duration": "1d"}}
        )
        spender: Final = scenario.key(team_id=team, models=[model])
        bystander: Final = scenario.key(team_id=team, models=[model])
        served(ask(gateway, model, spender))
        refused: Final = eventually(
            lambda: ask(gateway, model, bystander),
            lambda response: response.status_code != 200,
            seconds=LANDING_SECONDS,
        )
        assert refusal(refused) == f"LiteLLM Team: {team}, exceeded budget for model={model}"


def test_user_model_budget_refuses_the_users_key_for_that_model(gateway: Gateway) -> None:
    """A user has `model_max_budget` for one model. After one call with a personal key the next call to that
    model is refused with `LiteLLM User: <user_id>, exceeded budget for model=<model>`.
    UNVERIFIED: assumes `/user/new` accepts `model_max_budget` in the same shape as keys."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        user: Final = scenario.user(model_max_budget={model: {"max_budget": TINY_BUDGET, "budget_duration": "1d"}})
        key: Final = scenario.key(models=[model], user_id=user)
        served(ask(gateway, model, key))
        refused: Final = eventually(
            lambda: ask(gateway, model, key), lambda response: response.status_code != 200, seconds=LANDING_SECONDS
        )
        assert refusal(refused) == f"LiteLLM User: {user}, exceeded budget for model={model}"


@pytest.mark.parametrize("surface", SURFACES, ids=_IDS)
def test_a_key_with_zero_budget_is_refused_on_every_surface_before_the_provider(
    gateway: Gateway, surface: Surface
) -> None:
    """A key with `max_budget` 0 calls chat, embeddings, `/v1/messages`, `/v1/responses` and text completions.
    Each is refused with status 422 and a body that says `Budget has been exceeded`, and the provider receives
    nothing. Only the status and the phrase are pinned because `/v1/messages` may wrap errors differently."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = surface.model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=0)
        response: Final = gateway.request("POST", surface.path, surface.body(model), key=key)
        assert response.status_code == 422, f"{response.status_code} {response.text}"
        assert "Budget has been exceeded" in response.text, response.text
        assert provider_calls(wire) == ()


def test_a_streaming_request_over_budget_gets_a_json_error_not_an_event_stream(gateway: Gateway) -> None:
    """A key with `max_budget` 0 asks for `stream: true`. The refusal is a plain JSON body with status 422 and
    `content-type: application/json`, never a 200 event stream that carries the error inside."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=0)
        response: Final = gateway.request("POST", CHAT.path, CHAT.body(model, streaming=True), key=key)
        assert response.headers["content-type"].startswith("application/json"), response.headers
        assert refusal(response).startswith("Budget has been exceeded! Key="), response.text
        assert provider_calls(wire) == ()


def test_budget_refusal_error_body_has_the_openai_error_shape(gateway: Gateway) -> None:
    """The refusal body is `{"error": {"message", "type", "param", "code"}}` with `type` `budget_exceeded`,
    `param` null and `code` the string `"422"`. OpenAI SDKs raise `UnprocessableEntityError` from this."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=0)
        response: Final = ask(gateway, model, key)
        error: Final = object_value(JSON_OBJECT.validate_json(response.content)["error"])
        assert response.status_code == 422, response.text
        assert {"message", "type", "param", "code"} <= set(error), error
        assert (error["type"], error["param"], error["code"]) == ("budget_exceeded", None, "422"), error
        assert string_value(error["message"]).endswith("Current cost: 0.0, Max budget: 0.0"), error


def test_budget_exceeded_status_code_setting_restores_429(gateway: Gateway, tmp_path: Path) -> None:
    """A proxy started with `litellm_settings.budget_exceeded_status_code: 429` refuses a zero-budget key with
    HTTP 429 and `error.code` `"429"` while keeping `error.type` `budget_exceeded`. This is the documented way
    back to the status older clients retry on. UNVERIFIED: assumes `litellm_settings` sets the module value."""
    config: Final = proxy_config(tmp_path, litellm_settings={"budget_exceeded_status_code": 429})
    with (
        scripted_upstream() as wire,
        owned_proxy(gateway, tmp_path, {}, config=config, state=OwnedState.SHARED_WITH_WORKER) as owned,
        owned.scenario() as scenario,
    ):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=0)
        response: Final = ask(owned, model, key)
        error: Final = object_value(JSON_OBJECT.validate_json(response.content)["error"])
        assert response.status_code == 429, response.text
        assert (error["type"], error["code"]) == ("budget_exceeded", "429"), error


def test_global_proxy_budget_refuses_every_key_once_total_spend_passes_it(gateway: Gateway, tmp_path: Path) -> None:
    """A proxy with its own database is started with `litellm_settings.max_budget: 0.00000001`. The first call is
    served. Once its 0.02 is recorded against the proxy-wide total, a different key with no budget of its own is
    refused with 422 and `Budget has been exceeded! Current cost: <total>, Max budget: 1e-08`.
    UNVERIFIED: how fast the proxy-wide total is refreshed, the test polls for up to 70 seconds."""
    config: Final = proxy_config(tmp_path, litellm_settings={"max_budget": 0.00000001, "budget_duration": "30d"})
    with (
        scripted_upstream() as wire,
        owned_proxy(gateway, tmp_path, {}, config=config, state=OwnedState.PRIVATE) as owned,
        owned.scenario() as scenario,
    ):
        model: Final = chat_model(scenario, wire)
        spender: Final = scenario.key(models=[model])
        bystander: Final = scenario.key(models=[model])
        served(ask(owned, model, spender))
        refused: Final = eventually(
            lambda: ask(owned, model, bystander), lambda response: response.status_code != 200, seconds=LANDING_SECONDS
        )
        message: Final = refusal(refused)
        assert message.startswith("Budget has been exceeded! Current cost: "), message
        assert message.endswith(", Max budget: 1e-08"), message
