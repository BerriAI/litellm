"""Routing behaviours observable on the shared integration proxy: runtime model changes, team models, guards.

These tests use the suite's shared proxy (`gateway` fixture), whose config sets
`router_settings.disable_cooldowns: true` and does not allow mock testing params. Models are created at
runtime through `POST /model/new` and point at a scripted upstream started by each test, so the upstream
model name in a response's text (`served <upstream model>`) names the deployment that served it.

Pinned here: a model added through `/model/new` becomes routable without a restart and one removed through
`/model/delete` stops being routable; a model created for a team is reachable under its public name by that
team's keys only; `disable_cooldowns` keeps a rate limited deployment in rotation; and `mock_testing_*`
request params are refused unless an admin opted in.
"""

from __future__ import annotations

import uuid
from typing import Final

import pytest
from integration.routing.dec1_routing_support import (
    INVALID_MODEL,
    NO_DEPLOYMENTS,
    PROVIDER_KEY,
    chat,
    content_of,
    scripted_upstream,
    statuses,
)
from pydantic import JsonValue

from tests.integration._support.client import Gateway, eventually, list_value, object_value, string_value

pytestmark: Final = pytest.mark.timeout(300)

RELOAD_SECONDS: Final = 60
NO_RETRIES: Final = {"x-litellm-num-retries": "0"}


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def _create(gateway: Gateway, name: str, upstream_model: str, api_base: str, **model_info: JsonValue) -> str:
    created: Final = gateway.post(
        "/model/new",
        {
            "model_name": name,
            "litellm_params": {"model": f"openai/{upstream_model}", "api_key": PROVIDER_KEY, "api_base": api_base},
            "model_info": dict(model_info),
        },
    )
    return string_value(object_value(created["model_info"])["id"])


def _model_rows(gateway: Gateway, deployment_id: str) -> list[dict[str, JsonValue]]:
    return gateway.rows(
        'SELECT model_id, model_name FROM "LiteLLM_ProxyModelTable" WHERE model_id = %s', (deployment_id,)
    )


def test_a_model_added_through_model_new_is_routable_and_stored(gateway: Gateway) -> None:
    """A group that did not exist is created with `POST /model/new`, pointing at a healthy scripted upstream.

    Within the config reload window a chat request for the new name returns 200 with the upstream's text, the
    response's `x-litellm-model-id` is the id `/model/new` returned, and one row with that `model_id` exists in
    `LiteLLM_ProxyModelTable`.
    """
    name: Final = _unique("dec1-runtime-new")
    upstream_model: Final = _unique("ok-runtime-new")
    with gateway.scenario() as scenario, scripted_upstream() as upstream:
        deployment_id: Final = _create(gateway, name, upstream_model, upstream.api_base)
        scenario.cleanups.callback(scenario.delete_model, deployment_id)
        response: Final = eventually(
            lambda: chat(gateway, name), lambda answer: answer.status_code == 200, seconds=RELOAD_SECONDS
        )
        assert content_of(response) == f"served {upstream_model}"
        assert response.headers["x-litellm-model-id"] == deployment_id
        assert [row["model_id"] for row in _model_rows(gateway, deployment_id)] == [deployment_id]


def test_a_model_removed_through_model_delete_is_no_longer_routable(gateway: Gateway) -> None:
    """A single-deployment group is created through `/model/new`, served once, then removed with
    `POST /model/delete`.

    Within the config reload window a chat request for the name returns HTTP 400 whose body contains
    `Invalid model name passed in model=<name>`, its row is gone from `LiteLLM_ProxyModelTable`, and the
    upstream receives no request after the one that was served before the delete.
    """
    name: Final = _unique("dec1-runtime-delete")
    upstream_model: Final = _unique("ok-runtime-delete")
    with scripted_upstream() as upstream:
        deployment_id: Final = _create(gateway, name, upstream_model, upstream.api_base)
        eventually(lambda: chat(gateway, name).status_code, lambda status: status == 200, seconds=RELOAD_SECONDS)
        gateway.post("/model/delete", {"id": deployment_id})
        refused: Final = eventually(
            lambda: chat(gateway, name), lambda answer: answer.status_code == 400, seconds=RELOAD_SECONDS
        )
        assert f"{INVALID_MODEL}{name}" in refused.text
        assert _model_rows(gateway, deployment_id) == []
        served_before: Final = upstream.count(upstream_model)
        assert chat(gateway, name).status_code == 400
        assert upstream.count(upstream_model) == served_before


def test_deleting_one_of_two_deployments_leaves_the_group_served_by_the_other(gateway: Gateway) -> None:
    """A group gets two deployments through two `/model/new` calls, then one of them is deleted.

    Before the delete both upstream models serve traffic. After it, a batch of ten requests must all return
    200 from the remaining deployment, and its `x-litellm-model-id` is the only one seen.
    """
    name: Final = _unique("dec1-runtime-pair")
    kept_model: Final = _unique("ok-runtime-kept")
    removed_model: Final = _unique("ok-runtime-removed")
    with gateway.scenario() as scenario, scripted_upstream() as upstream:
        kept_id: Final = _create(gateway, name, kept_model, upstream.api_base)
        scenario.cleanups.callback(scenario.delete_model, kept_id)
        removed_id: Final = _create(gateway, name, removed_model, upstream.api_base)

        def served_by() -> frozenset[str]:
            responses: Final = tuple(chat(gateway, name) for _ in range(10))
            return frozenset(
                response.headers.get("x-litellm-model-id", str(response.status_code)) for response in responses
            )

        eventually(served_by, lambda ids: ids == {kept_id, removed_id}, seconds=RELOAD_SECONDS)
        gateway.post("/model/delete", {"id": removed_id})
        remaining: Final = eventually(served_by, lambda ids: ids == {kept_id}, seconds=RELOAD_SECONDS)
        assert remaining == {kept_id}
        assert gateway.rows('SELECT model_id FROM "LiteLLM_ProxyModelTable" WHERE model_id = %s', (removed_id,)) == []


def test_a_team_model_is_reachable_under_its_public_name_with_that_teams_key(gateway: Gateway) -> None:
    """A model is created through `/model/new` with `model_info.team_id` set to a team.

    The proxy stores it under a generated internal `model_name` and keeps the requested name in
    `model_info.team_public_model_name`. A key of that team calling the public name gets 200 from the
    deployment. `GET /model/info` shows the deployment with `team_id` and with both `team_public_model_name`
    and `model_name` equal to the public name: the listing swaps the internal routing key for the public name
    (litellm/proxy/common_utils/model_listing_utils.py:1). The stored row keeps the internal
    `model_name_{team_id}_{uuid}` name (litellm/proxy/management_endpoints/model_management_endpoints.py:2204).
    """
    public_name: Final = _unique("dec1-team-model")
    upstream_model: Final = _unique("ok-team-model")
    with gateway.scenario() as scenario, scripted_upstream() as upstream:
        team: Final = scenario.team()
        deployment_id: Final = _create(gateway, public_name, upstream_model, upstream.api_base, team_id=team)
        scenario.cleanups.callback(scenario.delete_model, deployment_id)
        key: Final = scenario.key(team_id=team)
        response: Final = eventually(
            lambda: chat(gateway, public_name, key=key),
            lambda answer: answer.status_code == 200,
            seconds=RELOAD_SECONDS,
        )
        assert content_of(response) == f"served {upstream_model}"
        assert response.headers["x-litellm-model-id"] == deployment_id
        entries: Final = tuple(
            object_value(entry)
            for entry in list_value(gateway.get("/model/info")["data"])
            if object_value(object_value(entry)["model_info"])["id"] == deployment_id
        )
        assert len(entries) == 1
        info: Final = object_value(entries[0]["model_info"])
        assert info["team_id"] == team
        assert info["team_public_model_name"] == public_name
        assert entries[0]["model_name"] == public_name
        stored: Final = gateway.rows(
            "SELECT model_name FROM \"LiteLLM_ProxyModelTable\" WHERE model_info->>'team_public_model_name' = %s",
            (public_name,),
        )
        assert len(stored) == 1, stored
        assert string_value(stored[0]["model_name"]).startswith(f"model_name_{team}_"), stored


def test_a_team_model_is_not_reachable_with_another_teams_key(gateway: Gateway) -> None:
    """A model is created for one team; a key that belongs to a different team calls its public name.

    Once the owning team's key is being served, the other team's key must get a 4xx refusal and the upstream
    must receive no request from it. UNVERIFIED: the exact status, 400 with `Invalid model name` or 401/403
    from the key's model access check; see `filter_team_based` in
    `litellm-rust/crates/router-core/src/select.rs` and `litellm/proxy/auth/auth_checks.py`.
    """
    public_name: Final = _unique("dec1-team-private")
    upstream_model: Final = _unique("ok-team-private")
    with gateway.scenario() as scenario, scripted_upstream() as upstream:
        owner: Final = scenario.team()
        deployment_id: Final = _create(gateway, public_name, upstream_model, upstream.api_base, team_id=owner)
        scenario.cleanups.callback(scenario.delete_model, deployment_id)
        owner_key: Final = scenario.key(team_id=owner)
        eventually(
            lambda: chat(gateway, public_name, key=owner_key).status_code,
            lambda status: status == 200,
            seconds=RELOAD_SECONDS,
        )
        served_for_owner: Final = upstream.count(upstream_model)
        outsider_key: Final = scenario.key(team_id=scenario.team())
        refused: Final = chat(gateway, public_name, key=outsider_key)
        assert refused.status_code in (400, 401, 403, 404), refused.text
        assert upstream.count(upstream_model) == served_for_owner


def test_two_teams_can_use_the_same_public_model_name_for_different_deployments(gateway: Gateway) -> None:
    """Two teams each create a model with the same public name, pointing at different upstream models.

    Five requests with each team's key must all be served by that team's own deployment and never by the
    other team's.
    """
    public_name: Final = _unique("dec1-team-shared-name")
    first_model: Final = _unique("ok-team-first")
    second_model: Final = _unique("ok-team-second")
    with gateway.scenario() as scenario, scripted_upstream() as upstream:
        first_team: Final = scenario.team()
        second_team: Final = scenario.team()
        first_id: Final = _create(gateway, public_name, first_model, upstream.api_base, team_id=first_team)
        scenario.cleanups.callback(scenario.delete_model, first_id)
        second_id: Final = _create(gateway, public_name, second_model, upstream.api_base, team_id=second_team)
        scenario.cleanups.callback(scenario.delete_model, second_id)
        first_key: Final = scenario.key(team_id=first_team)
        second_key: Final = scenario.key(team_id=second_team)
        for key, model in ((first_key, first_model), (second_key, second_model)):
            eventually(
                lambda key=key: chat(gateway, public_name, key=key).status_code,
                lambda status: status == 200,
                seconds=RELOAD_SECONDS,
            )
            served: Final = {content_of(chat(gateway, public_name, key=key)) for _ in range(5)}
            assert served == {f"served {model}"}


def test_disable_cooldowns_keeps_a_rate_limited_deployment_in_rotation(gateway: Gateway) -> None:
    """The shared proxy runs with `disable_cooldowns: true`; a group has a deployment with `weight: 1` that
    answers 429 and a healthy sibling without a weight.

    Five requests sent with `x-litellm-num-retries: 0` must all return the upstream's 429 and all five must
    reach the failing upstream. With cooldowns enabled the first 429 would move traffic to the sibling.
    """
    name: Final = _unique("dec1-runtime-nocooldown")
    failing: Final = f"status-429-{uuid.uuid4().hex}"
    healthy: Final = _unique("ok-runtime-nocooldown")
    with gateway.scenario() as scenario, scripted_upstream() as upstream:
        scenario.model(name=name, model=f"openai/{failing}", api_base=upstream.api_base, weight=1)
        scenario.model(name=name, model=f"openai/{healthy}", api_base=upstream.api_base)
        eventually(
            lambda: chat(gateway, name, headers=NO_RETRIES).status_code,
            lambda status: status == 429,
            seconds=RELOAD_SECONDS,
        )
        before: Final = upstream.count(failing)
        responses: Final = tuple(chat(gateway, name, headers=NO_RETRIES) for _ in range(5))
        assert statuses(responses) == (429,) * 5
        assert all(NO_DEPLOYMENTS not in response.text for response in responses)
        assert upstream.count(failing) == before + 5
        assert upstream.count(healthy) == 0


@pytest.mark.parametrize(
    "param",
    [
        "mock_testing_fallbacks",
        "mock_testing_context_fallbacks",
        "mock_testing_content_policy_fallbacks",
        "mock_testing_rate_limit_error",
    ],
)
def test_mock_testing_params_are_refused_with_400_unless_an_admin_enabled_them(gateway: Gateway, param: str) -> None:
    """The shared proxy does not set `general_settings.dangerously_allow_mock_testing_request_params`.

    A chat request that carries one of the `mock_testing_*` params returns HTTP 400 whose body contains
    `Mock testing request params are disabled on this proxy: <param>`, and the upstream receives nothing. The
    request is refused rather than served, so a fallback drill cannot pass without having run.
    """
    with gateway.scenario() as scenario, scripted_upstream() as upstream:
        upstream_model: Final = _unique("ok-runtime-mock")
        model: Final = scenario.model(model=f"openai/{upstream_model}", api_base=upstream.api_base)
        response: Final = chat(gateway, model, **{param: True})
        assert response.status_code == 400, response.text
        assert f"Mock testing request params are disabled on this proxy: {param}" in response.text
        assert upstream.count(upstream_model) == 0


def test_request_body_fallbacks_serve_a_failing_group_and_report_one_fallback(gateway: Gateway) -> None:
    """The primary group's upstream always answers 503; the caller sends `fallbacks: [<backup group>]` in the
    body with `x-litellm-num-retries: 0`.

    The caller gets 200 with the backup's text and `x-litellm-attempted-fallbacks: 1`. The failing upstream
    model received exactly one request and the backup's exactly one.
    """
    failing: Final = f"status-503-{uuid.uuid4().hex}"
    backup_model: Final = _unique("ok-runtime-clientfb")
    with gateway.scenario() as scenario, scripted_upstream() as upstream:
        primary: Final = scenario.model(model=f"openai/{failing}", api_base=upstream.api_base)
        backup: Final = scenario.model(model=f"openai/{backup_model}", api_base=upstream.api_base)
        response: Final = chat(gateway, primary, headers=NO_RETRIES, fallbacks=[backup])
        assert response.status_code == 200, response.text
        assert content_of(response) == f"served {backup_model}"
        assert response.headers["x-litellm-attempted-fallbacks"] == "1"
        assert upstream.order(failing, backup_model) == (failing, backup_model)
