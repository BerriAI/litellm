"""Model access groups: config tags on deployments, the `/access_group/*` tagging API, and access group objects.

Contract:

* A deployment's `model_info.access_groups` names the groups it belongs to. A key, team or user list that
  names a group grants every member deployment (`check_model_access_helper`, which consults
  `registry().access_groups(model_name=...)`). Group names are matched exactly; a group name never acts as a
  pattern.
* `POST /access_group/new` tags existing deployments with a group, `DELETE /access_group/{g}/delete` removes
  the tag, and both validate their input with fixed messages
  (litellm/proxy/management_endpoints/model_access_group_management_endpoints.py).
* Access group OBJECTS (`/v1/access_group`, `LiteLLM_AccessGroupTable`) are a separate mechanism: a key's
  `access_group_ids` or a team's `access_group_ids` add the object's `access_model_names` to what that key
  or team may call, after its own list refused (`can_key_call_model` and `can_team_access_model` fallbacks
  in litellm/proxy/auth/auth_checks.py).

The DB stores behind both mechanisms are being ported to Rust (drafts #52173-#52178), so the grant
semantics are pinned per layer.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Final

from pydantic import JsonValue

from tests.integration._support.client import Gateway, eventually, string_value
from tests.integration.authorization.dec1_modelaccess_support import (
    KEY_DENIED,
    TEAM_DENIED,
    USER_DENIED,
    assert_model_denied,
    assert_served,
    body_of,
    chat,
    unique_name,
)


@contextmanager
def _tag_group(gateway: Gateway, group: str, models: list[str]) -> Iterator[dict[str, JsonValue]]:
    """`POST /access_group/new` for `models`, deleted again on exit."""
    created: Final = gateway.post("/access_group/new", {"access_group": group, "model_names": models})
    try:
        yield created
    finally:
        response: Final = gateway.request("DELETE", f"/access_group/{group}/delete")
        assert response.status_code in (200, 404), response.text


@contextmanager
def _group_object(gateway: Gateway, models: list[str], **fields: JsonValue) -> Iterator[str]:
    """`POST /v1/access_group` with `access_model_names: models`, deleted again on exit."""
    created: Final = gateway.request(
        "POST",
        "/v1/access_group",
        {"access_group_name": unique_name("dec1-ma-object"), "access_model_names": models, **fields},
    )
    assert created.status_code == 201, created.text
    identity: Final = string_value(body_of(created)["access_group_id"])
    try:
        yield identity
    finally:
        deleted: Final = gateway.request("DELETE", f"/v1/access_group/{identity}")
        assert deleted.status_code in (204, 404), deleted.text


def test_a_user_list_naming_a_config_group_grants_its_members_to_a_personal_key(gateway: Gateway) -> None:
    """A deployment declares `access_groups: [<g>]`; a user has `models: [<g>]` and a personal key with no list.

    The member must be served and an outside model refused with 403 `user_model_access_denied`.
    """
    with gateway.scenario() as scenario:
        group: Final = unique_name("dec1-ma-group")
        member: Final = scenario.model(model_info={"access_groups": [group]})
        outsider: Final = scenario.model()
        user: Final = scenario.user(models=[group])
        key: Final = scenario.key(user_id=user)
        assert_served(chat(gateway, key, member))
        assert_model_denied(chat(gateway, key, outsider), outsider, USER_DENIED)


def test_a_deployment_in_two_groups_is_reachable_through_either(gateway: Gateway) -> None:
    """A deployment declares `access_groups: [<g1>, <g2>]`; one key lists `<g1>`, another lists `<g2>`.

    Both keys must be served on that deployment.
    """
    with gateway.scenario() as scenario:
        first_group: Final = unique_name("dec1-ma-group")
        second_group: Final = unique_name("dec1-ma-group")
        member: Final = scenario.model(model_info={"access_groups": [first_group, second_group]})
        assert_served(chat(gateway, scenario.key(models=[first_group]), member))
        assert_served(chat(gateway, scenario.key(models=[second_group]), member))


def test_a_group_name_is_matched_exactly_and_never_as_a_prefix(gateway: Gateway) -> None:
    """A deployment declares `access_groups: [<g>-prod]`; a key lists `<g>` (a prefix of the group name).

    The key must be refused with 403 `key_model_access_denied`: group membership is an exact name match.
    """
    with gateway.scenario() as scenario:
        group: Final = unique_name("dec1-ma-group")
        member: Final = scenario.model(model_info={"access_groups": [f"{group}-prod"]})
        key: Final = scenario.key(models=[group])
        assert_model_denied(chat(gateway, key, member), member, KEY_DENIED)


def test_a_group_on_the_key_and_a_plain_model_on_the_team_must_both_allow_the_request(gateway: Gateway) -> None:
    """Deployment M declares `access_groups: [<g>]`; a team allows only model N; its key lists `<g>`.

    Calling M passes the key but must be refused with 403 `team_model_access_denied`, because the team does
    not allow M by name or group.
    """
    with gateway.scenario() as scenario:
        group: Final = unique_name("dec1-ma-group")
        member: Final = scenario.model(model_info={"access_groups": [group]})
        other: Final = scenario.model()
        team: Final = scenario.team(models=[other])
        key: Final = scenario.key(team_id=team, models=[group])
        assert_model_denied(chat(gateway, key, member), member, TEAM_DENIED)


def test_access_group_new_tags_a_deployment_so_a_key_naming_the_group_is_served(gateway: Gateway) -> None:
    """An admin calls `POST /access_group/new` with `{access_group: <g>, model_names: [M]}`; a key lists `<g>`.

    The answer must be 200 with `access_group == <g>`, `model_names == [M]` and `models_updated == 1`, and the
    key must then be served on M (polled, because tagging reloads the router).
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        group: Final = unique_name("dec1-ma-tag")
        with _tag_group(gateway, group, [model]) as created:
            assert created["access_group"] == group and created["model_names"] == [model], created
            assert created["models_updated"] == 1, created
            key: Final = scenario.key(models=[group])
            assert_served(eventually(lambda: chat(gateway, key, model), lambda response: response.status_code == 200))


def test_deleting_a_tagged_group_revokes_the_keys_that_name_it(gateway: Gateway) -> None:
    """A group is created with `/access_group/new` for M and a key listing it is served; then
    `DELETE /access_group/<g>/delete` is called.

    After the delete the same key must be refused on M with 403 `key_model_access_denied` (polled).
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        group: Final = unique_name("dec1-ma-tag")
        with _tag_group(gateway, group, [model]):
            key: Final = scenario.key(models=[group])
            eventually(lambda: chat(gateway, key, model), lambda response: response.status_code == 200)
            deleted: Final = gateway.request("DELETE", f"/access_group/{group}/delete")
            assert deleted.status_code == 200, deleted.text
            refused: Final = eventually(
                lambda: chat(gateway, key, model), lambda response: response.status_code == 403, seconds=30
            )
            assert_model_denied(refused, model, KEY_DENIED)


def test_access_group_new_rejects_an_empty_group_name(gateway: Gateway) -> None:
    """An admin posts `/access_group/new` with `access_group: "  "` and one model.

    The answer must be 400 with body `{"detail": {"error": "access_group is required and cannot be empty"}}`.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        response: Final = gateway.request("POST", "/access_group/new", {"access_group": "  ", "model_names": [model]})
        assert response.status_code == 400, response.text
        assert body_of(response) == {"detail": {"error": "access_group is required and cannot be empty"}}


def test_access_group_new_rejects_a_request_without_models(gateway: Gateway) -> None:
    """An admin posts `/access_group/new` with a name and `model_names: []`.

    The answer must be 400 with body
    `{"detail": {"error": "Either model_names or model_ids must be provided and non-empty"}}`.
    """
    response: Final = gateway.request(
        "POST", "/access_group/new", {"access_group": unique_name("dec1-ma-tag"), "model_names": []}
    )
    assert response.status_code == 400, response.text
    assert body_of(response) == {"detail": {"error": "Either model_names or model_ids must be provided and non-empty"}}


def test_access_group_new_rejects_a_model_the_router_does_not_serve(gateway: Gateway) -> None:
    """An admin posts `/access_group/new` naming one model that does not exist.

    The answer must be 400 with body `{"detail": {"error": "Model(s) not found: <name>"}}`.
    """
    ghost: Final = unique_name("dec1-ma-ghost")
    response: Final = gateway.request(
        "POST", "/access_group/new", {"access_group": unique_name("dec1-ma-tag"), "model_names": [ghost]}
    )
    assert response.status_code == 400, response.text
    assert body_of(response) == {"detail": {"error": f"Model(s) not found: {ghost}"}}


def test_access_group_new_refuses_a_name_that_already_exists(gateway: Gateway) -> None:
    """An admin creates group `<g>` for M and then posts `/access_group/new` again with the same name.

    The second answer must be 409 with body `{"detail": {"error": "Access group '<g>' already exists. Use PUT
    /access_group/<g>/update to modify it."}}`.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        group: Final = unique_name("dec1-ma-tag")
        with _tag_group(gateway, group, [model]):
            again: Final = gateway.request("POST", "/access_group/new", {"access_group": group, "model_names": [model]})
            assert again.status_code == 409, again.text
            assert body_of(again) == {
                "detail": {
                    "error": f"Access group '{group}' already exists. Use PUT /access_group/{group}/update to modify it."
                }
            }


def test_deleting_an_unknown_tagged_group_is_a_404(gateway: Gateway) -> None:
    """An admin calls `DELETE /access_group/<unknown>/delete`.

    The answer must be 404 with body `{"detail": {"error": "Access group '<unknown>' not found"}}`.
    """
    unknown: Final = unique_name("dec1-ma-tag")
    response: Final = gateway.request("DELETE", f"/access_group/{unknown}/delete")
    assert response.status_code == 404, response.text
    assert body_of(response) == {"detail": {"error": f"Access group '{unknown}' not found"}}


def test_a_key_access_group_object_widens_a_restricted_key(gateway: Gateway) -> None:
    """An access group object grants model B; a key has `models: [A]` and `access_group_ids: [<object>]`.

    The key must be served on A and on B: after the key list refuses B, `can_key_call_model` checks the
    models of the key's access group objects. A third model C must still be refused with 403
    `key_model_access_denied`.
    """
    with gateway.scenario() as scenario:
        first: Final = scenario.model()
        second: Final = scenario.model()
        third: Final = scenario.model()
        with _group_object(gateway, [second]) as group_id:
            key: Final = scenario.key(models=[first], access_group_ids=[group_id])
            assert_served(chat(gateway, key, first))
            assert_served(chat(gateway, key, second))
            assert_model_denied(chat(gateway, key, third), third, KEY_DENIED)


def test_a_team_access_group_object_adds_to_the_team_list(gateway: Gateway) -> None:
    """An access group object grants model B; a team has `models: [A]` and `access_group_ids: [<object>]`.

    A key of that team with no list must be served on A and B, and refused on C with 403
    `team_model_access_denied`. UNVERIFIED: assumes `/team/new` accepts `access_group_ids` directly (the
    existing suite assigns teams through `assigned_team_ids` on the object instead).
    """
    with gateway.scenario() as scenario:
        first: Final = scenario.model()
        second: Final = scenario.model()
        third: Final = scenario.model()
        with _group_object(gateway, [second]) as group_id:
            team: Final = scenario.team(models=[first], access_group_ids=[group_id])
            key: Final = scenario.key(team_id=team)
            assert_served(chat(gateway, key, first))
            assert_served(chat(gateway, key, second))
            assert_model_denied(chat(gateway, key, third), third, TEAM_DENIED)


def test_an_access_group_object_assigned_to_a_key_by_id_lets_it_pass_its_teams_list(gateway: Gateway) -> None:
    """A team allows only A; an access group object grants B with `assigned_key_ids: [<hashed key>]`.

    A key of that team (no list) must be served on B: a team denial is waived when an access group assigned
    to the key grants the model (`key_access_group_grants_model`). A sibling key of the same team that is not
    assigned must still get 403 `team_model_access_denied` on B.
    """
    from hashlib import sha256

    with gateway.scenario() as scenario:
        first: Final = scenario.model()
        second: Final = scenario.model()
        team: Final = scenario.team(models=[first])
        assigned: Final = scenario.key(team_id=team)
        sibling: Final = scenario.key(team_id=team)
        hashed: Final = sha256(assigned.encode()).hexdigest()
        with _group_object(gateway, [second], assigned_key_ids=[hashed]):
            assert_served(
                eventually(lambda: chat(gateway, assigned, second), lambda response: response.status_code == 200)
            )
            assert_model_denied(chat(gateway, sibling, second), second, TEAM_DENIED)


def test_an_access_group_object_naming_a_config_group_grants_the_group_members(gateway: Gateway) -> None:
    """Deployment M declares `access_groups: [<g>]`; an access group object has `access_model_names: [<g>]`;
    a key has `models: [<other>]` and `access_group_ids: [<object>]`.

    The key must be served on M: the object's model names go through the same check, which expands config
    group names. UNVERIFIED: read from `can_key_call_model` passing the object's names as `models`.
    """
    with gateway.scenario() as scenario:
        group: Final = unique_name("dec1-ma-group")
        member: Final = scenario.model(model_info={"access_groups": [group]})
        other: Final = scenario.model()
        with _group_object(gateway, [group]) as group_id:
            key: Final = scenario.key(models=[other], access_group_ids=[group_id])
            assert_served(chat(gateway, key, member))


def test_a_key_access_group_id_that_does_not_exist_grants_nothing(gateway: Gateway) -> None:
    """A key has `models: [A]` and `access_group_ids: [<random id never created>]` and calls model B.

    The call must be refused with 403 `key_model_access_denied`: an unknown group resolves to no models.
    UNVERIFIED: `/key/generate` may validate the id and answer 4xx at creation instead; in that case pin
    the creation error.
    """
    with gateway.scenario() as scenario:
        first: Final = scenario.model()
        second: Final = scenario.model()
        response: Final = gateway.request(
            "POST", "/key/generate", {"models": [first], "access_group_ids": [f"ag-{uuid.uuid4().hex}"]}
        )
        if response.status_code != 200:
            assert 400 <= response.status_code < 500, response.text
            return
        key: Final = string_value(body_of(response)["key"])
        scenario.cleanups.callback(scenario.delete_key, key)
        assert_model_denied(chat(gateway, key, second), second, KEY_DENIED)
