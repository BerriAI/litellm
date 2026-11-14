"""Which models a virtual key may call, and which layer's restriction wins when key, team, member and user disagree.

Contract (litellm/proxy/auth/auth_checks.py `can_key_call_model`, `check_model_access_helper`,
`common_checks`; litellm/proxy/auth/user_api_key_auth.py `enforce_key_and_fallback_model_access`):

* A key with an empty or unset `models` list, `["*"]` or `["all-proxy-models"]` is not restricted by the key.
* `["all-team-models"]` on a key skips the key check and inherits the team's list. Without a team it means
  unrestricted.
* Every layer must allow the model. The key list is checked first and answers `key_model_access_denied`. The
  team list and the per member `allowed_models` answer `team_model_access_denied`. A personal key (no team)
  is also checked against its user's `models` and answers `user_model_access_denied`.
* The user's list is not consulted for a team key.
* An organization's `models` list is checked when a team is attached to it, not on each request.
* Client supplied `fallbacks` in the body are checked against the key list like the primary model.
* Every model denial is HTTP 403 with `error.param == "model"`, `error.code == "403"` and a fixed message
  that names the requested model but never the list of allowed models.
"""

from __future__ import annotations

from typing import Final

import pytest

from tests.integration._support.client import Gateway, eventually
from tests.integration.authorization.dec1_keys_support import (
    MODEL_DENIED_TEMPLATE,
    assert_denied,
    assert_model_denied,
    assert_served,
    chat,
    chat_body,
    marker,
    model_ids,
    upstream_requests,
)


def test_a_key_denial_has_the_exact_openai_error_body(gateway: Gateway) -> None:
    """A key limited to one model sends a chat completion for another existing model.

    The proxy must answer 403 with exactly `{"error": {"message": "The requested model '<model>' is not
    available for this API key, or the model name is invalid. Check the models available to you and try
    again.", "type": "key_model_access_denied", "param": "model", "code": "403"}}`.
    """
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        other: Final = scenario.model()
        key: Final = scenario.key(models=[allowed])
        refused: Final = chat(gateway, key, other)
        assert refused.status_code == 403, refused.text
        assert refused.json() == {
            "error": {
                "message": MODEL_DENIED_TEMPLATE.format(model=other),
                "type": "key_model_access_denied",
                "param": "model",
                "code": "403",
            }
        }, refused.text


def test_a_model_denial_does_not_reveal_the_allowed_models_or_call_the_upstream(gateway: Gateway) -> None:
    """A key limited to one model asks for another, and the refusal text and the upstream are inspected.

    The response must not contain the name of the model the key is allowed to use, and the upstream must
    not receive the request.
    """
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        other: Final = scenario.model()
        key: Final = scenario.key(models=[allowed])
        text: Final = marker()
        refused: Final = chat(gateway, key, other, text)
        assert refused.status_code == 403, refused.text
        assert allowed not in refused.text
        assert upstream_requests(gateway, text) == []


def test_a_denied_model_that_does_not_exist_gets_the_same_403_as_one_that_does(gateway: Gateway) -> None:
    """A key limited to one model asks for a model name that no deployment serves.

    The answer must be the same 403 `key_model_access_denied` body as for an existing but disallowed model,
    so a restricted key cannot probe which model names are configured.
    """
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        key: Final = scenario.key(models=[allowed])
        ghost: Final = f"ghost-{marker()}"
        assert_model_denied(chat(gateway, key, ghost), ghost, "key_model_access_denied")


@pytest.mark.parametrize("sentinel", ["*", "all-proxy-models"])
def test_a_key_with_an_everything_sentinel_reaches_any_model(gateway: Gateway, sentinel: str) -> None:
    """A key is created with `models: ["*"]` or `models: ["all-proxy-models"]` and calls a new model.

    Both sentinels mean every model on the proxy, so the request must be served with 200.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[sentinel])
        assert_served(chat(gateway, key, model))


def test_all_team_models_on_a_key_inherits_the_team_list(gateway: Gateway) -> None:
    """A team allows one model and its key is created with `models: ["all-team-models"]`.

    The key must be served on the team's model and must get 403 `team_model_access_denied` on another
    model. The denial type is the team's because the key level check is skipped for this sentinel.
    """
    with gateway.scenario() as scenario:
        team_model: Final = scenario.model()
        other: Final = scenario.model()
        team: Final = scenario.team(models=[team_model])
        key: Final = scenario.key(team_id=team, models=["all-team-models"])
        assert_served(chat(gateway, key, team_model))
        assert_model_denied(chat(gateway, key, other), other, "team_model_access_denied")


def test_all_team_models_on_a_key_without_a_team_is_unrestricted(gateway: Gateway) -> None:
    """A key with no team is created with `models: ["all-team-models"]` and calls a new model.

    With no team to inherit from, the sentinel means no key level restriction and the request must be
    served with 200.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=["all-team-models"])
        assert_served(chat(gateway, key, model))


def test_all_team_models_follows_a_later_change_to_the_team_list(gateway: Gateway) -> None:
    """A key with `all-team-models` is denied a model, then the team's list is updated to include it.

    After `/team/update` the same key must be served on that model without touching the key. The sentinel
    is resolved against the team's current list on each request.
    """
    with gateway.scenario() as scenario:
        first: Final = scenario.model()
        second: Final = scenario.model()
        team: Final = scenario.team(models=[first])
        key: Final = scenario.key(team_id=team, models=["all-team-models"])
        assert_model_denied(chat(gateway, key, second), second, "team_model_access_denied")
        gateway.post("/team/update", {"team_id": team, "models": [first, second]})
        assert_served(eventually(lambda: chat(gateway, key, second), lambda response: response.status_code == 200))


def test_the_key_list_wins_when_it_is_narrower_than_the_team_list(gateway: Gateway) -> None:
    """A team allows two models and its key lists only the first.

    The key must be served on the first and get 403 `key_model_access_denied` on the second, although the
    team allows it. A key can only narrow what its team grants.
    """
    with gateway.scenario() as scenario:
        first: Final = scenario.model()
        second: Final = scenario.model()
        team: Final = scenario.team(models=[first, second])
        key: Final = scenario.key(team_id=team, models=[first])
        assert_served(chat(gateway, key, first))
        assert_model_denied(chat(gateway, key, second), second, "key_model_access_denied")


def test_the_team_list_wins_when_the_key_list_is_wider(gateway: Gateway) -> None:
    """A team is narrowed to one model after its key was created with two models.

    The key must get 403 `team_model_access_denied` on the model the team no longer allows, although the
    key's own list still names it. A key cannot exceed its team.
    """
    with gateway.scenario() as scenario:
        first: Final = scenario.model()
        second: Final = scenario.model()
        team: Final = scenario.team(models=[first, second])
        key: Final = scenario.key(team_id=team, models=[first, second])
        assert_served(chat(gateway, key, second))
        gateway.post("/team/update", {"team_id": team, "models": [first]})
        refused: Final = eventually(lambda: chat(gateway, key, second), lambda response: response.status_code == 403)
        assert_model_denied(refused, second, "team_model_access_denied")
        assert_served(chat(gateway, key, first))


def test_a_team_key_with_no_models_of_its_own_gets_exactly_the_team_list(gateway: Gateway) -> None:
    """A team allows one model and its key is created without `models`.

    The key must be served on the team's model and get 403 `team_model_access_denied` on another.
    """
    with gateway.scenario() as scenario:
        team_model: Final = scenario.model()
        other: Final = scenario.model()
        team: Final = scenario.team(models=[team_model])
        key: Final = scenario.key(team_id=team)
        assert_served(chat(gateway, key, team_model))
        assert_model_denied(chat(gateway, key, other), other, "team_model_access_denied")


def test_a_team_member_allowed_models_list_narrows_the_team_for_that_members_key(gateway: Gateway) -> None:
    """A team allows two models and one member is updated with `allowed_models` naming only the first.

    That member's team key must be served on the first model and get 403 `team_model_access_denied` on the
    second. A key of the same team without a user must still reach the second model.
    """
    with gateway.scenario() as scenario:
        first: Final = scenario.model()
        second: Final = scenario.model()
        team: Final = scenario.team(models=[first, second])
        member: Final = scenario.member(team)
        gateway.post("/team/member_update", {"team_id": team, "user_id": member, "allowed_models": [first]})
        member_key: Final = scenario.key(team_id=team, user_id=member)
        service_key: Final = scenario.key(team_id=team)
        assert_served(chat(gateway, member_key, first))
        refused: Final = eventually(
            lambda: chat(gateway, member_key, second), lambda response: response.status_code == 403
        )
        assert_model_denied(refused, second, "team_model_access_denied")
        assert_served(chat(gateway, service_key, second))


def test_clearing_a_members_allowed_models_restores_the_team_list(gateway: Gateway) -> None:
    """A member restricted to one model is updated with `allowed_models: []`.

    An empty per member list means no extra restriction, so the member's key must be served again on the
    second team model.
    """
    with gateway.scenario() as scenario:
        first: Final = scenario.model()
        second: Final = scenario.model()
        team: Final = scenario.team(models=[first, second])
        member: Final = scenario.member(team)
        gateway.post("/team/member_update", {"team_id": team, "user_id": member, "allowed_models": [first]})
        key: Final = scenario.key(team_id=team, user_id=member)
        eventually(lambda: chat(gateway, key, second), lambda response: response.status_code == 403)
        gateway.post("/team/member_update", {"team_id": team, "user_id": member, "allowed_models": []})
        assert_served(eventually(lambda: chat(gateway, key, second), lambda response: response.status_code == 200))


def test_a_personal_key_is_limited_by_its_users_models(gateway: Gateway) -> None:
    """A user allows one model and owns a personal key (no team) created without `models`.

    The key must be served on the user's model and get 403 `user_model_access_denied` on another model.
    """
    with gateway.scenario() as scenario:
        user_model: Final = scenario.model()
        other: Final = scenario.model()
        user: Final = scenario.user(models=[user_model])
        key: Final = scenario.key(user_id=user)
        assert_served(chat(gateway, key, user_model))
        assert_model_denied(chat(gateway, key, other), other, "user_model_access_denied")


def test_a_personal_key_cannot_exceed_its_user_even_when_its_own_list_is_wider(gateway: Gateway) -> None:
    """An admin creates a personal key naming two models for a user who is then narrowed to the first.

    The key check passes for the second model but the user check must refuse it with 403
    `user_model_access_denied`. UNVERIFIED: `/user/update` is assumed to accept `models` and to take effect
    within the polling window; see `can_user_call_model`.
    """
    with gateway.scenario() as scenario:
        first: Final = scenario.model()
        second: Final = scenario.model()
        user: Final = scenario.user(models=[first, second])
        key: Final = scenario.key(user_id=user, models=[first, second])
        assert_served(chat(gateway, key, second))
        gateway.post("/user/update", {"user_id": user, "models": [first]})
        refused: Final = eventually(lambda: chat(gateway, key, second), lambda response: response.status_code == 403)
        assert_model_denied(refused, second, "user_model_access_denied")


def test_a_users_model_list_is_not_applied_to_their_team_key(gateway: Gateway) -> None:
    """A user allowed only model A is a member of a team that allows model B, and holds a key in that team.

    The team key must be served on model B. The user's personal list only applies to keys without a team.
    """
    with gateway.scenario() as scenario:
        user_model: Final = scenario.model()
        team_model: Final = scenario.model()
        team: Final = scenario.team(models=[team_model])
        user: Final = scenario.user(models=[user_model], user_role="internal_user")
        gateway.post("/team/member_add", {"team_id": team, "member": {"role": "user", "user_id": user}})
        key: Final = scenario.key(team_id=team, user_id=user)
        assert_served(chat(gateway, key, team_model))


def test_a_no_default_models_user_cannot_use_a_personal_key_but_can_use_a_team_key(gateway: Gateway) -> None:
    """A user has `models: ["no-default-models"]`, one personal key and one key in a team that allows a model.

    The personal key must get 403 with type `key_model_access_denied` (this sentinel reuses the key type),
    and the team key must be served on the team's model.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(models=[model])
        user: Final = scenario.user(models=["no-default-models"], user_role="internal_user")
        gateway.post("/team/member_add", {"team_id": team, "member": {"role": "user", "user_id": user}})
        personal: Final = scenario.key(user_id=user)
        team_key: Final = scenario.key(team_id=team, user_id=user)
        assert_model_denied(chat(gateway, personal, model), model, "key_model_access_denied")
        assert_served(chat(gateway, team_key, model))


def test_a_config_access_group_name_on_a_key_grants_its_member_models(gateway: Gateway) -> None:
    """A deployment declares `model_info.access_groups: [<group>]` and a key lists the group name in `models`.

    The key must be served on the member model and get 403 `key_model_access_denied` on a model that is
    not in the group.
    """
    with gateway.scenario() as scenario:
        group: Final = f"group-{marker()}"
        member: Final = scenario.model(model_info={"access_groups": [group]})
        outsider: Final = scenario.model()
        key: Final = scenario.key(models=[group])
        assert_served(chat(gateway, key, member))
        assert_model_denied(chat(gateway, key, outsider), outsider, "key_model_access_denied")


def test_a_config_access_group_name_on_a_team_grants_its_member_models(gateway: Gateway) -> None:
    """A deployment declares an access group and a team lists the group name in `models`.

    A key of that team must be served on the member model and get 403 `team_model_access_denied` on a
    model outside the group.
    """
    with gateway.scenario() as scenario:
        group: Final = f"group-{marker()}"
        member: Final = scenario.model(model_info={"access_groups": [group]})
        outsider: Final = scenario.model()
        team: Final = scenario.team(models=[group])
        key: Final = scenario.key(team_id=team)
        assert_served(chat(gateway, key, member))
        assert_model_denied(chat(gateway, key, outsider), outsider, "team_model_access_denied")


def test_a_client_side_fallback_outside_the_key_list_is_refused(gateway: Gateway) -> None:
    """A key limited to model A sends a chat completion for A with `fallbacks: [B]` in the body.

    The proxy must answer 403 `key_model_access_denied` naming B and must not call the upstream, even
    though the primary model is allowed. Otherwise fallbacks would be a way around the key's list.
    """
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        other: Final = scenario.model()
        key: Final = scenario.key(models=[allowed])
        text: Final = marker()
        refused: Final = chat(gateway, key, allowed, text, fallbacks=[other])
        assert_model_denied(refused, other, "key_model_access_denied")
        assert upstream_requests(gateway, text) == []


def test_a_client_side_fallback_inside_the_key_list_is_accepted(gateway: Gateway) -> None:
    """A key allowed models A and B sends a chat completion for A with `fallbacks: [B]`.

    Both names pass the key check, so the request must be served with 200.
    """
    with gateway.scenario() as scenario:
        first: Final = scenario.model()
        second: Final = scenario.model()
        key: Final = scenario.key(models=[first, second])
        assert_served(chat(gateway, key, first, fallbacks=[second]))


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/v1/completions", {"prompt": "dec1 keys"}),
        ("/v1/embeddings", {"input": "dec1 keys"}),
        ("/v1/responses", {"input": "dec1 keys"}),
        ("/chat/completions", {"messages": [{"role": "user", "content": "dec1 keys"}]}),
        ("/v1/moderations", {"input": "dec1 keys"}),
    ],
)
def test_the_key_model_list_is_enforced_on_every_openai_shaped_llm_route(
    gateway: Gateway, path: str, payload: dict[str, object]
) -> None:
    """A key limited to one model calls another model on completions, embeddings, responses, moderations and
    the unversioned chat route.

    Each route must answer 403 with type `key_model_access_denied`, param `model` and the fixed denial
    message. The check reads `model` from the body on all of them.
    """
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        other: Final = scenario.model()
        key: Final = scenario.key(models=[allowed])
        refused: Final = gateway.client.post(
            path, json={"model": other, **payload}, headers={"Authorization": f"Bearer {key}"}
        )
        assert_model_denied(refused, other, "key_model_access_denied")


def test_a_model_denial_on_the_anthropic_messages_route_is_a_403(gateway: Gateway) -> None:
    """A key limited to one model calls `/v1/messages` for another model.

    The proxy must answer HTTP 403 and the body must name the requested model in the fixed denial message.
    UNVERIFIED: the envelope is assumed to be Anthropic shaped (`{"type": "error", "error": {...}}`) as it
    is for 401s on this route, so only the status and message text are asserted.
    """
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        other: Final = scenario.model()
        key: Final = scenario.key(models=[allowed])
        refused: Final = gateway.request(
            "POST",
            "/v1/messages",
            {"model": other, "max_tokens": 16, "messages": [{"role": "user", "content": "dec1 keys"}]},
            key=key,
        )
        assert refused.status_code == 403, refused.text
        assert MODEL_DENIED_TEMPLATE.format(model=other) in refused.text, refused.text


def test_v1_models_lists_exactly_the_models_on_the_key(gateway: Gateway) -> None:
    """A key limited to one model calls `GET /v1/models`.

    The listing must contain that model and must not contain another model that exists on the proxy.
    Clients build their model pickers from this list.
    """
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        other: Final = scenario.model()
        key: Final = scenario.key(models=[allowed])
        listed: Final = model_ids(gateway, key)
        assert allowed in listed and other not in listed, sorted(listed)


def test_v1_models_for_a_team_key_without_its_own_list_shows_the_team_models(gateway: Gateway) -> None:
    """A team allows one model and its key, created without `models`, calls `GET /v1/models`.

    The listing must contain the team's model and must not contain another model that exists on the proxy.
    """
    with gateway.scenario() as scenario:
        team_model: Final = scenario.model()
        other: Final = scenario.model()
        team: Final = scenario.team(models=[team_model])
        key: Final = scenario.key(team_id=team)
        listed: Final = model_ids(gateway, key)
        assert team_model in listed and other not in listed, sorted(listed)


def test_a_request_without_a_model_is_not_a_model_denial(gateway: Gateway) -> None:
    """A key limited to one model sends a chat completion body with no `model` field.

    Auth has no model to check, so the answer must not be a 401 or a 403. The request fails later as a
    client error from the endpoint itself.
    """
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        key: Final = scenario.key(models=[allowed])
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "no model"}]}, key=key
        )
        assert response.status_code not in (200, 401, 403), response.text
        assert 400 <= response.status_code < 500, response.text


def test_a_team_created_under_an_organization_cannot_list_a_model_the_organization_lacks(gateway: Gateway) -> None:
    """An organization allows model A and an admin tries to create a team in it with model B.

    The organization's list is enforced when the team is attached, so `/team/new` must fail with a 4xx
    and no team row may be created. UNVERIFIED: the exact status (expected 400 or 403) and message; see
    `can_org_access_model` callers in `team_endpoints.py`.
    """
    with gateway.scenario() as scenario:
        org_model: Final = scenario.model()
        other: Final = scenario.model()
        organization: Final = scenario.organization(models=[org_model])
        alias: Final = marker()
        refused: Final = gateway.request(
            "POST", "/team/new", {"team_alias": alias, "organization_id": organization, "models": [other]}
        )
        assert 400 <= refused.status_code < 500, refused.text
        assert gateway.rows('SELECT team_id FROM "LiteLLM_TeamTable" WHERE team_alias=%s', (alias,)) == []


def test_a_key_in_an_organization_team_is_limited_by_the_team_list_it_was_given(gateway: Gateway) -> None:
    """An organization allows models A and B, its team allows A, and a team key calls B.

    The key must get 403 `team_model_access_denied`. The organization's wider list does not widen the
    team at request time.
    """
    with gateway.scenario() as scenario:
        first: Final = scenario.model()
        second: Final = scenario.model()
        organization: Final = scenario.organization(models=[first, second])
        team: Final = scenario.team(organization_id=organization, models=[first])
        key: Final = scenario.key(team_id=team)
        assert_served(chat(gateway, key, first))
        assert_model_denied(chat(gateway, key, second), second, "team_model_access_denied")


def test_the_master_key_ignores_model_lists(gateway: Gateway) -> None:
    """The master key calls a model that exists on the proxy but is on no key, team or user list.

    The request must be served with 200 and reach the upstream exactly once.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        text: Final = marker()
        assert_served(gateway.request("POST", "/v1/chat/completions", chat_body(model, text)))
        assert len(upstream_requests(gateway, text)) == 1


def test_a_model_denial_is_a_403_and_not_a_401(gateway: Gateway) -> None:
    """A valid key is refused a model, and an unknown key asks for the same model.

    The valid key must get 403 (authenticated but not allowed) and the unknown key 401 (not authenticated).
    SDKs map these to different exception classes, so the two must not be merged.
    """
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        other: Final = scenario.model()
        key: Final = scenario.key(models=[allowed])
        assert_denied(chat(gateway, key, other), status=403, error_type="key_model_access_denied", param="model")
        assert_denied(
            chat(gateway, "sk-dec1-keys-never-issued-model-access", other),
            status=401,
            error_type="token_not_found_in_db",
            param="key",
        )
