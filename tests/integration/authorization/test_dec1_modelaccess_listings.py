"""Which model names each caller sees on `/v1/models`, `/models`, `/model/info` and `/model_group/info`.

Contract (litellm/proxy/proxy_server.py `model_list`, `_key_model_listing`, `model_info_v1`,
`model_group_info`; litellm/proxy/utils.py `get_available_models_for_user`; litellm/proxy/auth/model_checks.py
`get_key_models`, `get_team_models`, `get_complete_model_list`;
litellm/proxy/common_utils/discoverable_model_filter.py):

* The listing source is the key list when it is non-empty, otherwise the team list, otherwise every proxy
  model. The key list wins even when it is wider than the team list, so a team key can list a model it would
  be refused (pinned as current behaviour).
* Names keep the order of the list they come from, duplicates removed.
* A config access group name in a list is replaced by its member models, unless `include_model_access_groups`
  is true, and `only_model_access_groups=true` returns only the group names.
* `scope=expand` is the only accepted scope; anything else is a 400 with a fixed message.
* A deployment with `model_info.discoverable: false` is hidden from non-admin listings on all three routes but
  stays callable; admins still see it.
* `/model/info` lists router deployments filtered to the caller's names and never returns the provider
  `api_key`. `/model_group/info` lists one row per allowed group.

Clients (SDK model pickers, Claude Code, Codex, the UI) build their model menus from these answers, so the
exact set of names per caller is the contract.
"""

from __future__ import annotations

from typing import Final

import pytest

from tests.integration._support.client import Gateway, list_value, object_value
from tests.integration.authorization.dec1_modelaccess_support import (
    TEAM_DENIED,
    assert_model_denied,
    assert_served,
    body_of,
    chat,
    get_as,
    listed_ids,
    model_group_names,
    model_group_rows,
    model_ids,
    model_info_names,
    unique_name,
)


def test_v1_models_lists_the_key_models_in_the_order_the_key_names_them(gateway: Gateway) -> None:
    """A key has `models: [C, A, B]` (three fresh deployments) and calls `GET /v1/models`.

    The ids must be exactly `[C, A, B]`: the key's own order, not alphabetical and not deployment creation
    order. A port that sorts or uses a set changes what pickers show first.
    """
    with gateway.scenario() as scenario:
        first: Final = scenario.model()
        second: Final = scenario.model()
        third: Final = scenario.model()
        key: Final = scenario.key(models=[third, first, second])
        assert model_ids(gateway, key) == [third, first, second]


def test_a_repeated_key_entry_is_listed_once(gateway: Gateway) -> None:
    """A key has `models: [A, B, A]` and calls `GET /v1/models`.

    The ids must be exactly `[A, B]`: duplicates are dropped keeping the first position.
    """
    with gateway.scenario() as scenario:
        first: Final = scenario.model()
        second: Final = scenario.model()
        key: Final = scenario.key(models=[first, second, first])
        assert model_ids(gateway, key) == [first, second]


def test_models_and_v1_models_answer_the_same_list(gateway: Gateway) -> None:
    """A key limited to two models calls `GET /models` and `GET /v1/models`.

    Both answers must list the same ids in the same order with `object: "list"`, since both paths are the
    same handler.
    """
    with gateway.scenario() as scenario:
        first: Final = scenario.model()
        second: Final = scenario.model()
        key: Final = scenario.key(models=[first, second])
        versioned: Final = get_as(gateway, key, "/v1/models")
        unversioned: Final = get_as(gateway, key, "/models")
        assert listed_ids(versioned) == listed_ids(unversioned) == [first, second]
        assert body_of(versioned)["object"] == "list"


def test_each_listed_model_has_the_openai_model_object_shape(gateway: Gateway) -> None:
    """A key limited to one model calls `GET /v1/models`.

    The single entry must carry `id` (the model name), `object: "model"`, an integer `created` and
    `owned_by: "openai"` (the listing always reports `openai` as owner, `create_model_info_response`).
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        response: Final = get_as(gateway, key, "/v1/models")
        entries: Final = list_value(body_of(response)["data"])
        assert len(entries) == 1, response.text
        entry: Final = object_value(entries[0])
        assert entry["id"] == model and entry["object"] == "model" and entry["owned_by"] == "openai", entry
        assert isinstance(entry["created"], int), entry


def test_a_team_key_without_its_own_list_lists_the_team_models_in_team_order(gateway: Gateway) -> None:
    """A team has `models: [B, A]` and its key has no list; the key calls `GET /v1/models`.

    The ids must be exactly `[B, A]`.
    """
    with gateway.scenario() as scenario:
        first: Final = scenario.model()
        second: Final = scenario.model()
        team: Final = scenario.team(models=[second, first])
        key: Final = scenario.key(team_id=team)
        assert model_ids(gateway, key) == [second, first]


def test_a_key_list_wider_than_its_team_is_listed_in_full_although_the_extra_model_is_refused(
    gateway: Gateway,
) -> None:
    """A team allows A and B, its key is created with `models: [A, B]`, then the team is narrowed to A.

    `GET /v1/models` still lists `[A, B]` because the key list is the listing source whenever it is non-empty,
    while a call to B is refused with 403 `team_model_access_denied`. Pinned as current behaviour (listing and
    enforcement disagree); see `get_complete_model_list`, which ignores `team_models` when `key_models` is set.
    """
    with gateway.scenario() as scenario:
        first: Final = scenario.model()
        second: Final = scenario.model()
        team: Final = scenario.team(models=[first, second])
        key: Final = scenario.key(team_id=team, models=[first, second])
        gateway.post("/team/update", {"team_id": team, "models": [first]})
        assert_model_denied(chat(gateway, key, second), second, TEAM_DENIED)
        assert model_ids(gateway, key) == [first, second]


def test_a_config_access_group_on_a_key_is_listed_as_its_member_models(gateway: Gateway) -> None:
    """Two deployments declare `model_info.access_groups: [<group>]` and a key has `models: [<group>]`.

    `GET /v1/models` must list both member models and must not list `<group>` itself.
    """
    with gateway.scenario() as scenario:
        group: Final = unique_name("dec1-ma-group")
        first: Final = scenario.model(model_info={"access_groups": [group]})
        second: Final = scenario.model(model_info={"access_groups": [group]})
        key: Final = scenario.key(models=[group])
        listed: Final = model_ids(gateway, key)
        assert first in listed and second in listed and group not in listed, listed


def test_include_model_access_groups_also_lists_the_group_name(gateway: Gateway) -> None:
    """The same key calls `GET /v1/models?include_model_access_groups=true`.

    The listing must contain `<group>` as well as both member models.
    """
    with gateway.scenario() as scenario:
        group: Final = unique_name("dec1-ma-group")
        first: Final = scenario.model(model_info={"access_groups": [group]})
        second: Final = scenario.model(model_info={"access_groups": [group]})
        key: Final = scenario.key(models=[group])
        listed: Final = model_ids(gateway, key, {"include_model_access_groups": "true"})
        assert {group, first, second} <= set(listed), listed


def test_only_model_access_groups_lists_just_the_group_names(gateway: Gateway) -> None:
    """A key has `models: [<group>, <plain model>]` and calls
    `GET /v1/models?include_model_access_groups=true&only_model_access_groups=true`.

    The ids must be exactly `[<group>]`: plain models and member models are dropped.
    """
    with gateway.scenario() as scenario:
        group: Final = unique_name("dec1-ma-group")
        scenario.model(model_info={"access_groups": [group]})
        plain: Final = scenario.model()
        key: Final = scenario.key(models=[group, plain])
        listed: Final = model_ids(
            gateway, key, {"include_model_access_groups": "true", "only_model_access_groups": "true"}
        )
        assert listed == [group], listed


def test_a_name_that_is_both_a_deployed_model_and_an_access_group_is_kept_in_the_listing(gateway: Gateway) -> None:
    """A deployment is named N and another deployment declares `access_groups: [N]`; a key has `models: [N]`.

    The listing must contain N itself and the member model: a grant naming both a deployed model and a group
    means both (`_get_models_from_access_groups` keeps deployed literals).
    """
    with gateway.scenario() as scenario:
        name: Final = scenario.model()
        member: Final = scenario.model(model_info={"access_groups": [name]})
        key: Final = scenario.key(models=[name])
        listed: Final = model_ids(gateway, key)
        assert name in listed and member in listed, listed
        assert_served(chat(gateway, key, name))
        assert_served(chat(gateway, key, member))


def test_an_unknown_scope_is_a_400_with_the_fixed_message(gateway: Gateway) -> None:
    """A key calls `GET /v1/models?scope=everything`.

    The proxy must answer 400 and the body must carry exactly
    `Invalid scope parameter. Only 'expand' is currently supported. Received: everything`. UNVERIFIED: the
    envelope is assumed to be FastAPI's `{"detail": <message>}` because it is a plain `HTTPException`.
    """
    with gateway.scenario() as scenario:
        key: Final = scenario.key(models=[scenario.model()])
        response: Final = get_as(gateway, key, "/v1/models", {"scope": "everything"})
        assert response.status_code == 400, response.text
        assert body_of(response) == {
            "detail": "Invalid scope parameter. Only 'expand' is currently supported. Received: everything"
        }, response.text


def test_scope_expand_is_ignored_for_a_plain_key(gateway: Gateway) -> None:
    """A non-admin key limited to one model calls `GET /v1/models?scope=expand`.

    Expansion is only for proxy, team and org admins, so the ids must still be exactly the key's one model.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        assert model_ids(gateway, key, {"scope": "expand"}) == [model]


def test_scope_expand_for_the_master_key_lists_every_proxy_model(gateway: Gateway) -> None:
    """The master key calls `GET /v1/models?scope=expand` after two deployments are added.

    Both new deployments must be listed, since the expanded view is the whole proxy model list.
    """
    with gateway.scenario() as scenario:
        first: Final = scenario.model()
        second: Final = scenario.model()
        listed: Final = model_ids(gateway, gateway.key, {"scope": "expand"})
        assert first in listed and second in listed


def test_an_undiscoverable_model_is_hidden_from_an_unrestricted_key_but_still_callable(gateway: Gateway) -> None:
    """A deployment has `model_info.discoverable: false`; a key with no model list calls `GET /v1/models` and
    then calls the model.

    The listing must not contain it, and the chat call must be served with 200: hiding is presentation only.
    """
    with gateway.scenario() as scenario:
        hidden: Final = scenario.model(model_info={"discoverable": False})
        visible: Final = scenario.model()
        key: Final = scenario.key()
        listed: Final = model_ids(gateway, key)
        assert visible in listed and hidden not in listed, listed
        assert_served(chat(gateway, key, hidden))


def test_an_undiscoverable_model_named_on_the_key_is_still_hidden(gateway: Gateway) -> None:
    """A key has `models: [<undiscoverable>, <visible>]` and calls `GET /v1/models`.

    The ids must be exactly `[<visible>]`: being granted the model does not make it discoverable.
    """
    with gateway.scenario() as scenario:
        hidden: Final = scenario.model(model_info={"discoverable": False})
        visible: Final = scenario.model()
        key: Final = scenario.key(models=[hidden, visible])
        assert model_ids(gateway, key) == [visible]


def test_the_master_key_still_lists_an_undiscoverable_model(gateway: Gateway) -> None:
    """The master key calls `GET /v1/models` after an undiscoverable deployment is added.

    Admins see everything, so the model must be listed (`undiscoverable_model_names` returns nothing for an
    admin view).
    """
    with gateway.scenario() as scenario:
        hidden: Final = scenario.model(model_info={"discoverable": False})
        assert hidden in model_ids(gateway, gateway.key)


def test_an_undiscoverable_model_is_hidden_on_model_info_and_model_group_info(gateway: Gateway) -> None:
    """A key with no model list calls `GET /model/info` and `GET /model_group/info` after an undiscoverable
    deployment is added.

    Neither answer may contain that model name, while a normal deployment added at the same time appears on
    both.
    """
    with gateway.scenario() as scenario:
        hidden: Final = scenario.model(model_info={"discoverable": False})
        visible: Final = scenario.model()
        key: Final = scenario.key()
        info_names: Final = model_info_names(gateway, key)
        group_names: Final = model_group_names(gateway, key)
        assert visible in info_names and hidden not in info_names
        assert visible in group_names and hidden not in group_names


def test_model_info_for_a_restricted_key_lists_only_its_deployments(gateway: Gateway) -> None:
    """A key limited to model A calls `GET /model/info`; model B also exists.

    Every row's `model_name` must be A (one row per deployment of A), and B must not appear.
    """
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        scenario.model()
        key: Final = scenario.key(models=[allowed])
        assert model_info_names(gateway, key) == [allowed]


def test_model_info_never_returns_the_provider_api_key(gateway: Gateway) -> None:
    """A key limited to one model calls `GET /model/info`.

    The row's `litellm_params` must not contain `api_key`, and the configured provider key
    `integration-provider-key` must not appear anywhere in the response text.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        response: Final = get_as(gateway, key, "/model/info")
        assert response.status_code == 200, response.text
        rows: Final = list_value(body_of(response)["data"])
        assert len(rows) == 1, response.text
        assert "api_key" not in object_value(object_value(rows[0])["litellm_params"]), response.text
        assert "integration-provider-key" not in response.text


def test_model_info_by_an_unknown_id_is_a_400_naming_the_id(gateway: Gateway) -> None:
    """A key calls `GET /model/info?litellm_model_id=<unknown>`.

    The proxy must answer 400 and the body must contain `Model id = <unknown> not found on litellm proxy`.
    UNVERIFIED: only the status and message text are asserted because the envelope comes from a dict detail.
    """
    with gateway.scenario() as scenario:
        key: Final = scenario.key(models=[scenario.model()])
        unknown: Final = unique_name("dec1-ma-unknown-id")
        response: Final = get_as(gateway, key, "/model/info", {"litellm_model_id": unknown})
        assert response.status_code == 400, response.text
        assert f"Model id = {unknown} not found on litellm proxy" in response.text


def test_model_info_by_id_returns_a_deployment_the_key_cannot_call(gateway: Gateway) -> None:
    """A key limited to model A calls `GET /model/info?litellm_model_id=<id of a deployment of B>`.

    The single-id lookup returns B's row with 200 today: it filters only team BYOK rows, not the caller's
    model list. Pinned as current behaviour and flagged in the index as a possible metadata leak; a port that
    applies the listing filter here would answer 400 instead.
    """
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        other: Final = scenario.model()
        key: Final = scenario.key(models=[allowed])
        master_rows: Final = list_value(gateway.get("/model/info")["data"])
        other_id: Final = next(
            object_value(object_value(row)["model_info"])["id"]
            for row in master_rows
            if object_value(row)["model_name"] == other
        )
        assert isinstance(other_id, str)
        assert model_info_names(gateway, key, {"litellm_model_id": other_id}) == [other]


def test_model_group_info_for_a_restricted_key_lists_exactly_its_groups(gateway: Gateway) -> None:
    """A key with `models: [B, A]` calls `GET /model_group/info`.

    The `model_group` values must be exactly `[B, A]` and each row must report `providers: ["openai"]`.
    """
    with gateway.scenario() as scenario:
        first: Final = scenario.model()
        second: Final = scenario.model()
        key: Final = scenario.key(models=[second, first])
        rows: Final = model_group_rows(gateway, key)
        assert [row["model_group"] for row in rows] == [second, first], rows
        assert all(row["providers"] == ["openai"] for row in rows), rows


def test_model_group_info_filtered_to_a_group_the_key_cannot_use_is_empty(gateway: Gateway) -> None:
    """A key limited to A calls `GET /model_group/info?model_group=<B>` for an existing group B.

    The answer must be 200 with `data: []`: the filter is applied inside the caller's allowed names, so it
    cannot be used to read a group outside them.
    """
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        other: Final = scenario.model()
        key: Final = scenario.key(models=[allowed])
        assert model_group_rows(gateway, key, other) == []


def test_model_group_info_lists_a_key_entry_that_no_deployment_serves_with_no_providers(gateway: Gateway) -> None:
    """A key has `models: [<ghost>]`, a name no deployment serves, and calls `GET /model_group/info`.

    The answer contains one row for `<ghost>` with `providers: []` today (`get_model_group_info` emits a stub
    row for a name the router does not know). UNVERIFIED: `_describable_model_groups` may drop it; pin
    whichever the current proxy answers before porting.
    """
    with gateway.scenario() as scenario:
        ghost: Final = unique_name("dec1-ma-ghost")
        key: Final = scenario.key(models=[ghost])
        rows: Final = model_group_rows(gateway, key)
        assert [(row["model_group"], row["providers"]) for row in rows] == [(ghost, [])], rows


def test_a_team_model_is_listed_under_its_public_name_on_model_info_for_its_team(gateway: Gateway) -> None:
    """A deployment is created with `model_info.team_id: <team>`; a key of that team calls `GET /model/info`.

    The row's `model_name` must be the public name the deployment was created with, never the internal
    routing key `model_name_<team>_<uuid>` (`_translate_model_name_for_response`).
    """
    with gateway.scenario() as scenario:
        team: Final = scenario.team(models=[])
        model: Final = scenario.model(model_info={"team_id": team})
        key: Final = scenario.key(team_id=team)
        names: Final = model_info_names(gateway, key)
        assert model in names, names
        assert not any(name.startswith(f"model_name_{team}_") for name in names), names


def test_a_team_model_is_absent_from_another_teams_model_group_info(gateway: Gateway) -> None:
    """A deployment belongs to team one; a key of team two (no list) calls `GET /model_group/info`.

    The team one model must not be listed for team two.
    """
    with gateway.scenario() as scenario:
        owner: Final = scenario.team(models=[])
        stranger: Final = scenario.team(models=[])
        model: Final = scenario.model(model_info={"team_id": owner})
        key: Final = scenario.key(team_id=stranger)
        assert model not in model_group_names(gateway, key)


@pytest.mark.parametrize("path", ["/v1/models", "/model/info", "/model_group/info"])
def test_a_key_with_no_default_models_still_lists_every_proxy_model(gateway: Gateway, path: str) -> None:
    """A user has `models: ["no-default-models"]` and a personal key with no list calls a listing route.

    Pinned current behaviour: the listing reads only the key and team lists. With both empty,
    `get_complete_model_list` lists every proxy model (litellm/proxy/auth/model_checks.py:310-317) and drops only the
    sentinel name itself (model_checks.py:307), so a fresh model created just before is listed although the same key
    is refused on it at inference (`can_user_call_model`, auth_checks.py:6247), and `no-default-models` never
    appears. Same layering as `test_a_key_list_wider_than_its_team_is_listed_in_full_although_the_extra_model_is_refused`.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        user: Final = scenario.user(models=["no-default-models"])
        key: Final = scenario.key(user_id=user)
        response: Final = get_as(gateway, key, path)
        assert response.status_code == 200, response.text
        assert model in response.text and "no-default-models" not in response.text, response.text
