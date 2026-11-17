"""Model aliases and model access: which name each alias layer checks, and what the caller then reaches.

Contract (litellm/proxy/auth/auth_checks.py `_can_object_call_model`, `_model_in_team_aliases`,
`check_model_access_helper`; litellm/proxy/common_utils/model_listing_utils.py `_alias_layers`,
`alias_listing_entries`):

* Key `aliases` resolve BEFORE the access check. When a key alias applies, only the model it dispatches to is
  checked against every list, never the alias name, so listing the alias name in `models` grants nothing and
  a denial still names the alias the caller sent.
* Team `model_aliases` are checked AS the alias. Any name that is a key of the team's alias map passes the
  key, team and member checks outright, so a team alias reaches its target even when the target is on no
  list. Pinned as current behaviour: a port that resolves team aliases first would tighten access.
* Router `model_group_alias` and the global `litellm_settings.model_alias_map` are checked as "either": the
  request passes when the alias name or its target is allowed.
* `/v1/models` adds one row per key or team alias whose target it lists, after the plain models, and
  `/model_group/info` lists those aliases too (#51754). A `model_group_alias` entry marked `hidden: true` is
  callable but never listed.

The router alias tests boot one owned proxy for the module whose deployments point at an owned wire peer, so
the alias targets the upstream receives can be read exactly.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final

import pytest

from tests.integration._support.client import JSON_OBJECT, Gateway, gateway_from_environment, string_value
from tests.integration._support.process import owned_proxy
from tests.integration._support.wire import Reply, Request, Wire, wire_server
from tests.integration.authorization.dec1_modelaccess_support import (
    KEY_DENIED,
    TEAM_DENIED,
    assert_model_denied,
    assert_served,
    chat,
    chat_reply,
    marker,
    model_group_names,
    model_ids,
    owned_config,
    unique_name,
    upstream_models,
    wire_deployment,
)


def test_a_key_alias_is_checked_as_its_target_so_the_target_on_the_list_serves_it(gateway: Gateway) -> None:
    """A key has `models: [<target>]` and `aliases: {<alias>: <target>}`, and calls `<alias>`.

    The call must be served with 200 and the upstream must receive the target deployment's provider model,
    because the key alias resolves before the access check.
    """
    with gateway.scenario() as scenario:
        provider: Final = f"dec1-ma-target-{uuid.uuid4().hex}"
        target: Final = scenario.model(model=f"openai/{provider}")
        alias: Final = unique_name("dec1-ma-alias")
        key: Final = scenario.key(models=[target], aliases={alias: target})
        text: Final = marker()
        assert_served(chat(gateway, key, alias, text))
        assert upstream_models(gateway, text) == [provider]


def test_listing_only_the_alias_name_on_a_key_grants_nothing(gateway: Gateway) -> None:
    """A key has `models: [<alias>]` and `aliases: {<alias>: <target>}`, and calls `<alias>`.

    The call must be refused with the exact 403 `key_model_access_denied` body naming `<alias>`: once a key
    alias applies only its target is checked, and the target is not on the list.
    """
    with gateway.scenario() as scenario:
        target: Final = scenario.model()
        alias: Final = unique_name("dec1-ma-alias")
        key: Final = scenario.key(models=[alias], aliases={alias: target})
        text: Final = marker()
        assert_model_denied(chat(gateway, key, alias, text), alias, KEY_DENIED)


def test_a_key_alias_can_shadow_a_real_group_and_redirect_it(gateway: Gateway) -> None:
    """Two deployments A and B exist; a key has `models: [B]` and `aliases: {A: B}`, and calls `A`.

    The call must be served and the upstream must receive B's provider model, not A's: the key alias wins
    over a deployed group of the same name. Calling A without the alias would be a 403.
    """
    with gateway.scenario() as scenario:
        provider_b: Final = f"dec1-ma-b-{uuid.uuid4().hex}"
        first: Final = scenario.model()
        second: Final = scenario.model(model=f"openai/{provider_b}")
        key: Final = scenario.key(models=[second], aliases={first: second})
        text: Final = marker()
        assert_served(chat(gateway, key, first, text))
        assert upstream_models(gateway, text) == [provider_b]


def test_a_team_list_is_checked_against_the_key_alias_target(gateway: Gateway) -> None:
    """A team allows only `<target>`; its key has no list and `aliases: {<alias>: <other>}` where `<other>` exists.

    Calling `<alias>` must be refused with 403 `team_model_access_denied` naming `<alias>`: the team check
    also follows the key alias to `<other>`, which the team does not allow.
    """
    with gateway.scenario() as scenario:
        target: Final = scenario.model()
        other: Final = scenario.model()
        team: Final = scenario.team(models=[target])
        alias: Final = unique_name("dec1-ma-alias")
        key: Final = scenario.key(team_id=team, aliases={alias: other})
        assert_model_denied(chat(gateway, key, alias), alias, TEAM_DENIED)


def test_a_key_alias_to_a_team_allowed_model_is_served_for_a_team_key(gateway: Gateway) -> None:
    """A team allows `<target>`; its key has no list and `aliases: {<alias>: <target>}`.

    Calling `<alias>` must be served with 200 and reach `<target>`'s provider model.
    """
    with gateway.scenario() as scenario:
        provider: Final = f"dec1-ma-team-target-{uuid.uuid4().hex}"
        target: Final = scenario.model(model=f"openai/{provider}")
        team: Final = scenario.team(models=[target])
        alias: Final = unique_name("dec1-ma-alias")
        key: Final = scenario.key(team_id=team, aliases={alias: target})
        text: Final = marker()
        assert_served(chat(gateway, key, alias, text))
        assert upstream_models(gateway, text) == [provider]


def test_a_team_alias_passes_the_key_check_even_when_the_key_list_lacks_its_target(gateway: Gateway) -> None:
    """A team allows A and B and has `model_aliases: {<alias>: B}`; its key has `models: [A]`.

    Calling `<alias>` is served with 200 and reaches B today, although calling B directly is a 403
    `key_model_access_denied`: a team alias name passes `check_model_access_helper` through
    `_model_in_team_aliases` before the key list is consulted. Pinned as current behaviour; a port that
    resolves team aliases before the key check would refuse it.
    """
    with gateway.scenario() as scenario:
        provider_b: Final = f"dec1-ma-team-alias-{uuid.uuid4().hex}"
        first: Final = scenario.model()
        second: Final = scenario.model(model=f"openai/{provider_b}")
        alias: Final = unique_name("dec1-ma-team-alias")
        team: Final = scenario.team(models=[first, second], model_aliases={alias: second})
        key: Final = scenario.key(team_id=team, models=[first])
        assert_model_denied(chat(gateway, key, second), second, KEY_DENIED)
        text: Final = marker()
        assert_served(chat(gateway, key, alias, text))
        assert upstream_models(gateway, text) == [provider_b]


def test_a_team_alias_reaches_a_target_the_team_list_does_not_name(gateway: Gateway) -> None:
    """A team allows only A and has `model_aliases: {<alias>: B}`; its key has no list.

    Calling B is a 403 `team_model_access_denied`, but calling `<alias>` is served and reaches B: the team
    alias map is part of what the team grants. UNVERIFIED that no later step re-checks the dispatched model
    on this route; read from `_can_object_call_model` and `common_checks`, where the team check receives the
    alias name.
    """
    with gateway.scenario() as scenario:
        provider_b: Final = f"dec1-ma-team-only-alias-{uuid.uuid4().hex}"
        first: Final = scenario.model()
        second: Final = scenario.model(model=f"openai/{provider_b}")
        alias: Final = unique_name("dec1-ma-team-alias")
        team: Final = scenario.team(models=[first], model_aliases={alias: second})
        key: Final = scenario.key(team_id=team)
        assert_model_denied(chat(gateway, key, second), second, TEAM_DENIED)
        text: Final = marker()
        assert_served(chat(gateway, key, alias, text))
        assert upstream_models(gateway, text) == [provider_b]


def test_a_team_alias_does_not_help_a_key_of_another_team(gateway: Gateway) -> None:
    """Team one has `model_aliases: {<alias>: A}`; a key of team two (which allows only B) calls `<alias>`.

    Aliases belong to the team the key authenticated as, so the call must be refused with 403
    `team_model_access_denied` naming `<alias>`.
    """
    with gateway.scenario() as scenario:
        first: Final = scenario.model()
        second: Final = scenario.model()
        alias: Final = unique_name("dec1-ma-team-alias")
        scenario.team(models=[first], model_aliases={alias: first})
        other_team: Final = scenario.team(models=[second])
        key: Final = scenario.key(team_id=other_team)
        assert_model_denied(chat(gateway, key, alias), alias, TEAM_DENIED)


def test_v1_models_lists_aliases_after_the_plain_models(gateway: Gateway) -> None:
    """A key has `models: [A, B]` and `aliases: {<alias>: A}` and calls `GET /v1/models`.

    The ids must be exactly `[A, B, <alias>]` in that order: plain models in key list order, then one row
    per alias whose target is listed.
    """
    with gateway.scenario() as scenario:
        first: Final = scenario.model()
        second: Final = scenario.model()
        alias: Final = unique_name("dec1-ma-alias")
        key: Final = scenario.key(models=[first, second], aliases={alias: first})
        assert model_ids(gateway, key) == [first, second, alias]


def test_a_team_alias_whose_target_is_listed_appears_in_v1_models(gateway: Gateway) -> None:
    """A team allows A and has `model_aliases: {<alias>: A}`; its key has no list and calls `GET /v1/models`.

    The listing must contain both A and `<alias>`.
    """
    with gateway.scenario() as scenario:
        first: Final = scenario.model()
        alias: Final = unique_name("dec1-ma-team-alias")
        team: Final = scenario.team(models=[first], model_aliases={alias: first})
        key: Final = scenario.key(team_id=team)
        listed: Final = model_ids(gateway, key)
        assert first in listed and alias in listed, listed


def test_a_team_alias_to_an_unlisted_target_is_callable_but_not_listed(gateway: Gateway) -> None:
    """A team allows A and has `model_aliases: {<alias>: B}`; its key has no list.

    `<alias>` is served (see above) but `GET /v1/models` must not list it, because aliases are only added
    when their target is listed. Pinned as current behaviour: listing and enforcement disagree here.
    """
    with gateway.scenario() as scenario:
        first: Final = scenario.model()
        second: Final = scenario.model()
        alias: Final = unique_name("dec1-ma-team-alias")
        team: Final = scenario.team(models=[first], model_aliases={alias: second})
        key: Final = scenario.key(team_id=team)
        assert_served(chat(gateway, key, alias))
        assert alias not in model_ids(gateway, key)


def test_model_group_info_lists_a_key_alias_as_its_own_group(gateway: Gateway) -> None:
    """A key has `models: [A]` and `aliases: {<alias>: A}` and calls `GET /model_group/info`.

    The `model_group` values must include both A and `<alias>` (#51754 made key and team aliases visible on
    this route, landed in the last week).
    """
    with gateway.scenario() as scenario:
        first: Final = scenario.model()
        alias: Final = unique_name("dec1-ma-alias")
        key: Final = scenario.key(models=[first], aliases={alias: first})
        names: Final = model_group_names(gateway, key)
        assert first in names and alias in names, names


def test_model_group_info_filtered_by_a_key_alias_returns_one_row_named_after_the_alias(gateway: Gateway) -> None:
    """The same key calls `GET /model_group/info?model_group=<alias>`.

    The answer must hold exactly one row whose `model_group` is `<alias>` (not the target's name), described
    from the target's deployments, so its `providers` is `["openai"]`.
    """
    with gateway.scenario() as scenario:
        first: Final = scenario.model()
        alias: Final = unique_name("dec1-ma-alias")
        key: Final = scenario.key(models=[first], aliases={alias: first})
        response: Final = gateway.request("GET", "/model_group/info", key=key, params={"model_group": alias})
        assert response.status_code == 200, response.text
        rows: Final = JSON_OBJECT.validate_json(response.content)["data"]
        assert isinstance(rows, list) and len(rows) == 1, response.text
        row: Final = rows[0]
        assert isinstance(row, dict) and row["model_group"] == alias and row["providers"] == ["openai"], row


def test_a_key_level_model_group_alias_is_resolved_before_the_key_check(gateway: Gateway) -> None:
    """A key has `models: [<target>]` and `router_settings: {"model_group_alias": {<alias>: <target>}}`.

    Calling `<alias>` must be served and reach `<target>`'s provider model: a key's or team's own
    `router_settings.model_group_alias` rewrites the body model before the allowlist checks
    (`_resolve_router_settings_model_group_alias`).
    """
    with gateway.scenario() as scenario:
        provider: Final = f"dec1-ma-key-router-{uuid.uuid4().hex}"
        target: Final = scenario.model(model=f"openai/{provider}")
        alias: Final = unique_name("dec1-ma-key-router-alias")
        key: Final = scenario.key(models=[target], router_settings={"model_group_alias": {alias: target}})
        text: Final = marker()
        assert_served(chat(gateway, key, alias, text))
        assert upstream_models(gateway, text) == [provider]


def test_a_key_level_model_group_alias_name_on_the_list_does_not_grant_its_target(gateway: Gateway) -> None:
    """A key has `models: [<alias>]` and `router_settings: {"model_group_alias": {<alias>: <target>}}`.

    Calling `<alias>` must be refused with 403 `key_model_access_denied`, and because the body was
    rewritten before the check the denial names `<target>`, not the alias the caller sent. This is the
    opposite of the proxy-wide `model_group_alias`, which allows either name. UNVERIFIED: the message names
    the rewritten model because `rewrite_request_model` runs before `_get_model_from_request_context`.
    """
    with gateway.scenario() as scenario:
        target: Final = scenario.model()
        alias: Final = unique_name("dec1-ma-key-router-alias")
        key: Final = scenario.key(models=[alias], router_settings={"model_group_alias": {alias: target}})
        assert_model_denied(chat(gateway, key, alias), target, KEY_DENIED)


# --- router model_group_alias and the global model_alias_map, on an owned proxy -------------------------


@dataclass(frozen=True, slots=True)
class _AliasRig:
    proxy: Gateway
    wire: Wire
    target: str
    other: str
    router_alias: str
    hidden_alias: str
    global_alias: str


def _reply(request: Request) -> Reply:
    if request.method != "POST" or not request.body:
        return Reply(body=b'{"object":"list","data":[]}')
    body: Final = JSON_OBJECT.validate_json(request.body)
    return Reply(body=chat_reply(string_value(body["model"])))


@pytest.fixture(scope="module")
def alias_rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_AliasRig]:
    directory: Final = tmp_path_factory.mktemp("dec1-modelaccess-aliases")
    target: Final = unique_name("dec1-ma-router-target")
    other: Final = unique_name("dec1-ma-router-other")
    router_alias: Final = unique_name("dec1-ma-router-alias")
    hidden_alias: Final = unique_name("dec1-ma-hidden-alias")
    global_alias: Final = unique_name("dec1-ma-global-alias")
    with wire_server(_reply) as wire:
        config: Final = owned_config(
            directory,
            [
                wire_deployment(target, f"provider-{target}", wire.url),
                wire_deployment(other, f"provider-{other}", wire.url),
            ],
            router_settings={
                "model_group_alias": {router_alias: target, hidden_alias: {"model": target, "hidden": True}},
            },
            litellm_settings={"model_alias_map": {global_alias: target}},
        )
        with gateway_from_environment() as shared, owned_proxy(shared, directory / "proxy", {}, config=config) as owned:
            yield _AliasRig(owned, wire, target, other, router_alias, hidden_alias, global_alias)


def _wire_models(wire: Wire, text: str) -> list[str]:
    return [
        string_value(JSON_OBJECT.validate_json(request.body)["model"])
        for request in wire.drain()
        if text.encode() in request.body
    ]


def test_a_key_allowed_the_target_may_call_a_router_alias(alias_rig: _AliasRig) -> None:
    """`router_settings.model_group_alias` maps `<alias>` to `<target>`, and a key has `models: [<target>]`.

    Calling `<alias>` must be served and the peer must receive `provider-<target>`: the alias target is one
    of the names checked.
    """
    with alias_rig.proxy.scenario() as scenario:
        key: Final = scenario.key(models=[alias_rig.target])
        text: Final = marker()
        assert_served(chat(alias_rig.proxy, key, alias_rig.router_alias, text))
        assert _wire_models(alias_rig.wire, text) == [f"provider-{alias_rig.target}"]


def test_a_key_allowed_only_the_router_alias_name_reaches_its_target(alias_rig: _AliasRig) -> None:
    """A key has `models: [<alias>]` only and calls `<alias>`.

    The call must be served and reach `provider-<target>`: unlike a key alias, a router alias is allowed
    when either the alias name or the target is on the list. Calling `<target>` directly with the same key
    must be a 403 `key_model_access_denied`.
    """
    with alias_rig.proxy.scenario() as scenario:
        key: Final = scenario.key(models=[alias_rig.router_alias])
        text: Final = marker()
        assert_served(chat(alias_rig.proxy, key, alias_rig.router_alias, text))
        assert _wire_models(alias_rig.wire, text) == [f"provider-{alias_rig.target}"]
        assert_model_denied(chat(alias_rig.proxy, key, alias_rig.target), alias_rig.target, KEY_DENIED)


def test_a_key_allowed_neither_name_is_refused_on_a_router_alias(alias_rig: _AliasRig) -> None:
    """A key has `models: [<other>]` and calls the router alias of `<target>`.

    The answer must be the exact 403 `key_model_access_denied` body naming the alias.
    """
    with alias_rig.proxy.scenario() as scenario:
        key: Final = scenario.key(models=[alias_rig.other])
        assert_model_denied(chat(alias_rig.proxy, key, alias_rig.router_alias), alias_rig.router_alias, KEY_DENIED)


def test_a_hidden_router_alias_is_callable_but_never_listed(alias_rig: _AliasRig) -> None:
    """`model_group_alias` maps `<hidden>` to `{"model": <target>, "hidden": true}`; a key allowed `<target>`
    calls `<hidden>` and then `GET /v1/models`.

    The call must be served and reach `provider-<target>`, and the listing must not contain `<hidden>`.
    """
    with alias_rig.proxy.scenario() as scenario:
        key: Final = scenario.key(models=[alias_rig.target])
        text: Final = marker()
        assert_served(chat(alias_rig.proxy, key, alias_rig.hidden_alias, text))
        assert _wire_models(alias_rig.wire, text) == [f"provider-{alias_rig.target}"]
        assert alias_rig.hidden_alias not in model_ids(alias_rig.proxy, key)


def test_a_global_model_alias_map_entry_is_allowed_through_its_target(alias_rig: _AliasRig) -> None:
    """`litellm_settings.model_alias_map` maps `<global>` to `<target>`, and a key has `models: [<target>]`.

    Calling `<global>` must be served and reach `provider-<target>`.
    """
    with alias_rig.proxy.scenario() as scenario:
        key: Final = scenario.key(models=[alias_rig.target])
        text: Final = marker()
        assert_served(chat(alias_rig.proxy, key, alias_rig.global_alias, text))
        assert _wire_models(alias_rig.wire, text) == [f"provider-{alias_rig.target}"]


def test_a_key_allowed_only_the_global_alias_name_reaches_its_target(alias_rig: _AliasRig) -> None:
    """A key has `models: [<global>]` only and calls `<global>`.

    The call must be served and reach `provider-<target>`: the global alias map, like the router alias, is
    allowed when either name is on the list. Pinned because the key alias layer behaves the opposite way.
    """
    with alias_rig.proxy.scenario() as scenario:
        key: Final = scenario.key(models=[alias_rig.global_alias])
        text: Final = marker()
        assert_served(chat(alias_rig.proxy, key, alias_rig.global_alias, text))
        assert _wire_models(alias_rig.wire, text) == [f"provider-{alias_rig.target}"]


def test_a_key_alias_wins_over_a_router_alias_of_the_same_name(alias_rig: _AliasRig) -> None:
    """A key has `models: [<other>]` and `aliases: {<router alias>: <other>}`, and calls the router alias name.

    The key alias must win: the call is served and reaches `provider-<other>`, not the router target. The
    proxy-wide `model_group_alias` is applied by the router after the key alias rewrote the body.
    UNVERIFIED: ordering read from `CallerAliases.rewrite` and `_alias_layers`, which list the router layer
    on both ends.
    """
    with alias_rig.proxy.scenario() as scenario:
        key: Final = scenario.key(models=[alias_rig.other], aliases={alias_rig.router_alias: alias_rig.other})
        text: Final = marker()
        assert_served(chat(alias_rig.proxy, key, alias_rig.router_alias, text))
        assert _wire_models(alias_rig.wire, text) == [f"provider-{alias_rig.other}"]
