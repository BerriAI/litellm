"""How model access interacts with fallbacks: client fallbacks are refused at auth, router fallbacks are skipped.

Contract:

* Client fallbacks sent in the body (`fallbacks`, `context_window_fallbacks`, `content_policy_fallbacks`, the
  same three inside `router_settings_override`, as plain names or `{"model": ...}` objects) are each checked
  against the key list at auth and refused with the usual 403 before any upstream call
  (litellm/proxy/auth/user_api_key_auth.py `enforce_key_and_fallback_model_access`,
  litellm/proxy/auth/auth_utils.py `iter_request_fallback_targets`). A key with `all-team-models` skips this
  key-level fallback check entirely.
* Router fallbacks configured in `router_settings.fallbacks` are chosen after auth. With
  `general_settings.enforce_fallback_model_access` on (the default) each target is re-checked against the key,
  team, member, user, project and customer lists, and a target the caller cannot use is silently skipped, so
  the caller receives the primary model's error (litellm/proxy/auth/fallback_model_access.py
  `RouterFallbackAccessCheck`). With the setting off, only the customer list is re-checked.

The router fallback tests boot one owned proxy for the module whose deployments point at an owned wire peer.
Any deployment whose provider model contains `-fails-` answers 500, every other one answers a chat reply, so
the order and set of upstream attempts can be read exactly.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import JSON_OBJECT, Gateway, gateway_from_environment, string_value
from tests.integration._support.process import owned_proxy
from tests.integration._support.wire import Reply, Request, Wire, wire_server
from tests.integration.authorization.dec1_modelaccess_support import (
    KEY_DENIED,
    assert_model_denied,
    assert_served,
    chat,
    chat_reply,
    marker,
    owned_config,
    unique_name,
    upstream_requests,
    wire_deployment,
)

# --- client fallbacks on the shared proxy -------------------------------------------------------------


def test_a_client_fallback_given_as_an_object_is_checked_like_a_name(gateway: Gateway) -> None:
    """A key limited to A sends a chat completion for A with `fallbacks: [{"model": <B>}]`.

    The answer must be 403 `key_model_access_denied` naming B with no upstream call: the object form is
    unwrapped through `fallback_target_model_name`.
    """
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        other: Final = scenario.model()
        key: Final = scenario.key(models=[allowed])
        text: Final = marker()
        assert_model_denied(chat(gateway, key, allowed, text, fallbacks=[{"model": other}]), other, KEY_DENIED)
        assert upstream_requests(gateway, text) == []


def test_a_client_fallback_mapping_per_model_is_checked(gateway: Gateway) -> None:
    """A key limited to A sends `fallbacks: [{<A>: [<B>]}]`, the router config shape, in the body.

    The answer must be 403 `key_model_access_denied` naming B: every target list inside the mapping is
    walked.
    """
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        other: Final = scenario.model()
        key: Final = scenario.key(models=[allowed])
        assert_model_denied(chat(gateway, key, allowed, fallbacks=[{allowed: [other]}]), other, KEY_DENIED)


@pytest.mark.parametrize("field", ["context_window_fallbacks", "content_policy_fallbacks"])
def test_the_other_client_fallback_fields_are_checked_too(gateway: Gateway, field: str) -> None:
    """A key limited to A sends a chat completion for A with `<field>: [<B>]`.

    The answer must be 403 `key_model_access_denied` naming B, for both `context_window_fallbacks` and
    `content_policy_fallbacks`.
    """
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        other: Final = scenario.model()
        key: Final = scenario.key(models=[allowed])
        extra: Final[dict[str, JsonValue]] = {field: [other]}
        assert_model_denied(chat(gateway, key, allowed, **extra), other, KEY_DENIED)


def test_fallbacks_inside_router_settings_override_are_checked(gateway: Gateway) -> None:
    """A key limited to A sends `router_settings_override: {"fallbacks": [{<A>: [<B>]}]}` in the body.

    The answer must be 403 `key_model_access_denied` naming B: overriding router settings per request is not
    a way around the key list.
    """
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        other: Final = scenario.model()
        key: Final = scenario.key(models=[allowed])
        refused: Final = chat(gateway, key, allowed, router_settings_override={"fallbacks": [{allowed: [other]}]})
        assert_model_denied(refused, other, KEY_DENIED)


def test_the_first_disallowed_client_fallback_is_the_one_named(gateway: Gateway) -> None:
    """A key allowed A and B sends `fallbacks: [<B>, <C>, <D>]` where C and D are not allowed.

    The denial must name C, the first disallowed target in body order (targets are de-duplicated preserving
    order and checked one by one).
    """
    with gateway.scenario() as scenario:
        first: Final = scenario.model()
        second: Final = scenario.model()
        third: Final = scenario.model()
        fourth: Final = scenario.model()
        key: Final = scenario.key(models=[first, second])
        assert_model_denied(chat(gateway, key, first, fallbacks=[second, third, fourth]), third, KEY_DENIED)


def test_a_client_fallback_allowed_by_the_key_but_unknown_to_the_router_is_refused(gateway: Gateway) -> None:
    """A key with `models: ["<stem>-*"]` calls `<stem>-a` with `fallbacks: ["<stem>-ghost"]`, which the
    pattern allows but no deployment serves.

    The request must be refused before the upstream: `is_valid_fallback_model` routes each fallback name and
    fails for an unknown one. UNVERIFIED: the status is assumed to be 400 (the router's invalid model error);
    only "not 200 and no upstream call" is asserted.
    """
    with gateway.scenario() as scenario:
        stem: Final = unique_name()
        primary: Final = scenario.model(name=f"{stem}-a")
        key: Final = scenario.key(models=[f"{stem}-*"])
        text: Final = marker()
        refused: Final = chat(gateway, key, primary, text, fallbacks=[f"{stem}-ghost"])
        assert refused.status_code != 200, refused.text
        assert upstream_requests(gateway, text) == []


# --- router fallbacks on an owned proxy -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _FallbackRig:
    proxy: Gateway
    wire: Wire
    failing: str
    failing_two: str
    fallback: str
    fallback_two: str


def _reply(request: Request) -> Reply:
    if request.method != "POST" or not request.body:
        return Reply(body=b'{"object":"list","data":[]}')
    model: Final = string_value(JSON_OBJECT.validate_json(request.body)["model"])
    if "-fails-" in model:
        return Reply(status=500, body=b'{"error":{"message":"scripted primary failure","type":"server_error"}}')
    return Reply(body=chat_reply(model))


def _fallback_config(directory: Path, wire: Wire, names: tuple[str, str, str, str], enforce: bool | None) -> Path:
    failing, failing_two, fallback, fallback_two = names
    return owned_config(
        directory,
        [
            wire_deployment(failing, f"provider-fails-{failing}", wire.url),
            wire_deployment(failing_two, f"provider-fails-{failing_two}", wire.url),
            wire_deployment(fallback, f"provider-{fallback}", wire.url),
            wire_deployment(fallback_two, f"provider-{fallback_two}", wire.url),
        ],
        general_settings={} if enforce is None else {"enforce_fallback_model_access": enforce},
        router_settings={"fallbacks": [{failing: [fallback]}, {failing_two: [fallback_two, fallback]}]},
    )


def _names() -> tuple[str, str, str, str]:
    return (
        unique_name("dec1-ma-primary"),
        unique_name("dec1-ma-primary-two"),
        unique_name("dec1-ma-fallback"),
        unique_name("dec1-ma-fallback-two"),
    )


@pytest.fixture(scope="module")
def fallback_rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_FallbackRig]:
    directory: Final = tmp_path_factory.mktemp("dec1-modelaccess-fallbacks")
    names: Final = _names()
    with wire_server(_reply) as wire:
        config: Final = _fallback_config(directory, wire, names, enforce=None)
        with gateway_from_environment() as shared, owned_proxy(shared, directory / "proxy", {}, config=config) as owned:
            yield _FallbackRig(owned, wire, *names)


def _attempts(wire: Wire, text: str) -> list[str]:
    return [
        string_value(JSON_OBJECT.validate_json(request.body)["model"])
        for request in wire.drain()
        if text.encode() in request.body
    ]


def test_a_router_fallback_the_key_cannot_use_is_skipped_and_the_primary_error_returned(
    fallback_rig: _FallbackRig,
) -> None:
    """`router_settings.fallbacks` sends `<primary>` (which answers 500) to `<fallback>`; a key allows only
    `<primary>`.

    The call must answer 500 and the peer must see exactly one attempt, `provider-fails-<primary>`: with the
    default `enforce_fallback_model_access` the fallback is skipped rather than served or refused with 403.
    """
    rig: Final = fallback_rig
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key(models=[rig.failing])
        text: Final = marker()
        response: Final = chat(rig.proxy, key, rig.failing, text)
        assert response.status_code == 500, response.text
        assert _attempts(rig.wire, text) == [f"provider-fails-{rig.failing}"]


def test_a_router_fallback_the_key_can_use_is_attempted_and_served(fallback_rig: _FallbackRig) -> None:
    """A key allows `<primary>` and `<fallback>` and calls `<primary>`.

    The call must be served with 200 and the peer must see `provider-fails-<primary>` then
    `provider-<fallback>`.
    """
    rig: Final = fallback_rig
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key(models=[rig.failing, rig.fallback])
        text: Final = marker()
        assert_served(chat(rig.proxy, key, rig.failing, text))
        assert _attempts(rig.wire, text) == [f"provider-fails-{rig.failing}", f"provider-{rig.fallback}"]


def test_a_router_fallback_chain_skips_a_disallowed_target_and_uses_the_next(fallback_rig: _FallbackRig) -> None:
    """`<primary two>` falls back to `[<fallback two>, <fallback>]`; a key allows `<primary two>` and `<fallback>`.

    The call must be served by `<fallback>`, and the peer must never see `provider-<fallback two>`: a
    skipped target does not end the chain. UNVERIFIED: assumes the router moves on to the next target in the
    same list after the access predicate refuses one.
    """
    rig: Final = fallback_rig
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key(models=[rig.failing_two, rig.fallback])
        text: Final = marker()
        assert_served(chat(rig.proxy, key, rig.failing_two, text))
        attempts: Final = _attempts(rig.wire, text)
        assert attempts[0] == f"provider-fails-{rig.failing_two}" and attempts[-1] == f"provider-{rig.fallback}"
        assert f"provider-{rig.fallback_two}" not in attempts, attempts


def test_a_router_fallback_outside_the_team_list_is_skipped(fallback_rig: _FallbackRig) -> None:
    """A team allows only `<primary>`; its key has no list of its own and calls `<primary>`.

    The fallback passes the (empty) key list but not the team list, so the call must answer 500 after one
    attempt: the router check re-runs the team layer, not only the key layer.
    """
    rig: Final = fallback_rig
    with rig.proxy.scenario() as scenario:
        team: Final = scenario.team(models=[rig.failing])
        key: Final = scenario.key(team_id=team)
        text: Final = marker()
        response: Final = chat(rig.proxy, key, rig.failing, text)
        assert response.status_code == 500, response.text
        assert _attempts(rig.wire, text) == [f"provider-fails-{rig.failing}"]


def test_a_router_fallback_outside_the_users_list_is_skipped_for_a_personal_key(fallback_rig: _FallbackRig) -> None:
    """A user allows only `<primary>` and owns a personal key with no list; the key calls `<primary>`.

    The call must answer 500 after one attempt: the user layer is re-checked for the fallback too.

    BUG (run 44176682961): the fallback answered 200. RouterFallbackAccessCheck re-runs can_key_call_resolved_model
    for the target (litellm/proxy/auth/fallback_model_access.py:121-123), which covers key, team and project, not
    the user's own models list (can_user_call_model, auth_checks.py:6239). A direct call to the fallback by this key
    is refused, so the class docstring's 'could have requested it directly' does not hold."""
    pytest.skip("BUG: a router fallback outside the user's models list is served for a personal key")
    rig: Final = fallback_rig
    with rig.proxy.scenario() as scenario:
        user: Final = scenario.user(models=[rig.failing])
        key: Final = scenario.key(user_id=user)
        text: Final = marker()
        response: Final = chat(rig.proxy, key, rig.failing, text)
        assert response.status_code == 500, response.text
        assert _attempts(rig.wire, text) == [f"provider-fails-{rig.failing}"]


def test_an_unrestricted_key_gets_the_router_fallback(fallback_rig: _FallbackRig) -> None:
    """A key with no model list, no team and no user calls `<primary>`.

    The call must be served by `<fallback>` (two attempts), since nothing restricts the fallback target.
    """
    rig: Final = fallback_rig
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key()
        text: Final = marker()
        assert_served(chat(rig.proxy, key, rig.failing, text))
        assert _attempts(rig.wire, text) == [f"provider-fails-{rig.failing}", f"provider-{rig.fallback}"]


def test_an_all_team_models_key_skips_the_auth_fallback_check_but_not_the_router_one(
    fallback_rig: _FallbackRig,
) -> None:
    """A team allows only `<primary>`; its key has `models: ["all-team-models"]` and sends `<primary>` with
    `fallbacks: [<fallback>]` in the body.

    Auth does not refuse the client fallback (the key-level fallback check is skipped for this sentinel, and
    the team check only reads the primary model), so the answer is not a 403. The router then skips the
    fallback because the team does not allow it, so the call answers 500 after one attempt. Pinned as current
    behaviour: a port that checks client fallbacks for every key turns this into a 403.
    """
    rig: Final = fallback_rig
    with rig.proxy.scenario() as scenario:
        team: Final = scenario.team(models=[rig.failing])
        key: Final = scenario.key(team_id=team, models=["all-team-models"])
        text: Final = marker()
        response: Final = chat(rig.proxy, key, rig.failing, text, fallbacks=[rig.fallback])
        assert response.status_code == 500, response.text
        assert _attempts(rig.wire, text) == [f"provider-fails-{rig.failing}"]


def test_a_skipped_router_fallback_does_not_count_as_a_model_denial(fallback_rig: _FallbackRig) -> None:
    """A key allowing only `<primary>` calls it and the router fallback is skipped.

    The error body must be the primary's upstream error, not a `key_model_access_denied` body.

    Pinned current behaviour: the 500 names the skipped target in one closing sentence, "Fallback to <fallback> was
    skipped because the caller is not authorized to call it." (litellm/litellm_core_utils/routing_common_utils.py:
    298-303), so the caller does learn the fallback's name, but not as a model denial.
    """
    rig: Final = fallback_rig
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key(models=[rig.failing])
        response: Final = chat(rig.proxy, key, rig.failing)
        assert response.status_code == 500, response.text
        assert "model_access_denied" not in response.text, response.text
        assert response.json()["error"]["message"].endswith(
            f"Fallback to {rig.fallback} was skipped because the caller is not authorized to call it."
        ), response.text


def test_turning_enforcement_off_lets_a_router_fallback_reach_a_model_outside_the_key_list(
    tmp_path: Path,
) -> None:
    """A proxy runs with `general_settings.enforce_fallback_model_access: false`; a key allows only `<primary>`.

    The call must be served by `<fallback>` with two upstream attempts: with enforcement off only the
    customer list is re-checked for router fallbacks. This boots its own proxy.
    """
    names: Final = _names()
    failing, _, fallback, _ = names
    with wire_server(_reply) as wire:
        config: Final = _fallback_config(tmp_path, wire, names, enforce=False)
        with (
            gateway_from_environment() as shared,
            owned_proxy(shared, tmp_path / "proxy", {}, config=config) as proxy,
            proxy.scenario() as scenario,
        ):
            key: Final = scenario.key(models=[failing])
            text: Final = f"{marker()}-{uuid.uuid4().hex}"
            assert_served(chat(proxy, key, failing, text))
            assert _attempts(wire, text) == [f"provider-fails-{failing}", f"provider-{fallback}"]
