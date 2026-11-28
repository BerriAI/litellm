"""Where a key's per-model limit comes from, and which model names share one per-model window.

The contract, read from ``get_key_own_model_rate_limit`` / ``get_key_model_rpm_limit`` in
``litellm/proxy/auth/auth_utils.py`` (lines 1673-1722) and ``_add_model_per_key_rate_limit_descriptor``,
``RateLimitedModel.limit_in``, ``_rate_limited_model`` and ``create_rate_limit_descriptors`` in
``litellm/proxy/hooks/parallel_request_limiter_v3.py`` (lines 162-171, 2748-2766, 2890-2937, 3375-3431):

- A key's per-model map is read in order: key ``metadata.model_rpm_limit`` / ``model_tpm_limit`` when it is a
  non-empty map, else the ``rpm_limit`` / ``tpm_limit`` entries of the key's ``model_max_budget``, else the team's
  ``metadata.model_rpm_limit``, else the deployment's ``default_api_key_rpm_limit``. Only the first source found is
  used; an empty metadata map counts as absent (``if result:``).
- The descriptor is ``model_per_key`` with value ``<hashed key>:<model group>``, where the group is the
  ``model_group_alias`` target of the requested name. The limit is looked up by the requested name first and the
  group second, but the counter is always the group's, so an alias and its group share one window.
- A request for a ``-latest`` alias that the cost map points at another model (for example ``gpt-mini-latest`` to
  ``gpt-5.4-mini``) is also held to the key's limit for that target (``expand_model_latest_alias``), and duplicate
  descriptors are merged keeping the smallest limit (``coalesce_rate_limit_descriptors``).

``model_max_budget`` as a per-model limit source was listed as not covered in ``limits.md``; the ``-latest``
expansion is new with #51608. The Rust limiter (#51537) re-implements every lookup here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from pydantic import JsonValue

from tests.integration._support.client import Gateway
from tests.integration._support.wire import wire_server
from tests.integration.authorization.dec1_limits_support import (
    assert_limited,
    chat,
    hashed,
    limits_proxy,
    posts,
    provider,
    scope_headers,
    statuses,
)

LATEST_ALIAS: Final = "gpt-mini-latest"
LATEST_TARGET: Final = "gpt-5.4-mini"
"""Cost map row ``gpt-mini-latest`` has ``alias_of: gpt-5.4-mini`` (model_prices_and_context_window.json)."""


def _budget_entry(**limits: int) -> dict[str, JsonValue]:
    return {"budget_limit": 100.0, "time_period": "1d", **limits}


def test_a_keys_model_max_budget_rpm_limit_is_enforced_as_a_per_model_limit(gateway: Gateway) -> None:
    """A key has no ``model_rpm_limit`` but ``model_max_budget: {<model>: {rpm_limit: 1, budget_limit: 100}}``. The
    first request is served and the second refused with ``Rate limit exceeded for model_per_key: <hash>:<model>.
    Limit type: requests. Current limit: 1`` (auth_utils.py:1682-1690). Only one request reaches the provider."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], model_max_budget={model: _budget_entry(rpm_limit=1)})
        assert chat(gateway, model, key).status_code == 200
        assert_limited(chat(gateway, model, key), "model_per_key", f"{hashed(key)}:{model}", "requests", 1)
        assert len(posts(wire)) == 1


def test_a_keys_model_max_budget_tpm_limit_refuses_a_request_too_large_for_it(gateway: Gateway) -> None:
    """A key has ``model_max_budget: {<model>: {tpm_limit: 30}}`` and asks for ``max_tokens: 50`` (estimate 60). The
    request is refused with the ``model_per_key`` tokens 429 reading ``Current limit: 30, Remaining: 30`` and never
    reaches the provider (auth_utils.py:1684, parallel_request_limiter_v3.py:2915)."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], model_max_budget={model: _budget_entry(tpm_limit=30)})
        refused: Final = chat(gateway, model, key, max_tokens=50)
        assert_limited(refused, "model_per_key", f"{hashed(key)}:{model}", "tokens", 30, remaining=30)
        assert posts(wire) == ()


def test_a_served_response_reports_the_model_max_budget_limits_as_model_per_key_headers(gateway: Gateway) -> None:
    """A key has ``model_max_budget: {<model>: {rpm_limit: 4, tpm_limit: 1000}}`` and sends ``max_tokens: 20``. The 200
    carries ``x-ratelimit-model_per_key-limit-requests: 4``, ``remaining-requests: 3``, ``limit-tokens: 1000`` and
    ``remaining-tokens: 970``: a budget-sourced limit reports exactly like a metadata-sourced one."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], model_max_budget={model: _budget_entry(rpm_limit=4, tpm_limit=1000)})
        served: Final = chat(gateway, model, key, max_tokens=20)
        assert served.status_code == 200, served.text
        assert scope_headers(served, "model_per_key") == {
            "limit-requests": "4",
            "remaining-requests": "3",
            "limit-tokens": "1000",
            "remaining-tokens": "970",
        }


def test_a_keys_metadata_model_rpm_limit_wins_over_its_model_max_budget(gateway: Gateway) -> None:
    """A key has ``model_rpm_limit: {<model>: 5}`` and ``model_max_budget: {<model>: {rpm_limit: 1}}``. Metadata is read
    first and the budget is never consulted (auth_utils.py:1677-1680), so two requests are served and the second
    reports ``x-ratelimit-model_per_key-limit-requests: 5``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(
            models=[model], model_rpm_limit={model: 5}, model_max_budget={model: _budget_entry(rpm_limit=1)}
        )
        responses: Final = [chat(gateway, model, key) for _ in range(2)]
        assert statuses(responses) == [200, 200], [response.text for response in responses]
        assert scope_headers(responses[1], "model_per_key")["limit-requests"] == "5"


def test_a_metadata_model_rpm_limit_for_another_model_still_hides_the_model_max_budget(gateway: Gateway) -> None:
    """A key has ``model_rpm_limit: {<other>: 5}`` and ``model_max_budget: {<model>: {rpm_limit: 1}}``. The metadata
    map is non-empty, so it is the only source and has no entry for ``<model>``: two requests to ``<model>`` are both
    served with no ``x-ratelimit-model_per_key-*`` header (auth_utils.py:1677-1680 return the map before the budget
    is read). Pinned as current behaviour; a port that merges the two maps per model would refuse the second."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        other: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(
            models=[model, other],
            model_rpm_limit={other: 5},
            model_max_budget={model: _budget_entry(rpm_limit=1)},
        )
        responses: Final = [chat(gateway, model, key) for _ in range(2)]
        assert statuses(responses) == [200, 200], [response.text for response in responses]
        assert scope_headers(responses[1], "model_per_key") == {}


def test_a_keys_model_max_budget_wins_over_its_teams_model_rpm_limit(gateway: Gateway) -> None:
    """A team has ``model_rpm_limit: {<model>: 5}`` and its key has ``model_max_budget: {<model>: {rpm_limit: 1}}``.
    The key's budget is source 2 and the team metadata source 3 (auth_utils.py:1700-1714), so the key's second
    request is refused naming ``model_per_key: <hash>:<model>`` with limit 1, not the team's 5."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[model], model_rpm_limit={model: 5})
        key: Final = scenario.key(team_id=team, model_max_budget={model: _budget_entry(rpm_limit=1)})
        assert chat(gateway, model, key).status_code == 200
        assert_limited(chat(gateway, model, key), "model_per_key", f"{hashed(key)}:{model}", "requests", 1)


def test_a_limit_on_the_group_holds_a_model_group_alias_on_the_groups_counter(gateway: Gateway, tmp_path: Path) -> None:
    """A proxy maps ``model_group_alias: {<alias>: <group>}`` and a key has ``model_rpm_limit: {<group>: 1}``. A request
    for the alias is served, then a request for the group itself is refused naming ``model_per_key: <hash>:<group>``:
    both names resolve to the group's counter (parallel_request_limiter_v3.py:2748-2754 and 2928)."""
    group: Final = f"limcap3-group-{tmp_path.name[-8:]}"
    alias: Final = f"{group}-alias"
    with (
        wire_server(provider()) as wire,
        limits_proxy(gateway, tmp_path, router_settings={"model_group_alias": {alias: group}}) as candidate,
        candidate.scenario() as scenario,
    ):
        scenario.model(name=group, api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[group, alias], model_rpm_limit={group: 1})
        assert chat(candidate, alias, key).status_code == 200
        assert_limited(chat(candidate, group, key), "model_per_key", f"{hashed(key)}:{group}", "requests", 1)


def test_a_limit_named_for_the_alias_applies_to_alias_requests_on_the_shared_group_counter(
    gateway: Gateway, tmp_path: Path
) -> None:
    """Same alias, and the key has ``model_rpm_limit: {<alias>: 3, <group>: 1}``. One group request uses the shared
    counter (1 of 1). An alias request is then checked against its own limit 3 on that same counter and served with
    ``x-ratelimit-model_per_key-limit-requests: 3`` and ``remaining-requests: 1``; a second group request is refused
    at limit 1 (``RateLimitedModel.limit_in`` prefers the requested name, parallel_request_limiter_v3.py:166-171).
    One counter judged by two limits is surprising; pinned so a port changes it on purpose."""
    group: Final = f"limcap3-group-{tmp_path.name[-8:]}"
    alias: Final = f"{group}-alias"
    with (
        wire_server(provider()) as wire,
        limits_proxy(gateway, tmp_path, router_settings={"model_group_alias": {alias: group}}) as candidate,
        candidate.scenario() as scenario,
    ):
        scenario.model(name=group, api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[group, alias], model_rpm_limit={alias: 3, group: 1})
        assert chat(candidate, group, key).status_code == 200
        through_alias: Final = chat(candidate, alias, key)
        assert through_alias.status_code == 200, through_alias.text
        assert scope_headers(through_alias, "model_per_key") == {"limit-requests": "3", "remaining-requests": "1"}
        assert_limited(chat(candidate, group, key), "model_per_key", f"{hashed(key)}:{group}", "requests", 1)


def test_a_latest_alias_request_is_held_to_the_keys_limit_for_its_target(gateway: Gateway, tmp_path: Path) -> None:
    """A proxy serves a model named ``gpt-mini-latest`` and a key has ``model_rpm_limit: {gpt-5.4-mini: 1}``. The first
    ``gpt-mini-latest`` request is served; the second is refused naming ``model_per_key: <hash>:gpt-5.4-mini``,
    because ``expand_model_latest_alias`` adds the cost map target as a second candidate
    (model_latest_aliases.py:60, parallel_request_limiter_v3.py:3375-3381). UNVERIFIED: that the loaded cost map
    entry's ``names[0]`` is the ``alias_of`` target."""
    with (
        wire_server(provider()) as wire,
        limits_proxy(gateway, tmp_path) as candidate,
        candidate.scenario() as scenario,
    ):
        scenario.model(name=LATEST_ALIAS, api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[LATEST_ALIAS], model_rpm_limit={LATEST_TARGET: 1})
        assert chat(candidate, LATEST_ALIAS, key).status_code == 200
        assert_limited(
            chat(candidate, LATEST_ALIAS, key), "model_per_key", f"{hashed(key)}:{LATEST_TARGET}", "requests", 1
        )
        assert len(posts(wire)) == 1


def test_a_latest_alias_and_its_target_share_one_per_model_window(gateway: Gateway, tmp_path: Path) -> None:
    """The proxy serves both ``gpt-5.4-mini`` and ``gpt-mini-latest``; a key has ``model_rpm_limit: {gpt-5.4-mini: 1}``.
    A request for ``gpt-5.4-mini`` is served, and a following ``gpt-mini-latest`` request is refused naming
    ``model_per_key: <hash>:gpt-5.4-mini``: the alias request counts on the target's window. UNVERIFIED as above."""
    with (
        wire_server(provider()) as wire,
        limits_proxy(gateway, tmp_path) as candidate,
        candidate.scenario() as scenario,
    ):
        scenario.model(name=LATEST_TARGET, api_base=f"{wire.url}/v1")
        scenario.model(name=LATEST_ALIAS, api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[LATEST_TARGET, LATEST_ALIAS], model_rpm_limit={LATEST_TARGET: 1})
        assert chat(candidate, LATEST_TARGET, key).status_code == 200
        assert_limited(
            chat(candidate, LATEST_ALIAS, key), "model_per_key", f"{hashed(key)}:{LATEST_TARGET}", "requests", 1
        )


def test_a_limit_named_for_the_latest_alias_does_not_hold_its_target(gateway: Gateway, tmp_path: Path) -> None:
    """A key has ``model_rpm_limit: {gpt-mini-latest: 1}`` and calls ``gpt-5.4-mini`` three times. All three are
    served with no ``x-ratelimit-model_per_key-*`` header: expansion runs from the alias to its target, never back
    (``expand_model_latest_alias`` of a plain model name is the name alone, model_latest_aliases.py:60-66)."""
    with (
        wire_server(provider()) as wire,
        limits_proxy(gateway, tmp_path) as candidate,
        candidate.scenario() as scenario,
    ):
        scenario.model(name=LATEST_TARGET, api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[LATEST_TARGET], model_rpm_limit={LATEST_ALIAS: 1})
        responses: Final = [chat(candidate, LATEST_TARGET, key) for _ in range(3)]
        assert statuses(responses) == [200, 200, 200], [response.text for response in responses]
        assert all(scope_headers(response, "model_per_key") == {} for response in responses)


def test_a_latest_alias_with_limits_on_both_names_is_held_to_the_smaller_on_each_window(
    gateway: Gateway, tmp_path: Path
) -> None:
    """A key has ``model_rpm_limit: {gpt-mini-latest: 5, gpt-5.4-mini: 1}`` and calls ``gpt-mini-latest`` twice. The
    request makes two descriptors, ``<hash>:gpt-mini-latest`` (limit 5) and ``<hash>:gpt-5.4-mini`` (limit 1), so the
    second call is refused naming the target at limit 1 although the alias's own window has room
    (parallel_request_limiter_v3.py:3431, ``coalesce_rate_limit_descriptors`` keeps both because their values differ).
    UNVERIFIED as above."""
    with (
        wire_server(provider()) as wire,
        limits_proxy(gateway, tmp_path) as candidate,
        candidate.scenario() as scenario,
    ):
        scenario.model(name=LATEST_ALIAS, api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[LATEST_ALIAS], model_rpm_limit={LATEST_ALIAS: 5, LATEST_TARGET: 1})
        assert chat(candidate, LATEST_ALIAS, key).status_code == 200
        assert_limited(
            chat(candidate, LATEST_ALIAS, key), "model_per_key", f"{hashed(key)}:{LATEST_TARGET}", "requests", 1
        )
