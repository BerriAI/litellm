"""Rate limits on the owners of a key: user, team, team member, end user (customer) and organization.

The contract: ``rpm_limit`` and ``tpm_limit`` can be set on an internal user (``/user/new``), a team
(``/team/new``), one member of a team (``/team/member_update``), an end user (through a budget attached with
``/end_user/new``) and an organization (``/organization/new``). Each owner has ONE window shared by every key
and request that resolves to it, independent of the windows of other owners. A request is charged to every
scope it belongs to.

The 429 names the scope and its identity exactly:

- ``Rate limit exceeded for user: <user_id>.``
- ``Rate limit exceeded for team: <team_id>.``
- ``Rate limit exceeded for team_member: <team_id>:<user_id>.``
- ``Rate limit exceeded for end_user: <end user id>.``
- ``Rate limit exceeded for organization: <organization_id>.``

and served responses carry ``x-ratelimit-<scope>-limit-requests|tokens`` and
``x-ratelimit-<scope>-remaining-requests|tokens`` with the same scope word. Redis keys follow the same naming,
``{<scope>:<identity>}:requests``.

Read from ``litellm/proxy/hooks/parallel_request_limiter_v3.py`` (``create_rate_limit_descriptors``,
``create_organization_rate_limit_descriptor``) and ``litellm/proxy/auth/user_api_key_auth.py``.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Final

import pytest

from tests.integration._support.client import Gateway, Scenario
from tests.integration._support.wire import wire_server
from tests.integration.authorization.dec1_limits_support import (
    assert_limited,
    chat,
    counter_key,
    limits_proxy,
    posts,
    provider,
    redis_integer,
    scope_headers,
    shared_redis,
    statuses,
)


def _end_user(gateway: Gateway, scenario: Scenario, **budget: int) -> str:
    """An end user whose budget carries the given limits. The budget is created first so it is deleted last."""
    budget_id: Final = scenario.budget(**budget)
    identity: Final = f"limits-customer-{uuid.uuid4().hex}"
    gateway.post("/end_user/new", {"user_id": identity, "budget_id": budget_id})
    scenario.cleanups.callback(gateway.post, "/end_user/delete", {"user_ids": [identity]})
    return identity


def _limited_member(gateway: Gateway, scenario: Scenario, team: str, **limits: int) -> str:
    user: Final = scenario.member(team)
    gateway.post("/team/member_update", {"team_id": team, "user_id": user, **limits})
    return user


def test_two_keys_of_one_user_share_the_users_rpm_window(gateway: Gateway) -> None:
    """A user has rpm_limit=1 and two keys. The first key's request is served; a request with the second key in
    the same window must be refused with ``Rate limit exceeded for user: <user_id>. Limit type: requests.
    Current limit: 1, Remaining: 0``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        user: Final = scenario.user(rpm_limit=1)
        first: Final = scenario.key(models=[model], user_id=user)
        second: Final = scenario.key(models=[model], user_id=user)
        assert chat(gateway, model, first).status_code == 200
        assert_limited(chat(gateway, model, second), "user", user, "requests", 1)
        assert len(posts(wire)) == 1


def test_a_users_tpm_limit_refuses_a_request_too_large_for_it(gateway: Gateway) -> None:
    """A user has tpm_limit=20 and their key asks for max_tokens=50 (estimate 60). The request must be refused
    with ``Rate limit exceeded for user: <user_id>. Limit type: tokens. Current limit: 20, Remaining: 20`` before
    the provider is called."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        user: Final = scenario.user(tpm_limit=20)
        key: Final = scenario.key(models=[model], user_id=user)
        assert_limited(chat(gateway, model, key, max_tokens=50), "user", user, "tokens", 20, remaining=20)
        assert posts(wire) == ()


def test_a_served_response_reports_the_users_request_and_token_limits(gateway: Gateway) -> None:
    """A user has rpm_limit=5 and tpm_limit=1000 and their key sends one request estimated at 30 tokens. The 200
    must carry exactly ``x-ratelimit-user-limit-requests: 5``, ``-remaining-requests: 4``, ``-limit-tokens:
    1000`` and ``-remaining-tokens: 970``, and no ``x-ratelimit-api_key-*`` header."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        user: Final = scenario.user(rpm_limit=5, tpm_limit=1000)
        key: Final = scenario.key(models=[model], user_id=user)
        served: Final = chat(gateway, model, key, max_tokens=20)
        assert served.status_code == 200, served.text
        assert scope_headers(served, "user") == {
            "limit-requests": "5",
            "remaining-requests": "4",
            "limit-tokens": "1000",
            "remaining-tokens": "970",
        }
        assert scope_headers(served, "api_key") == {}


def test_one_users_exhausted_window_does_not_touch_another_user(gateway: Gateway) -> None:
    """Two users each have rpm_limit=1. After the first user is refused, the second user's key must still be
    served and report ``remaining-requests: 0`` of its own window."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        first: Final = scenario.key(models=[model], user_id=scenario.user(rpm_limit=1))
        second: Final = scenario.key(models=[model], user_id=scenario.user(rpm_limit=1))
        assert statuses([chat(gateway, model, first), chat(gateway, model, first)]) == [200, 429]
        served: Final = chat(gateway, model, second)
        assert served.status_code == 200, served.text
        assert scope_headers(served, "user") == {"limit-requests": "1", "remaining-requests": "0"}


def test_two_keys_of_one_team_share_the_teams_rpm_window(gateway: Gateway) -> None:
    """A team has rpm_limit=2 and two keys. One request per key is served; a third request with either key must
    be refused with ``Rate limit exceeded for team: <team_id>. Limit type: requests. Current limit: 2,
    Remaining: 0``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[model], rpm_limit=2)
        first: Final = scenario.key(team_id=team)
        second: Final = scenario.key(team_id=team)
        assert statuses([chat(gateway, model, first), chat(gateway, model, second)]) == [200, 200]
        assert_limited(chat(gateway, model, first), "team", team, "requests", 2)
        assert len(posts(wire)) == 2


def test_a_teams_tpm_limit_refuses_a_request_too_large_for_it(gateway: Gateway) -> None:
    """A team has tpm_limit=20 and its key asks for max_tokens=50 (estimate 60). The request must be refused with
    ``Rate limit exceeded for team: <team_id>. Limit type: tokens. Current limit: 20, Remaining: 20`` and never
    reach the provider."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[model], tpm_limit=20)
        key: Final = scenario.key(team_id=team)
        assert_limited(chat(gateway, model, key, max_tokens=50), "team", team, "tokens", 20, remaining=20)
        assert posts(wire) == ()


def test_a_served_response_reports_the_teams_limits_and_redis_counts_the_team(gateway: Gateway) -> None:
    """A team has rpm_limit=5 and tpm_limit=1000 and its key sends one request estimated at 30 tokens. The 200
    must carry the four ``x-ratelimit-team-*`` headers (5, 4, 1000, 970) and Redis must hold
    ``{team:<team_id>}:requests`` equal to 1."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario, shared_redis() as redis:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[model], rpm_limit=5, tpm_limit=1000)
        served: Final = chat(gateway, model, scenario.key(team_id=team), max_tokens=20)
        assert served.status_code == 200, served.text
        assert scope_headers(served, "team") == {
            "limit-requests": "5",
            "remaining-requests": "4",
            "limit-tokens": "1000",
            "remaining-tokens": "970",
        }
        assert redis_integer(redis, counter_key("team", team, "requests")) == 1


def test_a_team_limit_does_not_apply_to_a_key_outside_the_team(gateway: Gateway) -> None:
    """A team with rpm_limit=1 is exhausted by its key. A key that belongs to no team must still be served on the
    same model and carry no ``x-ratelimit-team-*`` header."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        team_key: Final = scenario.key(team_id=scenario.team(models=[model], rpm_limit=1))
        outsider: Final = scenario.key(models=[model])
        assert statuses([chat(gateway, model, team_key), chat(gateway, model, team_key)]) == [200, 429]
        served: Final = chat(gateway, model, outsider)
        assert served.status_code == 200, served.text
        assert scope_headers(served, "team") == {}


def test_a_team_members_rpm_limit_refuses_that_member_and_names_team_and_user(gateway: Gateway) -> None:
    """A team member is given rpm_limit=1 with ``/team/member_update`` before their team key is first used. The
    key's second request must be refused with ``Rate limit exceeded for team_member: <team_id>:<user_id>. Limit
    type: requests. Current limit: 1, Remaining: 0``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[model])
        user: Final = _limited_member(gateway, scenario, team, rpm_limit=1)
        key: Final = scenario.key(team_id=team, user_id=user)
        assert chat(gateway, model, key).status_code == 200
        assert_limited(chat(gateway, model, key), "team_member", f"{team}:{user}", "requests", 1)
        assert len(posts(wire)) == 1


def test_a_team_members_tpm_limit_refuses_a_request_too_large_for_it(gateway: Gateway) -> None:
    """A team member is given tpm_limit=20. Their team key asks for max_tokens=50 (estimate 60) and must be
    refused with ``Rate limit exceeded for team_member: <team_id>:<user_id>. Limit type: tokens. Current limit:
    20, Remaining: 20``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[model])
        user: Final = _limited_member(gateway, scenario, team, tpm_limit=20)
        key: Final = scenario.key(team_id=team, user_id=user)
        refused: Final = chat(gateway, model, key, max_tokens=50)
        assert_limited(refused, "team_member", f"{team}:{user}", "tokens", 20, remaining=20)
        assert posts(wire) == ()


def test_a_served_response_reports_the_team_members_request_limit(gateway: Gateway) -> None:
    """A team member has rpm_limit=3. Their team key's first request must answer 200 with
    ``x-ratelimit-team_member-limit-requests: 3`` and ``x-ratelimit-team_member-remaining-requests: 2``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[model])
        user: Final = _limited_member(gateway, scenario, team, rpm_limit=3)
        served: Final = chat(gateway, model, scenario.key(team_id=team, user_id=user))
        assert served.status_code == 200, served.text
        assert scope_headers(served, "team_member") == {"limit-requests": "3", "remaining-requests": "2"}


def test_one_members_limit_does_not_apply_to_another_member_of_the_team(gateway: Gateway) -> None:
    """In one team, member A has rpm_limit=1 and member B has no member limit. After A is refused, B's team key
    must still be served and carry no ``x-ratelimit-team_member-*`` header."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[model])
        limited: Final = _limited_member(gateway, scenario, team, rpm_limit=1)
        free: Final = scenario.member(team)
        limited_key: Final = scenario.key(team_id=team, user_id=limited)
        free_key: Final = scenario.key(team_id=team, user_id=free)
        assert statuses([chat(gateway, model, limited_key), chat(gateway, model, limited_key)]) == [200, 429]
        served: Final = chat(gateway, model, free_key)
        assert served.status_code == 200, served.text
        assert scope_headers(served, "team_member") == {}


def test_a_member_limit_does_not_follow_the_user_outside_the_team(gateway: Gateway) -> None:
    """A user has rpm_limit=1 as a member of a team and also owns a personal key with no team. After the team key
    is refused, the personal key must still be served: the member limit is scoped to ``<team_id>:<user_id>``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[model])
        user: Final = _limited_member(gateway, scenario, team, rpm_limit=1)
        team_key: Final = scenario.key(team_id=team, user_id=user)
        personal_key: Final = scenario.key(models=[model], user_id=user)
        assert statuses([chat(gateway, model, team_key), chat(gateway, model, team_key)]) == [200, 429]
        assert chat(gateway, model, personal_key).status_code == 200


def test_the_teams_default_member_rpm_limit_applies_to_a_member_without_their_own(gateway: Gateway) -> None:
    """A team is created with ``team_member_rpm_limit=1`` and a member is added without a limit of their own. The
    member's second request must be refused for ``team_member: <team_id>:<user_id>`` requests with limit 1.
    UNVERIFIED: assumed from ``team_endpoints.py`` creating a member budget from ``team_member_rpm_limit``; if
    the default is only applied to members added through another route, adjust the setup."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[model], team_member_rpm_limit=1)
        user: Final = scenario.member(team)
        key: Final = scenario.key(team_id=team, user_id=user)
        assert chat(gateway, model, key).status_code == 200
        assert_limited(chat(gateway, model, key), "team_member", f"{team}:{user}", "requests", 1)


def test_an_end_users_rpm_limit_follows_the_user_field_across_keys(gateway: Gateway) -> None:
    """An end user has a budget with rpm_limit=1. A request with ``user: <id>`` through one key is served; a
    request with the same ``user`` through a different key must be refused with ``Rate limit exceeded for
    end_user: <id>. Limit type: requests. Current limit: 1, Remaining: 0``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        customer: Final = _end_user(gateway, scenario, rpm_limit=1)
        first: Final = scenario.key(models=[model])
        second: Final = scenario.key(models=[model])
        assert chat(gateway, model, first, user=customer).status_code == 200
        assert_limited(chat(gateway, model, second, user=customer), "end_user", customer, "requests", 1)
        assert len(posts(wire)) == 1


def test_an_end_users_tpm_limit_refuses_a_request_too_large_for_it(gateway: Gateway) -> None:
    """An end user has a budget with tpm_limit=20. A request with ``user: <id>`` and max_tokens=50 (estimate 60)
    must be refused with ``Rate limit exceeded for end_user: <id>. Limit type: tokens. Current limit: 20,
    Remaining: 20`` and never reach the provider."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        customer: Final = _end_user(gateway, scenario, tpm_limit=20)
        key: Final = scenario.key(models=[model])
        refused: Final = chat(gateway, model, key, user=customer, max_tokens=50)
        assert_limited(refused, "end_user", customer, "tokens", 20, remaining=20)
        assert posts(wire) == ()


def test_a_served_response_reports_the_end_users_limits(gateway: Gateway) -> None:
    """An end user has a budget with rpm_limit=5 and tpm_limit=1000. One request estimated at 30 tokens must
    answer 200 with ``x-ratelimit-end_user-limit-requests: 5``, ``-remaining-requests: 4``, ``-limit-tokens:
    1000`` and ``-remaining-tokens: 970``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        customer: Final = _end_user(gateway, scenario, rpm_limit=5, tpm_limit=1000)
        served: Final = chat(gateway, model, scenario.key(models=[model]), user=customer, max_tokens=20)
        assert served.status_code == 200, served.text
        assert scope_headers(served, "end_user") == {
            "limit-requests": "5",
            "remaining-requests": "4",
            "limit-tokens": "1000",
            "remaining-tokens": "970",
        }


def test_an_exhausted_end_user_does_not_block_the_same_key_for_other_callers(gateway: Gateway) -> None:
    """An end user with rpm_limit=1 is exhausted through a key. The same key must still be served when it sends no
    ``user`` field, and its response must carry no ``x-ratelimit-end_user-*`` header."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        customer: Final = _end_user(gateway, scenario, rpm_limit=1)
        key: Final = scenario.key(models=[model])
        limited: Final = [chat(gateway, model, key, user=customer) for _ in range(2)]
        assert statuses(limited) == [200, 429], [response.text for response in limited]
        served: Final = chat(gateway, model, key)
        assert served.status_code == 200, served.text
        assert scope_headers(served, "end_user") == {}


def test_an_end_user_limit_applies_to_the_master_key_too(gateway: Gateway) -> None:
    """An end user has rpm_limit=1. Two requests with ``user: <id>`` sent with the proxy master key must answer
    200 then 429 for ``end_user``: the admin key has no limits of its own but does not waive the customer's."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        customer: Final = _end_user(gateway, scenario, rpm_limit=1)
        assert chat(gateway, model, gateway.key, user=customer).status_code == 200
        assert_limited(chat(gateway, model, gateway.key, user=customer), "end_user", customer, "requests", 1)


def test_the_default_end_user_budget_limits_customers_that_were_never_created(gateway: Gateway, tmp_path: Path) -> None:
    """A proxy is started with ``litellm_settings.max_end_user_budget_id`` pointing at a budget with rpm_limit=1.
    A request with a ``user`` value the proxy has never seen is served once and then refused for ``end_user:
    <that value>``, while a second unseen ``user`` value still gets its own first request served. The candidate
    runs on the shared database (``shared=True``) because the budget row is created through the shared gateway."""
    with gateway.scenario() as setup:
        default_budget: Final = setup.budget(rpm_limit=1)
        with (
            wire_server(provider()) as wire,
            limits_proxy(
                gateway, tmp_path, litellm_settings={"max_end_user_budget_id": default_budget}, shared=True
            ) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(api_base=f"{wire.url}/v1")
            key: Final = scenario.key(models=[model])
            first: Final = f"limits-walk-in-{uuid.uuid4().hex}"
            second: Final = f"limits-walk-in-{uuid.uuid4().hex}"
            assert chat(candidate, model, key, user=first).status_code == 200
            assert_limited(chat(candidate, model, key, user=first), "end_user", first, "requests", 1)
            assert chat(candidate, model, key, user=second).status_code == 200
            assert len(posts(wire)) == 2


def test_two_teams_of_one_organization_share_the_organizations_rpm_window(gateway: Gateway) -> None:
    """An organization has rpm_limit=1 and two teams, each with a key. The first team's request is served; the
    second team's request must be refused with ``Rate limit exceeded for organization: <organization_id>. Limit
    type: requests. Current limit: 1, Remaining: 0``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        organization: Final = scenario.organization(models=[model], rpm_limit=1)
        first: Final = scenario.key(team_id=scenario.team(organization_id=organization, models=[model]))
        second: Final = scenario.key(team_id=scenario.team(organization_id=organization, models=[model]))
        assert chat(gateway, model, first).status_code == 200
        assert_limited(chat(gateway, model, second), "organization", organization, "requests", 1)
        assert len(posts(wire)) == 1


def test_an_organizations_tpm_limit_refuses_a_request_too_large_for_it(gateway: Gateway) -> None:
    """An organization has tpm_limit=20. A key of one of its teams asks for max_tokens=50 (estimate 60) and must
    be refused with ``Rate limit exceeded for organization: <organization_id>. Limit type: tokens. Current
    limit: 20, Remaining: 20``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        organization: Final = scenario.organization(models=[model], tpm_limit=20)
        key: Final = scenario.key(team_id=scenario.team(organization_id=organization, models=[model]))
        refused: Final = chat(gateway, model, key, max_tokens=50)
        assert_limited(refused, "organization", organization, "tokens", 20, remaining=20)
        assert posts(wire) == ()


def test_a_served_response_reports_the_organizations_limits(gateway: Gateway) -> None:
    """An organization has rpm_limit=5 and tpm_limit=1000. A key of one of its teams sends one request estimated
    at 30 tokens and must get 200 with the four ``x-ratelimit-organization-*`` headers (5, 4, 1000, 970)."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        organization: Final = scenario.organization(models=[model], rpm_limit=5, tpm_limit=1000)
        key: Final = scenario.key(team_id=scenario.team(organization_id=organization, models=[model]))
        served: Final = chat(gateway, model, key, max_tokens=20)
        assert served.status_code == 200, served.text
        assert scope_headers(served, "organization") == {
            "limit-requests": "5",
            "remaining-requests": "4",
            "limit-tokens": "1000",
            "remaining-tokens": "970",
        }


@pytest.mark.parametrize("owner", ["user", "team"])
def test_a_request_is_charged_to_the_key_and_to_its_owner_side_by_side(gateway: Gateway, owner: str) -> None:
    """A user (or team) has rpm_limit=5 and its key has rpm_limit=3. One request must answer 200 carrying both
    ``x-ratelimit-api_key-*`` (3, 2) and ``x-ratelimit-<owner>-*`` (5, 4): every scope with a limit is charged
    and reported, the key's limit does not replace the owner's."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = (
            scenario.key(models=[model], rpm_limit=3, user_id=scenario.user(rpm_limit=5))
            if owner == "user"
            else scenario.key(rpm_limit=3, team_id=scenario.team(models=[model], rpm_limit=5))
        )
        served: Final = chat(gateway, model, key)
        assert served.status_code == 200, served.text
        assert scope_headers(served, "api_key") == {"limit-requests": "3", "remaining-requests": "2"}
        assert scope_headers(served, owner) == {"limit-requests": "5", "remaining-requests": "4"}
