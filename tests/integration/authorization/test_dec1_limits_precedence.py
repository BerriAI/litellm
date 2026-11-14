"""Which limit wins when several apply to one request.

The contract: limits do not override each other. A request is checked against every scope it belongs to (key,
user, team, team member, end user, per-model, organization) and is refused as soon as any one of them is
exhausted, so the tightest limit is the effective one wherever it is set. The 429 names exactly one scope:

- Kinds are checked in this order: requests (rpm) first, then ``max_parallel_requests``, then tokens (tpm).
- Within one kind the scopes are checked in a fixed order and the first exhausted one is named: ``api_key``,
  ``user``, ``team``, ``team_member``, ``end_user``, ``model_per_key``, ``model_per_team``, ...,
  ``organization``.
- The request check charges every scope in one step, so a request refused by one scope has still used one
  request of every other scope. The token reservation is all or nothing: a request refused for tokens by one
  scope leaves no reservation on the others.
- The output cap the proxy adds to an uncapped request is sized from the smallest token limit among the scopes.

Read from ``litellm/proxy/hooks/parallel_request_limiter_v3.py`` (``create_rate_limit_descriptors``,
``_build_request_rate_limit_descriptors``, ``async_pre_call_hook``, ``atomic_check_and_increment_by_n``).
"""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Final

from tests.integration._support.client import Gateway, Scenario, eventually
from tests.integration._support.wire import wire_server
from tests.integration.authorization.dec1_limits_support import (
    assert_limited,
    chat,
    counter_key,
    hashed,
    posts,
    provider,
    ratelimit_header_names,
    redis_integer,
    scope_headers,
    shared_redis,
    statuses,
)


def _end_user(gateway: Gateway, scenario: Scenario, **budget: int) -> str:
    budget_id: Final = scenario.budget(**budget)
    identity: Final = f"limits-customer-{uuid.uuid4().hex}"
    gateway.post("/end_user/new", {"user_id": identity, "budget_id": budget_id})
    scenario.cleanups.callback(gateway.post, "/end_user/delete", {"user_ids": [identity]})
    return identity


def test_a_tighter_team_limit_refuses_a_key_that_still_has_requests_left(gateway: Gateway) -> None:
    """A key has rpm_limit=5 and its team has rpm_limit=1. The key's second request must be refused naming the
    team (``Rate limit exceeded for team: <team_id>`` with limit 1), although the key itself has 3 requests
    left. A generous key limit cannot lift its team's limit."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[model], rpm_limit=1)
        key: Final = scenario.key(team_id=team, rpm_limit=5)
        assert chat(gateway, model, key).status_code == 200
        assert_limited(chat(gateway, model, key), "team", team, "requests", 1)
        assert len(posts(wire)) == 1


def test_a_tighter_key_limit_refuses_a_key_whose_team_still_has_requests_left(gateway: Gateway) -> None:
    """A key has rpm_limit=1 and its team has rpm_limit=5. The key's second request must be refused naming the
    key (``api_key: <hash>`` with limit 1), and a second key of the same team must still be served."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[model], rpm_limit=5)
        key: Final = scenario.key(team_id=team, rpm_limit=1)
        sibling: Final = scenario.key(team_id=team)
        assert chat(gateway, model, key).status_code == 200
        assert_limited(chat(gateway, model, key), "api_key", hashed(key), "requests", 1)
        assert chat(gateway, model, sibling).status_code == 200


def test_the_tightest_of_key_user_and_team_limits_is_the_one_that_refuses(gateway: Gateway) -> None:
    """A key has rpm_limit=5, its user rpm_limit=2 and its team rpm_limit=3. Two requests are served and the
    third must be refused naming the user with limit 2, the smallest of the three."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[model], rpm_limit=3)
        user: Final = scenario.user(rpm_limit=2)
        key: Final = scenario.key(team_id=team, user_id=user, rpm_limit=5)
        assert statuses([chat(gateway, model, key), chat(gateway, model, key)]) == [200, 200]
        assert_limited(chat(gateway, model, key), "user", user, "requests", 2)
        assert len(posts(wire)) == 2


def test_when_every_scope_is_exhausted_at_once_the_refusal_names_the_key(gateway: Gateway) -> None:
    """A key, its user, its team, its organization and the end user all have rpm_limit=1. The second request
    exhausts all five at once and the 429 must name ``api_key``, the first scope in the fixed order."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        organization: Final = scenario.organization(models=[model], rpm_limit=1)
        team: Final = scenario.team(organization_id=organization, models=[model], rpm_limit=1)
        user: Final = scenario.user(rpm_limit=1)
        customer: Final = _end_user(gateway, scenario, rpm_limit=1)
        key: Final = scenario.key(team_id=team, user_id=user, rpm_limit=1)
        assert chat(gateway, model, key, user=customer).status_code == 200
        assert_limited(chat(gateway, model, key, user=customer), "api_key", hashed(key), "requests", 1)


def test_when_user_team_and_organization_are_exhausted_at_once_the_refusal_names_the_user(
    gateway: Gateway,
) -> None:
    """A key without its own limit belongs to a user, a team and an organization that all have rpm_limit=1. The
    second request must be refused naming ``user``, which comes before ``team`` and ``organization``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        organization: Final = scenario.organization(models=[model], rpm_limit=1)
        team: Final = scenario.team(organization_id=organization, models=[model], rpm_limit=1)
        user: Final = scenario.user(rpm_limit=1)
        key: Final = scenario.key(team_id=team, user_id=user)
        assert chat(gateway, model, key).status_code == 200
        assert_limited(chat(gateway, model, key), "user", user, "requests", 1)


def test_when_team_and_organization_are_exhausted_at_once_the_refusal_names_the_team(gateway: Gateway) -> None:
    """A key belongs to a team with rpm_limit=1 inside an organization with rpm_limit=1. The second request must
    be refused naming ``team``: the organization is checked last."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        organization: Final = scenario.organization(models=[model], rpm_limit=1)
        team: Final = scenario.team(organization_id=organization, models=[model], rpm_limit=1)
        key: Final = scenario.key(team_id=team)
        assert chat(gateway, model, key).status_code == 200
        assert_limited(chat(gateway, model, key), "team", team, "requests", 1)


def test_an_exhausted_key_is_named_before_its_exhausted_per_model_limit(gateway: Gateway) -> None:
    """A key has rpm_limit=1 and ``model_rpm_limit`` of 1 for the model it calls. The second request exhausts
    both and must be refused naming ``api_key``, not ``model_per_key``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=1, model_rpm_limit={model: 1})
        assert chat(gateway, model, key).status_code == 200
        assert_limited(chat(gateway, model, key), "api_key", hashed(key), "requests", 1)


def test_every_limited_scope_reports_its_own_headers_on_one_response(gateway: Gateway) -> None:
    """A key, its user, its team, its organization and the end user each have rpm_limit=9. One served response
    must carry a limit and a remaining request header for each of ``api_key``, ``user``, ``team``, ``end_user``
    and ``organization``, ten ``x-ratelimit-*`` headers in total, each reading 9 and 8."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        organization: Final = scenario.organization(models=[model], rpm_limit=9)
        team: Final = scenario.team(organization_id=organization, models=[model], rpm_limit=9)
        user: Final = scenario.user(rpm_limit=9)
        customer: Final = _end_user(gateway, scenario, rpm_limit=9)
        key: Final = scenario.key(team_id=team, user_id=user, rpm_limit=9)
        served: Final = chat(gateway, model, key, user=customer)
        assert served.status_code == 200, served.text
        scopes: Final = ("api_key", "user", "team", "end_user", "organization")
        assert ratelimit_header_names(served) == {
            f"x-ratelimit-{scope}-{field}-requests" for scope in scopes for field in ("limit", "remaining")
        }
        assert {scope: scope_headers(served, scope) for scope in scopes} == {
            scope: {"limit-requests": "9", "remaining-requests": "8"} for scope in scopes
        }


def test_requests_are_checked_before_tokens(gateway: Gateway) -> None:
    """A key has rpm_limit=1 and tpm_limit=1. Its first request passes the request check and is refused for
    tokens; its second request is refused for requests, because the first refusal already used the window's one
    request and rpm is checked before tpm."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=1, tpm_limit=1)
        assert_limited(chat(gateway, model, key, max_tokens=5), "api_key", hashed(key), "tokens", 1, remaining=1)
        assert_limited(chat(gateway, model, key, max_tokens=5), "api_key", hashed(key), "requests", 1)
        assert posts(wire) == ()


def test_requests_are_checked_before_the_parallel_slot(gateway: Gateway) -> None:
    """A key has rpm_limit=1 and max_parallel_requests=1 and its one request is held at the provider. A second
    request is over both limits and must be refused with ``Limit type: requests``, not max_parallel_requests."""
    release: Final = threading.Event()
    with (
        wire_server(provider(hold=release)) as wire,
        gateway.scenario() as scenario,
        ThreadPoolExecutor(1) as pool,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=1, max_parallel_requests=1)
        first: Final = pool.submit(chat, gateway, model, key)
        try:
            eventually(wire.received.qsize, lambda count: count == 1)
            assert_limited(chat(gateway, model, key), "api_key", hashed(key), "requests", 1)
        finally:
            release.set()
        assert first.result().status_code == 200


def test_the_parallel_slot_is_checked_before_tokens(gateway: Gateway) -> None:
    """A key has max_parallel_requests=1 and tpm_limit=20 and one small request is held at the provider. A second
    request that is also too large for the token limit must be refused with ``Limit type:
    max_parallel_requests``, because the slot is checked before the token reservation."""
    release: Final = threading.Event()
    with (
        wire_server(provider(hold=release)) as wire,
        gateway.scenario() as scenario,
        ThreadPoolExecutor(1) as pool,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], max_parallel_requests=1, tpm_limit=20)
        first: Final = pool.submit(chat, gateway, model, key, max_tokens=5)
        try:
            eventually(wire.received.qsize, lambda count: count == 1)
            refused: Final = chat(gateway, model, key, max_tokens=500)
            assert_limited(refused, "api_key", hashed(key), "max_parallel_requests", 1)
        finally:
            release.set()
        assert first.result().status_code == 200


def test_a_request_refused_by_the_team_has_already_used_one_of_the_keys_requests(gateway: Gateway) -> None:
    """A key has rpm_limit=10 and its team rpm_limit=1. The key's second request is refused by the team. Redis
    must then show ``{api_key:<hash>}:requests`` at 2: the request check charges every scope together, so the
    refused request counted against the key as well."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario, shared_redis() as redis:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[model], rpm_limit=1)
        key: Final = scenario.key(team_id=team, rpm_limit=10)
        assert chat(gateway, model, key).status_code == 200
        assert_limited(chat(gateway, model, key), "team", team, "requests", 1)
        assert redis_integer(redis, counter_key("api_key", hashed(key), "requests")) == 2
        assert len(posts(wire)) == 1


def test_a_token_refusal_by_the_team_leaves_no_reservation_on_the_key(gateway: Gateway) -> None:
    """A key has tpm_limit=1000 and its team tpm_limit=20. A request estimated at 60 tokens is refused by the
    team. The next request, estimated at 15, must be served with ``x-ratelimit-api_key-remaining-tokens: 985``
    and ``x-ratelimit-team-remaining-tokens: 5``, so the refused 60 were rolled back from the key's window."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[model], tpm_limit=20)
        key: Final = scenario.key(team_id=team, tpm_limit=1000)
        assert_limited(chat(gateway, model, key, max_tokens=50), "team", team, "tokens", 20, remaining=20)
        served: Final = chat(gateway, model, key, max_tokens=5)
        assert served.status_code == 200, served.text
        assert scope_headers(served, "api_key") == {"limit-tokens": "1000", "remaining-tokens": "985"}
        assert scope_headers(served, "team") == {"limit-tokens": "20", "remaining-tokens": "5"}


def test_the_added_output_cap_is_sized_from_the_smallest_token_limit_in_play(gateway: Gateway) -> None:
    """A key has tpm_limit=10000 and its team tpm_limit=100. An uncapped request must reach the provider with
    ``max_tokens: 25`` (a quarter of the team's 100), not uncapped as the key's own limit would allow, and both
    windows are charged the same 35 token reservation."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[model], tpm_limit=100)
        key: Final = scenario.key(team_id=team, tpm_limit=10000)
        served: Final = chat(gateway, model, key)
        assert served.status_code == 200, served.text
        (forwarded,) = posts(wire)
        assert forwarded["max_tokens"] == 25
        assert scope_headers(served, "api_key") == {"limit-tokens": "10000", "remaining-tokens": "9965"}
        assert scope_headers(served, "team") == {"limit-tokens": "100", "remaining-tokens": "65"}


def test_a_token_limit_on_the_key_and_a_request_limit_on_the_team_are_both_enforced(gateway: Gateway) -> None:
    """A key has tpm_limit=1000 and its team rpm_limit=2. Two small requests are served; the third must be refused
    naming the team for requests even though the key has plenty of tokens left."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[model], rpm_limit=2)
        key: Final = scenario.key(team_id=team, tpm_limit=1000)
        served: Final = [chat(gateway, model, key, max_tokens=5) for _ in range(2)]
        assert statuses(served) == [200, 200], [response.text for response in served]
        assert_limited(chat(gateway, model, key, max_tokens=5), "team", team, "requests", 2)
        assert len(posts(wire)) == 2
