"""Limits changed on a team, user, organization, end-user budget or project apply to the next requests.

The contract: the limiter reads limits from the auth object of each request, so a change through the management
API must reach enforcement without a restart and without waiting for the window to end. ``/key/update`` and
``/team/member_update`` are covered by ``test_dec1_limits_updates_and_windows.py``; this file covers the owners
whose objects are cached separately (team, user, organization, end user via its budget, project).

The DB object stores behind these caches are being ported to Rust (addendum, Nov 17-24), which is exactly where a
stale-cache regression would appear. How fast a change lands is UNVERIFIED for every owner: each test polls for
up to 10 seconds, which is the harness default and longer than any invalidation path read in
``litellm/proxy/management_endpoints``. When raising a limit, the new value is 100 so that refused polls (which
still bump the request counter) cannot use it up.
"""

from __future__ import annotations

from typing import Final

from tests.integration._support.client import Gateway, eventually
from tests.integration._support.wire import wire_server
from tests.integration.authorization.dec1_limits_support import chat, provider, scope_headers
from tests.integration.authorization.dec1_ratelim_support import end_user


def _served_again(gateway: Gateway, model: str, key: str, user: str | None = None) -> None:
    extra: Final = {} if user is None else {"user": user}
    eventually(lambda: chat(gateway, model, key, **extra).status_code, lambda status: status == 200)


def _refused(gateway: Gateway, model: str, key: str) -> None:
    eventually(lambda: chat(gateway, model, key).status_code, lambda status: status == 429)


def test_raising_a_teams_rpm_limit_admits_its_key_again(gateway: Gateway) -> None:
    """A team with rpm_limit=1 has used its request and been refused. After ``/team/update`` sets rpm_limit=100 the
    team's key must be served again within the same window."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[model], rpm_limit=1)
        key: Final = scenario.key(team_id=team)
        assert [chat(gateway, model, key).status_code for _ in range(2)] == [200, 429]
        gateway.post("/team/update", {"team_id": team, "rpm_limit": 100})
        _served_again(gateway, model, key)


def test_lowering_a_teams_rpm_limit_refuses_its_key(gateway: Gateway) -> None:
    """A team with rpm_limit=100 has served one request. After ``/team/update`` sets rpm_limit=1 the next request
    must be refused for ``team`` in the same window."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[model], rpm_limit=100)
        key: Final = scenario.key(team_id=team)
        assert chat(gateway, model, key).status_code == 200
        gateway.post("/team/update", {"team_id": team, "rpm_limit": 1})
        _refused(gateway, model, key)


def test_a_teams_new_limit_is_reported_in_the_headers(gateway: Gateway) -> None:
    """A team with rpm_limit=5 is updated to rpm_limit=7. A later request must report
    ``x-ratelimit-team-limit-requests: 7``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[model], rpm_limit=5)
        key: Final = scenario.key(team_id=team)
        assert chat(gateway, model, key).status_code == 200
        gateway.post("/team/update", {"team_id": team, "rpm_limit": 7})
        eventually(
            lambda: scope_headers(chat(gateway, model, key), "team").get("limit-requests"),
            lambda limit: limit == "7",
        )


def test_a_per_model_limit_added_by_team_update_is_enforced(gateway: Gateway) -> None:
    """A team without limits has a key that has made one request. After ``/team/update`` sets
    ``model_rpm_limit={<model>: 1}``, a request must be refused within 10 seconds, named ``model_per_team`` or
    ``model_per_key`` (a team per-model limit is also applied to each key on its own, so either window may be the
    first to fill). The per-model windows only start counting once the limit exists."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[model])
        key: Final = scenario.key(team_id=team)
        assert chat(gateway, model, key).status_code == 200
        gateway.post("/team/update", {"team_id": team, "model_rpm_limit": {model: 1}})
        refused: Final = eventually(lambda: chat(gateway, model, key), lambda response: response.status_code == 429)
        assert refused.headers["rate_limited_by"] in ("model_per_team", "model_per_key"), dict(refused.headers)


def test_raising_a_users_rpm_limit_admits_their_key_again(gateway: Gateway) -> None:
    """A user with rpm_limit=1 is refused on their second request. After ``/user/update`` sets rpm_limit=100 their
    key must be served again in the same window."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        user: Final = scenario.user(rpm_limit=1)
        key: Final = scenario.key(models=[model], user_id=user)
        assert [chat(gateway, model, key).status_code for _ in range(2)] == [200, 429]
        gateway.post("/user/update", {"user_id": user, "rpm_limit": 100})
        _served_again(gateway, model, key)


def test_lowering_a_users_tpm_limit_refuses_a_request_that_used_to_fit(gateway: Gateway) -> None:
    """A user with tpm_limit=10000 is served a request asking max_tokens=50. After ``/user/update`` sets
    tpm_limit=20 the same request (estimate 60) must be refused for ``user`` tokens."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        user: Final = scenario.user(tpm_limit=10000)
        key: Final = scenario.key(models=[model], user_id=user)
        assert chat(gateway, model, key, max_tokens=50).status_code == 200
        gateway.post("/user/update", {"user_id": user, "tpm_limit": 20})
        refused: Final = eventually(
            lambda: chat(gateway, model, key, max_tokens=50), lambda response: response.status_code == 429
        )
        assert refused.headers["rate_limited_by"] == "user"


def test_raising_an_organizations_rpm_limit_admits_its_keys_again(gateway: Gateway) -> None:
    """An organization with rpm_limit=1 refuses its team key's second request. After ``/organization/update`` sets
    rpm_limit=100 the key must be served again in the same window."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        organization: Final = scenario.organization(models=[model], rpm_limit=1)
        team: Final = scenario.team(models=[model], organization_id=organization)
        key: Final = scenario.key(team_id=team)
        assert [chat(gateway, model, key).status_code for _ in range(2)] == [200, 429]
        response: Final = gateway.request(
            "PATCH", "/organization/update", {"organization_id": organization, "rpm_limit": 100}
        )
        assert response.status_code == 200, response.text
        _served_again(gateway, model, key)


def test_raising_an_end_users_budget_rpm_admits_them_again(gateway: Gateway) -> None:
    """An end user whose budget has rpm_limit=1 is refused on a second request. After ``/budget/update`` sets that
    budget's rpm_limit=100 a request naming the end user must be served again."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        customer, budget_id = end_user(gateway, scenario, rpm_limit=1)
        key: Final = scenario.key(models=[model])
        assert [chat(gateway, model, key, user=customer).status_code for _ in range(2)] == [200, 429]
        gateway.post("/budget/update", {"budget_id": budget_id, "rpm_limit": 100})
        _served_again(gateway, model, key, customer)


def test_raising_a_projects_model_limit_admits_its_keys_again(gateway: Gateway) -> None:
    """A project with ``model_rpm_limit={<model>: 1}`` refuses its key's second request. After ``/project/update``
    sets ``model_rpm_limit={<model>: 100}`` the key must be served again. UNVERIFIED: the update route and method
    are assumed to be ``POST /project/update`` with ``project_id``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[model])
        project: Final = scenario.project(team, models=[model], model_rpm_limit={model: 1})
        key: Final = scenario.key(team_id=team, project_id=project)
        assert [chat(gateway, model, key).status_code for _ in range(2)] == [200, 429]
        gateway.post("/project/update", {"project_id": project, "model_rpm_limit": {model: 100}})
        _served_again(gateway, model, key)


def test_moving_a_key_to_another_team_moves_it_to_that_teams_window(gateway: Gateway) -> None:
    """Team A (rpm_limit=1) has refused its key. After ``/key/update`` moves the key to team B (rpm_limit=5) the key
    must be served and report ``x-ratelimit-team-limit-requests: 5``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        exhausted: Final = scenario.team(models=[model], rpm_limit=1)
        fresh: Final = scenario.team(models=[model], rpm_limit=5)
        key: Final = scenario.key(team_id=exhausted)
        assert [chat(gateway, model, key).status_code for _ in range(2)] == [200, 429]
        gateway.post("/key/update", {"key": key, "team_id": fresh})
        served: Final = eventually(lambda: chat(gateway, model, key), lambda response: response.status_code == 200)
        assert scope_headers(served, "team")["limit-requests"] == "5"
