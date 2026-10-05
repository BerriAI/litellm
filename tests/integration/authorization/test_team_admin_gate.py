"""Status-code matrix for every management route that admits a team admin today.

Each route is called as a proxy admin, an admin of the target team, a plain member, an admin of another team
and a teamless user. It is called again on a team that belongs to an organization, as an admin of that
organization and as an admin of another organization. The expected codes pin current behaviour so the shared
team-admin gate can prove parity.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import Final, Literal, assert_never

import httpx
import pytest
from pydantic import JsonValue

from tests.integration._support.client import (
    Gateway,
    Scenario,
    delete_key_if_present,
    eventually,
    gateway_from_environment,
    object_value,
    string_value,
    team_admin_permissions,
)
from tests.integration._support.database import read_rows

Caller = Literal["proxy_admin", "team_admin", "member", "other_team_admin", "outsider", "org_admin", "other_org_admin"]
CALLERS: Final[tuple[Caller, ...]] = (
    "proxy_admin",
    "team_admin",
    "member",
    "other_team_admin",
    "outsider",
    "org_admin",
    "other_org_admin",
)
ORG_CALLERS: Final[frozenset[Caller]] = frozenset({"org_admin", "other_org_admin"})


@dataclass(frozen=True, slots=True)
class Call:
    method: str
    path: str
    body: Mapping[str, JsonValue] | None = None


@dataclass(frozen=True, slots=True)
class TeamScenario:
    """The shared team as one test case sees it: its ids, a key per caller, and fresh things to act on."""

    scenario: Scenario
    team_id: str
    keys: Mapping[Caller, str]
    request_id: str
    since: datetime
    until: datetime

    @property
    def gateway(self) -> Gateway:
        return self.scenario.gateway

    def user(self) -> str:
        return self.scenario.user(user_role="internal_user")

    def member(self) -> str:
        return self.scenario.member(self.team_id)

    def member_key(self) -> str:
        created: Final = self.gateway.post("/key/generate", {"user_id": self.member(), "team_id": self.team_id})
        return string_value(created["key"])

    def service_key(self) -> str:
        created: Final = self.gateway.post(
            "/key/service-account/generate", {"team_id": self.team_id, "key_alias": f"matrix-{uuid.uuid4().hex}"}
        )
        token: Final = string_value(created["key"])
        self.scenario.cleanups.callback(delete_key_if_present, self.gateway, token)
        return token

    def model(self) -> str:
        created: Final = self.gateway.post("/model/new", _team_model_body(self, f"matrix-{uuid.uuid4().hex}"))
        model_id: Final = string_value(object_value(created["model_info"])["id"])
        self.scenario.cleanups.callback(_delete_model_if_present, self.gateway, model_id)
        return model_id

    def callback_name(self) -> str:
        name: Final = f"matrix-{uuid.uuid4().hex}"
        self.scenario.cleanups.callback(self.gateway.request, "DELETE", f"/team/{self.team_id}/callback/{name}")
        return name

    def callback(self) -> str:
        name: Final = self.callback_name()
        self.gateway.post(f"/team/{self.team_id}/callback", _callback_body(name))
        return name

    def invitation(self) -> str:
        created: Final = self.gateway.post("/invitation/new", {"user_id": self.member()}, key=self.keys["team_admin"])
        return string_value(created["id"])


@dataclass(frozen=True, slots=True)
class Route:
    name: str
    call: Callable[[TeamScenario], Call]
    team_admin: int
    others: int
    proxy_admin: int | None = 200
    member: int | None = None
    other_team_admin: int | None = None
    outsider: int | None = None
    org_admin: int | None = None
    other_org_admin: int | None = None
    permission: str = ""
    cleanup: Callable[[TeamScenario, dict[str, JsonValue]], None] | None = None

    def expected(self, caller: Caller) -> int | None:
        match caller:
            case "proxy_admin":
                return self.proxy_admin
            case "team_admin":
                return self.team_admin
            case "member":
                return self.others if self.member is None else self.member
            case "other_team_admin":
                return self.others if self.other_team_admin is None else self.other_team_admin
            case "outsider":
                return self.others if self.outsider is None else self.outsider
            case "org_admin":
                return self.others if self.org_admin is None else self.org_admin
            case "other_org_admin":
                return self.others if self.other_org_admin is None else self.other_org_admin
            case _:
                assert_never(caller)


def _day(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%d")


def _stamp(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%d %H:%M:%S")


def _spend_rows(gateway: Gateway, team_id: str, since: datetime, until: datetime) -> list[JsonValue]:
    page: Final = gateway.get(
        "/spend/logs/ui", {"team_id": team_id, "start_date": _stamp(since), "end_date": _stamp(until)}
    )
    rows: Final = page["data"]
    assert isinstance(rows, list)
    return rows


def _team_model_body(s: TeamScenario, name: str) -> dict[str, JsonValue]:
    return {
        "model_name": name,
        "litellm_params": {
            "model": "openai/gpt-4o-mini",
            "api_key": "integration-provider-key",
            "api_base": f"{s.gateway.upstream_url}/v1",
        },
        "model_info": {"team_id": s.team_id},
    }


def _callback_body(name: str) -> dict[str, JsonValue]:
    return {
        "callback_name": name,
        "callback_type": "success",
        "callback_vars": {
            "langfuse_public_key": "pk-matrix",
            "langfuse_secret_key": "sk-matrix",
            "langfuse_host": "http://127.0.0.1:9",
        },
    }


def _delete_model_if_present(gateway: Gateway, model_id: str) -> None:
    if read_rows('SELECT model_id FROM "LiteLLM_ProxyModelTable" WHERE model_id = %s', (model_id,)):
        gateway.post("/model/delete", {"id": model_id})


def _delete_project(s: TeamScenario, created: dict[str, JsonValue]) -> None:
    response: Final = s.gateway.request("DELETE", "/project/delete", {"project_ids": [created["project_id"]]})
    assert response.status_code == 200, response.text


def _delete_key(s: TeamScenario, created: dict[str, JsonValue]) -> None:
    delete_key_if_present(s.gateway, string_value(created["key"]))


def _delete_model(s: TeamScenario, created: dict[str, JsonValue]) -> None:
    _delete_model_if_present(s.gateway, string_value(object_value(created["model_info"])["id"]))


# fmt: off
ROUTES: Final[tuple[Route, ...]] = (
    Route("member_add_user",
          lambda s: Call("POST", "/team/member_add", {"team_id": s.team_id, "member": {"role": "user", "user_id": s.user()}}),
          team_admin=200, others=403, org_admin=200),
    Route("member_add_admin",
          lambda s: Call("POST", "/team/member_add", {"team_id": s.team_id, "member": {"role": "admin", "user_id": s.user()}}),
          team_admin=200, others=403, org_admin=200),
    Route("member_update_budget",
          lambda s: Call("POST", "/team/member_update", {"team_id": s.team_id, "user_id": s.member(), "max_budget_in_team": 5}),
          team_admin=200, others=403, org_admin=200),
    Route("member_update_role_admin",
          lambda s: Call("POST", "/team/member_update", {"team_id": s.team_id, "user_id": s.member(), "role": "admin"}),
          team_admin=200, others=403, org_admin=200),
    Route("member_delete",
          lambda s: Call("POST", "/team/member_delete", {"team_id": s.team_id, "user_id": s.member()}),
          team_admin=200, others=403, org_admin=200),
    Route("members_bulk_delete",
          lambda s: Call("POST", f"/management/v1/teams/{s.team_id}/members/bulk_delete", {"members": [{"user_id": s.member()}]}),
          team_admin=200, others=403, org_admin=200),
    Route("members_bulk_update",
          lambda s: Call("POST", f"/management/v1/teams/{s.team_id}/members/bulk_update",
                         {"members": [{"user_id": s.member(), "max_budget_in_team": 10}]}),
          team_admin=200, others=403, org_admin=200),
    Route("member_reset_spend",
          lambda s: Call("POST", f"/team/{s.team_id}/member/{s.member()}/reset_spend", {"reset_to": 0}),
          team_admin=200, others=403, org_admin=200),
    Route("member_reset_budget",
          lambda s: Call("POST", f"/team/{s.team_id}/member/{s.member()}/reset_budget"),
          team_admin=200, others=403, org_admin=200),
    Route("invitation_new",
          lambda s: Call("POST", "/invitation/new", {"user_id": s.member()}),
          team_admin=200, others=400),
    Route("invitation_delete",
          lambda s: Call("POST", "/invitation/delete", {"invitation_id": s.invitation()}),
          team_admin=200, others=400, other_team_admin=403, org_admin=403, other_org_admin=403),
    Route("user_info_v2",
          lambda s: Call("GET", f"/v2/user/info?user_id={s.member()}"),
          team_admin=200, others=404),
    Route("permissions_update",
          lambda s: Call("POST", "/team/permissions_update",
                         {"team_id": s.team_id, "team_member_permissions": ["/key/info", "/key/health"]}),
          team_admin=200, others=403, org_admin=200),
    Route("permissions_list",
          lambda s: Call("GET", f"/team/permissions_list?team_id={s.team_id}"),
          team_admin=200, others=403, org_admin=200),
    Route("key_generate_team",
          lambda s: Call("POST", "/key/generate", {"team_id": s.team_id}),
          team_admin=200, others=400, member=401, cleanup=_delete_key),
    Route("service_account_generate",
          lambda s: Call("POST", "/key/service-account/generate", {"team_id": s.team_id, "key_alias": f"matrix-{uuid.uuid4().hex}"}),
          team_admin=200, others=400, member=401, cleanup=_delete_key),
    Route("key_update_service_account",
          lambda s: Call("POST", "/key/update", {"key": s.service_key(), "max_budget": 5}),
          team_admin=200, others=401),
    Route("key_update_member_key",
          lambda s: Call("POST", "/key/update", {"key": s.member_key(), "max_budget": 5}),
          team_admin=403, others=403),
    Route("key_update_member_key_permitted",
          lambda s: Call("POST", "/key/update", {"key": s.member_key(), "max_budget": 5}),
          team_admin=200, others=403, permission="member_key_budgets"),
    Route("team_key_bulk_update",
          lambda s: Call("POST", "/team/key/bulk_update",
                         {"team_id": s.team_id, "all_keys_in_team": True, "update_fields": {"max_budget": 5}}),
          team_admin=200, others=401),
    Route("key_delete",
          lambda s: Call("POST", "/key/delete", {"keys": [s.member_key()]}),
          team_admin=200, others=403),
    Route("key_regenerate",
          lambda s: Call("POST", "/key/regenerate", {"key": s.member_key()}),
          team_admin=200, others=401),
    Route("key_reset_spend",
          lambda s: Call("POST", f"/key/{s.member_key()}/reset_spend", {"reset_to": 0}),
          team_admin=200, others=403),
    Route("key_block",
          lambda s: Call("POST", "/key/block", {"key": s.member_key()}),
          team_admin=200, others=403, org_admin=200),
    Route("key_unblock",
          lambda s: Call("POST", "/key/unblock", {"key": s.member_key()}),
          team_admin=200, others=403, org_admin=200),
    Route("key_list_team",
          lambda s: Call("GET", f"/key/list?team_id={s.team_id}&include_team_keys=true&return_full_object=true"),
          team_admin=200, others=403, member=200),
    Route("spend_logs_ui",
          lambda s: Call("GET", f"/spend/logs/ui?team_id={s.team_id}&start_date={_stamp(s.since)}&end_date={_stamp(s.until)}"),
          team_admin=200, others=403),
    Route("spend_log_payload",
          lambda s: Call("GET", f"/spend/logs/ui/{s.request_id}"),
          team_admin=200, others=403),
    Route("team_daily_activity",
          lambda s: Call("GET", f"/team/daily/activity?team_ids={s.team_id}&start_date={_day(s.since)}&end_date={_day(s.until)}"),
          team_admin=200, others=404, member=200),
    Route("team_spend_by_user",
          lambda s: Call("GET", f"/team/spend/by_user?team_ids={s.team_id}&start_date={_day(s.since)}&end_date={_day(s.until)}"),
          team_admin=200, others=404, member=200),
    Route("model_new_team",
          lambda s: Call("POST", "/model/new", _team_model_body(s, f"matrix-{uuid.uuid4().hex}")),
          team_admin=200, others=403, cleanup=_delete_model),
    Route("model_update_team",
          lambda s: Call("POST", "/model/update", {"model_info": {"id": s.model(), "team_id": s.team_id}, "litellm_params": {"rpm": 10}}),
          team_admin=200, others=403),
    Route("model_delete_team",
          lambda s: Call("POST", "/model/delete", {"id": s.model()}),
          team_admin=200, others=403),
    Route("auto_router_availability",
          lambda s: Call("POST", "/auto_router/availability", {"team_id": s.team_id}),
          team_admin=200, others=403),
    Route("callback_add",
          lambda s: Call("POST", f"/team/{s.team_id}/callback", _callback_body(s.callback_name())),
          team_admin=200, others=403, org_admin=200),
    Route("callback_get",
          lambda s: Call("GET", f"/team/{s.team_id}/callback"),
          team_admin=200, others=403, org_admin=200),
    Route("callback_delete",
          lambda s: Call("DELETE", f"/team/{s.team_id}/callback/{s.callback()}"),
          team_admin=200, others=403, org_admin=200),
    Route("disable_logging",
          lambda s: Call("POST", f"/team/{s.team_id}/disable_logging"),
          team_admin=401, others=401),
    Route("team_info",
          lambda s: Call("GET", f"/team/info?team_id={s.team_id}"),
          team_admin=200, others=403, member=200, org_admin=200),
    Route("team_update_budget",
          lambda s: Call("POST", "/team/update", {"team_id": s.team_id, "max_budget": 5}),
          team_admin=403, others=403, org_admin=200),
    Route("team_update_budget_permitted",
          lambda s: Call("POST", "/team/update", {"team_id": s.team_id, "max_budget": 4}),
          team_admin=200, others=403, org_admin=200, permission="max_budget"),
    Route("project_new",
          lambda s: Call("POST", "/project/new", {"team_id": s.team_id, "project_alias": f"matrix-{uuid.uuid4().hex}"}),
          team_admin=403, others=403, cleanup=_delete_project),
    Route("project_new_permitted",
          lambda s: Call("POST", "/project/new", {"team_id": s.team_id, "project_alias": f"matrix-{uuid.uuid4().hex}"}),
          team_admin=200, others=403, permission="projects", cleanup=_delete_project),
    Route("team_delete",
          lambda s: Call("POST", "/team/delete", {"team_ids": [s.team_id]}),
          team_admin=401, others=401, proxy_admin=None),
    Route("team_block",
          lambda s: Call("POST", "/team/block", {"team_id": s.team_id}),
          team_admin=401, others=401, proxy_admin=None),
)
# fmt: on


def _cases() -> Iterator[tuple[Route, Caller]]:
    for route in ROUTES:
        for caller in CALLERS:
            if route.expected(caller) is not None:
                yield route, caller


CASES: Final = tuple(_cases())


def _team_scenario(scenario: Scenario, team_id: str, keys: Mapping[Caller, str]) -> TeamScenario:
    since: Final = datetime.now(timezone.utc) - timedelta(days=1)
    until: Final = since + timedelta(days=2)
    scenario.gateway.chat(scenario.model(), key=keys["team_admin"])
    rows: Final = eventually(
        lambda: _spend_rows(scenario.gateway, team_id, since, until), lambda found: len(found) > 0, seconds=30
    )
    return TeamScenario(
        scenario=scenario,
        team_id=team_id,
        keys=keys,
        request_id=string_value(object_value(rows[0])["request_id"]),
        since=since,
        until=until,
    )


@pytest.fixture(scope="module")
def shared() -> Iterator[TeamScenario]:
    with gateway_from_environment() as gateway, gateway.scenario() as scenario:
        team_id: Final = scenario.team()
        other_team_id: Final = scenario.team()
        team_admin: Final = scenario.member(team_id, role="admin")
        member: Final = scenario.member(team_id)
        other_team_admin: Final = scenario.member(other_team_id, role="admin")
        outsider: Final = scenario.user(user_role="internal_user")
        keys: Final[Mapping[Caller, str]] = MappingProxyType(
            {
                "proxy_admin": gateway.key,
                "team_admin": scenario.key(user_id=team_admin, team_id=team_id),
                "member": scenario.key(user_id=member, team_id=team_id),
                "other_team_admin": scenario.key(user_id=other_team_admin, team_id=other_team_id),
                "outsider": scenario.key(user_id=outsider),
            }
        )
        yield _team_scenario(scenario, team_id, keys)


@pytest.fixture(scope="module")
def org_team() -> Iterator[TeamScenario]:
    with gateway_from_environment() as gateway, gateway.scenario() as scenario:
        organization_id: Final = scenario.organization()
        other_organization_id: Final = scenario.organization()
        team_id: Final = scenario.team(organization_id=organization_id)
        team_admin: Final = scenario.member(team_id, role="admin")
        org_admin: Final = scenario.org_member(organization_id, role="org_admin")
        other_org_admin: Final = scenario.org_member(other_organization_id, role="org_admin")
        keys: Final[Mapping[Caller, str]] = MappingProxyType(
            {
                "team_admin": scenario.key(user_id=team_admin, team_id=team_id),
                "org_admin": scenario.key(user_id=org_admin),
                "other_org_admin": scenario.key(user_id=other_org_admin),
            }
        )
        yield _team_scenario(scenario, team_id, keys)


@pytest.mark.parametrize(("route", "caller"), CASES, ids=tuple(f"{route.name}[{caller}]" for route, caller in CASES))
def test_status_code(shared: TeamScenario, org_team: TeamScenario, route: Route, caller: Caller) -> None:
    team: Final = org_team if caller in ORG_CALLERS else shared
    with team.gateway.scenario() as scenario:
        s: Final = replace(team, scenario=scenario)
        if route.name == "team_update_budget_permitted":
            s.gateway.post("/team/update", {"team_id": s.team_id, "max_budget": 5})
        if route.permission:
            scenario.cleanups.enter_context(team_admin_permissions(s.gateway, (route.permission,)))
        call: Final = route.call(s)
        response: Final = s.gateway.request(call.method, call.path, call.body, key=s.keys[caller])
        assert response.status_code == route.expected(caller), (
            f"{caller} {call.method} {call.path}: {response.status_code} {response.text}"
        )
        if route.name == "team_update_budget_permitted":
            assert read_rows(
                'SELECT max_budget FROM "LiteLLM_TeamTable" WHERE team_id = %s', (s.team_id,)
            ) == [{"max_budget": 4.0 if response.status_code == 200 else 5.0}]
        if response.status_code == 200 and route.cleanup is not None:
            route.cleanup(s, object_value(response.json()))


ADMIN_ONLY_CALLERS: Final[tuple[tuple[str, Caller], ...]] = (
    ("shared", "team_admin"),
    ("shared", "member"),
    ("shared", "outsider"),
    ("org_team", "org_admin"),
)
ADMIN_ONLY_IDS: Final = tuple(caller for _, caller in ADMIN_ONLY_CALLERS)


def _admin_only_team(request: pytest.FixtureRequest, fixture: str, scenario: Scenario) -> TeamScenario:
    team: Final = request.getfixturevalue(fixture)
    assert isinstance(team, TeamScenario)
    return replace(team, scenario=scenario)


def _team_model(s: TeamScenario) -> tuple[str, str]:
    name: Final = f"matrix-{uuid.uuid4().hex}"
    created: Final = s.gateway.post("/model/new", _team_model_body(s, name))
    model_id: Final = string_value(object_value(created["model_info"])["id"])
    s.scenario.cleanups.callback(_delete_model_if_present, s.gateway, model_id)
    return model_id, name


def _blocked(model_id: str) -> list[dict[str, object]]:
    return read_rows('SELECT blocked FROM "LiteLLM_ProxyModelTable" WHERE model_id = %s', (model_id,))


def _serve(s: TeamScenario, name: str, key: str) -> httpx.Response:
    return s.gateway.request(
        "POST", "/v1/chat/completions", {"model": name, "messages": [{"role": "user", "content": "serving state"}]}, key=key
    )


def _refusal(response: httpx.Response) -> JsonValue:
    """The error body with the caller's masked user id elided, since it differs per caller."""
    body: Final = response.json()
    error: Final = object_value(body).get("error")
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return {**body, "error": {**error, "message": re.sub(r"Your user_id=\S+", "Your user_id=<caller>", error["message"])}}
    return body


def _admin_only_route(route: str) -> tuple[int, JsonValue]:
    return 401, {
        "error": {
            "message": "Authentication Error, Only proxy admin can be used to generate, delete, update info for new "
            f"keys/users/teams. Route={route}. Your role=internal_user. Your user_id=<caller>",
            "type": "auth_error",
            "param": "None",
            "code": "401",
        }
    }


_BLOCK_REFUSALS: Final[Mapping[tuple[str, str], tuple[int, JsonValue]]] = MappingProxyType(
    {
        ("POST", "/model/block"): _admin_only_route("/model/block"),
        ("POST", "/model/unblock"): _admin_only_route("/model/unblock"),
    }
)
_PATCH_BLOCKED_REFUSED: Final[tuple[int, JsonValue]] = (
    403,
    {"error": {"message": "Only proxy admins can change a model's blocked flag.", "type": "auth_error", "param": "blocked", "code": "403"}},
)
_PATCH_NOT_TEAM_ADMIN: Final[tuple[int, JsonValue]] = (
    403,
    {"detail": "This team does not allow you to manage your own auto routers."},
)


def _block_refusal(caller: Caller, call: Call) -> tuple[int, JsonValue]:
    if call.method == "PATCH":
        return _PATCH_BLOCKED_REFUSED if caller == "team_admin" else _PATCH_NOT_TEAM_ADMIN
    return _BLOCK_REFUSALS[call.method, call.path]


@pytest.mark.parametrize(("fixture", "caller"), ADMIN_ONLY_CALLERS, ids=ADMIN_ONLY_IDS)
def test_only_proxy_admin_flips_a_models_blocked_flag(request: pytest.FixtureRequest, fixture: str, caller: Caller) -> None:
    with gateway_from_environment() as gateway, gateway.scenario() as scenario:
        s: Final = _admin_only_team(request, fixture, scenario)
        model_id, name = _team_model(s)
        for blocked in (True, False):
            s.gateway.post("/model/block" if blocked else "/model/unblock", {"model_id": model_id})
            attempts: Final = (
                Call("POST", "/model/unblock" if blocked else "/model/block", {"model_id": model_id}),
                Call("PATCH", f"/model/{model_id}/update", {"blocked": not blocked}),
            )
            for call in attempts:
                response = s.gateway.request(call.method, call.path, call.body, key=s.keys[caller])
                assert (response.status_code, _refusal(response)) == _block_refusal(caller, call), (
                    f"{caller} {call.method} {call.path}: {response.text}"
                )
                assert _blocked(model_id) == [{"blocked": blocked}], f"{caller} {call.method} {call.path} flipped blocked"
            served = _serve(s, name, s.keys["team_admin"])
            assert served.status_code == (403 if blocked else 200), served.text


@contextmanager
def _model_creation_disabled_for_internal_users(gateway: Gateway) -> Iterator[None]:
    setting: Final = "disable_model_add_for_internal_users"
    original: Final = object_value(gateway.get("/get/ui_settings")["values"]).get(setting, False)
    response: Final = gateway.request("PATCH", "/update/ui_settings", {setting: True})
    assert response.status_code == 200, response.text
    try:
        yield
    finally:
        restored: Final = gateway.request("PATCH", "/update/ui_settings", {setting: original})
        assert restored.status_code == 200, restored.text


def _model_rows(name: str) -> list[dict[str, object]]:
    return read_rows(
        'SELECT model_id FROM "LiteLLM_ProxyModelTable" '
        "WHERE model_name = %s OR model_info->>'team_public_model_name' = %s",
        (name, name),
    )


def _listed_names(gateway: Gateway, key: str) -> tuple[frozenset[str], frozenset[str]]:
    info: Final = gateway.get("/model/info")["data"]
    models: Final = gateway.request("GET", "/v1/models", key=key)
    assert models.status_code == 200, models.text
    listed: Final = models.json()["data"]
    assert isinstance(info, list) and isinstance(listed, list), models.text
    info_names: Final = frozenset(
        name
        for entry in map(object_value, info)
        for name in (entry["model_name"], object_value(entry["model_info"]).get("team_public_model_name"))
        if isinstance(name, str)
    )
    return info_names, frozenset(string_value(object_value(entry)["id"]) for entry in listed)


def test_internal_user_model_creation_prohibition_refuses_team_admin_but_not_proxy_admin(shared: TeamScenario) -> None:
    with shared.gateway.scenario() as scenario:
        s: Final = replace(shared, scenario=scenario)
        team_admin: Final = s.keys["team_admin"]
        refused_name: Final = f"matrix-{uuid.uuid4().hex}"
        admin_name: Final = f"matrix-{uuid.uuid4().hex}"
        with _model_creation_disabled_for_internal_users(s.gateway):
            refused: Final = s.gateway.request("POST", "/model/new", _team_model_body(s, refused_name), key=team_admin)
            assert (refused.status_code, refused.json()) == (
                403,
                {
                    "error": {
                        "message": "Model creation is disabled for internal users by disable_model_add_for_internal_users.",
                        "type": "auth_error",
                        "param": "None",
                        "code": "403",
                    }
                },
            ), refused.text
            assert _model_rows(refused_name) == []
            info_names, listed = _listed_names(s.gateway, team_admin)
            assert refused_name not in info_names and refused_name not in listed, (info_names, listed)
            created: Final = s.gateway.post("/model/new", _team_model_body(s, admin_name))
            s.scenario.cleanups.callback(
                _delete_model_if_present, s.gateway, string_value(object_value(created["model_info"])["id"])
            )
            assert _model_rows(admin_name) == [{"model_id": object_value(created["model_info"])["id"]}]
            assert admin_name in _listed_names(s.gateway, team_admin)[1]
        allowed_name: Final = f"matrix-{uuid.uuid4().hex}"
        allowed: Final = s.gateway.request("POST", "/model/new", _team_model_body(s, allowed_name), key=team_admin)
        assert allowed.status_code == 200, allowed.text
        s.scenario.cleanups.callback(
            _delete_model_if_present, s.gateway, string_value(object_value(allowed.json()["model_info"])["id"])
        )
        assert len(_model_rows(allowed_name)) == 1


def _credential_row(name: str) -> list[dict[str, object]]:
    return read_rows(
        'SELECT credential_name, credential_values, credential_info, updated_at::text AS updated_at FROM "LiteLLM_CredentialsTable" '
        "WHERE credential_name = %s",
        (name,),
    )


@pytest.mark.parametrize(("fixture", "caller"), ADMIN_ONLY_CALLERS, ids=ADMIN_ONLY_IDS)
def test_only_proxy_admin_reads_or_changes_credentials(request: pytest.FixtureRequest, fixture: str, caller: Caller) -> None:
    secret: Final = f"synthetic-shared-credential-{uuid.uuid4().hex}"
    with gateway_from_environment() as gateway, gateway.scenario() as scenario:
        s: Final = _admin_only_team(request, fixture, scenario)
        name: Final = f"credential-{uuid.uuid4().hex}"
        attempted: Final = f"credential-{uuid.uuid4().hex}"
        s.gateway.post(
            "/credentials",
            {"credential_name": name, "credential_values": {"api_key": secret}, "credential_info": {"team": "shared"}},
        )
        s.scenario.cleanups.callback(s.gateway.request, "DELETE", f"/credentials/{name}")
        s.scenario.cleanups.callback(s.gateway.request, "DELETE", f"/credentials/{attempted}")
        stored: Final = _credential_row(name)
        assert len(stored) == 1, stored
        attempts: Final = (
            Call("POST", "/credentials", {"credential_name": attempted, "credential_values": {"api_key": "k"}, "credential_info": {}}),
            Call("PATCH", f"/credentials/{name}", {"credential_name": name, "credential_values": {"api_key": "overwritten"}, "credential_info": {}}),
            Call("DELETE", f"/credentials/{name}"),
            Call("GET", "/credentials"),
            Call("GET", f"/credentials/by_name/{name}"),
        )
        for call in attempts:
            response = s.gateway.request(call.method, call.path, call.body, key=s.keys[caller])
            assert (response.status_code, _refusal(response)) == _admin_only_route(call.path), (
                f"{caller} {call.method} {call.path}: {response.text}"
            )
            assert secret not in response.text, f"{caller} {call.method} {call.path} exposed the credential"
            assert _credential_row(name) == stored, f"{caller} {call.method} {call.path} changed the credential row"
            assert _credential_row(attempted) == [], f"{caller} {call.method} {call.path} wrote a credential"
