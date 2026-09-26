"""Status-code matrix for every management route that admits a team admin today.

Each route is called as a proxy admin, an admin of the target team, a plain member, an admin of another team
and a teamless user. The expected codes pin current behaviour so the shared team-admin gate can prove parity.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import Final, Literal, assert_never

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

Caller = Literal["proxy_admin", "team_admin", "member", "other_team_admin", "outsider"]
CALLERS: Final[tuple[Caller, ...]] = ("proxy_admin", "team_admin", "member", "other_team_admin", "outsider")


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
    other_team_id: str
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
          team_admin=200, others=403),
    Route("member_add_admin",
          lambda s: Call("POST", "/team/member_add", {"team_id": s.team_id, "member": {"role": "admin", "user_id": s.user()}}),
          team_admin=200, others=403),
    Route("member_update_budget",
          lambda s: Call("POST", "/team/member_update", {"team_id": s.team_id, "user_id": s.member(), "max_budget_in_team": 5}),
          team_admin=200, others=403),
    Route("member_update_role_admin",
          lambda s: Call("POST", "/team/member_update", {"team_id": s.team_id, "user_id": s.member(), "role": "admin"}),
          team_admin=200, others=403),
    Route("member_delete",
          lambda s: Call("POST", "/team/member_delete", {"team_id": s.team_id, "user_id": s.member()}),
          team_admin=200, others=403),
    Route("members_bulk_delete",
          lambda s: Call("POST", f"/management/v1/teams/{s.team_id}/members/bulk_delete", {"members": [{"user_id": s.member()}]}),
          team_admin=200, others=403),
    Route("members_bulk_update",
          lambda s: Call("POST", f"/management/v1/teams/{s.team_id}/members/bulk_update",
                         {"members": [{"user_id": s.member(), "max_budget_in_team": 10}]}),
          team_admin=200, others=403),
    Route("member_reset_spend",
          lambda s: Call("POST", f"/team/{s.team_id}/member/{s.member()}/reset_spend", {"reset_to": 0}),
          team_admin=200, others=403),
    Route("member_reset_budget",
          lambda s: Call("POST", f"/team/{s.team_id}/member/{s.member()}/reset_budget"),
          team_admin=200, others=403),
    Route("invitation_new",
          lambda s: Call("POST", "/invitation/new", {"user_id": s.member()}),
          team_admin=200, others=400),
    Route("invitation_delete",
          lambda s: Call("POST", "/invitation/delete", {"invitation_id": s.invitation()}),
          team_admin=200, others=400, other_team_admin=403),
    Route("user_info_v2",
          lambda s: Call("GET", f"/v2/user/info?user_id={s.member()}"),
          team_admin=200, others=404),
    Route("permissions_update",
          lambda s: Call("POST", "/team/permissions_update",
                         {"team_id": s.team_id, "team_member_permissions": ["/key/info", "/key/health"]}),
          team_admin=200, others=403),
    Route("permissions_list",
          lambda s: Call("GET", f"/team/permissions_list?team_id={s.team_id}"),
          team_admin=200, others=403),
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
          team_admin=200, others=403),
    Route("key_unblock",
          lambda s: Call("POST", "/key/unblock", {"key": s.member_key()}),
          team_admin=200, others=403),
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
          team_admin=200, others=403),
    Route("callback_get",
          lambda s: Call("GET", f"/team/{s.team_id}/callback"),
          team_admin=200, others=403),
    Route("callback_delete",
          lambda s: Call("DELETE", f"/team/{s.team_id}/callback/{s.callback()}"),
          team_admin=200, others=403),
    Route("disable_logging",
          lambda s: Call("POST", f"/team/{s.team_id}/disable_logging"),
          team_admin=401, others=401),
    Route("team_info",
          lambda s: Call("GET", f"/team/info?team_id={s.team_id}"),
          team_admin=200, others=403, member=200),
    Route("team_update_budget",
          lambda s: Call("POST", "/team/update", {"team_id": s.team_id, "max_budget": 5}),
          team_admin=403, others=403),
    Route("team_update_budget_permitted",
          lambda s: Call("POST", "/team/update", {"team_id": s.team_id, "max_budget": 7}),
          team_admin=200, others=403, permission="max_budget"),
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
        since: Final = datetime.now(timezone.utc) - timedelta(days=1)
        until: Final = since + timedelta(days=2)
        gateway.chat(scenario.model(), key=keys["team_admin"])
        rows: Final = eventually(
            lambda: _spend_rows(gateway, team_id, since, until), lambda found: len(found) > 0, seconds=30
        )
        yield TeamScenario(
            scenario=scenario,
            team_id=team_id,
            other_team_id=other_team_id,
            keys=keys,
            request_id=string_value(object_value(rows[0])["request_id"]),
            since=since,
            until=until,
        )


@pytest.mark.parametrize(("route", "caller"), CASES, ids=tuple(f"{route.name}[{caller}]" for route, caller in CASES))
def test_status_code(shared: TeamScenario, route: Route, caller: Caller) -> None:
    with shared.gateway.scenario() as scenario:
        s: Final = replace(shared, scenario=scenario)
        if route.permission:
            scenario.cleanups.enter_context(team_admin_permissions(s.gateway, (route.permission,)))
        call: Final = route.call(s)
        response: Final = s.gateway.request(call.method, call.path, call.body, key=s.keys[caller])
        assert response.status_code == route.expected(caller), (
            f"{caller} {call.method} {call.path}: {response.status_code} {response.text}"
        )
        if response.status_code == 200 and route.cleanup is not None:
            route.cleanup(s, object_value(response.json()))
