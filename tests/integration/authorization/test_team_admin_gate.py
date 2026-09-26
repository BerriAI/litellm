"""Status-code matrix for every management door that admits a team admin today.

Each door is called as a proxy admin, an admin of the target team, a plain member, an admin of another team
and a teamless user. The expected codes pin current behaviour so the shared team-admin gate can prove parity.
"""

from __future__ import annotations

import enum
import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import Final

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


class Actor(enum.Enum):
    PROXY_ADMIN = "proxy_admin"
    TEAM_ADMIN = "team_admin"
    MEMBER = "member"
    OTHER_TEAM_ADMIN = "other_team_admin"
    OUTSIDER = "outsider"


class Need(enum.Enum):
    VICTIM = "victim"
    NEWCOMER = "newcomer"
    SERVICE_KEY = "service_key"
    MODEL = "model"
    CALLBACK = "callback"
    INVITATION = "invitation"
    PERMISSIONS = "permissions"


PERMITTED_FIELDS: Final = ("max_budget", "projects", "member_key_budgets")


@dataclass(frozen=True, slots=True)
class World:
    gateway: Gateway
    team: str
    other_team: str
    keys: Mapping[Actor, str]
    request_id: str
    since: datetime
    until: datetime

    def day(self, moment: datetime) -> str:
        return moment.strftime("%Y-%m-%d")

    def stamp(self, moment: datetime) -> str:
        return moment.strftime("%Y-%m-%d %H:%M:%S")


@dataclass(frozen=True, slots=True)
class Target:
    nonce: str
    victim: str = ""
    victim_key: str = ""
    newcomer: str = ""
    service_key: str = ""
    model_id: str = ""
    callback: str = ""
    invitation: str = ""


@dataclass(frozen=True, slots=True)
class Call:
    method: str
    path: str
    body: Mapping[str, JsonValue] | None = None


@dataclass(frozen=True, slots=True)
class Door:
    name: str
    call: Callable[[World, Target], Call]
    expected: Mapping[Actor, int]
    needs: frozenset[Need] = frozenset()
    dispose: Callable[[World, Target, dict[str, JsonValue]], None] | None = None


def verdict(
    team_admin: int, member: int, other_team_admin: int, outsider: int, proxy_admin: int | None = 200
) -> Mapping[Actor, int]:
    return MappingProxyType(
        {
            **({Actor.PROXY_ADMIN: proxy_admin} if proxy_admin is not None else {}),
            Actor.TEAM_ADMIN: team_admin,
            Actor.MEMBER: member,
            Actor.OTHER_TEAM_ADMIN: other_team_admin,
            Actor.OUTSIDER: outsider,
        }
    )


def _spend_rows(gateway: Gateway, team: str, since: datetime, until: datetime) -> list[JsonValue]:
    page: Final = gateway.get(
        "/spend/logs/ui",
        {
            "team_id": team,
            "start_date": since.strftime("%Y-%m-%d %H:%M:%S"),
            "end_date": until.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    rows: Final = page["data"]
    assert isinstance(rows, list)
    return rows


def _team_model_body(world: World, name: str) -> dict[str, JsonValue]:
    return {
        "model_name": name,
        "litellm_params": {
            "model": "openai/gpt-4o-mini",
            "api_key": "integration-provider-key",
            "api_base": f"{world.gateway.upstream_url}/v1",
        },
        "model_info": {"team_id": world.team},
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


def _delete_callback_if_present(gateway: Gateway, team: str, name: str) -> None:
    gateway.request("DELETE", f"/team/{team}/callback/{name}")


def _dispose_project(world: World, target: Target, created: dict[str, JsonValue]) -> None:
    response: Final = world.gateway.request("DELETE", "/project/delete", {"project_ids": [created["project_id"]]})
    assert response.status_code == 200, response.text


def _dispose_key(world: World, target: Target, created: dict[str, JsonValue]) -> None:
    delete_key_if_present(world.gateway, string_value(created["key"]))


def _dispose_model(world: World, target: Target, created: dict[str, JsonValue]) -> None:
    _delete_model_if_present(world.gateway, string_value(object_value(created["model_info"])["id"]))


def _dispose_callback(world: World, target: Target, created: dict[str, JsonValue]) -> None:
    _delete_callback_if_present(world.gateway, world.team, f"matrix-{target.nonce}")


def prepare(world: World, scenario: Scenario, needs: frozenset[Need]) -> Target:
    gateway: Final = world.gateway
    if Need.PERMISSIONS in needs:
        scenario.cleanups.enter_context(team_admin_permissions(gateway, PERMITTED_FIELDS))
    wants_victim: Final = Need.VICTIM in needs or Need.INVITATION in needs
    victim: Final = scenario.user(user_role="internal_user") if wants_victim else ""
    if victim:
        gateway.post("/team/member_add", {"team_id": world.team, "member": {"role": "user", "user_id": victim}})
    victim_key: Final = (
        string_value(gateway.post("/key/generate", {"user_id": victim, "team_id": world.team})["key"]) if victim else ""
    )
    newcomer: Final = scenario.user(user_role="internal_user") if Need.NEWCOMER in needs else ""
    service_key: Final = (
        string_value(
            gateway.post(
                "/key/service-account/generate", {"team_id": world.team, "key_alias": f"matrix-{uuid.uuid4().hex}"}
            )["key"]
        )
        if Need.SERVICE_KEY in needs
        else ""
    )
    if service_key:
        scenario.cleanups.callback(delete_key_if_present, gateway, service_key)
    model_id: Final = (
        string_value(
            object_value(
                gateway.post("/model/new", _team_model_body(world, f"matrix-{uuid.uuid4().hex}"))["model_info"]
            )["id"]
        )
        if Need.MODEL in needs
        else ""
    )
    if model_id:
        scenario.cleanups.callback(_delete_model_if_present, gateway, model_id)
    callback: Final = f"matrix-{uuid.uuid4().hex}" if Need.CALLBACK in needs else ""
    if callback:
        gateway.post(f"/team/{world.team}/callback", _callback_body(callback))
        scenario.cleanups.callback(_delete_callback_if_present, gateway, world.team, callback)
    invitation: Final = (
        string_value(gateway.post("/invitation/new", {"user_id": victim}, key=world.keys[Actor.TEAM_ADMIN])["id"])
        if Need.INVITATION in needs
        else ""
    )
    return Target(
        nonce=uuid.uuid4().hex,
        victim=victim,
        victim_key=victim_key,
        newcomer=newcomer,
        service_key=service_key,
        model_id=model_id,
        callback=callback,
        invitation=invitation,
    )


DOORS: Final[tuple[Door, ...]] = (
    Door(
        "member_add_user",
        lambda w, t: Call(
            "POST", "/team/member_add", {"team_id": w.team, "member": {"role": "user", "user_id": t.newcomer}}
        ),
        verdict(team_admin=200, member=403, other_team_admin=403, outsider=403),
        needs=frozenset({Need.NEWCOMER}),
    ),
    Door(
        "member_add_admin",
        lambda w, t: Call(
            "POST", "/team/member_add", {"team_id": w.team, "member": {"role": "admin", "user_id": t.newcomer}}
        ),
        verdict(team_admin=200, member=403, other_team_admin=403, outsider=403),
        needs=frozenset({Need.NEWCOMER}),
    ),
    Door(
        "member_update_budget",
        lambda w, t: Call(
            "POST", "/team/member_update", {"team_id": w.team, "user_id": t.victim, "max_budget_in_team": 5}
        ),
        verdict(team_admin=200, member=403, other_team_admin=403, outsider=403),
        needs=frozenset({Need.VICTIM}),
    ),
    Door(
        "member_update_role_admin",
        lambda w, t: Call("POST", "/team/member_update", {"team_id": w.team, "user_id": t.victim, "role": "admin"}),
        verdict(team_admin=200, member=403, other_team_admin=403, outsider=403),
        needs=frozenset({Need.VICTIM}),
    ),
    Door(
        "member_delete",
        lambda w, t: Call("POST", "/team/member_delete", {"team_id": w.team, "user_id": t.victim}),
        verdict(team_admin=200, member=403, other_team_admin=403, outsider=403),
        needs=frozenset({Need.VICTIM}),
    ),
    Door(
        "members_bulk_delete",
        lambda w, t: Call(
            "POST", f"/management/v1/teams/{w.team}/members/bulk_delete", {"members": [{"user_id": t.victim}]}
        ),
        verdict(team_admin=200, member=403, other_team_admin=403, outsider=403),
        needs=frozenset({Need.VICTIM}),
    ),
    Door(
        "members_bulk_update",
        lambda w, t: Call(
            "POST",
            f"/management/v1/teams/{w.team}/members/bulk_update",
            {"members": [{"user_id": t.victim, "max_budget_in_team": 10}]},
        ),
        verdict(team_admin=200, member=403, other_team_admin=403, outsider=403),
        needs=frozenset({Need.VICTIM}),
    ),
    Door(
        "member_reset_spend",
        lambda w, t: Call("POST", f"/team/{w.team}/member/{t.victim}/reset_spend", {"reset_to": 0}),
        verdict(team_admin=200, member=403, other_team_admin=403, outsider=403),
        needs=frozenset({Need.VICTIM}),
    ),
    Door(
        "member_reset_budget",
        lambda w, t: Call("POST", f"/team/{w.team}/member/{t.victim}/reset_budget"),
        verdict(team_admin=200, member=403, other_team_admin=403, outsider=403),
        needs=frozenset({Need.VICTIM}),
    ),
    Door(
        "invitation_new",
        lambda w, t: Call("POST", "/invitation/new", {"user_id": t.victim}),
        verdict(team_admin=200, member=400, other_team_admin=400, outsider=400),
        needs=frozenset({Need.VICTIM}),
    ),
    Door(
        "invitation_delete",
        lambda w, t: Call("POST", "/invitation/delete", {"invitation_id": t.invitation}),
        verdict(team_admin=200, member=400, other_team_admin=403, outsider=400),
        needs=frozenset({Need.INVITATION}),
    ),
    Door(
        "user_info_v2",
        lambda w, t: Call("GET", f"/v2/user/info?user_id={t.victim}"),
        verdict(team_admin=200, member=404, other_team_admin=404, outsider=404),
        needs=frozenset({Need.VICTIM}),
    ),
    Door(
        "permissions_update",
        lambda w, t: Call(
            "POST",
            "/team/permissions_update",
            {"team_id": w.team, "team_member_permissions": ["/key/info", "/key/health"]},
        ),
        verdict(team_admin=200, member=403, other_team_admin=403, outsider=403),
    ),
    Door(
        "permissions_list",
        lambda w, t: Call("GET", f"/team/permissions_list?team_id={w.team}"),
        verdict(team_admin=200, member=403, other_team_admin=403, outsider=403),
    ),
    Door(
        "key_generate_team",
        lambda w, t: Call("POST", "/key/generate", {"team_id": w.team}),
        verdict(team_admin=200, member=401, other_team_admin=400, outsider=400),
        dispose=_dispose_key,
    ),
    Door(
        "service_account_generate",
        lambda w, t: Call(
            "POST", "/key/service-account/generate", {"team_id": w.team, "key_alias": f"matrix-{t.nonce}"}
        ),
        verdict(team_admin=200, member=401, other_team_admin=400, outsider=400),
        dispose=_dispose_key,
    ),
    Door(
        "key_update_service_account",
        lambda w, t: Call("POST", "/key/update", {"key": t.service_key, "max_budget": 5}),
        verdict(team_admin=200, member=401, other_team_admin=401, outsider=401),
        needs=frozenset({Need.SERVICE_KEY}),
    ),
    Door(
        "key_update_member_key",
        lambda w, t: Call("POST", "/key/update", {"key": t.victim_key, "max_budget": 5}),
        verdict(team_admin=403, member=403, other_team_admin=403, outsider=403),
        needs=frozenset({Need.VICTIM}),
    ),
    Door(
        "key_update_member_key_permitted",
        lambda w, t: Call("POST", "/key/update", {"key": t.victim_key, "max_budget": 5}),
        verdict(team_admin=200, member=403, other_team_admin=403, outsider=403),
        needs=frozenset({Need.VICTIM, Need.PERMISSIONS}),
    ),
    Door(
        "team_key_bulk_update",
        lambda w, t: Call(
            "POST",
            "/team/key/bulk_update",
            {"team_id": w.team, "all_keys_in_team": True, "update_fields": {"max_budget": 5}},
        ),
        verdict(team_admin=200, member=401, other_team_admin=401, outsider=401),
    ),
    Door(
        "key_delete",
        lambda w, t: Call("POST", "/key/delete", {"keys": [t.victim_key]}),
        verdict(team_admin=200, member=403, other_team_admin=403, outsider=403),
        needs=frozenset({Need.VICTIM}),
    ),
    Door(
        "key_regenerate",
        lambda w, t: Call("POST", "/key/regenerate", {"key": t.victim_key}),
        verdict(team_admin=200, member=401, other_team_admin=401, outsider=401),
        needs=frozenset({Need.VICTIM}),
    ),
    Door(
        "key_reset_spend",
        lambda w, t: Call("POST", f"/key/{t.victim_key}/reset_spend", {"reset_to": 0}),
        verdict(team_admin=200, member=403, other_team_admin=403, outsider=403),
        needs=frozenset({Need.VICTIM}),
    ),
    Door(
        "key_block",
        lambda w, t: Call("POST", "/key/block", {"key": t.victim_key}),
        verdict(team_admin=200, member=403, other_team_admin=403, outsider=403),
        needs=frozenset({Need.VICTIM}),
    ),
    Door(
        "key_unblock",
        lambda w, t: Call("POST", "/key/unblock", {"key": t.victim_key}),
        verdict(team_admin=200, member=403, other_team_admin=403, outsider=403),
        needs=frozenset({Need.VICTIM}),
    ),
    Door(
        "key_list_team",
        lambda w, t: Call("GET", f"/key/list?team_id={w.team}&include_team_keys=true&return_full_object=true"),
        verdict(team_admin=200, member=200, other_team_admin=403, outsider=403),
    ),
    Door(
        "spend_logs_ui",
        lambda w, t: Call(
            "GET",
            f"/spend/logs/ui?team_id={w.team}&start_date={w.stamp(w.since)}&end_date={w.stamp(w.until)}",
        ),
        verdict(team_admin=200, member=403, other_team_admin=403, outsider=403),
    ),
    Door(
        "spend_log_payload",
        lambda w, t: Call("GET", f"/spend/logs/ui/{w.request_id}"),
        verdict(team_admin=200, member=403, other_team_admin=403, outsider=403),
    ),
    Door(
        "team_daily_activity",
        lambda w, t: Call(
            "GET", f"/team/daily/activity?team_ids={w.team}&start_date={w.day(w.since)}&end_date={w.day(w.until)}"
        ),
        verdict(team_admin=200, member=200, other_team_admin=404, outsider=404),
    ),
    Door(
        "team_spend_by_user",
        lambda w, t: Call(
            "GET", f"/team/spend/by_user?team_ids={w.team}&start_date={w.day(w.since)}&end_date={w.day(w.until)}"
        ),
        verdict(team_admin=200, member=200, other_team_admin=404, outsider=404),
    ),
    Door(
        "model_new_team",
        lambda w, t: Call("POST", "/model/new", _team_model_body(w, f"matrix-{t.nonce}")),
        verdict(team_admin=200, member=403, other_team_admin=403, outsider=403),
        dispose=_dispose_model,
    ),
    Door(
        "model_update_team",
        lambda w, t: Call(
            "POST",
            "/model/update",
            {"model_info": {"id": t.model_id, "team_id": w.team}, "litellm_params": {"rpm": 10}},
        ),
        verdict(team_admin=200, member=403, other_team_admin=403, outsider=403),
        needs=frozenset({Need.MODEL}),
    ),
    Door(
        "model_delete_team",
        lambda w, t: Call("POST", "/model/delete", {"id": t.model_id}),
        verdict(team_admin=200, member=403, other_team_admin=403, outsider=403),
        needs=frozenset({Need.MODEL}),
    ),
    Door(
        "auto_router_availability",
        lambda w, t: Call("POST", "/auto_router/availability", {"team_id": w.team}),
        verdict(team_admin=200, member=403, other_team_admin=403, outsider=403),
    ),
    Door(
        "callback_add",
        lambda w, t: Call("POST", f"/team/{w.team}/callback", _callback_body(f"matrix-{t.nonce}")),
        verdict(team_admin=200, member=403, other_team_admin=403, outsider=403),
        dispose=_dispose_callback,
    ),
    Door(
        "callback_get",
        lambda w, t: Call("GET", f"/team/{w.team}/callback"),
        verdict(team_admin=200, member=403, other_team_admin=403, outsider=403),
        needs=frozenset({Need.CALLBACK}),
    ),
    Door(
        "callback_delete",
        lambda w, t: Call("DELETE", f"/team/{w.team}/callback/{t.callback}"),
        verdict(team_admin=200, member=403, other_team_admin=403, outsider=403),
        needs=frozenset({Need.CALLBACK}),
    ),
    Door(
        "disable_logging",
        lambda w, t: Call("POST", f"/team/{w.team}/disable_logging"),
        verdict(team_admin=401, member=401, other_team_admin=401, outsider=401),
    ),
    Door(
        "team_info",
        lambda w, t: Call("GET", f"/team/info?team_id={w.team}"),
        verdict(team_admin=200, member=200, other_team_admin=403, outsider=403),
    ),
    Door(
        "team_update_budget",
        lambda w, t: Call("POST", "/team/update", {"team_id": w.team, "max_budget": 5}),
        verdict(team_admin=403, member=403, other_team_admin=403, outsider=403),
    ),
    Door(
        "team_update_budget_permitted",
        lambda w, t: Call("POST", "/team/update", {"team_id": w.team, "max_budget": 7}),
        verdict(team_admin=200, member=403, other_team_admin=403, outsider=403),
        needs=frozenset({Need.PERMISSIONS}),
    ),
    Door(
        "project_new",
        lambda w, t: Call("POST", "/project/new", {"team_id": w.team, "project_alias": f"matrix-{t.nonce}"}),
        verdict(team_admin=403, member=403, other_team_admin=403, outsider=403),
        dispose=_dispose_project,
    ),
    Door(
        "project_new_permitted",
        lambda w, t: Call("POST", "/project/new", {"team_id": w.team, "project_alias": f"matrix-{t.nonce}"}),
        verdict(team_admin=200, member=403, other_team_admin=403, outsider=403),
        needs=frozenset({Need.PERMISSIONS}),
        dispose=_dispose_project,
    ),
    Door(
        "team_delete",
        lambda w, t: Call("POST", "/team/delete", {"team_ids": [w.team]}),
        verdict(team_admin=401, member=401, other_team_admin=401, outsider=401, proxy_admin=None),
    ),
    Door(
        "team_block",
        lambda w, t: Call("POST", "/team/block", {"team_id": w.team}),
        verdict(team_admin=401, member=401, other_team_admin=401, outsider=401, proxy_admin=None),
    ),
)

CASES: Final = tuple((door, actor) for door in DOORS for actor in door.expected)


@pytest.fixture(scope="module")
def world() -> Iterator[World]:
    with gateway_from_environment() as gateway, gateway.scenario() as scenario:
        team: Final = scenario.team()
        other_team: Final = scenario.team()
        users: Final = MappingProxyType(
            {
                actor: scenario.user(user_role="internal_user")
                for actor in (Actor.TEAM_ADMIN, Actor.MEMBER, Actor.OTHER_TEAM_ADMIN, Actor.OUTSIDER)
            }
        )
        gateway.post(
            "/team/member_add",
            {
                "team_id": team,
                "member": [
                    {"role": "admin", "user_id": users[Actor.TEAM_ADMIN]},
                    {"role": "user", "user_id": users[Actor.MEMBER]},
                ],
            },
        )
        gateway.post(
            "/team/member_add",
            {"team_id": other_team, "member": {"role": "admin", "user_id": users[Actor.OTHER_TEAM_ADMIN]}},
        )
        keys: Final = MappingProxyType(
            {
                Actor.PROXY_ADMIN: gateway.key,
                Actor.TEAM_ADMIN: scenario.key(user_id=users[Actor.TEAM_ADMIN], team_id=team),
                Actor.MEMBER: scenario.key(user_id=users[Actor.MEMBER], team_id=team),
                Actor.OTHER_TEAM_ADMIN: scenario.key(user_id=users[Actor.OTHER_TEAM_ADMIN], team_id=other_team),
                Actor.OUTSIDER: scenario.key(user_id=users[Actor.OUTSIDER]),
            }
        )
        since: Final = datetime.now(timezone.utc) - timedelta(days=1)
        until: Final = since + timedelta(days=2)
        gateway.chat(scenario.model(), key=keys[Actor.TEAM_ADMIN])
        rows: Final = eventually(
            lambda: _spend_rows(gateway, team, since, until), lambda found: len(found) > 0, seconds=30
        )
        yield World(
            gateway=gateway,
            team=team,
            other_team=other_team,
            keys=keys,
            request_id=string_value(object_value(rows[0])["request_id"]),
            since=since,
            until=until,
        )


@pytest.mark.parametrize(("door", "actor"), CASES, ids=tuple(f"{door.name}[{actor.value}]" for door, actor in CASES))
def test_door_status(world: World, door: Door, actor: Actor) -> None:
    with world.gateway.scenario() as scenario:
        target: Final = prepare(world, scenario, door.needs)
        call: Final = door.call(world, target)
        response: Final = world.gateway.request(call.method, call.path, call.body, key=world.keys[actor])
        assert response.status_code == door.expected[actor], (
            f"{actor.value} {call.method} {call.path}: {response.status_code} {response.text}"
        )
        if response.status_code == 200 and door.dispose is not None:
            door.dispose(world, target, object_value(response.json()))
