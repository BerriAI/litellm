import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Final, Literal

import pytest
from pydantic import JsonValue

from tests.integration._support.client import Gateway, Scenario, delete_key_if_present, object_value, string_value
from tests.integration._support.database import read_rows

MemberDimension = Literal["user_id", "user_email"]
UNCHANGED_BY_ALIAS_UPDATE_SKIP: Final = frozenset(
    {
        "team_alias",
        "members_with_roles",
        "created_at",
        "updated_at",
        "model_spend",
        "model_max_budget",
        "model_id",
        "litellm_organization_table",
        "object_permission_id",
        "object_permission",
        "litellm_model_table",
        "policies",
        "allow_team_guardrail_config",
        "projects",
    }
)


def _team_info(gateway: Gateway, team_id: str) -> dict[str, JsonValue]:
    return object_value(gateway.get("/team/info", {"team_id": team_id})["team_info"])


def _member_ids(gateway: Gateway, team_id: str) -> list[str | None]:
    members: Final = _team_info(gateway, team_id)["members_with_roles"]
    assert isinstance(members, list)
    return [
        member_id if isinstance(member_id := object_value(member).get("user_id"), str) else None for member in members
    ]


def _user_team_ids(gateway: Gateway, user_id: str) -> list[str]:
    teams: Final = gateway.get("/user/info", {"user_id": user_id})["teams"]
    assert isinstance(teams, list)
    return [string_value(object_value(team)["team_id"]) for team in teams]


def _delete_users_if_present(gateway: Gateway, user_ids: list[str]) -> None:
    present: Final = [
        user_id
        for user_id in user_ids
        if read_rows('SELECT user_id FROM "LiteLLM_UserTable" WHERE user_id = %s', (user_id,))
    ]
    if present:
        response: Final = gateway.request("POST", "/user/delete", {"user_ids": list[JsonValue](present)})
        assert response.status_code == 200, response.text


def _member_delete(gateway: Gateway, team_id: str, user_id: str) -> int:
    return gateway.request("POST", "/team/member_delete", {"team_id": team_id, "user_id": user_id}).status_code


def _owned_email_member(gateway: Gateway, scenario: Scenario) -> str:
    user_id: Final = f"integration_{uuid.uuid4().hex}@example.com"
    scenario.cleanups.callback(_delete_users_if_present, gateway, [user_id])
    return user_id


def test_concurrent_team_creation_records_the_named_member(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        user: Final = scenario.user()
        with ThreadPoolExecutor(max_workers=10) as pool:
            teams: Final = list(
                pool.map(lambda _: scenario.team(members_with_roles=[{"role": "user", "user_id": user}]), range(10))
            )
        assert len(set(teams)) == 10
        for team in teams:
            assert user in _member_ids(gateway, team)


def test_team_info_serves_admin_and_team_keys_and_denies_other_keys(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        assert _team_info(gateway, team)["team_id"] == team
        team_key: Final = scenario.key(team_id=team)
        as_team: Final = gateway.request("GET", "/team/info", params={"team_id": team}, key=team_key)
        assert as_team.status_code == 200, as_team.text
        assert object_value(object_value(as_team.json())["team_info"])["team_id"] == team
        outsider: Final = scenario.key()
        denied: Final = gateway.request("GET", "/team/info", params={"team_id": team}, key=outsider)
        assert denied.status_code in {401, 403}, denied.text


def test_team_alias_update_keeps_every_other_team_field(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        admin: Final = scenario.user()
        created: Final = gateway.post(
            "/team/new",
            {
                "team_alias": f"integration-{uuid.uuid4().hex}",
                "members_with_roles": [{"role": "admin", "user_id": admin}],
            },
        )
        team: Final = string_value(created["team_id"])
        scenario.cleanups.callback(scenario.delete_team, team)
        initial_size: Final = len(_member_ids(gateway, team))
        new_members: Final = [_owned_email_member(gateway, scenario) for _ in range(10)]
        gateway.post(
            "/team/member_add",
            {"team_id": team, "member": [{"role": "user", "user_id": member} for member in new_members]},
        )
        members: Final = _member_ids(gateway, team)
        assert len(members) == initial_size + 10
        assert set(new_members) <= set(members)
        new_alias: Final = f"integration-{uuid.uuid4().hex}"
        updated: Final = object_value(gateway.post("/team/update", {"team_id": team, "team_alias": new_alias})["data"])
        assert updated["team_alias"] == new_alias
        updated_members: Final = updated["members_with_roles"]
        assert isinstance(updated_members, list)
        assert len(updated_members) == len(members)
        compared: Final = (set(created) | set(updated)) - UNCHANGED_BY_ALIAS_UPDATE_SKIP
        for field in compared:
            assert updated.get(field) == created.get(field), field
        assert compared


def test_member_added_by_email_sees_the_team_in_user_info(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        email: Final = f"integration-{uuid.uuid4().hex}@example.com"
        user: Final = scenario.user(user_email=email)
        team: Final = scenario.team()
        gateway.post("/team/member_add", {"team_id": team, "member": {"role": "user", "user_email": email}})
        assert team in _user_team_ids(gateway, user)


def test_team_delete_detaches_members_and_hides_the_team(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        first: Final = scenario.user()
        second: Final = scenario.user()
        created: Final = gateway.post(
            "/team/new",
            {
                "team_alias": f"integration-{uuid.uuid4().hex}",
                "members_with_roles": [{"role": "admin", "user_id": first}, {"role": "user", "user_id": second}],
            },
        )
        team: Final = string_value(created["team_id"])
        scenario.cleanups.callback(gateway.request, "POST", "/team/delete", {"team_ids": [team]})
        team_key: Final = string_value(gateway.post("/key/generate", {"team_id": team, "user_id": second})["key"])
        scenario.cleanups.callback(delete_key_if_present, gateway, team_key)
        assert _user_team_ids(gateway, second) == [team]
        gateway.post("/team/delete", {"team_ids": [team]})
        assert _user_team_ids(gateway, second) == []
        missing: Final = gateway.request("GET", "/team/info", params={"team_id": team})
        assert missing.status_code == 404, missing.text


@pytest.mark.parametrize("dimension", ["user_id", "user_email"])
def test_member_delete_removes_the_member_by_id_or_email(gateway: Gateway, dimension: MemberDimension) -> None:
    with gateway.scenario() as scenario:
        email: Final = f"integration-{uuid.uuid4().hex}@example.com"
        user: Final = scenario.user(user_email=email)
        selector: Final[dict[str, JsonValue]] = {"user_id": user} if dimension == "user_id" else {"user_email": email}
        team: Final = scenario.team(members_with_roles=[{"role": "user", **selector}])
        assert user in _member_ids(gateway, team)
        deleted: Final = gateway.request("POST", "/team/member_delete", {"team_id": team, **selector})
        assert deleted.status_code == 200, deleted.text
        assert user not in _member_ids(gateway, team)
        assert team not in _user_team_ids(gateway, user)


def test_member_add_with_email_shaped_user_id_grows_team_by_one(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        initial: Final = _member_ids(gateway, team)
        member: Final = _owned_email_member(gateway, scenario)
        gateway.post("/team/member_add", {"team_id": team, "member": {"role": "user", "user_id": member}})
        after: Final = _member_ids(gateway, team)
        assert member in after
        assert len(after) == len(initial) + 1


def test_member_delete_removes_once_and_rejects_repeats(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        member: Final = f"integration-{uuid.uuid4().hex}"
        scenario.cleanups.callback(_delete_users_if_present, gateway, [member])
        gateway.post("/team/member_add", {"team_id": team, "member": {"role": "user", "user_id": member}})
        before: Final = _member_ids(gateway, team)
        assert member in before
        assert _member_delete(gateway, team, member) == 200
        assert [_member_delete(gateway, team, member) for _ in range(4)] == [400, 400, 400, 400]
        after: Final = _member_ids(gateway, team)
        assert member not in after
        assert len(after) == len(before) - 1


def test_member_delete_of_a_non_member_is_a_bad_request(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        outsider: Final = f"integration-{uuid.uuid4().hex}"
        assert outsider not in _member_ids(gateway, team)
        assert _member_delete(gateway, team, outsider) == 400
