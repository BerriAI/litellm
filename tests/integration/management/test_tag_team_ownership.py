"""
Optional team ownership on registered tags: persistence, FK and uniqueness constraints,
and the proxy-admin-only guard for assigning or releasing an owner.
"""

import uuid
from typing import Final

import psycopg.errors
import pytest
from pydantic import JsonValue

from tests.integration._support.client import Gateway, Scenario, object_value, string_value
from tests.integration._support.database import read_rows, write_rows


def _tag_rows(name: str) -> list[dict[str, JsonValue]]:
    return read_rows('SELECT tag_name, team_id FROM "LiteLLM_TagTable" WHERE tag_name = %s', (name,))


def _tag_name() -> str:
    return f"integration-{uuid.uuid4().hex}"


def _create_tag(scenario: Scenario, name: str, **fields: JsonValue) -> None:
    scenario.gateway.post("/tag/new", {"name": name, "models": [], **fields})
    scenario.cleanups.callback(_delete_tag, scenario.gateway, name)


def _delete_tag(gateway: Gateway, name: str) -> None:
    gateway.post("/tag/delete", {"name": name})
    assert _tag_rows(name) == []


def _tag_info(gateway: Gateway, name: str) -> dict[str, JsonValue]:
    return object_value(gateway.post("/tag/info", {"names": [name]})[name])


def _new_team(gateway: Gateway) -> str:
    created: Final = gateway.post("/team/new", {"team_alias": f"integration-{uuid.uuid4().hex}"})
    return string_value(created["team_id"])


def test_deleting_the_owning_team_nulls_out_tag_ownership(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        team_id: Final = _new_team(gateway)
        name: Final = _tag_name()
        _create_tag(scenario, name, team_id=team_id)
        assert _tag_rows(name) == [{"tag_name": name, "team_id": team_id}]

        gateway.post("/team/delete", {"team_ids": [team_id]})

        assert _tag_rows(name) == [{"tag_name": name, "team_id": None}]
        assert _tag_info(gateway, name)["team_id"] is None


def test_tag_team_id_foreign_key_and_index(gateway: Gateway) -> None:
    name: Final = _tag_name()
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        write_rows(
            'INSERT INTO "LiteLLM_TagTable" (tag_name, models, team_id) VALUES (%s, %s, %s)',
            (name, [], "no-such-team"),
        )

    indexes: Final = read_rows("SELECT indexdef FROM pg_indexes WHERE tablename = 'LiteLLM_TagTable'", ())
    assert any("team_id" in str(index["indexdef"]) for index in indexes)


def test_tag_name_is_globally_unique_across_team_owners(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        team_a: Final = scenario.team()
        team_b: Final = scenario.team()
        name: Final = _tag_name()
        _create_tag(scenario, name, team_id=team_a)

        response: Final = gateway.request("POST", "/tag/new", {"name": name, "team_id": team_b})
        assert response.status_code == 400, response.text
        assert _tag_rows(name) == [{"tag_name": name, "team_id": team_a}]

        with pytest.raises(psycopg.errors.UniqueViolation):
            write_rows(
                'INSERT INTO "LiteLLM_TagTable" (tag_name, models) VALUES (%s, %s)',
                (name, []),
            )


def test_non_admin_key_cannot_set_tag_team_ownership(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        team_id: Final = scenario.team()
        user_id: Final = scenario.user(user_role="internal_user")
        key: Final = scenario.key(user_id=user_id, allowed_routes=["/tag/new", "/tag/update"])
        name: Final = _tag_name()

        create: Final = gateway.request("POST", "/tag/new", {"name": name, "team_id": team_id}, key=key)
        assert create.status_code == 403, create.text
        assert _tag_rows(name) == []

        _create_tag(scenario, name, team_id=team_id)

        assign: Final = gateway.request("POST", "/tag/update", {"name": name, "team_id": scenario.team()}, key=key)
        assert assign.status_code == 403, assign.text
        assert _tag_rows(name) == [{"tag_name": name, "team_id": team_id}]

        release: Final = gateway.request("POST", "/tag/update", {"name": name, "team_id": None}, key=key)
        assert release.status_code == 403, release.text
        assert _tag_rows(name) == [{"tag_name": name, "team_id": team_id}]

        unrelated: Final = gateway.request(
            "POST", "/tag/update", {"name": name, "description": "denied edit"}, key=key
        )
        assert unrelated.status_code == 403, unrelated.text
        assert _tag_rows(name) == [{"tag_name": name, "team_id": team_id}]
        assert _tag_info(gateway, name)["team_id"] == team_id


def test_team_admin_manages_their_own_team_tags(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        admin_id: Final = scenario.user()
        team_a: Final = scenario.team(members_with_roles=[{"role": "admin", "user_id": admin_id}])
        team_b: Final = scenario.team()
        admin_key: Final = scenario.key(user_id=admin_id, team_id=team_a)

        name: Final = _tag_name()
        create: Final = gateway.request(
            "POST", "/tag/new", {"name": name, "team_id": team_a, "models": []}, key=admin_key
        )
        assert create.status_code == 200, create.text
        scenario.cleanups.callback(lambda: gateway.request("POST", "/tag/delete", {"name": name}))
        assert _tag_rows(name) == [{"tag_name": name, "team_id": team_a}]

        update: Final = gateway.request(
            "POST", "/tag/update", {"name": name, "description": "admin edit"}, key=admin_key
        )
        assert update.status_code == 200, update.text
        assert _tag_rows(name) == [{"tag_name": name, "team_id": team_a}]

        transfer: Final = gateway.request(
            "POST", "/tag/update", {"name": name, "team_id": team_b}, key=admin_key
        )
        assert transfer.status_code == 403, transfer.text
        assert _tag_rows(name) == [{"tag_name": name, "team_id": team_a}]

        foreign_tag: Final = _tag_name()
        _create_tag(scenario, foreign_tag, team_id=team_b)
        foreign: Final = gateway.request("POST", "/tag/delete", {"name": foreign_tag}, key=admin_key)
        assert foreign.status_code == 403, foreign.text
        assert _tag_rows(foreign_tag) == [{"tag_name": foreign_tag, "team_id": team_b}]

        unowned_tag: Final = _tag_name()
        _create_tag(scenario, unowned_tag)
        unowned: Final = gateway.request("POST", "/tag/delete", {"name": unowned_tag}, key=admin_key)
        assert unowned.status_code == 403, unowned.text
        assert _tag_rows(unowned_tag) == [{"tag_name": unowned_tag, "team_id": None}]

        delete: Final = gateway.request("POST", "/tag/delete", {"name": name}, key=admin_key)
        assert delete.status_code == 200, delete.text
        assert _tag_rows(name) == []


def test_regular_team_member_cannot_manage_tags_and_sees_only_own_team_tags(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        member_id: Final = scenario.user()
        team_a: Final = scenario.team(members_with_roles=[{"role": "user", "user_id": member_id}])
        team_b: Final = scenario.team()
        member_key: Final = scenario.key(user_id=member_id, team_id=team_a)

        team_a_tag: Final = _tag_name()
        _create_tag(scenario, team_a_tag, team_id=team_a)
        team_b_tag: Final = _tag_name()
        _create_tag(scenario, team_b_tag, team_id=team_b)

        member_tag: Final = _tag_name()
        create: Final = gateway.request(
            "POST", "/tag/new", {"name": member_tag, "team_id": team_a, "models": []}, key=member_key
        )
        assert create.status_code == 403, create.text
        assert _tag_rows(member_tag) == []

        update: Final = gateway.request(
            "POST", "/tag/update", {"name": team_a_tag, "description": "member edit"}, key=member_key
        )
        assert update.status_code == 403, update.text

        delete: Final = gateway.request("POST", "/tag/delete", {"name": team_a_tag}, key=member_key)
        assert delete.status_code == 403, delete.text
        assert _tag_rows(team_a_tag) == [{"tag_name": team_a_tag, "team_id": team_a}]

        listing: Final = gateway.request("GET", "/tag/list", key=member_key)
        assert listing.status_code == 200, listing.text
        entries: Final = {entry["name"]: entry for entry in listing.json()}
        assert team_a_tag in entries
        assert entries[team_a_tag]["team_id"] == team_a
        assert team_b_tag not in entries


def test_deleting_two_owning_teams_nulls_all_their_tags(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        team_a: Final = _new_team(gateway)
        team_b: Final = _new_team(gateway)
        team_c: Final = scenario.team()
        tag_a: Final = _tag_name()
        tag_b: Final = _tag_name()
        tag_c: Final = _tag_name()
        _create_tag(scenario, tag_a, team_id=team_a)
        _create_tag(scenario, tag_b, team_id=team_b)
        _create_tag(scenario, tag_c, team_id=team_c)

        gateway.post("/team/delete", {"team_ids": [team_a, team_b]})

        assert _tag_rows(tag_a) == [{"tag_name": tag_a, "team_id": None}]
        assert _tag_rows(tag_b) == [{"tag_name": tag_b, "team_id": None}]
        assert _tag_rows(tag_c) == [{"tag_name": tag_c, "team_id": team_c}]
        assert _tag_info(gateway, tag_a)["team_id"] is None
        assert _tag_info(gateway, tag_b)["team_id"] is None
        assert _tag_info(gateway, tag_c)["team_id"] == team_c
