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

    indexes: Final = read_rows(
        "SELECT indexdef FROM pg_indexes WHERE tablename = 'LiteLLM_TagTable'", ()
    )
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

        create: Final = gateway.request(
            "POST", "/tag/new", {"name": name, "team_id": team_id}, key=key
        )
        assert create.status_code == 403, create.text
        assert _tag_rows(name) == []

        _create_tag(scenario, name, team_id=team_id)

        assign: Final = gateway.request(
            "POST", "/tag/update", {"name": name, "team_id": scenario.team()}, key=key
        )
        assert assign.status_code == 403, assign.text
        assert _tag_rows(name) == [{"tag_name": name, "team_id": team_id}]

        release: Final = gateway.request(
            "POST", "/tag/update", {"name": name, "team_id": None}, key=key
        )
        assert release.status_code == 403, release.text
        assert _tag_rows(name) == [{"tag_name": name, "team_id": team_id}]

        unrelated: Final = gateway.request(
            "POST", "/tag/update", {"name": name, "description": "allowed edit"}, key=key
        )
        assert unrelated.status_code == 200, unrelated.text
        assert _tag_rows(name) == [{"tag_name": name, "team_id": team_id}]
        assert _tag_info(gateway, name)["team_id"] == team_id
