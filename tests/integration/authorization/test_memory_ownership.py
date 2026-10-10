import uuid
from typing import Final

import httpx
from integration._support.client import Gateway, Scenario
from integration._support.database import read_rows, write_rows
from pydantic import JsonValue, TypeAdapter

from litellm.types.memory_management import LiteLLM_MemoryRow, MemoryDeleteResponse, MemoryListResponse

_MEMORY_ROUTE: Final = ["/v1/memory"]
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


def _admin_only_error(route: str, user_id: str) -> dict[str, JsonValue]:
    masked_user_id: Final = f"{user_id[:6]}{'*' * (len(user_id) - 8)}{user_id[-2:]}"
    return {
        "error": {
            "message": (
                "Authentication Error, Only proxy admin can be used to generate, delete, update info for new "
                f"keys/users/teams. Route={route}. Your role=internal_user. Your user_id={masked_user_id}"
            ),
            "type": "auth_error",
            "param": "None",
            "code": "401",
        }
    }


def _delete_memory_row(key: str) -> None:
    write_rows('DELETE FROM "LiteLLM_MemoryTable" WHERE key = %s', (key,))


def _memory_rows(key: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT key, value, user_id, team_id, created_by, updated_by FROM "LiteLLM_MemoryTable" WHERE key = %s',
        (key,),
    )


def _memory_create_rows(key: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        "SELECT memory_id, key, value, metadata, user_id, team_id, created_by, updated_by "
        'FROM "LiteLLM_MemoryTable" WHERE key = %s',
        (key,),
    )


def _memory_request(
    gateway: Gateway,
    method: str,
    key: str,
    *,
    caller: str,
    body: dict[str, JsonValue] | None = None,
) -> httpx.Response:
    return gateway.request(method, f"/v1/memory/{key}", body, key=caller)


def _register_memory_cleanup(scenario: Scenario, key: str) -> None:
    scenario.cleanups.callback(_delete_memory_row, key)


def test_memory_create_enforces_personal_and_team_scope(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        other_team: Final = scenario.team()
        member_a: Final = scenario.member(team)
        member_b: Final = scenario.member(team)
        key_a: Final = scenario.key(user_id=member_a, team_id=team, allowed_routes=_MEMORY_ROUTE)
        own_key: Final = f"memory-create-own-{uuid.uuid4().hex}"
        other_user_key: Final = f"memory-create-other-user-{uuid.uuid4().hex}"
        other_team_key: Final = f"memory-create-other-team-{uuid.uuid4().hex}"
        admin_key: Final = f"memory-create-admin-{uuid.uuid4().hex}"

        _register_memory_cleanup(scenario, own_key)
        own_response: Final = gateway.request(
            "POST",
            "/v1/memory",
            {"key": own_key, "value": "own"},
            key=key_a,
        )
        assert own_response.status_code == 200, own_response.text
        own: Final = LiteLLM_MemoryRow.model_validate_json(own_response.content)
        assert (
            own.key,
            own.value,
            own.metadata,
            own.user_id,
            own.team_id,
            own.created_by,
            own.updated_by,
        ) == (own_key, "own", None, member_a, team, member_a, member_a), own_response.text
        assert own.created_at is not None and own.updated_at is not None, own_response.text
        assert _memory_create_rows(own_key) == [
            {
                "memory_id": own.memory_id,
                "key": own_key,
                "value": "own",
                "metadata": None,
                "user_id": member_a,
                "team_id": team,
                "created_by": member_a,
                "updated_by": member_a,
            }
        ], own_response.text

        _register_memory_cleanup(scenario, other_user_key)
        other_user_response: Final = gateway.request(
            "POST",
            "/v1/memory",
            {"key": other_user_key, "value": "x", "user_id": member_b},
            key=key_a,
        )
        assert other_user_response.status_code == 403, other_user_response.text
        assert other_user_response.text == '{"detail":"Only proxy admins may set user_id to a different user."}', (
            other_user_response.text
        )
        assert _memory_rows(other_user_key) == []

        _register_memory_cleanup(scenario, other_team_key)
        other_team_response: Final = gateway.request(
            "POST",
            "/v1/memory",
            {"key": other_team_key, "value": "x", "team_id": other_team},
            key=key_a,
        )
        assert other_team_response.status_code == 403, other_team_response.text
        assert other_team_response.text == '{"detail":"Only proxy admins may set team_id to a different team."}', (
            other_team_response.text
        )
        assert _memory_rows(other_team_key) == []

        _register_memory_cleanup(scenario, admin_key)
        admin_response: Final = gateway.request(
            "POST",
            "/v1/memory",
            {"key": admin_key, "value": "x", "user_id": member_b},
            key=gateway.key,
        )
        assert admin_response.status_code == 200, admin_response.text
        admin_memory: Final = LiteLLM_MemoryRow.model_validate_json(admin_response.content)
        assert (
            admin_memory.key,
            admin_memory.value,
            admin_memory.user_id,
            admin_memory.created_by,
            admin_memory.updated_by,
        ) == (admin_key, "x", member_b, "default_user_id", "default_user_id"), admin_response.text
        assert admin_memory.created_at is not None and admin_memory.updated_at is not None, admin_response.text
        assert _memory_create_rows(admin_key) == [
            {
                "memory_id": admin_memory.memory_id,
                "key": admin_key,
                "value": "x",
                "metadata": None,
                "user_id": member_b,
                "team_id": None,
                "created_by": "default_user_id",
                "updated_by": "default_user_id",
            }
        ], admin_response.text


def test_memory_routes_enforce_personal_and_team_write_ownership(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        member_a: Final = scenario.member(team)
        member_b: Final = scenario.member(team)
        team_admin: Final = scenario.member(team, role="admin")
        outsider: Final = scenario.user(user_role="internal_user")
        default_key: Final = scenario.key(user_id=member_a)
        key_a: Final = scenario.key(user_id=member_a, team_id=team, allowed_routes=_MEMORY_ROUTE)
        key_b: Final = scenario.key(user_id=member_b, team_id=team, allowed_routes=_MEMORY_ROUTE)
        admin_key: Final = scenario.key(user_id=team_admin, team_id=team, allowed_routes=_MEMORY_ROUTE)
        team_key: Final = scenario.key(team_id=team, allowed_routes=_MEMORY_ROUTE)
        outsider_key: Final = scenario.key(user_id=outsider, allowed_routes=_MEMORY_ROUTE)
        personal_key: Final = f"agent/{team}-{member_a}-alpha"
        team_only_key: Final = f"agent/{team}-shared"

        default_access: Final = _memory_request(
            gateway,
            "PUT",
            personal_key,
            caller=default_key,
            body={"value": "default access"},
        )
        assert default_access.status_code == 401, default_access.text
        assert _JSON_OBJECT.validate_json(default_access.content) == _admin_only_error(
            f"/v1/memory/{personal_key}", member_a
        ), default_access.text

        created: Final = _memory_request(
            gateway,
            "PUT",
            personal_key,
            caller=key_a,
            body={"value": "a1"},
        )
        assert created.status_code == 200, created.text
        _register_memory_cleanup(scenario, personal_key)
        created_row: Final = LiteLLM_MemoryRow.model_validate_json(created.content)
        assert (created_row.key, created_row.value, created_row.user_id, created_row.team_id) == (
            personal_key,
            "a1",
            member_a,
            team,
        ), created.text
        assert _memory_rows(personal_key) == [
            {
                "key": personal_key,
                "value": "a1",
                "user_id": member_a,
                "team_id": team,
                "created_by": member_a,
                "updated_by": member_a,
            }
        ]

        visible_to_member: Final = _memory_request(gateway, "GET", personal_key, caller=key_b)
        assert visible_to_member.status_code == 200, visible_to_member.text
        assert LiteLLM_MemoryRow.model_validate_json(visible_to_member.content).value == "a1", visible_to_member.text
        overwrite_refused: Final = _memory_request(
            gateway,
            "PUT",
            personal_key,
            caller=key_b,
            body={"value": "b-overwrite"},
        )
        assert overwrite_refused.status_code == 403, overwrite_refused.text
        assert overwrite_refused.text == ('{"detail":"You do not have permission to modify this memory entry."}'), (
            overwrite_refused.text
        )
        delete_refused: Final = _memory_request(gateway, "DELETE", personal_key, caller=key_b)
        assert delete_refused.status_code == 403, delete_refused.text
        assert _memory_rows(personal_key) == [
            {
                "key": personal_key,
                "value": "a1",
                "user_id": member_a,
                "team_id": team,
                "created_by": member_a,
                "updated_by": member_a,
            }
        ]

        outsider_read: Final = _memory_request(gateway, "GET", personal_key, caller=outsider_key)
        assert outsider_read.status_code == 404, outsider_read.text
        assert outsider_read.text == f"""{{"detail":"Memory with key '{personal_key}' not found"}}""", (
            outsider_read.text
        )
        outsider_list: Final = gateway.request("GET", "/v1/memory", key=outsider_key)
        assert outsider_list.status_code == 200, outsider_list.text
        outsider_memories: Final = MemoryListResponse.model_validate_json(outsider_list.content)
        assert outsider_memories.total == 0 and outsider_memories.memories == [], outsider_list.text

        overwritten: Final = _memory_request(
            gateway,
            "PUT",
            personal_key,
            caller=key_a,
            body={"value": "a2"},
        )
        assert overwritten.status_code == 200, overwritten.text
        assert LiteLLM_MemoryRow.model_validate_json(overwritten.content).value == "a2", overwritten.text
        assert _memory_rows(personal_key) == [
            {
                "key": personal_key,
                "value": "a2",
                "user_id": member_a,
                "team_id": team,
                "created_by": member_a,
                "updated_by": member_a,
            }
        ]

        team_created: Final = _memory_request(
            gateway,
            "PUT",
            team_only_key,
            caller=team_key,
            body={"value": "team value"},
        )
        assert team_created.status_code == 200, team_created.text
        _register_memory_cleanup(scenario, team_only_key)
        team_row: Final = LiteLLM_MemoryRow.model_validate_json(team_created.content)
        assert (team_row.value, team_row.user_id, team_row.team_id) == ("team value", None, team), team_created.text
        assert _memory_rows(team_only_key) == [
            {
                "key": team_only_key,
                "value": "team value",
                "user_id": None,
                "team_id": team,
                "created_by": None,
                "updated_by": None,
            }
        ]
        member_team_read: Final = _memory_request(gateway, "GET", team_only_key, caller=key_b)
        assert member_team_read.status_code == 200, member_team_read.text
        assert (
            LiteLLM_MemoryRow.model_validate_json(member_team_read.content).model_dump(include={"key", "value"})
        ) == {"key": team_only_key, "value": "team value"}, member_team_read.text
        member_list: Final = gateway.request("GET", "/v1/memory", key=key_b)
        assert member_list.status_code == 200, member_list.text
        member_memories: Final = MemoryListResponse.model_validate_json(member_list.content)
        assert (member_memories.total, sorted((row.key, row.value) for row in member_memories.memories)) == (
            2,
            sorted(((personal_key, "a2"), (team_only_key, "team value"))),
        ), member_list.text
        outsider_team_read: Final = _memory_request(gateway, "GET", team_only_key, caller=outsider_key)
        assert outsider_team_read.status_code == 404, outsider_team_read.text

        member_team_write: Final = _memory_request(
            gateway,
            "PUT",
            team_only_key,
            caller=key_a,
            body={"value": "member overwrite"},
        )
        assert member_team_write.status_code == 403, member_team_write.text
        assert _memory_rows(team_only_key) == [
            {
                "key": team_only_key,
                "value": "team value",
                "user_id": None,
                "team_id": team,
                "created_by": None,
                "updated_by": None,
            }
        ]
        admin_write: Final = _memory_request(
            gateway,
            "PUT",
            team_only_key,
            caller=admin_key,
            body={"value": "admin overwrite"},
        )
        assert admin_write.status_code == 200, admin_write.text
        assert LiteLLM_MemoryRow.model_validate_json(admin_write.content).value == "admin overwrite", admin_write.text

        team_admin_personal_write: Final = _memory_request(
            gateway,
            "PUT",
            personal_key,
            caller=admin_key,
            body={"value": "team admin overwrite"},
        )
        assert team_admin_personal_write.status_code == 403, team_admin_personal_write.text
        assert team_admin_personal_write.text == (
            '{"detail":"You do not have permission to modify this memory entry."}'
        ), team_admin_personal_write.text
        assert _memory_rows(personal_key) == [
            {
                "key": personal_key,
                "value": "a2",
                "user_id": member_a,
                "team_id": team,
                "created_by": member_a,
                "updated_by": member_a,
            }
        ]

        master_read: Final = _memory_request(gateway, "GET", personal_key, caller=gateway.key)
        assert master_read.status_code == 200, master_read.text
        assert LiteLLM_MemoryRow.model_validate_json(master_read.content).value == "a2", master_read.text
        master_write: Final = _memory_request(
            gateway,
            "PUT",
            personal_key,
            caller=gateway.key,
            body={"value": "master overwrite"},
        )
        assert master_write.status_code == 200, master_write.text
        assert _memory_rows(personal_key) == [
            {
                "key": personal_key,
                "value": "master overwrite",
                "user_id": member_a,
                "team_id": team,
                "created_by": member_a,
                "updated_by": "default_user_id",
            }
        ]

        deleted: Final = _memory_request(gateway, "DELETE", personal_key, caller=key_a)
        assert deleted.status_code == 200, deleted.text
        assert MemoryDeleteResponse.model_validate_json(deleted.content).deleted is True, deleted.text
        assert _memory_rows(personal_key) == []
        missing: Final = _memory_request(gateway, "GET", personal_key, caller=key_a)
        assert missing.status_code == 404, missing.text
        assert missing.text == f"""{{"detail":"Memory with key '{personal_key}' not found"}}""", missing.text
