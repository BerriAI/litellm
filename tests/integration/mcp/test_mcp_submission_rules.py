import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Final

import httpx
import psycopg
from integration._support.client import JSON_OBJECT, Gateway, eventually, string_value
from integration._support.database import read_rows
from integration._support.mcp import McpPeer, forget_mcp, mcp_peer
from pydantic import JsonValue, TypeAdapter

_REQUIRED_FIELDS: Final = ("description", "url", "alias")
_JSON_ROWS: Final = TypeAdapter(list[dict[str, JsonValue]])


@contextmanager
def _submission_rules_lock() -> Iterator[None]:
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as connection:
        connection.execute("SELECT pg_advisory_lock(%s, %s)", (9126, 1))
        try:
            yield
        finally:
            connection.execute("SELECT pg_advisory_unlock(%s, %s)", (9126, 1))


def _set_submission_rules(gateway: Gateway) -> httpx.Response:
    return gateway.request(
        "POST",
        "/config/field/update",
        {
            "field_name": "mcp_required_fields",
            "field_value": list(_REQUIRED_FIELDS),
            "config_type": "general_settings",
        },
    )


def _delete_submission_rules(gateway: Gateway) -> None:
    response: Final = gateway.request(
        "POST",
        "/config/field/delete",
        {"field_name": "mcp_required_fields", "config_type": "general_settings"},
    )
    assert response.status_code == 200, response.text


def _registration_body(peer: McpPeer, alias: str, description: str | None = None) -> dict[str, JsonValue]:
    registration: Final = JSON_OBJECT.validate_python(peer.registration())
    return {
        "server_name": alias,
        "alias": alias,
        **registration,
        **({"description": description} if description is not None else {}),
    }


def _submit_registration(
    gateway: Gateway,
    peer: McpPeer,
    team_key: str,
    alias: str,
    description: str | None = None,
) -> httpx.Response:
    return gateway.client.post(
        "/v1/mcp/server/register",
        json=_registration_body(peer, alias, description),
        headers={"x-litellm-api-key": team_key},
    )


def _submission_items(response: httpx.Response) -> list[dict[str, JsonValue]]:
    summary: Final = JSON_OBJECT.validate_json(response.content)
    return _JSON_ROWS.validate_python(summary["items"])


def test_saved_submission_rules_are_listed_as_a_top_level_array(gateway: Gateway) -> None:
    with _submission_rules_lock(), gateway.scenario() as scenario:
        updated: Final = _set_submission_rules(gateway)
        scenario.cleanups.callback(_delete_submission_rules, gateway)
        assert updated.status_code == 200, updated.text

        listed: Final = gateway.request("GET", "/config/list", params={"config_type": "general_settings"})
        assert listed.status_code == 200, listed.text
        payload: Final = TypeAdapter(JsonValue).validate_json(listed.content)
        assert isinstance(payload, list), listed.text
        rows: Final = _JSON_ROWS.validate_python(payload)
        matching: Final = tuple(row for row in rows if row.get("field_name") == "mcp_required_fields")
        assert len(matching) == 1, listed.text
        assert matching[0]["field_value"] == list(_REQUIRED_FIELDS), listed.text
        assert matching[0]["stored_in_db"] is True, listed.text

        stored: Final = read_rows(
            'SELECT param_value -> \'mcp_required_fields\' AS field_value FROM "LiteLLM_Config" '
            "WHERE param_name = %s",
            ("general_settings",),
        )
        assert stored == [{"field_value": list(_REQUIRED_FIELDS)}]


def test_team_submission_missing_a_required_field_is_rejected_and_complete_one_is_pending(
    gateway: Gateway,
) -> None:
    with mcp_peer() as peer, _submission_rules_lock(), gateway.scenario() as scenario:
        updated: Final = _set_submission_rules(gateway)
        scenario.cleanups.callback(_delete_submission_rules, gateway)
        assert updated.status_code == 200, updated.text

        team_id: Final = scenario.team()
        team_key: Final = scenario.key(team_id=team_id)

        def submit_without_description() -> tuple[httpx.Response, str]:
            alias: Final = f"mcp_rules_{uuid.uuid4().hex}"
            response: Final = _submit_registration(gateway, peer, team_key, alias)
            if response.status_code == 201:
                accepted: Final = JSON_OBJECT.validate_json(response.content)
                scenario.cleanups.callback(forget_mcp, gateway, string_value(accepted["server_id"]))
            return response, alias

        missing: Final = eventually(
            submit_without_description,
            lambda result: result[0].status_code == 400
            and "Submission is missing required fields: ['description']" in result[0].text,
            seconds=30,
        )
        missing_response, missing_alias = missing
        assert missing_response.status_code == 400, missing_response.text
        assert "Submission is missing required fields: ['description']" in missing_response.text
        assert (
            read_rows('SELECT server_id FROM "LiteLLM_MCPServerTable" WHERE alias = %s', (missing_alias,)) == []
        )

        alias: Final = f"mcp_rules_{uuid.uuid4().hex}"
        registered: Final = _submit_registration(gateway, peer, team_key, alias, "An integration MCP server")
        assert registered.status_code == 201, registered.text
        submission: Final = JSON_OBJECT.validate_json(registered.content)
        server_id: Final = string_value(submission["server_id"])
        scenario.cleanups.callback(forget_mcp, gateway, server_id)

        listed: Final = eventually(
            lambda: gateway.request("GET", "/v1/mcp/server/submissions"),
            lambda response: response.status_code == 200 and server_id in response.text,
        )
        assert listed.status_code == 200, listed.text
        assert any(
            item.get("server_id") == server_id and item.get("approval_status") == "pending_review"
            for item in _submission_items(listed)
        ), listed.text
