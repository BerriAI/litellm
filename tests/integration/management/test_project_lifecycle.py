import json
import os
import re
import signal
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from typing import Final
from uuid import uuid4

import httpx
import psutil
import psycopg
import pytest
from integration._support.client import (
    JSON_OBJECT,
    Gateway,
    delete_key_if_present,
    eventually,
    gateway_from_environment,
    object_value,
    string_value,
)
from integration._support.database import read_rows, write_rows
from integration._support.process import group_members, owned_proxy, owned_proxy_process
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

from litellm.models.user import LiteLLM_UserTable
from litellm.proxy.auth.auth_checks import ExperimentalUIJWTToken

_OWNED_PROXY_SALT_KEY: Final = "sk-integration-salt"
_OWNERSHIP_MARKER: Final = re.compile(rb"ownership-[0-9a-f]{32}")


def _project_rows(project_id: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT p.project_id, p.project_alias, p.description, p.team_id, p.models, p.blocked, '
        'p.budget_id, b.max_budget FROM "LiteLLM_ProjectTable" AS p '
        'LEFT JOIN "LiteLLM_BudgetTable" AS b ON b.budget_id = p.budget_id '
        'WHERE p.project_id = %s',
        (project_id,),
    )


def _key_rows(key: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT token, key_alias, team_id, project_id FROM "LiteLLM_VerificationToken" WHERE token = %s',
        (sha256(key.encode()).hexdigest(),),
    )


def _key_permission_and_budget_ids(key: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT object_permission_id, budget_id FROM "LiteLLM_VerificationToken" WHERE token = %s',
        (sha256(key.encode()).hexdigest(),),
    )


def _key_state_rows(key: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT to_jsonb(k) AS key_row, to_jsonb(b) AS budget_row FROM "LiteLLM_VerificationToken" AS k '
        'LEFT JOIN "LiteLLM_BudgetTable" AS b ON b.budget_id = k.budget_id WHERE k.token = %s',
        (sha256(key.encode()).hexdigest(),),
    )


def _project_state_rows(project_id: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT to_jsonb(p) AS project_row, to_jsonb(b) AS budget_row FROM "LiteLLM_ProjectTable" AS p '
        'LEFT JOIN "LiteLLM_BudgetTable" AS b ON b.budget_id = p.budget_id WHERE p.project_id = %s',
        (project_id,),
    )


def _object_permission_rows(permission_id: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT * FROM "LiteLLM_ObjectPermissionTable" WHERE object_permission_id = %s',
        (permission_id,),
    )


def _object_permission_table_rows() -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT to_jsonb(p) AS row FROM "LiteLLM_ObjectPermissionTable" AS p ORDER BY object_permission_id',
        (),
    )


def _budget_table_rows() -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT to_jsonb(b) AS row FROM "LiteLLM_BudgetTable" AS b ORDER BY budget_id',
        (),
    )


def _budget_rows(budget_id: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT to_jsonb(b) AS row FROM "LiteLLM_BudgetTable" AS b WHERE budget_id = %s',
        (budget_id,),
    )


def _clear_key_object_permission(key: str, permission_id: str) -> None:
    write_rows(
        'UPDATE "LiteLLM_VerificationToken" SET object_permission_id = NULL WHERE token = %s',
        (sha256(key.encode()).hexdigest(),),
    )
    write_rows(
        'DELETE FROM "LiteLLM_ObjectPermissionTable" WHERE object_permission_id = %s',
        (permission_id,),
    )


def _deleted_key_rows(key: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT to_jsonb(d) AS row FROM "LiteLLM_DeletedVerificationToken" AS d WHERE d.token = %s',
        (sha256(key.encode()).hexdigest(),),
    )


def _deprecated_key_rows(key: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT to_jsonb(d) AS row FROM "LiteLLM_DeprecatedVerificationToken" AS d WHERE d.token = %s',
        (sha256(key.encode()).hexdigest(),),
    )


def _cli_session_token(
    user_id: str,
    team_id: str | None,
    *,
    monkeypatch: pytest.MonkeyPatch,
    max_budget: float | None = None,
) -> str:
    monkeypatch.setenv("LITELLM_SALT_KEY", _OWNED_PROXY_SALT_KEY)
    user: Final = LiteLLM_UserTable(
        user_id=user_id,
        user_role="internal_user",
        teams=[team_id] if team_id is not None else [],
        models=[],
        max_budget=max_budget,
    )
    return ExperimentalUIJWTToken.get_cli_jwt_auth_token(
        user_info=user,
        team_id=team_id,
        team_alias="ownership-team" if team_id is not None else None,
        max_budget=max_budget,
    )


def _discard_unexpected_key(candidate: Gateway, response: httpx.Response) -> None:
    if response.status_code != 200:
        return
    body: Final = JSON_OBJECT.validate_json(response.content)
    candidate.post("/key/delete", {"keys": [string_value(body["key"])]})


def _ownership_chat_reply(identity: str, stream: bool) -> Reply:
    if not stream:
        return Reply(
            body=json.dumps(
                {
                    "id": identity,
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4.1-mini",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ownership ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
                }
            ).encode()
        )
    return Reply(
        content_type="text/event-stream",
        chunks=(
            f"data: {json.dumps({'id': identity, 'object': 'chat.completion.chunk', 'created': 1, 'model': 'gpt-4.1-mini', 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': 'ownership'}}]})}\n\n".encode(),
            f"data: {json.dumps({'id': identity, 'object': 'chat.completion.chunk', 'created': 1, 'model': 'gpt-4.1-mini', 'choices': [{'index': 0, 'delta': {'content': ' ok'}, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 7, 'completion_tokens': 2, 'total_tokens': 9}})}\n\n".encode(),
            b"data: [DONE]\n\n",
        ),
    )


def _ownership_responses_reply(identity: str, stream: bool) -> Reply:
    response: Final = {
        "id": identity,
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-4.1-mini",
        "output": [
            {
                "id": f"msg_{identity}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "ownership ok", "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 7, "output_tokens": 2, "total_tokens": 9},
    }
    if not stream:
        return Reply(body=json.dumps(response).encode())
    events: Final = (
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": {**response, "status": "in_progress", "output": []},
        },
        {
            "type": "response.output_text.delta",
            "sequence_number": 1,
            "item_id": f"msg_{identity}",
            "output_index": 0,
            "content_index": 0,
            "delta": "ownership ok",
        },
        {"type": "response.completed", "sequence_number": 2, "response": response},
    )
    return Reply(
        content_type="text/event-stream",
        chunks=tuple(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events),
    )


def _ownership_upstream(request: Request) -> Reply:
    found: Final = _OWNERSHIP_MARKER.search(request.body)
    if found is None:
        return Reply(status=400, body=b'{"error":"missing ownership marker"}')
    marker: Final = found.group(0).decode()
    body: Final = JSON_OBJECT.validate_json(request.body)
    stream: Final = body.get("stream") is True
    if request.target.endswith("/responses"):
        return _ownership_responses_reply(f"resp_{marker}", stream)
    return _ownership_chat_reply(f"chatcmpl-{marker}", stream)


def _ownership_sse_events(text: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        JSON_OBJECT.validate_json(line[6:])
        for line in text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    )


def _ownership_request_marker(request: Request) -> str:
    found: Final = _OWNERSHIP_MARKER.search(request.body)
    assert found is not None, request.body
    return found.group(0).decode()


def _ownership_request_payload(
    path: str, model: str, marker: str, stream: bool
) -> tuple[dict[str, JsonValue], dict[str, str]]:
    if path.endswith("/messages"):
        return (
            {
                "model": model,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": marker}],
                "stream": stream,
            },
            {"anthropic-version": "2023-06-01"},
        )
    if path.endswith("/responses"):
        return {"model": model, "input": marker, "stream": stream}, {}
    return {"model": model, "messages": [{"role": "user", "content": marker}], "stream": stream}, {}


def _ownership_response_id(response: httpx.Response, path: str) -> str:
    if not response.headers.get("content-type", "").startswith("text/event-stream"):
        return string_value(JSON_OBJECT.validate_json(response.content)["id"])
    events: Final = _ownership_sse_events(response.text)
    identities: Final = (
        tuple(
            string_value(object_value(event["message"])["id"])
            for event in events
            if event.get("type") == "message_start"
        )
        if path.endswith("/messages")
        else tuple(
            string_value(object_value(event["response"])["id"])
            for event in events
            if event.get("type") == "response.completed"
        )
        if path.endswith("/responses")
        else tuple(string_value(event["id"]) for event in events if "id" in event)
    )
    unique_identities: Final = frozenset(identities)
    assert len(unique_identities) == 1, response.text
    return next(iter(unique_identities))


def _ownership_serving_call(
    candidate: Gateway,
    key: str,
    model: str,
    index: int,
    marker: str,
) -> tuple[int, str, str]:
    stream: Final = index % 2 == 0
    route: Final = index % 3
    paths: Final = ("/v1/chat/completions", "/v1/responses", "/v1/messages")
    path: Final = paths[route]
    body, headers = _ownership_request_payload(path, model, marker, stream)
    response: Final = candidate.request("POST", path, body, key=key, headers=headers)
    response.read()
    if response.status_code != 200:
        return response.status_code, "", response.text
    return response.status_code, _ownership_response_id(response, path), response.text


def _create_mcp_server(candidate: Gateway, server_id: str, server_name: str, alias: str) -> None:
    response: Final = candidate.request(
        "POST",
        "/v1/mcp/server",
        {
            "server_id": server_id,
            "server_name": server_name,
            "alias": alias,
            "transport": "sse",
            "url": "http://127.0.0.1:9/mcp",
        },
    )
    assert response.status_code == 201, response.text


def _delete_mcp_server(candidate: Gateway, server_id: str) -> None:
    response: Final = candidate.request("DELETE", f"/v1/mcp/server/{server_id}")
    assert response.status_code == 202, response.text


def test_service_account_generate_rejects_foreign_team_project_without_writing_key(ownership_gateway: Gateway) -> None:
    with ownership_gateway.scenario() as scenario:
        model: Final = scenario.model()
        team_a: Final = scenario.team(models=[model])
        team_b: Final = scenario.team(models=[model])
        project_b: Final = scenario.project(team_b, models=[model])
        alias: Final = f"service-account-{uuid4().hex}"
        vector_store: Final = f"vector-store-{uuid4().hex}"
        budget_rows_before: Final = _budget_table_rows()
        object_permission_rows_before: Final = _object_permission_table_rows()
        response: Final = ownership_gateway.request(
            "POST",
            "/key/service-account/generate",
            {
                "team_id": team_a,
                "project_id": project_b,
                "key_alias": alias,
                "models": [model],
                "soft_budget": 3.5,
                "object_permission": {"vector_stores": [vector_store]},
            },
        )
        _discard_unexpected_key(ownership_gateway, response)
        assert response.status_code == 400, response.text
        assert _budget_table_rows() == budget_rows_before
        assert _object_permission_table_rows() == object_permission_rows_before
        assert (
            read_rows(
                'SELECT token FROM "LiteLLM_VerificationToken" WHERE project_id = %s AND key_alias = %s',
                (project_b, alias),
            )
            == []
        )


def test_key_generate_unowned_project_accepts_any_team(ownership_gateway: Gateway) -> None:
    with ownership_gateway.scenario() as scenario:
        model: Final = scenario.model()
        team_a: Final = scenario.team(models=[model])
        team_b: Final = scenario.team(models=[model])
        project: Final = scenario.project(team_a, models=[model])
        write_rows('UPDATE "LiteLLM_ProjectTable" SET team_id = NULL WHERE project_id = %s', (project,))
        team_key: Final = scenario.key(team_id=team_b, project_id=project, models=[model])
        teamless_key: Final = scenario.key(project_id=project, models=[model])
        chats: Final = tuple(
            ownership_gateway.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": "legacy unowned"}]},
                key=key,
            )
            for key in (team_key, teamless_key)
        )
        assert all(chat.status_code == 200 for chat in chats), tuple(chat.text for chat in chats)


@pytest.mark.parametrize(
    ("path", "stream"),
    (
        ("/v1/chat/completions", False),
        ("/v1/chat/completions", True),
        ("/v1/messages", False),
        ("/v1/messages", True),
        ("/v1/responses", False),
        ("/v1/responses", True),
    ),
)
def test_legacy_mismatched_key_keeps_serving_and_stays_editable(
    ownership_gateway: Gateway, path: str, stream: bool
) -> None:
    def respond(request: Request) -> Reply:
        return _ownership_upstream(request)

    with wire_server(respond) as upstream:
        with ownership_gateway.scenario() as scenario:
            model: Final = scenario.model(api_base=f"{upstream.url}/v1")
            team_a: Final = scenario.team(models=[model])
            team_b: Final = scenario.team(models=[model])
            project: Final = scenario.project(team_a, models=[model])
            key: Final = scenario.key(team_id=team_a, project_id=project, models=[model])
            write_rows(
                'UPDATE "LiteLLM_VerificationToken" SET team_id = %s WHERE token = %s',
                (team_b, sha256(key.encode()).hexdigest()),
            )
            marker: Final = f"ownership-{uuid4().hex}"
            body, headers = _ownership_request_payload(path, model, marker, stream)
            response: Final = ownership_gateway.request("POST", path, body, key=key, headers=headers)
            response.read()
            assert response.status_code == 200, response.text
            requests: Final = upstream.drain()
            assert len(requests) == 1
            expected_target: Final = "/chat/completions" if path.endswith("/chat/completions") else "/responses"
            assert requests[0].target.endswith(expected_target), requests[0].target
            assert marker.encode() in requests[0].body
            assert _ownership_response_id(response, path) != ""


def test_stored_team_mismatch_allows_key_edits_and_detach(ownership_gateway: Gateway) -> None:
    with ownership_gateway.scenario() as scenario:
        model: Final = scenario.model()
        team_a: Final = scenario.team(models=[model])
        team_b: Final = scenario.team(models=[model])
        project: Final = scenario.project(team_a, models=[model])
        key: Final = scenario.key(team_id=team_a, project_id=project, models=[model])
        write_rows(
            'UPDATE "LiteLLM_VerificationToken" SET team_id = %s WHERE token = %s',
            (team_b, sha256(key.encode()).hexdigest()),
        )
        alias: Final = f"stored-mismatch-{uuid4().hex}"
        alias_update: Final = ownership_gateway.request(
            "POST",
            "/key/update",
            {"key": key, "key_alias": alias},
        )
        assert alias_update.status_code == 200, alias_update.text
        aliased_key: Final = _key_rows(key)
        assert aliased_key[0]["key_alias"] == alias
        unchanged: Final = ownership_gateway.request(
            "POST",
            "/key/update",
            {"key": key, "team_id": team_b},
        )
        assert unchanged.status_code == 200, unchanged.text
        assert _key_rows(key) == aliased_key
        detached: Final = ownership_gateway.request(
            "POST",
            "/key/update",
            {"key": key, "project_id": None},
        )
        assert detached.status_code == 200, detached.text
        detached_key: Final = _key_rows(key)
        assert detached_key[0]["team_id"] == team_b
        assert detached_key[0]["project_id"] is None


def test_key_regenerate_cross_team_with_object_permission_writes_nothing(ownership_gateway: Gateway) -> None:
    with ownership_gateway.scenario() as scenario:
        model: Final = scenario.model()
        team_a: Final = scenario.team(models=[model])
        team_b: Final = scenario.team(models=[model])
        project_a: Final = scenario.project(team_a, models=[model])
        project_b: Final = scenario.project(team_b, models=[model])
        key: Final = scenario.key(
            team_id=team_a,
            project_id=project_a,
            models=[model],
            object_permission={"vector_stores": ["before"]},
        )
        ids: Final = _key_permission_and_budget_ids(key)
        assert len(ids) == 1
        permission_id: Final = string_value(ids[0]["object_permission_id"])
        scenario.cleanups.callback(_clear_key_object_permission, key, permission_id)
        before: Final = (
            _object_permission_rows(permission_id),
            _object_permission_table_rows(),
            _key_rows(key),
            _key_state_rows(key),
            _key_permission_and_budget_ids(key),
            _deleted_key_rows(key),
            _deprecated_key_rows(key),
        )
        response: Final = ownership_gateway.request(
            "POST",
            f"/key/{key}/regenerate",
            {"project_id": project_b, "object_permission": {"vector_stores": ["after"]}},
        )
        _discard_unexpected_key(ownership_gateway, response)
        assert response.status_code == 400, response.text
        assert (
            _object_permission_rows(permission_id),
            _object_permission_table_rows(),
            _key_rows(key),
            _key_state_rows(key),
            _key_permission_and_budget_ids(key),
            _deleted_key_rows(key),
            _deprecated_key_rows(key),
        ) == before


def test_key_bulk_update_cross_team_with_object_permission_preserves_permission(ownership_gateway: Gateway) -> None:
    with ownership_gateway.scenario() as scenario:
        model: Final = scenario.model()
        team_a: Final = scenario.team(models=[model])
        team_b: Final = scenario.team(models=[model])
        project: Final = scenario.project(team_a, models=[model])
        key: Final = scenario.key(
            team_id=team_a,
            project_id=project,
            models=[model],
            object_permission={"vector_stores": ["before"]},
        )
        ids: Final = _key_permission_and_budget_ids(key)
        assert len(ids) == 1
        permission_id: Final = string_value(ids[0]["object_permission_id"])
        scenario.cleanups.callback(_clear_key_object_permission, key, permission_id)
        before: Final = (
            _object_permission_rows(permission_id),
            _object_permission_table_rows(),
            _key_rows(key),
            _key_state_rows(key),
            _key_permission_and_budget_ids(key),
            _deleted_key_rows(key),
            _deprecated_key_rows(key),
        )
        response: Final = ownership_gateway.request(
            "POST",
            "/key/bulk_update",
            {
                "keys": [
                    {
                        "key": key,
                        "team_id": team_b,
                        "object_permission": {"vector_stores": ["after"]},
                    }
                ],
            },
        )
        assert response.status_code == 200, response.text
        failed_updates: Final = JSON_OBJECT.validate_json(response.content)["failed_updates"]
        assert isinstance(failed_updates, list)
        assert len(failed_updates) == 1
        failed_update: Final = object_value(failed_updates[0])
        assert f"Project {project} belongs to team {team_a}" in string_value(failed_update["failed_reason"])
        assert (
            _object_permission_rows(permission_id),
            _object_permission_table_rows(),
            _key_rows(key),
            _key_state_rows(key),
            _key_permission_and_budget_ids(key),
            _deleted_key_rows(key),
            _deprecated_key_rows(key),
        ) == before


def test_key_update_ambiguous_mcp_permission_error_precedes_project_ownership(ownership_gateway: Gateway) -> None:
    with ownership_gateway.scenario() as scenario:
        model: Final = scenario.model()
        identifier: Final = f"ambiguous{uuid4().hex}"
        first_id: Final = f"mcp{uuid4().hex}"
        second_id: Final = f"mcp{uuid4().hex}"
        _create_mcp_server(ownership_gateway, first_id, identifier, f"alias{uuid4().hex}")
        scenario.cleanups.callback(_delete_mcp_server, ownership_gateway, first_id)
        _create_mcp_server(ownership_gateway, second_id, f"name{uuid4().hex}", f"alias{uuid4().hex}")
        scenario.cleanups.callback(_delete_mcp_server, ownership_gateway, second_id)
        write_rows(
            'UPDATE "LiteLLM_MCPServerTable" SET alias = %s WHERE server_id = %s',
            (identifier, second_id),
        )
        team_permissions: Final = {"mcp_servers": [first_id, second_id]}
        team_a: Final = scenario.team(models=[model], object_permission=team_permissions)
        team_b: Final = scenario.team(models=[model], object_permission=team_permissions)
        project: Final = scenario.project(team_a, models=[model])
        key: Final = scenario.key(
            team_id=team_a,
            project_id=project,
            models=[model],
            object_permission={"vector_stores": ["before"]},
        )
        ids: Final = _key_permission_and_budget_ids(key)
        assert len(ids) == 1
        permission_id: Final = string_value(ids[0]["object_permission_id"])
        scenario.cleanups.callback(_clear_key_object_permission, key, permission_id)
        before: Final = (
            _object_permission_rows(permission_id),
            _object_permission_table_rows(),
            _key_rows(key),
            _key_state_rows(key),
            _key_permission_and_budget_ids(key),
            _deleted_key_rows(key),
            _deprecated_key_rows(key),
        )
        response: Final = ownership_gateway.request(
            "POST",
            "/key/update",
            {
                "key": key,
                "team_id": team_b,
                "object_permission": {"mcp_tool_permissions": {identifier: ["tool"]}},
            },
        )
        assert response.status_code == 400, response.text
        assert "ambiguous" in response.text.lower(), response.text
        assert "project" not in response.text.lower(), response.text
        assert (
            _object_permission_rows(permission_id),
            _object_permission_table_rows(),
            _key_rows(key),
            _key_state_rows(key),
            _key_permission_and_budget_ids(key),
            _deleted_key_rows(key),
            _deprecated_key_rows(key),
        ) == before


def test_team_key_bulk_update_rejects_foreign_team_project(ownership_gateway: Gateway) -> None:
    with ownership_gateway.scenario() as scenario:
        model: Final = scenario.model()
        team_a: Final = scenario.team(models=[model])
        team_b: Final = scenario.team(models=[model])
        project_b: Final = scenario.project(team_b, models=[model])
        key: Final = scenario.key(team_id=team_a, models=[model])
        before: Final = _key_rows(key)
        response: Final = ownership_gateway.request(
            "POST",
            "/team/key/bulk_update",
            {
                "team_id": team_a,
                "key_ids": [sha256(key.encode()).hexdigest()],
                "update_fields": {"project_id": project_b, "team_id": team_b},
            },
        )
        assert response.status_code == 422, response.text
        assert "project_id" in response.text, response.text
        assert "team_id" in response.text, response.text
        assert _key_rows(key) == before


def test_bulk_update_and_regenerate_new_object_permission_is_served(ownership_gateway: Gateway) -> None:
    with wire_server(_ownership_upstream) as upstream:
        with ownership_gateway.scenario() as scenario:
            model: Final = scenario.model(api_base=f"{upstream.url}/v1")
            team: Final = scenario.team(models=[model])
            project: Final = scenario.project(team, models=[model])
            bulk_key: Final = scenario.key(team_id=team, project_id=project, models=[model])
            bulk_response: Final = ownership_gateway.request(
                "POST",
                "/key/bulk_update",
                {
                    "keys": [
                        {
                            "key": bulk_key,
                            "object_permission": {"vector_stores": ["bulk"]},
                        }
                    ],
                },
            )
            assert bulk_response.status_code == 200, bulk_response.text
            bulk_info: Final = ownership_gateway.request("GET", "/key/info", params={"key": bulk_key})
            assert bulk_info.status_code == 200, bulk_info.text
            bulk_row: Final = object_value(JSON_OBJECT.validate_json(bulk_info.content)["info"])
            bulk_permission_id: Final = string_value(bulk_row["object_permission_id"])
            assert bulk_permission_id != ""
            assert len(_object_permission_rows(bulk_permission_id)) == 1
            scenario.cleanups.callback(_clear_key_object_permission, bulk_key, bulk_permission_id)
            bulk_chat: Final = ownership_gateway.chat(model, key=bulk_key, text=f"ownership-{uuid4().hex}")
            assert string_value(bulk_chat["id"]) != ""
            regenerated: Final = ownership_gateway.request(
                "POST",
                "/key/generate",
                {"team_id": team, "project_id": project, "models": [model]},
            )
            assert regenerated.status_code == 200, regenerated.text
            regenerated_key: Final = string_value(JSON_OBJECT.validate_json(regenerated.content)["key"])
            scenario.cleanups.callback(delete_key_if_present, ownership_gateway, regenerated_key)
            regeneration: Final = ownership_gateway.request(
                "POST",
                f"/key/{regenerated_key}/regenerate",
                {"object_permission": {"vector_stores": ["regenerated"]}},
            )
            assert regeneration.status_code == 200, regeneration.text
            new_key: Final = string_value(JSON_OBJECT.validate_json(regeneration.content)["key"])
            scenario.cleanups.callback(delete_key_if_present, ownership_gateway, new_key)
            info: Final = ownership_gateway.request("GET", "/key/info", params={"key": new_key})
            assert info.status_code == 200, info.text
            info_row: Final = object_value(JSON_OBJECT.validate_json(info.content)["info"])
            regenerated_permission_id: Final = string_value(info_row["object_permission_id"])
            assert regenerated_permission_id != ""
            assert regenerated_permission_id != bulk_permission_id
            assert len(_object_permission_rows(regenerated_permission_id)) == 1
            scenario.cleanups.callback(_clear_key_object_permission, new_key, regenerated_permission_id)
            chat: Final = ownership_gateway.chat(model, key=new_key, text=f"ownership-{uuid4().hex}")
            assert string_value(chat["id"]) != ""


def test_ownership_rejections_during_concurrent_traffic_burst(ownership_gateway: Gateway) -> None:
    with wire_server(_ownership_upstream) as upstream:
        with ownership_gateway.scenario() as scenario:
            model: Final = scenario.model(api_base=f"{upstream.url}/v1")
            team_a: Final = scenario.team(models=[model])
            team_b: Final = scenario.team(models=[model])
            project_a: Final = scenario.project(team_a, models=[model])
            serving_project: Final = scenario.project(team_a, models=[model])
            key: Final = scenario.key(team_id=team_a, project_id=project_a, models=[model])
            serving_key: Final = scenario.key(team_id=team_a, project_id=serving_project, models=[model])
            markers: Final = tuple(f"ownership-{uuid4().hex}" for _ in range(30))
            key_before: Final = (
                _object_permission_table_rows(),
                _key_rows(key),
                _key_state_rows(key),
                _key_permission_and_budget_ids(key),
                _deleted_key_rows(key),
                _deprecated_key_rows(key),
            )
            serving_key_before: Final = (
                _key_rows(serving_key),
                _key_permission_and_budget_ids(serving_key),
                _deleted_key_rows(serving_key),
                _deprecated_key_rows(serving_key),
            )
            project_before: Final = _project_state_rows(project_a)

            def serve(index: int) -> tuple[int, str, str]:
                return _ownership_serving_call(ownership_gateway, serving_key, model, index, markers[index])

            def update_key() -> httpx.Response:
                return ownership_gateway.request("POST", "/key/update", {"key": key, "team_id": team_b})

            def bulk_update() -> httpx.Response:
                return ownership_gateway.request(
                    "POST",
                    "/key/bulk_update",
                    {"keys": [{"key": key, "team_id": team_b}]},
                )

            def move_project() -> httpx.Response:
                return ownership_gateway.request(
                    "POST", "/project/update", {"project_id": project_a, "team_id": team_b}
                )

            with ThreadPoolExecutor(max_workers=33) as pool:
                serving_futures: Final = tuple(pool.submit(serve, index) for index in range(30))
                ownership_futures: Final = (
                    pool.submit(update_key),
                    pool.submit(bulk_update),
                    pool.submit(move_project),
                )
                serving_results: Final = tuple(future.result(timeout=90) for future in serving_futures)
            ownership_results: Final = tuple(future.result(timeout=90) for future in ownership_futures)
            assert all(status == 200 for status, _, _ in serving_results), serving_results
            assert all(response_id for _, response_id, _ in serving_results), serving_results
            assert len({response_id for _, response_id, _ in serving_results}) == 30
            key_update_response: Final = ownership_results[0]
            bulk_update_response: Final = ownership_results[1]
            project_update_response: Final = ownership_results[2]
            assert key_update_response.status_code == 400, key_update_response.text
            assert bulk_update_response.status_code == 200, bulk_update_response.text
            bulk_result: Final = JSON_OBJECT.validate_json(bulk_update_response.content)
            successful_updates: Final = bulk_result["successful_updates"]
            failed_updates: Final = bulk_result["failed_updates"]
            assert isinstance(successful_updates, list), bulk_update_response.text
            assert successful_updates == [], bulk_update_response.text
            assert isinstance(failed_updates, list), bulk_update_response.text
            assert len(failed_updates) == 1, bulk_update_response.text
            failed_update: Final = object_value(failed_updates[0])
            assert f"Project {project_a} belongs to team {team_a}" in string_value(failed_update["failed_reason"]), (
                bulk_update_response.text
            )
            assert project_update_response.status_code == 400, project_update_response.text
            upstream_requests: Final = upstream.drain()
            assert len(upstream_requests) == 30
            received_markers: Final = tuple(_ownership_request_marker(request) for request in upstream_requests)
            assert tuple(sorted(received_markers)) == tuple(sorted(markers))
            assert (
                _object_permission_table_rows(),
                _key_rows(key),
                _key_state_rows(key),
                _key_permission_and_budget_ids(key),
                _deleted_key_rows(key),
                _deprecated_key_rows(key),
            ) == key_before
            assert (
                _key_rows(serving_key),
                _key_permission_and_budget_ids(serving_key),
                _deleted_key_rows(serving_key),
                _deprecated_key_rows(serving_key),
            ) == serving_key_before
            assert _project_state_rows(project_a) == project_before


def test_ownership_checks_wait_for_row_locks_without_failing_readiness(ownership_gateway: Gateway) -> None:
    with ownership_gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(models=[model])
        project: Final = scenario.project(team, models=[model], max_budget=7)
        key: Final = scenario.key(team_id=team, project_id=project, models=[model])
        alias: Final = f"locked-{uuid4().hex}"
        locked: Final = threading.Event()
        release: Final = threading.Event()

        def hold_locks() -> None:
            with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
                connection.execute("BEGIN")
                connection.execute(
                    'SELECT project_id FROM "LiteLLM_ProjectTable" WHERE project_id = %s FOR UPDATE',
                    (project,),
                )
                connection.execute(
                    'SELECT token FROM "LiteLLM_VerificationToken" WHERE token = %s FOR UPDATE',
                    (sha256(key.encode()).hexdigest(),),
                )
                locked.set()
                assert release.wait(timeout=30)
                connection.commit()

        with ThreadPoolExecutor(max_workers=3) as pool:
            lock_future: Final = pool.submit(hold_locks)
            try:
                assert locked.wait(timeout=10)
                key_future: Final = pool.submit(
                    ownership_gateway.request,
                    "POST",
                    "/key/update",
                    {"key": key, "key_alias": alias},
                )
                project_future: Final = pool.submit(
                    ownership_gateway.request,
                    "POST",
                    "/project/update",
                    {"project_id": project, "team_id": team, "max_budget": 19},
                )
                lock_waiters: Final = eventually(
                    lambda: read_rows(
                        "SELECT pid FROM pg_stat_activity WHERE datname = current_database() "
                        "AND wait_event_type = 'Lock' AND cardinality(pg_blocking_pids(pid)) > 0",
                        (),
                    ),
                    lambda rows: len(rows) >= 2,
                    seconds=10,
                )
                assert len(lock_waiters) >= 2
                readiness: Final = eventually(
                    lambda: ownership_gateway.request("GET", "/health/readiness").status_code,
                    lambda status: status == 200,
                    seconds=10,
                )
                assert readiness == 200
            finally:
                release.set()
            key_response: Final = key_future.result(timeout=90)
            project_response: Final = project_future.result(timeout=90)
            lock_future.result(timeout=30)
        assert key_response.status_code == 200, key_response.text
        assert project_response.status_code == 200, project_response.text
        assert _key_rows(key)[0]["key_alias"] == alias
        assert _project_rows(project)[0]["max_budget"] == 19.0


def test_ownership_enforced_after_worker_kill(ownership_gateway: Gateway, tmp_path: Path) -> None:
    started: Final = threading.Event()
    release_stream: Final = threading.Event()

    def respond(request: Request) -> Reply:
        started.set()
        reply: Final = _ownership_upstream(request)
        return Reply(
            status=reply.status,
            body=reply.body,
            content_type=reply.content_type,
            chunks=reply.chunks,
            gate_after_first=release_stream,
        )

    with wire_server(respond) as upstream:
        with owned_proxy_process(
            ownership_gateway,
            tmp_path / "worker-kill",
            {"LITELLM_SALT_KEY": _OWNED_PROXY_SALT_KEY},
            workers=2,
        ) as owned:
            with owned.gateway.scenario() as scenario:
                model: Final = scenario.model(api_base=f"{upstream.url}/v1")
                team_a: Final = scenario.team(models=[model])
                team_b: Final = scenario.team(models=[model])
                project_a: Final = scenario.project(team_a, models=[model])
                project_b: Final = scenario.project(team_b, models=[model])
                key: Final = scenario.key(team_id=team_a, project_id=project_a, models=[model])
                markers: Final = tuple(f"ownership-{uuid4().hex}" for _ in range(24))

                def serve(index: int) -> tuple[int, int, str]:
                    try:
                        response: Final = owned.gateway.request(
                            "POST",
                            "/v1/chat/completions",
                            {
                                "model": model,
                                "messages": [{"role": "user", "content": markers[index]}],
                                "stream": True,
                            },
                            key=key,
                        )
                    except httpx.TransportError as error:
                        return index, 0, str(error)
                    return index, response.status_code, response.text

                candidate_port: Final = owned.gateway.client.base_url.port
                assert candidate_port is not None
                workers: Final = tuple(
                    process
                    for process in group_members(owned.process.pid)
                    if process.pid != owned.process.pid
                    and any(
                        connection.laddr.port == candidate_port and connection.status == psutil.CONN_LISTEN
                        for connection in process.net_connections(kind="inet")
                    )
                )
                assert len(workers) == 2, tuple(process.pid for process in workers)
                victim: Final = workers[0]
                survivor: Final = workers[1]
                with ThreadPoolExecutor(max_workers=24) as pool:
                    futures: Final = tuple(pool.submit(serve, index) for index in range(24))
                    try:
                        assert started.wait(timeout=10)
                        os.kill(victim.pid, signal.SIGKILL)
                        release_stream.set()
                        psutil.wait_procs((victim,), timeout=10)
                        assert not psutil.pid_exists(victim.pid), victim.pid
                    finally:
                        release_stream.set()
                    burst: Final = tuple(future.result(timeout=90) for future in futures)
                    assert psutil.pid_exists(survivor.pid), survivor.pid
                assert len(burst) == 24
                burst_requests: Final = upstream.drain()
                empty_body_count: Final = sum(not request.body for request in burst_requests)
                burst_markers: Final = tuple(
                    _ownership_request_marker(request) for request in burst_requests if request.body
                )
                successful_markers: Final = tuple(markers[index] for index, status, _ in burst if status == 200)
                assert burst_markers, f"empty_body_captures={empty_body_count}; burst={burst}"
                assert all(burst_markers.count(marker) == 1 for marker in successful_markers), (
                    f"empty_body_captures={empty_body_count}; "
                    f"successful_markers={successful_markers}; upstream_markers={burst_markers}; burst={burst}"
                )
                cross_team: Final = owned.gateway.request(
                    "POST",
                    "/key/generate",
                    {"team_id": team_a, "project_id": project_b, "models": [model]},
                )
                _discard_unexpected_key(owned.gateway, cross_team)
                assert cross_team.status_code == 400, cross_team.text
                marker: Final = f"ownership-{uuid4().hex}"
                chat: Final = owned.gateway.request(
                    "POST",
                    "/v1/chat/completions",
                    {"model": model, "messages": [{"role": "user", "content": marker}]},
                    key=key,
                )
                assert chat.status_code == 200, chat.text
                post_kill_requests: Final = upstream.drain()
                post_kill_empty_body_count: Final = sum(not request.body for request in post_kill_requests)
                post_kill_markers: Final = tuple(
                    _ownership_request_marker(request) for request in post_kill_requests if request.body
                )
                assert post_kill_markers.count(marker) == 1, (
                    f"empty_body_captures={post_kill_empty_body_count}; "
                    f"post_kill_markers={post_kill_markers}; response={chat.text}"
                )


@pytest.fixture(scope="module")
def ownership_gateway(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    with gateway_from_environment() as gateway:
        with owned_proxy(
            gateway,
            tmp_path_factory.mktemp("project-team-ownership"),
            {"LITELLM_SALT_KEY": _OWNED_PROXY_SALT_KEY},
            workers=2,
        ) as candidate:
            yield candidate


@pytest.mark.covers("mgmt.project.new.real_route_persists")
def test_project_new_persists_real_state(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(models=[model])
        budget: Final = scenario.budget(max_budget=7)
        project: Final = scenario.project(
            team, project_alias="new-project", budget_id=budget, models=[model], description="new project"
        )
        key: Final = scenario.key(team_id=team, project_id=project, models=[model])
        assert object_value(gateway.chat(model, key=key)["usage"])["total_tokens"] == 40
        rows: Final = _project_rows(project)
        assert rows != []
        assert len(rows) == 1
        row: Final = rows[0]
        assert row["project_id"] == project
        assert row["project_alias"] == "new-project"
        assert row["team_id"] == team
        assert row["description"] == "new project"
        assert row["models"] == [model]
        assert row["budget_id"] == budget
        assert row["blocked"] is False
        assert row["max_budget"] == 7.0


@pytest.mark.covers("mgmt.project.update.real_route_persists")
def test_project_update_persists_real_state(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(models=[model])
        budget: Final = scenario.budget(max_budget=3)
        project: Final = scenario.project(team, budget_id=budget, models=[model], description="before")
        key: Final = scenario.key(team_id=team, project_id=project, models=[model])
        updated: Final = gateway.post(
            "/project/update",
            {
                "project_id": project,
                "project_alias": "updated-project",
                "description": "after",
                "max_budget": 9,
                "blocked": True,
            },
        )
        assert string_value(updated["project_id"]) == project
        rows: Final = _project_rows(project)
        assert rows != []
        assert len(rows) == 1
        row: Final = rows[0]
        assert row["project_alias"] == "updated-project"
        assert row["description"] == "after"
        assert row["team_id"] == team
        assert row["models"] == [model]
        assert row["budget_id"] == budget
        assert row["blocked"] is True
        assert row["max_budget"] == 9.0
        blocked: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "blocked project"}]},
            key=key,
        )
        assert blocked.status_code == 401, blocked.text
        assert object_value(blocked.json()["error"])["type"] == "auth_error"
        gateway.post("/project/update", {"project_id": project, "blocked": False})
        assert object_value(gateway.chat(model, key=key)["usage"])["total_tokens"] == 40


@pytest.mark.covers("mgmt.project.delete.attached_key_refusal_preserves_state")
def test_project_delete_with_attached_key_refuses_and_preserves_state(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(models=[model])
        budget: Final = scenario.budget()
        project: Final = scenario.project(
            team, budget_id=budget, project_alias="delete-project", models=[model]
        )
        key: Final = scenario.key(team_id=team, project_id=project, models=[model])
        digest: Final = sha256(key.encode()).hexdigest()
        project_before: Final = _project_rows(project)
        key_before: Final = read_rows(
            'SELECT token, key_alias, models, metadata, max_budget, team_id, project_id, budget_id '
            'FROM "LiteLLM_VerificationToken" WHERE token = %s',
            (digest,),
        )
        assert len(project_before) == 1
        assert len(key_before) == 1
        assert key_before[0]["project_id"] == project
        assert key_before[0]["team_id"] == team
        denied: Final = gateway.request("DELETE", "/project/delete", {"project_ids": [project]})
        assert denied.status_code == 400, denied.text
        assert _project_rows(project) == project_before
        assert read_rows(
            'SELECT token, key_alias, models, metadata, max_budget, team_id, project_id, budget_id '
            'FROM "LiteLLM_VerificationToken" WHERE token = %s',
            (digest,),
        ) == key_before


def test_key_generate_rejects_foreign_team_project(ownership_gateway: Gateway) -> None:
    with ownership_gateway.scenario() as scenario:
        model: Final = scenario.model()
        team_a: Final = scenario.team(models=[model])
        team_b: Final = scenario.team(models=[model])
        project_b: Final = scenario.project(team_b, models=[model])
        alias: Final = f"cross-team-{uuid4().hex}"
        vector_store: Final = f"vector-store-{uuid4().hex}"
        budget_rows_before: Final = _budget_table_rows()
        object_permission_rows_before: Final = _object_permission_table_rows()
        cross_team: Final = ownership_gateway.request(
            "POST",
            "/key/generate",
            {
                "team_id": team_a,
                "project_id": project_b,
                "key_alias": alias,
                "models": [model],
                "soft_budget": 3.5,
                "object_permission": {"vector_stores": [vector_store]},
            },
        )
        _discard_unexpected_key(ownership_gateway, cross_team)
        assert cross_team.status_code == 400, cross_team.text
        assert _budget_table_rows() == budget_rows_before
        assert _object_permission_table_rows() == object_permission_rows_before
        assert (
            read_rows(
                'SELECT token FROM "LiteLLM_VerificationToken" WHERE project_id = %s AND team_id = %s',
                (project_b, team_a),
            )
            == []
        )


def test_key_generate_rejects_missing_team_for_owned_project(ownership_gateway: Gateway) -> None:
    with ownership_gateway.scenario() as scenario:
        model: Final = scenario.model()
        team_b: Final = scenario.team(models=[model])
        project_b: Final = scenario.project(team_b, models=[model])
        unbound: Final = ownership_gateway.request(
            "POST", "/key/generate", {"project_id": project_b, "models": [model]}
        )
        _discard_unexpected_key(ownership_gateway, unbound)
        assert unbound.status_code == 400, unbound.text
        assert read_rows(
            'SELECT token FROM "LiteLLM_VerificationToken" WHERE project_id = %s AND team_id IS NULL',
            (project_b,),
        ) == []


def test_key_generate_same_team_project_key_can_chat(ownership_gateway: Gateway) -> None:
    with ownership_gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(models=[model])
        project: Final = scenario.project(team, models=[model])
        generated: Final = ownership_gateway.request(
            "POST", "/key/generate", {"team_id": team, "project_id": project, "models": [model]}
        )
        assert generated.status_code == 200, generated.text
        key: Final = string_value(JSON_OBJECT.validate_json(generated.content)["key"])
        scenario.cleanups.callback(scenario.delete_key, key)
        generated_rows: Final = _key_rows(key)
        assert len(generated_rows) == 1
        assert generated_rows[0]["team_id"] == team
        assert generated_rows[0]["project_id"] == project
        chat: Final = ownership_gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "control"}]},
            key=key,
        )
        assert chat.status_code == 200, chat.text
        assert object_value(JSON_OBJECT.validate_json(chat.content)["usage"])["total_tokens"] == 40


def test_key_update_rejects_team_change_and_allows_unchanged_values(ownership_gateway: Gateway) -> None:
    with ownership_gateway.scenario() as scenario:
        model: Final = scenario.model()
        team_a: Final = scenario.team(models=[model])
        team_b: Final = scenario.team(models=[model])
        project_a: Final = scenario.project(team_a, models=[model])
        key: Final = scenario.key(team_id=team_a, project_id=project_a, models=[model], soft_budget=3.0)
        permission_seed: Final = ownership_gateway.request(
            "POST",
            "/key/update",
            {"key": key, "object_permission": {"vector_stores": ["existing-store"]}},
        )
        assert permission_seed.status_code == 200, permission_seed.text
        key_permission_state: Final = _key_permission_and_budget_ids(key)
        assert len(key_permission_state) == 1
        permission_id: Final = string_value(key_permission_state[0]["object_permission_id"])
        budget_id: Final = string_value(key_permission_state[0]["budget_id"])
        scenario.cleanups.callback(_clear_key_object_permission, key, permission_id)
        permission_before: Final = _object_permission_rows(permission_id)
        budget_before: Final = _budget_rows(budget_id)
        assert len(permission_before) == 1
        assert len(budget_before) == 1
        aliased: Final = ownership_gateway.request("POST", "/key/update", {"key": key, "key_alias": "updated"})
        assert aliased.status_code == 200, aliased.text
        after_alias: Final = _key_rows(key)
        assert len(after_alias) == 1
        assert after_alias[0]["key_alias"] == "updated"
        assert after_alias[0]["team_id"] == team_a
        assert after_alias[0]["project_id"] == project_a
        unchanged_team: Final = ownership_gateway.request("POST", "/key/update", {"key": key, "team_id": team_a})
        assert unchanged_team.status_code == 200, unchanged_team.text
        after_unchanged_team: Final = _key_rows(key)
        assert after_unchanged_team == after_alias
        before_reassignment: Final = _key_rows(key)
        assert len(before_reassignment) == 1
        reassigned: Final = ownership_gateway.request(
            "POST",
            "/key/update",
            {
                "key": key,
                "team_id": team_b,
                "soft_budget": 5.0,
                "object_permission": {"vector_stores": ["replacement-store"]},
            },
        )
        assert reassigned.status_code == 400, reassigned.text
        assert _key_rows(key) == before_reassignment
        assert _object_permission_rows(permission_id) == permission_before
        assert _budget_rows(budget_id) == budget_before


def test_key_update_can_detach_project_and_change_team(ownership_gateway: Gateway) -> None:
    with ownership_gateway.scenario() as scenario:
        model: Final = scenario.model()
        team_a: Final = scenario.team(models=[model])
        team_b: Final = scenario.team(models=[model])
        project_a: Final = scenario.project(team_a, models=[model])
        key: Final = scenario.key(team_id=team_a, project_id=project_a, models=[model])
        detached: Final = ownership_gateway.request(
            "POST", "/key/update", {"key": key, "project_id": None, "team_id": team_b}
        )
        assert detached.status_code == 200, detached.text
        after_detach: Final = _key_rows(key)
        assert len(after_detach) == 1
        assert after_detach[0]["project_id"] is None
        assert after_detach[0]["team_id"] == team_b
        assert after_detach[0]["key_alias"] is None


def test_key_regenerate_rejects_foreign_project_without_changing_key(ownership_gateway: Gateway) -> None:
    with ownership_gateway.scenario() as scenario:
        model: Final = scenario.model()
        team_a: Final = scenario.team(models=[model])
        team_b: Final = scenario.team(models=[model])
        project_b: Final = scenario.project(team_b, models=[model])
        generated: Final = ownership_gateway.request(
            "POST", "/key/generate", {"team_id": team_a, "models": [model]}
        )
        assert generated.status_code == 200, generated.text
        key: Final = string_value(JSON_OBJECT.validate_json(generated.content)["key"])
        scenario.cleanups.callback(scenario.delete_key, key)
        before: Final = _key_rows(key)
        before_deleted: Final = _deleted_key_rows(key)
        before_deprecated: Final = _deprecated_key_rows(key)
        assert len(before) == 1
        assert before_deleted == []
        assert before_deprecated == []
        response: Final = ownership_gateway.request(
            "POST", f"/key/{key}/regenerate", {"project_id": project_b, "grace_period": "1h"}
        )
        _discard_unexpected_key(ownership_gateway, response)
        assert response.status_code == 400, response.text
        assert _key_rows(key) == before
        assert _deleted_key_rows(key) == before_deleted
        assert _deprecated_key_rows(key) == before_deprecated


def _regenerate_as(candidate: Gateway, route: str, target: str, caller: str, body: dict[str, JsonValue]) -> httpx.Response:
    if route == "body":
        return candidate.request("POST", "/key/regenerate", {"key": target, **body}, key=caller)
    return candidate.request("POST", f"/key/{target}/regenerate", body, key=caller)


@pytest.mark.parametrize("route", ["path", "body"])
def test_key_regenerate_rejects_outsider_moving_own_key_into_foreign_team_project(
    ownership_gateway: Gateway, route: str
) -> None:
    with ownership_gateway.scenario() as scenario:
        model: Final = scenario.model()
        team_b: Final = scenario.team(models=[model])
        project_b: Final = scenario.project(team_b, models=[model])
        outsider: Final = scenario.user(user_role="internal_user")
        caller: Final = scenario.key(user_id=outsider)
        generated: Final = ownership_gateway.request("POST", "/key/generate", {"user_id": outsider, "models": [model]})
        assert generated.status_code == 200, generated.text
        target: Final = string_value(JSON_OBJECT.validate_json(generated.content)["key"])
        scenario.cleanups.callback(delete_key_if_present, ownership_gateway, target)
        before: Final = _key_rows(target)
        assert len(before) == 1
        assert before[0]["team_id"] is None
        assert before[0]["project_id"] is None

        response: Final = _regenerate_as(
            ownership_gateway, route, target, caller, {"team_id": team_b, "project_id": project_b}
        )
        _discard_unexpected_key(ownership_gateway, response)

        assert response.status_code == 403, response.text
        assert _key_rows(target) == before
        assert _deleted_key_rows(target) == []
        assert read_rows('SELECT token FROM "LiteLLM_VerificationToken" WHERE project_id = %s', (project_b,)) == []
        chat: Final = ownership_gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "rejected regeneration keeps the old key"}]},
            key=target,
        )
        assert chat.status_code == 200, chat.text


@pytest.mark.parametrize("route", ["path", "body"])
def test_key_regenerate_rejects_team_admin_moving_team_key_into_foreign_team_project(
    ownership_gateway: Gateway, route: str
) -> None:
    with ownership_gateway.scenario() as scenario:
        model: Final = scenario.model()
        team_a: Final = scenario.team(models=[model])
        team_b: Final = scenario.team(models=[model])
        project_b: Final = scenario.project(team_b, models=[model])
        team_a_admin: Final = scenario.member(team_a, "admin")
        caller: Final = scenario.key(user_id=team_a_admin)
        generated: Final = ownership_gateway.request("POST", "/key/generate", {"team_id": team_a, "models": [model]})
        assert generated.status_code == 200, generated.text
        target: Final = string_value(JSON_OBJECT.validate_json(generated.content)["key"])
        scenario.cleanups.callback(delete_key_if_present, ownership_gateway, target)
        before: Final = _key_rows(target)
        assert len(before) == 1
        assert before[0]["team_id"] == team_a

        response: Final = _regenerate_as(
            ownership_gateway, route, target, caller, {"team_id": team_b, "project_id": project_b}
        )
        _discard_unexpected_key(ownership_gateway, response)

        assert response.status_code == 403, response.text
        assert _key_rows(target) == before
        assert _deleted_key_rows(target) == []
        assert read_rows('SELECT token FROM "LiteLLM_VerificationToken" WHERE project_id = %s', (project_b,)) == []
        chat: Final = ownership_gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "rejected team key regeneration keeps the key"}]},
            key=target,
        )
        assert chat.status_code == 200, chat.text


@pytest.mark.parametrize(("route", "role"), [("path", "user"), ("body", "admin")])
def test_key_regenerate_lets_project_team_member_move_own_key_into_project(
    ownership_gateway: Gateway, route: str, role: str
) -> None:
    with ownership_gateway.scenario() as scenario:
        model: Final = scenario.model()
        team_b: Final = scenario.team(models=[model])
        project_b: Final = scenario.project(team_b, models=[model])
        member: Final = scenario.member(team_b, role)
        caller: Final = scenario.key(user_id=member)
        generated: Final = ownership_gateway.request("POST", "/key/generate", {"user_id": member, "models": [model]})
        assert generated.status_code == 200, generated.text
        target: Final = string_value(JSON_OBJECT.validate_json(generated.content)["key"])
        scenario.cleanups.callback(delete_key_if_present, ownership_gateway, target)

        response: Final = _regenerate_as(
            ownership_gateway, route, target, caller, {"team_id": team_b, "project_id": project_b}
        )

        assert response.status_code == 200, response.text
        new_key: Final = string_value(JSON_OBJECT.validate_json(response.content)["key"])
        scenario.cleanups.callback(delete_key_if_present, ownership_gateway, new_key)
        rows: Final = _key_rows(new_key)
        assert len(rows) == 1
        assert rows[0]["team_id"] == team_b
        assert rows[0]["project_id"] == project_b
        assert _key_rows(target) == []
        chat: Final = ownership_gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "member regeneration serves"}]},
            key=new_key,
        )
        assert chat.status_code == 200, chat.text


def test_key_generation_rejects_missing_project_without_writing_key(ownership_gateway: Gateway) -> None:
    with ownership_gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(models=[model])
        missing_project_id: Final = f"missing-{uuid4()}"
        key_generation: Final = ownership_gateway.request(
            "POST",
            "/key/generate",
            {"team_id": team, "project_id": missing_project_id, "models": [model]},
        )

        assert key_generation.status_code == 404, key_generation.text
        assert (
            read_rows(
                'SELECT token FROM "LiteLLM_VerificationToken" WHERE project_id = %s',
                (missing_project_id,),
            )
            == []
        )

        service_account_generation: Final = ownership_gateway.request(
            "POST",
            "/key/service-account/generate",
            {"team_id": team, "project_id": missing_project_id, "models": [model]},
        )

        assert service_account_generation.status_code == 404, service_account_generation.text
        assert (
            read_rows(
                'SELECT token FROM "LiteLLM_VerificationToken" WHERE project_id = %s',
                (missing_project_id,),
            )
            == []
        )


def test_key_regenerate_routes_reject_missing_project_without_changing_key(ownership_gateway: Gateway) -> None:
    with ownership_gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(models=[model])
        key: Final = scenario.key(team_id=team, models=[model])
        missing_project_id: Final = f"missing-{uuid4()}"
        before: Final = _key_rows(key)
        responses: Final = (
            ownership_gateway.request(
                "POST",
                "/key/regenerate",
                {"key": key, "project_id": missing_project_id},
            ),
            ownership_gateway.request(
                "POST",
                f"/key/{key}/regenerate",
                {"project_id": missing_project_id},
            ),
        )

        for response in responses:
            assert response.status_code == 404, response.text
            assert _key_rows(key) == before
            assert _deleted_key_rows(key) == []
            assert _deprecated_key_rows(key) == []
            assert (
                read_rows(
                    'SELECT token FROM "LiteLLM_VerificationToken" WHERE project_id = %s',
                    (missing_project_id,),
                )
                == []
            )

        chat: Final = ownership_gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "rejected regeneration preserves key"}]},
            key=key,
        )
        assert chat.status_code == 200, chat.text


def test_key_generate_nonmember_organization_error_precedes_project_ownership(
    ownership_gateway: Gateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    with ownership_gateway.scenario() as scenario:
        model: Final = scenario.model()
        organization_id: Final = scenario.organization()
        owner_team: Final = scenario.team(models=[model])
        project: Final = scenario.project(owner_team, models=[model])
        caller_id: Final = scenario.user(user_role="internal_user")
        caller_token: Final = _cli_session_token(caller_id, None, monkeypatch=monkeypatch)
        response: Final = ownership_gateway.request(
            "POST",
            "/key/generate",
            {
                "project_id": project,
                "organization_id": organization_id,
                "models": [model],
            },
            key=caller_token,
        )

        assert response.status_code == 403, response.text
        assert f"Caller is not a member of organization_id={organization_id}" in response.text
        assert (
            read_rows(
                'SELECT token FROM "LiteLLM_VerificationToken" WHERE project_id = %s',
                (project,),
            )
            == []
        )


def test_key_generate_duplicate_alias_error_precedes_project_ownership(
    ownership_gateway: Gateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    with ownership_gateway.scenario() as scenario:
        model: Final = scenario.model()
        caller_team: Final = scenario.team(models=[model])
        owner_team: Final = scenario.team(models=[model])
        project: Final = scenario.project(owner_team, models=[model])
        key_alias: Final = f"duplicate-{uuid4()}"
        existing_key: Final = scenario.key(team_id=caller_team, key_alias=key_alias, models=[model])
        caller_id: Final = scenario.member(caller_team, role="admin")
        caller_token: Final = _cli_session_token(caller_id, caller_team, monkeypatch=monkeypatch)
        existing_alias_rows: Final = read_rows(
            'SELECT token FROM "LiteLLM_VerificationToken" WHERE key_alias = %s',
            (key_alias,),
        )
        response: Final = ownership_gateway.request(
            "POST",
            "/key/generate",
            {
                "team_id": caller_team,
                "project_id": project,
                "key_alias": key_alias,
                "models": [model],
            },
            key=caller_token,
        )

        assert response.status_code == 400, response.text
        assert f"Key with alias '{key_alias}' already exists" in response.text
        assert len(existing_alias_rows) == 1
        assert read_rows(
            'SELECT token FROM "LiteLLM_VerificationToken" WHERE key_alias = %s',
            (key_alias,),
        ) == existing_alias_rows
        assert len(_key_rows(existing_key)) == 1
        assert (
            read_rows(
                'SELECT token FROM "LiteLLM_VerificationToken" WHERE project_id = %s',
                (project,),
            )
            == []
        )


def test_key_generate_budget_ceiling_precedes_project_ownership(
    ownership_gateway: Gateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    with ownership_gateway.scenario() as scenario:
        model: Final = scenario.model()
        caller_team: Final = scenario.team(models=[model])
        owner_team: Final = scenario.team(models=[model])
        project: Final = scenario.project(owner_team, models=[model])
        caller_id: Final = scenario.member(caller_team, role="admin")
        caller_token: Final = _cli_session_token(
            caller_id, caller_team, monkeypatch=monkeypatch, max_budget=1
        )
        response: Final = ownership_gateway.request(
            "POST",
            "/key/generate",
            {
                "team_id": caller_team,
                "project_id": project,
                "max_budget": 5,
                "models": [model],
            },
            key=caller_token,
        )

        assert response.status_code == 400, response.text
        assert "max_budget (5.0) cannot exceed the caller's own max_budget (1.0)" in response.text
        assert (
            read_rows(
                'SELECT token FROM "LiteLLM_VerificationToken" WHERE project_id = %s',
                (project,),
            )
            == []
        )


def test_key_regenerate_output_estimate_admin_error_precedes_project_ownership(
    ownership_gateway: Gateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    with ownership_gateway.scenario() as scenario:
        model: Final = scenario.model()
        project_team: Final = scenario.team(models=[model])
        destination_team: Final = scenario.team(models=[model])
        project: Final = scenario.project(project_team, models=[model])
        caller_id: Final = scenario.member(project_team, role="admin")
        key: Final = scenario.key(team_id=project_team, user_id=caller_id, models=[model])
        caller_token: Final = _cli_session_token(caller_id, project_team, monkeypatch=monkeypatch)
        before: Final = _key_rows(key)
        response: Final = ownership_gateway.request(
            "POST",
            f"/key/{key}/regenerate",
            {
                "team_id": destination_team,
                "project_id": project,
                "metadata": {"default_estimated_output_tokens": 1},
            },
            key=caller_token,
        )

        assert response.status_code == 403, response.text
        assert "Only proxy admins can set" in response.text
        assert _key_rows(key) == before
        chat: Final = ownership_gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "rejected regeneration keeps key valid"}]},
            key=key,
        )
        assert chat.status_code == 200, chat.text


def test_key_bulk_update_rejects_foreign_team_project_and_preserves_key(ownership_gateway: Gateway) -> None:
    with ownership_gateway.scenario() as scenario:
        model: Final = scenario.model()
        team_a: Final = scenario.team(models=[model])
        team_b: Final = scenario.team(models=[model])
        project_a: Final = scenario.project(team_a, models=[model])
        key: Final = scenario.key(team_id=team_a, project_id=project_a, models=[model])
        before: Final = _key_rows(key)
        assert len(before) == 1
        assert before[0]["team_id"] == team_a
        assert before[0]["project_id"] == project_a

        response: Final = ownership_gateway.request(
            "POST",
            "/key/bulk_update",
            {
                "keys": [
                    {"key": key, "team_id": team_b},
                    {"key": key, "max_budget": 10, "tags": ["bulk-update"]},
                ]
            },
        )
        assert response.status_code == 200, response.text
        body: Final = JSON_OBJECT.validate_json(response.content)
        failed_updates: Final = body["failed_updates"]
        successful_updates: Final = body["successful_updates"]
        assert isinstance(failed_updates, list), response.text
        assert isinstance(successful_updates, list), response.text
        assert len(failed_updates) == 1
        assert len(successful_updates) == 1
        failed_update: Final = object_value(failed_updates[0])
        successful_update: Final = object_value(successful_updates[0])
        assert string_value(failed_update["key"]) == key
        assert f"Project {project_a} belongs to team {team_a}" in string_value(failed_update["failed_reason"])
        assert string_value(successful_update["key"]) == key

        after: Final = _key_rows(key)
        assert len(after) == 1
        assert after[0]["team_id"] == before[0]["team_id"]
        assert after[0]["project_id"] == before[0]["project_id"]


def test_project_update_rejects_moving_project_with_attached_key(ownership_gateway: Gateway) -> None:
    with ownership_gateway.scenario() as scenario:
        model: Final = scenario.model()
        team_a: Final = scenario.team(models=[model])
        team_b: Final = scenario.team(models=[model])
        budget: Final = scenario.budget(max_budget=3)
        attached_project: Final = scenario.project(team_a, budget_id=budget, models=[model])
        key: Final = scenario.key(team_id=team_a, project_id=attached_project, models=[model])
        project_before: Final = _project_rows(attached_project)
        key_before: Final = _key_rows(key)
        budget_before: Final = _budget_rows(budget)
        moved_with_key: Final = ownership_gateway.request(
            "POST",
            "/project/update",
            {
                "project_id": attached_project,
                "team_id": team_b,
                "max_budget": 11,
            },
        )
        assert moved_with_key.status_code == 400, moved_with_key.text
        assert _project_rows(attached_project) == project_before
        assert _key_rows(key) == key_before
        assert _budget_rows(budget) == budget_before


def test_project_update_allows_moving_project_without_keys(ownership_gateway: Gateway) -> None:
    with ownership_gateway.scenario() as scenario:
        model: Final = scenario.model()
        team_a: Final = scenario.team(models=[model])
        team_b: Final = scenario.team(models=[model])
        unattached_project: Final = scenario.project(team_a, models=[model])
        moved_without_key: Final = ownership_gateway.request(
            "POST", "/project/update", {"project_id": unattached_project, "team_id": team_b}
        )
        assert moved_without_key.status_code == 200, moved_without_key.text
        moved_project: Final = _project_rows(unattached_project)
        assert len(moved_project) == 1
        assert moved_project[0]["team_id"] == team_b
        assert read_rows(
            'SELECT token FROM "LiteLLM_VerificationToken" WHERE project_id = %s',
            (unattached_project,),
        ) == []


def test_project_update_rejects_moving_project_with_teamless_key(ownership_gateway: Gateway) -> None:
    with ownership_gateway.scenario() as scenario:
        model: Final = scenario.model()
        team_a: Final = scenario.team(models=[model])
        team_b: Final = scenario.team(models=[model])
        project: Final = scenario.project(team_a, models=[model])
        key: Final = scenario.key(team_id=team_a, project_id=project, models=[model])
        write_rows(
            'UPDATE "LiteLLM_VerificationToken" SET team_id = NULL WHERE token = %s',
            (sha256(key.encode()).hexdigest(),),
        )
        project_before: Final = _project_rows(project)
        key_before: Final = _key_rows(key)
        assert key_before[0]["team_id"] is None
        moved: Final = ownership_gateway.request("POST", "/project/update", {"project_id": project, "team_id": team_b})
        assert moved.status_code == 400, moved.text
        assert _project_rows(project) == project_before
        assert _key_rows(key) == key_before
