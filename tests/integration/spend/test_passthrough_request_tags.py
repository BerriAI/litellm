import json
import uuid
from collections.abc import Callable, Mapping
from email.parser import BytesParser
from email.policy import HTTP
from hashlib import sha256
from pathlib import Path
from typing import Final
from urllib.parse import parse_qsl

import pytest
from integration._support.client import Gateway, JsonValue, Scenario, eventually, object_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server
from pydantic import TypeAdapter


def _chat_reply(marker: str) -> dict[str, JsonValue]:
    return {
        "id": f"chatcmpl-{marker}",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": marker}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }


def _anthropic_reply(marker: str) -> dict[str, JsonValue]:
    return {
        "id": f"msg_{marker}",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5",
        "content": [{"type": "text", "text": marker}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 5, "output_tokens": 3},
    }


def _chat_stream_frames(marker: str) -> tuple[bytes, ...]:
    chunk: Final = {"id": f"chatcmpl-{marker}", "object": "chat.completion.chunk", "created": 1, "model": "gpt-4o-mini"}
    return (
        f"data: {json.dumps({**chunk, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': marker}}]})}\n\n".encode(),
        f"data: {json.dumps({**chunk, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 5, 'completion_tokens': 3, 'total_tokens': 8}})}\n\n".encode(),
        b"data: [DONE]\n\n",
    )


def _spend_row(digest: str, call_type: str) -> dict[str, JsonValue]:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT request_tags, metadata, team_id FROM "LiteLLM_SpendLogs" WHERE api_key=%s AND call_type=%s',
            (digest, call_type),
        ),
        lambda values: len(values) == 1,
        seconds=70,
    )
    return rows[0]


def _spend_row_tagged(tag: str) -> dict[str, JsonValue]:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT request_tags, metadata, team_id, api_key FROM "LiteLLM_SpendLogs" WHERE request_tags::text LIKE %s',
            (f'%"{tag}"%',),
        ),
        lambda values: len(values) == 1,
        seconds=70,
    )
    return rows[0]


def _policy_tags(row: Mapping[str, JsonValue]) -> list[JsonValue]:
    raw: Final = row["request_tags"]
    tags: Final = json.loads(raw) if isinstance(raw, str) else raw
    assert isinstance(tags, list), row
    return [tag for tag in tags if not (isinstance(tag, str) and tag.startswith("User-Agent: "))]


def _spend_logs_metadata(row: Mapping[str, JsonValue]) -> JsonValue:
    metadata: Final = row["metadata"]
    return object_value(json.loads(metadata) if isinstance(metadata, str) else metadata).get("spend_logs_metadata")


def _tagged_key(scenario: Scenario, marker: str, **fields: JsonValue) -> tuple[str, str]:
    team: Final = scenario.team(metadata={"tags": [f"team-{marker}"], "spend_logs_metadata": {"team_field": marker}})
    project: Final = scenario.project(team, metadata={"tags": [f"project-{marker}"]})
    key: Final = scenario.key(
        team_id=team,
        project_id=project,
        metadata={"tags": [f"key-{marker}"], "spend_logs_metadata": {"cost_center": marker}},
        **fields,
    )
    return key, sha256(key.encode()).hexdigest()


def _digest(key: str) -> str:
    return sha256(key.encode()).hexdigest()


def _configured_passthrough(gateway: Gateway, scenario: Scenario, marker: str, target: str, *, auth: bool) -> str:
    path: Final = f"/integration-passthrough-{marker}"
    created: Final = gateway.post("/config/pass_through_endpoint", {"path": path, "target": target, "auth": auth})
    endpoints: Final = TypeAdapter(list[JsonValue]).validate_python(created["endpoints"])
    endpoint_id: Final = object_value(endpoints[0])["id"]
    scenario.cleanups.callback(
        lambda: gateway.request("DELETE", "/config/pass_through_endpoint", params={"endpoint_id": str(endpoint_id)})
    )
    return path


def _responses_reply(marker: str, stream: bool) -> Reply:
    response: Final[dict[str, JsonValue]] = {
        "id": f"resp_{marker}",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-4o-mini",
        "output": [
            {
                "id": f"msg_{marker}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": marker, "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
    }
    if not stream:
        return Reply(body=json.dumps(response).encode())
    events: Final[tuple[dict[str, JsonValue], ...]] = (
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": {**response, "status": "in_progress", "output": []},
        },
        {
            "type": "response.output_text.delta",
            "sequence_number": 1,
            "item_id": f"msg_{marker}",
            "output_index": 0,
            "content_index": 0,
            "delta": marker,
        },
        {"type": "response.completed", "sequence_number": 2, "response": response},
    )
    return Reply(
        content_type="text/event-stream",
        chunks=tuple(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events),
    )


def _echo_upstream(marker: str) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return Reply(body=json.dumps({"object": "list", "data": []}).encode())
        assert request.method == "POST", request
        body: Final = object_value(json.loads(request.body))
        assert marker in json.dumps(body), request
        if request.target == "/v1/responses":
            return _responses_reply(marker, body.get("stream") is True)
        assert body["messages"] == [{"role": "user", "content": marker}], request
        if body.get("stream") is True:
            return Reply(chunks=_chat_stream_frames(marker), content_type="text/event-stream")
        return Reply(body=json.dumps(_chat_reply(marker)).encode())

    return respond


def test_configured_passthrough_spend_row_matches_native_route_tags_and_spend_logs_metadata(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_echo_upstream(marker)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=wire.url + "/v1")
        path: Final = _configured_passthrough(gateway, scenario, marker, wire.url + "/echo", auth=True)
        key, digest = _tagged_key(scenario, marker, models=[model], allowed_passthrough_routes=[path])
        headers: Final = {"x-litellm-tags": f"caller-{marker},key-{marker}", "User-Agent": "integration-tags/1"}
        body: Final[dict[str, JsonValue]] = {"model": model, "messages": [{"role": "user", "content": marker}]}

        native: Final = gateway.request("POST", "/v1/chat/completions", body, key=key, headers=headers)
        assert native.status_code == 200, native.text
        passthrough: Final = gateway.request("POST", path, body, key=key, headers=headers)
        assert passthrough.status_code == 200, passthrough.text
        assert json.loads(passthrough.content) == _chat_reply(marker)

        native_row: Final = _spend_row(digest, "acompletion")
        passthrough_row: Final = _spend_row(digest, "pass_through_endpoint")
        expected: Final = [f"key-{marker}", f"team-{marker}", f"project-{marker}", f"caller-{marker}"]
        assert _policy_tags(native_row) == expected, native_row
        assert _policy_tags(passthrough_row) == expected, passthrough_row
        assert _spend_logs_metadata(native_row) == {"cost_center": marker, "team_field": marker}, native_row
        assert _spend_logs_metadata(passthrough_row) == {"cost_center": marker, "team_field": marker}, passthrough_row


@pytest.mark.parametrize("bucket", ["metadata", "litellm_metadata"])
def test_configured_passthrough_body_tags_lead_and_body_spend_logs_metadata_wins_over_key_and_team(
    gateway: Gateway, bucket: str
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_echo_upstream(marker)) as wire, gateway.scenario() as scenario:
        path: Final = _configured_passthrough(gateway, scenario, marker, wire.url + "/echo", auth=True)
        key, digest = _tagged_key(scenario, marker, allowed_passthrough_routes=[path])
        body: Final[dict[str, JsonValue]] = {
            "messages": [{"role": "user", "content": marker}],
            bucket: {
                "tags": [f"body-{marker}", f"team-{marker}"],
                "spend_logs_metadata": {"cost_center": f"body-{marker}"},
            },
        }
        response: Final = gateway.request("POST", path, body, key=key)
        assert response.status_code == 200, response.text
        row: Final = _spend_row(digest, "pass_through_endpoint")
        assert _policy_tags(row) == [f"body-{marker}", f"team-{marker}", f"key-{marker}", f"project-{marker}"], row
        assert _spend_logs_metadata(row) == {"cost_center": f"body-{marker}", "team_field": marker}, row


def test_configured_passthrough_streaming_upstream_row_carries_key_team_project_and_caller_tags(
    gateway: Gateway,
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_echo_upstream(marker)) as wire, gateway.scenario() as scenario:
        path: Final = _configured_passthrough(gateway, scenario, marker, wire.url + "/echo", auth=True)
        key, digest = _tagged_key(scenario, marker, allowed_passthrough_routes=[path])
        response: Final = gateway.request(
            "POST",
            path,
            {"stream": True, "messages": [{"role": "user", "content": marker}]},
            key=key,
            headers={"x-litellm-tags": f"caller-{marker}"},
        )
        assert response.status_code == 200, response.text
        assert response.content == b"".join(_chat_stream_frames(marker)), response.text
        row: Final = _spend_row(digest, "pass_through_endpoint")
        assert _policy_tags(row) == [f"key-{marker}", f"team-{marker}", f"project-{marker}", f"caller-{marker}"], row
        assert _spend_logs_metadata(row) == {"cost_center": marker, "team_field": marker}, row


def test_configured_passthrough_key_outside_any_team_carries_its_own_tags_and_spend_logs_metadata(
    gateway: Gateway,
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_echo_upstream(marker)) as wire, gateway.scenario() as scenario:
        path: Final = _configured_passthrough(gateway, scenario, marker, wire.url + "/echo", auth=True)
        key: Final = scenario.key(
            allowed_passthrough_routes=[path],
            metadata={"tags": [f"key-{marker}"], "spend_logs_metadata": {"cost_center": marker}},
        )
        response: Final = gateway.request(
            "POST",
            path,
            {"messages": [{"role": "user", "content": marker}]},
            key=key,
            headers={"x-litellm-tags": f"caller-{marker}"},
        )
        assert response.status_code == 200, response.text
        row: Final = _spend_row(_digest(key), "pass_through_endpoint")
        assert _policy_tags(row) == [f"key-{marker}", f"caller-{marker}"], row
        assert _spend_logs_metadata(row) == {"cost_center": marker}, row


def test_configured_passthrough_untagged_key_row_keeps_only_caller_tag_and_no_spend_logs_metadata(
    gateway: Gateway,
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_echo_upstream(marker)) as wire, gateway.scenario() as scenario:
        path: Final = _configured_passthrough(gateway, scenario, marker, wire.url + "/echo", auth=True)
        team: Final = scenario.team()
        key: Final = scenario.key(team_id=team, allowed_passthrough_routes=[path])
        response: Final = gateway.request(
            "POST",
            path,
            {"messages": [{"role": "user", "content": marker}]},
            key=key,
            headers={"x-litellm-tags": f"caller-{marker}"},
        )
        assert response.status_code == 200, response.text
        row: Final = _spend_row(_digest(key), "pass_through_endpoint")
        assert _policy_tags(row) == [f"caller-{marker}"], row
        assert _spend_logs_metadata(row) is None, row
        assert row["team_id"] == team, row


def test_open_passthrough_without_auth_row_carries_only_caller_tag(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_echo_upstream(marker)) as wire, gateway.scenario() as scenario:
        path: Final = _configured_passthrough(gateway, scenario, marker, wire.url + "/echo", auth=False)
        response: Final = gateway.client.post(
            path,
            json={"messages": [{"role": "user", "content": marker}]},
            headers={"x-litellm-tags": f"caller-{marker}"},
        )
        assert response.status_code == 200, response.text
        row: Final = _spend_row_tagged(f"caller-{marker}")
        assert _policy_tags(row) == [f"caller-{marker}"], row
        assert _spend_logs_metadata(row) is None, row
        assert row["api_key"] == "", row


@pytest.mark.parametrize(
    ("metadata", "leading_tags"),
    [
        ({"tags": "string-not-list"}, []),
        ({"tags": [1, None, "z"]}, [1, None, "z"]),
        ({"spend_logs_metadata": "string-not-object"}, []),
    ],
)
def test_configured_passthrough_hostile_body_metadata_shapes_still_carry_key_team_project_tags(
    gateway: Gateway, metadata: JsonValue, leading_tags: list[JsonValue]
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_echo_upstream(marker)) as wire, gateway.scenario() as scenario:
        path: Final = _configured_passthrough(gateway, scenario, marker, wire.url + "/echo", auth=True)
        key, digest = _tagged_key(scenario, marker, allowed_passthrough_routes=[path])
        response: Final = gateway.request(
            "POST", path, {"messages": [{"role": "user", "content": marker}], "metadata": metadata}, key=key
        )
        assert response.status_code == 200, response.text
        row: Final = _spend_row(digest, "pass_through_endpoint")
        assert _policy_tags(row) == [*leading_tags, f"key-{marker}", f"team-{marker}", f"project-{marker}"], row
        assert _spend_logs_metadata(row) == {"cost_center": marker, "team_field": marker}, row


def test_configured_passthrough_body_cannot_forge_user_api_key_attribution_fields(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_echo_upstream(marker)) as wire, gateway.scenario() as scenario:
        path: Final = _configured_passthrough(gateway, scenario, marker, wire.url + "/echo", auth=True)
        forged_team: Final = scenario.team()
        key, digest = _tagged_key(scenario, marker, allowed_passthrough_routes=[path])
        forged: Final[dict[str, JsonValue]] = {
            "user_api_key": "forged-" + marker,
            "user_api_key_team_id": forged_team,
            "user_api_key_user_id": "forged-" + marker,
            "user_api_key_alias": "forged-" + marker,
        }
        body: Final[dict[str, JsonValue]] = {"messages": [{"role": "user", "content": marker}], "metadata": forged}
        response: Final = gateway.request("POST", path, body, key=key)
        assert response.status_code == 200, response.text
        row: Final = _spend_row(digest, "pass_through_endpoint")
        assert row["team_id"] != forged_team, row
        assert _policy_tags(row) == [f"key-{marker}", f"team-{marker}", f"project-{marker}"], row
        assert read_rows('SELECT api_key FROM "LiteLLM_SpendLogs" WHERE team_id=%s', (forged_team,)) == [], forged_team


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("route", ["/v1/chat/completions", "/v1/messages"])
def test_native_routes_carry_key_team_project_and_caller_tags_and_key_over_team_spend_logs_metadata(
    gateway: Gateway, route: str, stream: bool
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_echo_upstream(marker)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=wire.url + "/v1")
        key, digest = _tagged_key(scenario, marker, models=[model])
        body: Final[dict[str, JsonValue]] = {
            "model": model,
            "max_tokens": 16,
            "stream": stream,
            "messages": [{"role": "user", "content": marker}],
        }
        response: Final = gateway.request(
            "POST",
            route,
            body,
            key=key,
            headers={"x-litellm-tags": f"caller-{marker}", "User-Agent": "integration-tags/1"},
        )
        assert response.status_code == 200, response.text
        rows: Final = eventually(
            lambda: read_rows('SELECT request_tags, metadata FROM "LiteLLM_SpendLogs" WHERE api_key=%s', (digest,)),
            lambda values: len(values) == 1,
            seconds=70,
        )
        assert _policy_tags(rows[0]) == [f"key-{marker}", f"team-{marker}", f"project-{marker}", f"caller-{marker}"], (
            rows
        )
        assert _spend_logs_metadata(rows[0]) == {"cost_center": marker, "team_field": marker}, rows


def test_anthropic_passthrough_spend_row_carries_key_team_project_tags_and_spend_logs_metadata(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/messages", request
        assert request.headers["x-api-key"] == "synthetic-anthropic-key"
        return Reply(body=json.dumps(_anthropic_reply(marker)).encode())

    config: Final = tmp_path / "proxy_config.yaml"
    config.write_text(
        "model_list: []\n"
        "general_settings:\n"
        "  master_key: os.environ/LITELLM_MASTER_KEY\n"
        "  database_url: os.environ/DATABASE_URL\n"
        "  store_model_in_db: true\n"
        "  disable_spend_logs: false\n"
        "  proxy_batch_write_at: 1\n"
        "router_settings:\n"
        "  disable_cooldowns: true\n"
    )
    with wire_server(respond) as wire:
        overrides: Final = {"ANTHROPIC_API_BASE": wire.url, "ANTHROPIC_API_KEY": "synthetic-anthropic-key"}
        with owned_proxy(gateway, tmp_path, overrides, config=config) as candidate, candidate.scenario() as scenario:
            key, digest = _tagged_key(scenario, marker)
            response: Final = candidate.request(
                "POST",
                "/anthropic/v1/messages",
                {
                    "model": "claude-sonnet-4-5",
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": marker}],
                },
                key=key,
                headers={"x-litellm-tags": f"caller-{marker}", "User-Agent": "integration-tags/1"},
            )
            assert response.status_code == 200, response.text
            assert json.loads(response.content) == _anthropic_reply(marker)
            row: Final = _spend_row(digest, "pass_through_endpoint")
            assert _policy_tags(row) == [f"key-{marker}", f"team-{marker}", f"project-{marker}", f"caller-{marker}"], (
                row
            )
            assert _spend_logs_metadata(row) == {"cost_center": marker, "team_field": marker}, row


_CONTRACT_TOKEN_ENV: Final = "INTEGRATION_PASSTHROUGH_CONTRACT_TOKEN"
_CONTRACT_TOKEN: Final = "scripted-configured-endpoint-token"
_ENDPOINTS: Final = TypeAdapter(list[dict[str, JsonValue]])


def _received_reply(request: Request) -> Reply:
    return Reply(body=json.dumps({"received": request.target}).encode())


def _create_endpoint(gateway: Gateway, scenario: Scenario, body: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    response: Final = gateway.request("POST", "/config/pass_through_endpoint", dict(body))
    assert response.status_code == 200, response.text
    created: Final = _ENDPOINTS.validate_python(response.json()["endpoints"])
    assert len(created) == 1, response.text
    endpoint_id: Final = str(created[0]["id"])
    scenario.cleanups.callback(
        lambda: gateway.request("DELETE", "/config/pass_through_endpoint", params={"endpoint_id": endpoint_id})
    )
    return created[0]


def _readback(gateway: Gateway, endpoint_id: str) -> list[dict[str, JsonValue]]:
    response: Final = gateway.request("GET", "/config/pass_through_endpoint", params={"endpoint_id": endpoint_id})
    assert response.status_code == 200, response.text
    return _ENDPOINTS.validate_python(response.json()["endpoints"])


def _stored_endpoint_ids() -> list[JsonValue]:
    rows: Final = read_rows('SELECT param_value FROM "LiteLLM_Config" WHERE param_name = %s', ("general_settings",))
    assert len(rows) == 1, rows
    settings: Final = object_value(rows[0]["param_value"])
    stored: Final = _ENDPOINTS.validate_python(settings.get("pass_through_endpoints") or [])
    return [endpoint["id"] for endpoint in stored]


def _stored_endpoint(endpoint_id: str) -> dict[str, JsonValue]:
    rows: Final = read_rows('SELECT param_value FROM "LiteLLM_Config" WHERE param_name = %s', ("general_settings",))
    assert len(rows) == 1, rows
    settings: Final = object_value(rows[0]["param_value"])
    stored: Final = _ENDPOINTS.validate_python(settings.get("pass_through_endpoints") or [])
    matches: Final = [endpoint for endpoint in stored if endpoint.get("id") == endpoint_id]
    assert len(matches) == 1, matches
    return matches[0]


def _query(request: Request) -> tuple[str, dict[str, str]]:
    path, _, query = request.target.partition("?")
    return path, dict(parse_qsl(query, keep_blank_values=True))


def _multipart_parts(content_type: str, body: bytes) -> list[tuple[str, str | None, bytes]]:
    parsed: Final = BytesParser(policy=HTTP).parsebytes(f"content-type: {content_type}\r\n\r\n".encode() + body)
    assert parsed.is_multipart(), content_type
    return [
        (str(part.get_param("name", header="content-disposition")), part.get_filename(), part.get_payload(decode=True))
        for part in parsed.iter_parts()
    ]


def test_configured_endpoint_forwards_json_multipart_and_query_and_an_update_keeps_omitted_fields(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = uuid.uuid4().hex
    path: Final = f"/integration-contract-{marker}"
    subpath_root: Final = f"/integration-contract-subpath-{marker}"
    with (
        wire_server(_received_reply) as wire,
        owned_proxy(gateway, tmp_path, {_CONTRACT_TOKEN_ENV: _CONTRACT_TOKEN}) as candidate,
        candidate.scenario() as scenario,
    ):
        created: Final = _create_endpoint(
            candidate,
            scenario,
            {
                "path": path,
                "target": f"{wire.url}/base",
                "headers": {"Authorization": f"Bearer os.environ/{_CONTRACT_TOKEN_ENV}", "x-static": "static-v1"},
                "methods": ["POST"],
                "default_query_params": {"api-version": "2024-01-01"},
                "auth": False,
            },
        )
        _create_endpoint(
            candidate,
            scenario,
            {"path": subpath_root, "target": f"{wire.url}/nested", "include_subpath": True, "auth": True},
        )
        endpoint_id: Final = str(created["id"])
        body: Final[dict[str, JsonValue]] = {"input": marker, "options": {"trace": marker, "stream": False}, "n": 2}

        sent_json: Final = candidate.client.post(path, params={"trace": "1"}, json=body)
        assert sent_json.status_code == 200, sent_json.text
        assert sent_json.json() == {"received": "/base?api-version=2024-01-01&trace=1"}, sent_json.text
        overridden: Final = candidate.client.post(path, params={"api-version": "2025-02-02"}, json=body)
        assert overridden.status_code == 200, overridden.text
        form: Final = candidate.client.post(
            path,
            files=[
                ("tag", (None, "first")),
                ("tag", (None, "second")),
                ("file", ("one.bin", b"\x00one", "application/octet-stream")),
                ("file", ("two.bin", b"\x00two", "application/octet-stream")),
            ],
        )
        assert form.status_code == 200, form.text
        refused_verb: Final = candidate.client.get(path)
        assert refused_verb.status_code == 405, refused_verb.text
        nested: Final = candidate.request("POST", f"{subpath_root}/sub/route", body, params={"q": "1"})
        assert nested.status_code == 200, nested.text
        before: Final = wire.drain()
        assert [(request.method, *_query(request)) for request in before] == [
            ("POST", "/base", {"api-version": "2024-01-01", "trace": "1"}),
            ("POST", "/base", {"api-version": "2025-02-02"}),
            ("POST", "/base", {"api-version": "2024-01-01"}),
            ("POST", "/nested/sub/route", {"q": "1"}),
        ], before
        for request in before[:3]:
            assert request.headers["authorization"] == f"Bearer {_CONTRACT_TOKEN}", request.headers
            assert request.headers["x-static"] == "static-v1", request.headers
        assert [json.loads(request.body) for request in (before[0], before[1], before[3])] == [body] * 3, before
        assert _multipart_parts(before[2].headers["content-type"], before[2].body) == [
            ("tag", None, b"first"),
            ("tag", None, b"second"),
            ("file", "one.bin", b"\x00one"),
            ("file", "two.bin", b"\x00two"),
        ], before[2].body

        retargeted: Final = candidate.request(
            "POST", f"/config/pass_through_endpoint/{endpoint_id}", {"path": path, "target": f"{wire.url}/v2"}
        )
        assert retargeted.status_code == 200, retargeted.text
        expected: Final = {**created, "target": f"{wire.url}/v2"}
        assert _ENDPOINTS.validate_python(retargeted.json()["endpoints"]) == [expected], retargeted.text
        retargeted_readback: Final = _readback(candidate, endpoint_id)
        assert retargeted_readback == [expected]
        assert _stored_endpoint(endpoint_id) == {
            key: value for key, value in retargeted_readback[0].items() if key != "is_from_config"
        }
        after_retarget: Final = candidate.client.post(path, json=body)
        assert after_retarget.status_code == 200, after_retarget.text
        assert after_retarget.json() == {"received": "/v2?api-version=2024-01-01"}, after_retarget.text
        retargeted_request: Final = wire.drain()
        assert [(request.method, request.target) for request in retargeted_request] == [
            ("POST", "/v2?api-version=2024-01-01")
        ], retargeted_request
        assert retargeted_request[0].headers["authorization"] == f"Bearer {_CONTRACT_TOKEN}", retargeted_request

        locked: Final = candidate.request(
            "POST",
            f"/config/pass_through_endpoint/{endpoint_id}",
            {"path": path, "target": f"{wire.url}/v2", "auth": True, "headers": {"x-static": "static-v2"}},
        )
        assert locked.status_code == 200, locked.text
        locked_expected: Final = {**expected, "auth": True, "headers": {"x-static": "static-v2"}}
        locked_readback: Final = _readback(candidate, endpoint_id)
        assert locked_readback == [locked_expected]
        assert _stored_endpoint(endpoint_id) == {
            key: value for key, value in locked_readback[0].items() if key != "is_from_config"
        }
        anonymous: Final = candidate.client.post(path, json=body)
        assert anonymous.status_code == 401, anonymous.text
        assert wire.drain() == ()
        keyed: Final = candidate.request("POST", path, body)
        assert keyed.status_code == 200, keyed.text
        locked_request: Final = wire.drain()
        assert [(request.method, request.target) for request in locked_request] == [
            ("POST", "/v2?api-version=2024-01-01")
        ], locked_request
        assert locked_request[0].headers["x-static"] == "static-v2", locked_request[0].headers
        assert "authorization" not in locked_request[0].headers, locked_request[0].headers


def test_renaming_a_subpath_endpoint_moves_both_routes_keeps_stored_fields_and_never_forwards_caller_headers(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = uuid.uuid4().hex
    path: Final = f"/integration-rename-{marker}"
    renamed_path: Final = f"/integration-renamed-{marker}"
    body: Final[dict[str, JsonValue]] = {"input": marker}
    with (
        wire_server(_received_reply) as wire,
        owned_proxy(gateway, tmp_path, {}, workers=1) as candidate,
        candidate.scenario() as scenario,
    ):
        created: Final = _create_endpoint(
            candidate,
            scenario,
            {
                "path": path,
                "target": f"{wire.url}/base",
                "include_subpath": True,
                "auth": True,
                "headers": {"x-static": "v1"},
            },
        )
        endpoint_id: Final = str(created["id"])
        expected: Final = {
            "id": endpoint_id,
            "path": path,
            "target": f"{wire.url}/base",
            "headers": {"x-static": "v1"},
            "default_query_params": {},
            "include_subpath": True,
            "cost_per_request": 0.0,
            "timeout": None,
            "auth": True,
            "guardrails": None,
            "is_from_config": False,
            "methods": None,
        }
        assert created == expected, created
        assert _readback(candidate, endpoint_id) == [expected]
        key: Final = scenario.key(allowed_passthrough_routes=[path, renamed_path])
        caller_headers: Final = {"x-caller": f"caller-{marker}"}
        first: Final = candidate.request("POST", path, body, key=key, headers=caller_headers)
        assert first.status_code == 200, first.text
        assert first.json() == {"received": "/base"}, first.text
        second: Final = candidate.request("POST", f"{path}/sub", body, key=key, headers=caller_headers)
        assert second.status_code == 200, second.text
        assert second.json() == {"received": "/base/sub"}, second.text
        initial_requests: Final = wire.drain()
        assert [(request.method, request.target) for request in initial_requests] == [
            ("POST", "/base"),
            ("POST", "/base/sub"),
        ], initial_requests

        updated: Final = candidate.request(
            "POST",
            f"/config/pass_through_endpoint/{endpoint_id}",
            {"path": renamed_path, "target": f"{wire.url}/base"},
        )
        assert updated.status_code == 200, updated.text
        updated_expected: Final = {**expected, "path": renamed_path}
        assert _ENDPOINTS.validate_python(updated.json()["endpoints"]) == [updated_expected], updated.text
        updated_readback: Final = _readback(candidate, endpoint_id)
        assert updated_readback == [updated_expected]
        assert _stored_endpoint(endpoint_id) == {
            key: value for key, value in updated_readback[0].items() if key != "is_from_config"
        }

        old_path: Final = candidate.request("POST", path, body, key=key, headers=caller_headers)
        old_subpath: Final = candidate.request("POST", f"{path}/sub", body, key=key, headers=caller_headers)
        assert old_path.status_code == 404, old_path.text
        assert old_subpath.status_code == 404, old_subpath.text
        assert wire.drain() == ()

        renamed: Final = candidate.request("POST", renamed_path, body, key=key, headers=caller_headers)
        assert renamed.status_code == 200, renamed.text
        assert renamed.json() == {"received": "/base"}, renamed.text
        renamed_subpath: Final = candidate.request("POST", f"{renamed_path}/sub", body, key=key, headers=caller_headers)
        assert renamed_subpath.status_code == 200, renamed_subpath.text
        assert renamed_subpath.json() == {"received": "/base/sub"}, renamed_subpath.text
        anonymous: Final = candidate.client.post(f"{renamed_path}/sub", json=body, headers=caller_headers)
        assert anonymous.status_code == 401, anonymous.text
        renamed_requests: Final = wire.drain()
        assert [(request.method, request.target) for request in (*initial_requests, *renamed_requests)] == [
            ("POST", "/base"),
            ("POST", "/base/sub"),
            ("POST", "/base"),
            ("POST", "/base/sub"),
        ], renamed_requests
        for upstream in (*initial_requests, *renamed_requests):
            assert upstream.headers["x-static"] == "v1", upstream.headers
            assert "x-caller" not in upstream.headers, upstream.headers
            assert {name: value for name, value in upstream.headers.items() if key in value} == {}, upstream.headers
            assert key not in upstream.target, upstream.target
        assert wire.drain() == ()


def test_configured_endpoint_without_auth_serves_its_subpaths_without_a_key(gateway: Gateway) -> None:
    pytest.skip("BUG: an auth=false include_subpath pass-through endpoint answers 401 on every subpath request")
    marker: Final = uuid.uuid4().hex
    path: Final = f"/integration-open-subpath-{marker}"
    with wire_server(_received_reply) as wire, gateway.scenario() as scenario:
        _create_endpoint(
            gateway, scenario, {"path": path, "target": f"{wire.url}/open", "include_subpath": True, "auth": False}
        )
        response: Final = gateway.client.post(f"{path}/sub", json={"input": marker})
        assert response.status_code == 200, response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/open/sub")]


def test_deleted_configured_endpoint_leaves_readback_storage_and_runtime_and_team_get_filters(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = uuid.uuid4().hex
    with (
        wire_server(_received_reply) as wire,
        owned_proxy(gateway, tmp_path, {}) as candidate,
        candidate.scenario() as scenario,
    ):
        allowed: Final = _create_endpoint(
            candidate, scenario, {"path": f"/integration-allowed-{marker}", "target": f"{wire.url}/allowed"}
        )
        hidden: Final = _create_endpoint(
            candidate, scenario, {"path": f"/integration-hidden-{marker}", "target": f"{wire.url}/hidden"}
        )
        team: Final = scenario.team(metadata={"allowed_passthrough_routes": [allowed["path"]]})
        team_view: Final = candidate.request("GET", f"/config/pass_through_endpoint/team/{team}")
        assert team_view.status_code == 200, team_view.text
        assert [endpoint["path"] for endpoint in _ENDPOINTS.validate_python(team_view.json()["endpoints"])] == [
            allowed["path"]
        ], team_view.text

        served: Final = candidate.request("POST", str(hidden["path"]), {"input": marker})
        assert served.status_code == 200, served.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/hidden")]

        deleted: Final = candidate.request(
            "DELETE", "/config/pass_through_endpoint", params={"endpoint_id": str(hidden["id"])}
        )
        assert deleted.status_code == 200, deleted.text
        assert _ENDPOINTS.validate_python(deleted.json()["endpoints"]) == [hidden], deleted.text
        assert _readback(candidate, str(hidden["id"])) == []
        listed: Final = candidate.request("GET", "/config/pass_through_endpoint")
        assert listed.status_code == 200, listed.text
        listed_ids: Final = [endpoint["id"] for endpoint in _ENDPOINTS.validate_python(listed.json()["endpoints"])]
        assert hidden["id"] not in listed_ids and allowed["id"] in listed_ids, listed.text
        stored_ids: Final = _stored_endpoint_ids()
        assert hidden["id"] not in stored_ids and allowed["id"] in stored_ids, stored_ids

        refused: Final = candidate.request("POST", str(hidden["path"]), {"input": marker})
        assert refused.status_code == 404, refused.text
        assert wire.drain() == ()
