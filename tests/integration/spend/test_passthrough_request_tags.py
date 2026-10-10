import json
import uuid
from collections.abc import Callable, Mapping
from hashlib import sha256
from pathlib import Path
from typing import Final

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
