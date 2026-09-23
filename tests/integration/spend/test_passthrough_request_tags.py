import json
import uuid
from hashlib import sha256
from pathlib import Path
from typing import Final

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


def _spend_row(digest: str, call_type: str) -> dict[str, object]:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT request_tags, metadata FROM "LiteLLM_SpendLogs" WHERE api_key=%s AND call_type=%s',
            (digest, call_type),
        ),
        lambda values: len(values) == 1,
        seconds=70,
    )
    return rows[0]


def _policy_tags(row: dict[str, object]) -> list[str]:
    raw: Final = row["request_tags"]
    tags: Final = json.loads(raw) if isinstance(raw, str) else raw
    assert isinstance(tags, list), row
    return [tag for tag in tags if not tag.startswith("User-Agent: ")]


def _spend_logs_metadata(row: dict[str, object]) -> object:
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


def test_configured_passthrough_spend_row_matches_native_route_tags_and_spend_logs_metadata(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return Reply(body=json.dumps({"object": "list", "data": []}).encode())
        assert request.method == "POST", request
        assert request.target in {"/v1/chat/completions", "/echo"}, request.target
        assert json.loads(request.body)["messages"] == [{"role": "user", "content": marker}], request
        return Reply(body=json.dumps(_chat_reply(marker)).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=wire.url + "/v1")
        path: Final = f"/integration-passthrough-{marker}"
        key, digest = _tagged_key(scenario, marker, models=[model], allowed_passthrough_routes=[path])
        created: Final = gateway.post(
            "/config/pass_through_endpoint", {"path": path, "target": wire.url + "/echo", "auth": True}
        )
        endpoints: Final = TypeAdapter(list[JsonValue]).validate_python(created["endpoints"])
        endpoint_id: Final = object_value(endpoints[0])["id"]
        scenario.cleanups.callback(
            lambda: gateway.request("DELETE", "/config/pass_through_endpoint", params={"endpoint_id": str(endpoint_id)})
        )
        headers: Final = {"x-litellm-tags": f"caller-{marker},key-{marker}", "User-Agent": "integration-tags/1"}
        body: Final = {"model": model, "messages": [{"role": "user", "content": marker}]}

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
