import base64
import json
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, Scenario
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

_PROJECT: Final = "scripted-project"
_LOCATION: Final = "us-central1"
_CHAT_BACKEND: Final = "gemini-3.7-flash"
_EMBED_BACKEND: Final = "text-embedding-005"
_CLAUDE_BACKEND: Final = "claude-haiku-5-5"
_MODELS_PATH: Final = f"/v1/projects/{_PROJECT}/locations/{_LOCATION}/publishers/google/models"
_CHAT_TARGET: Final = f"{_MODELS_PATH}/{_CHAT_BACKEND}:generateContent"
_STREAM_TARGET: Final = f"{_MODELS_PATH}/{_CHAT_BACKEND}:streamGenerateContent?alt=sse"
_EMBED_TARGET: Final = f"{_MODELS_PATH}/{_EMBED_BACKEND}:predict"
_CLAUDE_TARGET: Final = (
    f"/v1/projects/{_PROJECT}/locations/{_LOCATION}/publishers/anthropic/models/{_CLAUDE_BACKEND}:rawPredict"
)
_LABELS_HEADER: Final = "x-vertex-ai-labels"
_REPLY_TEXT: Final = "labels ok"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_GEMINI_REPLY: Final[dict[str, JsonValue]] = {
    "candidates": [{"content": {"role": "model", "parts": [{"text": _REPLY_TEXT}]}, "finishReason": "STOP"}],
    "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2, "totalTokenCount": 5},
    "modelVersion": _CHAT_BACKEND,
}
_EMBED_REPLY: Final[dict[str, JsonValue]] = {
    "predictions": [{"embeddings": {"values": [0.5, 0.25], "statistics": {"token_count": 2}}}]
}
_CLAUDE_REPLY: Final[dict[str, JsonValue]] = {
    "id": "msg_labels",
    "type": "message",
    "role": "assistant",
    "model": _CLAUDE_BACKEND,
    "content": [{"type": "text", "text": _REPLY_TEXT}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 3, "output_tokens": 2},
}


def _vertex_peer(request: Request) -> Reply:
    if request.target == "/_oauth/token":
        return Reply(
            body=json.dumps({"access_token": "scripted-token", "token_type": "Bearer", "expires_in": 3600}).encode()
        )
    if request.target == _STREAM_TARGET:
        return Reply(content_type="text/event-stream", chunks=(f"data: {json.dumps(_GEMINI_REPLY)}\n\n".encode(),))
    if request.target == _CHAT_TARGET:
        return Reply(body=json.dumps(_GEMINI_REPLY).encode())
    if request.target == _EMBED_TARGET:
        return Reply(body=json.dumps(_EMBED_REPLY).encode())
    if request.target == _CLAUDE_TARGET:
        return Reply(body=json.dumps(_CLAUDE_REPLY).encode())
    return Reply(status=404, body=json.dumps({"error": f"unscripted target {request.target}"}).encode())


def _deployment(scenario: Scenario, wire: Wire, backend: str) -> str:
    return scenario.model(
        model=f"vertex_ai/{backend}",
        api_base=wire.url,
        api_key=None,
        vertex_project=_PROJECT,
        vertex_location=_LOCATION,
        vertex_credentials=service_account_json(_PROJECT, wire.url),
        num_retries=0,
    )


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: Gateway
    wire: Wire
    chat_model: str
    embed_model: str
    claude_model: str
    key: str
    identity: Mapping[str, str]
    uppercase_team_key: str
    uppercase_team_identity: Mapping[str, str]

    def request_sent(self, marker: str) -> Request:
        (request,) = tuple(request for request in self.wire.drain() if marker.encode() in request.body)
        return request

    def labels_sent(self, marker: str) -> JsonValue:
        return _JSON_OBJECT.validate_json(self.request_sent(marker).body).get("labels")

    def header_labels_sent(self, marker: str) -> JsonValue:
        headers: Final = self.request_sent(marker).headers
        (encoded,) = tuple(value for name, value in headers.items() if name.lower() == _LABELS_HEADER)
        return _JSON_OBJECT.validate_json(base64.b64decode(encoded))


@pytest.fixture(scope="module")
def rig(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    suffix: Final = uuid.uuid4().hex[:12]
    directory: Final = tmp_path_factory.mktemp("vertex-request-labels")
    base_config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config: Final = {
        **base_config,
        "litellm_settings": {
            **base_config.get("litellm_settings", {}),
            "vertex_request_label_fields": [
                "user_api_key_alias",
                "user_api_key_team_alias",
                "user_api_key_org_alias",
                "user_api_key_project_alias",
                "spend_logs_metadata",
            ],
        },
    }
    config_path: Final = directory / "vertex-request-labels.yaml"
    config_path.write_text(yaml.safe_dump(config))
    with (
        wire_server(_vertex_peer) as wire,
        owned_proxy(session_gateway, directory, {}, config=config_path) as proxy,
        proxy.scenario() as scenario,
    ):
        chat_model: Final = _deployment(scenario, wire, _CHAT_BACKEND)
        embed_model: Final = _deployment(scenario, wire, _EMBED_BACKEND)
        claude_model: Final = _deployment(scenario, wire, _CLAUDE_BACKEND)
        org_alias: Final = f"ml-org-{suffix}"
        team_alias: Final = f"ai-platform-{suffix}"
        project_alias: Final = f"support-bot-{suffix}"
        key_alias: Final = f"chargeback-{suffix}"
        uppercase_key_alias: Final = f"support-{suffix}"
        team: Final = scenario.team(
            team_alias=team_alias,
            organization_id=scenario.organization(
                organization_alias=org_alias, models=[chat_model, embed_model, claude_model]
            ),
        )
        yield Rig(
            proxy=proxy,
            wire=wire,
            chat_model=chat_model,
            embed_model=embed_model,
            claude_model=claude_model,
            key=scenario.key(
                key_alias=key_alias,
                team_id=team,
                project_id=scenario.project(team, project_alias=project_alias),
                metadata={"spend_logs_metadata": {"cost-center": "cc-1234", "Owner": "Platform Team"}},
            ),
            identity={
                "user_api_key_alias": key_alias,
                "user_api_key_team_alias": team_alias,
                "user_api_key_org_alias": org_alias,
                "user_api_key_project_alias": project_alias,
            },
            uppercase_team_key=scenario.key(
                key_alias=uppercase_key_alias,
                team_id=scenario.team(team_alias=f"AI Platform {suffix}"),
            ),
            uppercase_team_identity={
                "user_api_key_alias": uppercase_key_alias,
                "user_api_key_team_alias": f"ai_platform_{suffix}",
            },
        )


def _chat(rig: Rig, key: str, marker: str, stream: bool = False, **extra: JsonValue) -> JsonValue:
    response: Final = rig.proxy.request(
        "POST",
        "/v1/chat/completions",
        {"model": rig.chat_model, "messages": [{"role": "user", "content": marker}], "stream": stream, **extra},
        key=key,
    )
    assert response.status_code == 200, response.text
    assert _REPLY_TEXT in response.text, response.text
    return rig.labels_sent(marker)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_labels_carry_the_key_team_org_and_project_identity_with_no_client_labels(rig: Rig, stream: bool) -> None:
    labels: Final = _chat(rig, rig.key, f"identity-{uuid.uuid4().hex}", stream)

    assert labels == {**rig.identity, "cost-center": "cc-1234", "owner": "platform_team"}, labels


def test_caller_cannot_forge_identity_and_its_labels_rank_above_spend_logs_metadata(rig: Rig) -> None:
    labels: Final = _chat(
        rig,
        rig.key,
        f"forged-{uuid.uuid4().hex}",
        labels={"user_api_key_team_alias": "forged", "cost-center": "caller", "ticket": "t-42"},
    )

    assert labels == {**rig.identity, "cost-center": "caller", "ticket": "t-42", "owner": "platform_team"}, labels


def test_request_metadata_labels_still_reach_vertex_under_the_identity(rig: Rig) -> None:
    labels: Final = _chat(rig, rig.key, f"requester-{uuid.uuid4().hex}", metadata={"env": "prod"})

    assert labels == {**rig.identity, "env": "prod", "cost-center": "cc-1234", "owner": "platform_team"}, labels


def test_team_alias_outside_googles_label_rules_is_rewritten_to_a_compliant_label(rig: Rig) -> None:
    labels: Final = _chat(rig, rig.uppercase_team_key, f"uppercase-{uuid.uuid4().hex}")

    assert labels == rig.uppercase_team_identity, labels


def test_embedding_labels_carry_the_caller_identity(rig: Rig) -> None:
    marker: Final = f"embed-{uuid.uuid4().hex}"
    response: Final = rig.proxy.request(
        "POST", "/v1/embeddings", {"model": rig.embed_model, "input": [marker]}, key=rig.key
    )

    assert response.status_code == 200, response.text
    assert rig.labels_sent(marker) == {**rig.identity, "cost-center": "cc-1234", "owner": "platform_team"}


def test_claude_chat_labels_header_carries_the_identity_and_drops_a_forged_label(rig: Rig) -> None:
    marker: Final = f"claude-chat-{uuid.uuid4().hex}"
    response: Final = rig.proxy.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": rig.claude_model,
            "messages": [{"role": "user", "content": marker}],
            "labels": {"user_api_key_team_alias": "forged", "ticket": "t-42"},
        },
        key=rig.key,
    )

    assert response.status_code == 200, response.text
    assert _REPLY_TEXT in response.text, response.text
    assert rig.header_labels_sent(marker) == {
        **rig.identity,
        "ticket": "t-42",
        "cost-center": "cc-1234",
        "owner": "platform_team",
    }


def test_claude_messages_labels_header_carries_the_identity(rig: Rig) -> None:
    marker: Final = f"claude-messages-{uuid.uuid4().hex}"
    response: Final = rig.proxy.request(
        "POST",
        "/v1/messages",
        {"model": rig.claude_model, "max_tokens": 16, "messages": [{"role": "user", "content": marker}]},
        key=rig.key,
    )

    assert response.status_code == 200, response.text
    assert _REPLY_TEXT in response.text, response.text
    assert rig.header_labels_sent(marker) == {**rig.identity, "cost-center": "cc-1234", "owner": "platform_team"}
