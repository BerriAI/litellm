import base64
import json
import uuid
from urllib.parse import unquote
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import boto3
import pytest
import yaml
from botocore.config import Config
from botocore.exceptions import ClientError
from integration._support.budget import case_timeout_seconds
from integration._support.client import Gateway
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.upstream import _aws_event_frame
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

_MODEL_ID: Final = "anthropic.claude-sonnet-5-v1:0"
_EVENT_STREAM: Final = "application/vnd.amazon.eventstream"
_METADATA_HEADER: Final = "x-amzn-bedrock-request-metadata"
_HEADER_MAX_CHARS: Final = 8500
_LONG_SPEND_LOGS_METADATA: Final = tuple((f"{index:02d}".ljust(256, "k"), "\n" * 256) for index in range(12))
_CONVERSE_REPLY: Final[Mapping[str, JsonValue]] = {
    "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
    "stopReason": "end_turn",
    "usage": {"inputTokens": 5, "outputTokens": 1, "totalTokens": 6},
    "metrics": {"latencyMs": 3},
}
_INVOKE_REPLY: Final[Mapping[str, JsonValue]] = {
    "id": "msg_bedrock",
    "type": "message",
    "role": "assistant",
    "model": "claude",
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 5, "output_tokens": 1},
}
_CONVERSE_STREAM_FRAMES: Final = (
    _aws_event_frame("messageStart", {"role": "assistant"}, "sc", "u"),
    _aws_event_frame("messageStop", {"stopReason": "end_turn"}, "sc", "u"),
    _aws_event_frame("metadata", {"usage": {"inputTokens": 5, "outputTokens": 1, "totalTokens": 6}}, "sc", "u"),
)


def _bedrock_peer(request: Request) -> Reply:
    if request.target.endswith("/converse-stream"):
        return Reply(chunks=_CONVERSE_STREAM_FRAMES, content_type=_EVENT_STREAM)
    if request.target.endswith("/invoke-with-response-stream"):
        chunk: Final = {"bytes": base64.b64encode(json.dumps(_INVOKE_REPLY).encode()).decode()}
        return Reply(chunks=(_aws_event_frame("chunk", chunk, "sc", "u"),), content_type=_EVENT_STREAM)
    if request.target.endswith("/converse"):
        return Reply(body=json.dumps(_CONVERSE_REPLY).encode())
    return Reply(body=json.dumps(_INVOKE_REPLY).encode())


def _messages(marker: str) -> Sequence[JsonValue]:
    return [{"role": "user", "content": [{"text": marker}]}]


def _invoke_body(marker: str) -> Mapping[str, JsonValue]:
    return {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": marker}],
    }


def _signed_header_names(request: Request) -> frozenset[str]:
    return frozenset(request.headers["authorization"].split("SignedHeaders=")[1].split(",")[0].split(";"))


def _header_value(pairs: tuple[tuple[str, str], ...]) -> str:
    return json.dumps(dict(pairs), separators=(",", ":"))


def _send_with_caller_metadata(
    runtime: object, action: str, model: str, marker: str, caller_metadata: Mapping[str, JsonValue]
) -> None:
    if action == "invoke":
        getattr(runtime, "invoke_model")(
            modelId=model, body=json.dumps({**_invoke_body(marker), "requestMetadata": caller_metadata})
        )
        return
    getattr(runtime, action.replace("-", "_"))(
        modelId=model, messages=_messages(marker), requestMetadata=caller_metadata
    )


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: Gateway
    wire: Wire
    strict_model: str
    dropping_model: str
    chat_model: str
    key: str
    identity: Mapping[str, str]
    long_key: str
    long_identity: tuple[tuple[str, str], ...]

    def runtime(self, key: str, monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201  # boto3 clients are untyped
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", key)
        return boto3.client(
            "bedrock-runtime",
            endpoint_url=f"{str(self.proxy.client.base_url).rstrip('/')}/bedrock",
            region_name="us-east-1",
            config=Config(
                read_timeout=case_timeout_seconds(),
                retries={"total_max_attempts": 1},
                parameter_validation=False,
            ),
        )

    def received(self, marker: str) -> tuple[Request, ...]:
        return tuple(request for request in self.wire.drain() if marker.encode() in request.body)


@pytest.fixture(scope="module")
def rig(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    suffix: Final = uuid.uuid4().hex[:12]
    strict_model: Final = f"bedrock-metadata-{suffix}"
    dropping_model: Final = f"bedrock-metadata-dropping-{suffix}"
    chat_model: Final = f"bedrock-metadata-chat-{suffix}"
    directory: Final = tmp_path_factory.mktemp("bedrock-request-metadata")
    base_config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    with wire_server(_bedrock_peer) as wire:
        deployment: Final = {
            "model": f"bedrock/{_MODEL_ID}",
            "api_base": wire.url,
            "aws_access_key_id": "AKIASCRIPTEDPROVIDER",
            "aws_secret_access_key": "scripted-secret",
            "aws_region_name": "us-east-1",
            "use_in_pass_through": True,
        }
        config: Final = {
            **base_config,
            "model_list": [
                *base_config.get("model_list", []),
                {"model_name": strict_model, "litellm_params": deployment},
                {"model_name": dropping_model, "litellm_params": {**deployment, "drop_params": True}},
                {
                    "model_name": chat_model,
                    "litellm_params": {**deployment, "model": f"bedrock/converse/{_MODEL_ID}", "drop_params": True},
                },
            ],
            "litellm_settings": {
                **base_config.get("litellm_settings", {}),
                "drop_params": False,
                "bedrock_request_metadata_fields": [
                    "user_api_key_alias",
                    "user_api_key_team_alias",
                    "spend_logs_metadata",
                ],
            },
        }
        config_path: Final = directory / "bedrock-request-metadata.yaml"
        config_path.write_text(yaml.safe_dump(config))
        with (
            owned_proxy(session_gateway, directory, {}, config=config_path, workers=2) as proxy,
            proxy.scenario() as scenario,
        ):
            team_alias: Final = f"platform-{suffix}"
            team_id: Final = scenario.team(team_alias=team_alias)
            key_alias: Final = f"prod-key-{suffix}"
            long_key_alias: Final = f"long-key-{suffix}"
            yield Rig(
                proxy=proxy,
                wire=wire,
                strict_model=strict_model,
                dropping_model=dropping_model,
                chat_model=chat_model,
                key=scenario.key(
                    key_alias=key_alias,
                    team_id=team_id,
                    metadata={"spend_logs_metadata": {"agent": "triage-bot"}},
                ),
                identity={
                    "user_api_key_alias": key_alias,
                    "user_api_key_team_alias": team_alias,
                    "agent": "triage-bot",
                },
                long_key=scenario.key(
                    key_alias=long_key_alias,
                    team_id=team_id,
                    metadata={"spend_logs_metadata": dict(_LONG_SPEND_LOGS_METADATA)},
                ),
                long_identity=(
                    ("user_api_key_alias", long_key_alias),
                    ("user_api_key_team_alias", team_alias),
                    *_LONG_SPEND_LOGS_METADATA,
                ),
            )


def test_boto3_passthrough_calls_carry_the_key_identity_in_signed_request_metadata(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker: Final = f"who-is-calling-{uuid.uuid4().hex}"
    forged: Final = {"user_api_key_team_alias": "forged", "ticket": "T-42", "bad!": "x"}
    runtime: Final = rig.runtime(rig.key, monkeypatch)
    model: Final = rig.dropping_model

    converse: Final = runtime.converse(modelId=model, messages=_messages(marker), requestMetadata=forged)
    converse_stream: Final = [
        next(iter(event))
        for event in runtime.converse_stream(modelId=model, messages=_messages(marker), requestMetadata=forged)[
            "stream"
        ]
    ]
    invoke: Final = json.loads(
        runtime.invoke_model(modelId=model, body=json.dumps(_invoke_body(marker)))["body"].read()
    )
    invoke_stream: Final = [
        json.loads(event["chunk"]["bytes"])
        for event in runtime.invoke_model_with_response_stream(modelId=model, body=json.dumps(_invoke_body(marker)))[
            "body"
        ]
    ]
    assert converse["output"] == _CONVERSE_REPLY["output"], converse
    assert converse_stream == ["messageStart", "messageStop", "metadata"], converse_stream
    assert invoke == _INVOKE_REPLY, invoke
    assert invoke_stream == [_INVOKE_REPLY], invoke_stream

    received: Final = rig.received(marker)
    assert [request.target for request in received] == [
        f"/model/{_MODEL_ID}/converse",
        f"/model/{_MODEL_ID}/converse-stream",
        f"/model/{_MODEL_ID}/invoke",
        f"/model/{_MODEL_ID}/invoke-with-response-stream",
    ], received
    converse_requests: Final = received[:2]
    invoke_requests: Final = received[2:]
    assert [json.loads(request.body) for request in converse_requests] == [
        {"messages": _messages(marker), "requestMetadata": {**rig.identity, "ticket": "T-42"}},
        {"messages": _messages(marker), "requestMetadata": {**rig.identity, "ticket": "T-42"}},
    ], [request.body for request in converse_requests]
    assert [_METADATA_HEADER in request.headers for request in converse_requests] == [False, False]
    assert [json.loads(request.body) for request in invoke_requests] == [_invoke_body(marker), _invoke_body(marker)]
    assert [json.loads(request.headers[_METADATA_HEADER]) for request in invoke_requests] == [
        rig.identity,
        rig.identity,
    ], [request.headers for request in invoke_requests]
    assert all(_METADATA_HEADER in _signed_header_names(request) for request in invoke_requests), [
        request.headers["authorization"] for request in invoke_requests
    ]


def test_invoke_metadata_too_long_for_the_header_signs_the_leading_pairs_that_fit(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker: Final = f"long-key-{uuid.uuid4().hex}"
    runtime: Final = rig.runtime(rig.long_key, monkeypatch)

    invoke: Final = json.loads(
        runtime.invoke_model(modelId=rig.strict_model, body=json.dumps(_invoke_body(marker)))["body"].read()
    )

    assert invoke == _INVOKE_REPLY, invoke
    (request,) = rig.received(marker)
    assert json.loads(request.body) == _invoke_body(marker)
    header: Final = request.headers[_METADATA_HEADER]
    sent_pairs: Final = len(json.loads(header))
    assert tuple(json.loads(header).items()) == rig.long_identity[:sent_pairs], header
    assert 2 <= sent_pairs < len(rig.long_identity), header
    assert len(header) <= _HEADER_MAX_CHARS < len(_header_value(rig.long_identity[: sent_pairs + 1]))
    assert _METADATA_HEADER in _signed_header_names(request), request.headers["authorization"]


@pytest.mark.parametrize(
    ("action", "caller_metadata", "error_code"),
    [
        pytest.param("converse", {"ticket": 42}, "SerializationException", id="converse-number-value"),
        pytest.param("converse-stream", {"bad!": "x"}, "ValidationException", id="converse-stream-bad-key"),
        pytest.param("invoke", {"bad!": "x"}, "ValidationException", id="invoke-bad-key"),
    ],
)
def test_malformed_caller_request_metadata_reaches_boto3_as_the_bedrock_error_without_an_upstream_call(
    rig: Rig, monkeypatch: pytest.MonkeyPatch, action: str, caller_metadata: Mapping[str, JsonValue], error_code: str
) -> None:
    marker: Final = f"malformed-{uuid.uuid4().hex}"
    runtime: Final = rig.runtime(rig.key, monkeypatch)

    with pytest.raises(ClientError) as raised:
        _send_with_caller_metadata(runtime, action, rig.strict_model, marker, caller_metadata)

    assert raised.value.response["Error"]["Code"] == error_code, raised.value.response
    assert raised.value.response["ResponseMetadata"]["HTTPStatusCode"] == 400, raised.value.response
    assert rig.received(marker) == ()


def test_chat_on_a_dropping_converse_deployment_sends_only_the_valid_caller_pairs(rig: Rig) -> None:
    marker: Final = f"chat-{uuid.uuid4().hex}"

    response: Final = rig.proxy.client.post(
        "/chat/completions",
        json={
            "model": rig.chat_model,
            "messages": [{"role": "user", "content": marker}],
            "requestMetadata": {"ticket": "T-42", "bad!": "x"},
        },
        headers={"Authorization": f"Bearer {rig.key}"},
        timeout=case_timeout_seconds(),
    )

    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == "ok", response.text
    (request,) = rig.received(marker)
    assert unquote(request.target) == f"/model/{_MODEL_ID}/converse", request.target
    assert json.loads(request.body)["requestMetadata"] == {**rig.identity, "ticket": "T-42"}, request.body
