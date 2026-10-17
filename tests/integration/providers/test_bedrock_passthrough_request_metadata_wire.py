import base64
import json
import uuid
from pathlib import Path
from typing import Final

import boto3
import pytest
import yaml
from botocore.awsrequest import AWSPreparedRequest
from botocore.config import Config
from integration._support.client import Gateway
from integration._support.process import owned_proxy
from integration._support.upstream import _aws_event_frame
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

_MODEL_ID: Final = "anthropic.claude-sonnet-5-v1:0"
_EVENT_STREAM: Final = "application/vnd.amazon.eventstream"
_METADATA_HEADER: Final = "x-amzn-bedrock-request-metadata"
_MESSAGES: Final[list[JsonValue]] = [{"role": "user", "content": [{"text": "who is calling"}]}]
_INVOKE_BODY: Final[dict[str, JsonValue]] = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 16,
    "messages": [{"role": "user", "content": "who is calling"}],
}
_CONVERSE_REPLY: Final[dict[str, JsonValue]] = {
    "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
    "stopReason": "end_turn",
    "usage": {"inputTokens": 5, "outputTokens": 1, "totalTokens": 6},
    "metrics": {"latencyMs": 3},
}
_INVOKE_REPLY: Final[dict[str, JsonValue]] = {
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


def _boto3_runtime(gateway: Gateway, key: str):  # noqa: ANN202 - boto3 clients are untyped
    client: Final = boto3.client(
        "bedrock-runtime",
        endpoint_url=f"{str(gateway.client.base_url).rstrip('/')}/bedrock",
        region_name="us-east-1",
        aws_access_key_id="AKIACALLERSIDE",
        aws_secret_access_key="caller-side-secret",
        config=Config(retries={"max_attempts": 1}),
    )

    def bearer(request: AWSPreparedRequest, **_: object) -> None:
        request.headers["Authorization"] = f"Bearer {key}"

    client.meta.events.register("before-send.bedrock-runtime.*", bearer)
    return client


def _signed_header_names(request: Request) -> frozenset[str]:
    return frozenset(request.headers["authorization"].split("SignedHeaders=")[1].split(",")[0].split(";"))


def test_boto3_passthrough_calls_carry_the_key_identity_in_signed_request_metadata(
    gateway: Gateway, tmp_path: Path
) -> None:
    suffix: Final = uuid.uuid4().hex[:12]
    router_name: Final = f"bedrock-metadata-{suffix}"
    base_config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    with wire_server(_bedrock_peer) as wire:
        config: Final = {
            **base_config,
            "model_list": [
                *base_config.get("model_list", []),
                {
                    "model_name": router_name,
                    "litellm_params": {
                        "model": f"bedrock/{_MODEL_ID}",
                        "api_base": wire.url,
                        "aws_access_key_id": "AKIASCRIPTEDPROVIDER",
                        "aws_secret_access_key": "scripted-secret",
                        "aws_region_name": "us-east-1",
                        "use_in_pass_through": True,
                    },
                },
            ],
            "litellm_settings": {
                **base_config.get("litellm_settings", {}),
                "drop_params": True,
                "bedrock_request_metadata_fields": [
                    "user_api_key_alias",
                    "user_api_key_team_alias",
                    "spend_logs_metadata",
                ],
            },
        }
        config_path: Final = tmp_path / "bedrock-request-metadata.yaml"
        config_path.write_text(yaml.safe_dump(config))
        with (
            owned_proxy(gateway, tmp_path, {}, config=config_path, workers=1) as candidate,
            candidate.scenario() as scenario,
        ):
            team_alias: Final = f"platform-{suffix}"
            key_alias: Final = f"prod-key-{suffix}"
            key: Final = scenario.key(
                key_alias=key_alias,
                team_id=scenario.team(team_alias=team_alias),
                metadata={"spend_logs_metadata": {"agent": "triage-bot"}},
            )
            identity: Final = {
                "user_api_key_alias": key_alias,
                "user_api_key_team_alias": team_alias,
                "agent": "triage-bot",
            }
            forged: Final = {"user_api_key_team_alias": "forged", "ticket": "T-42", "bad!": "x"}
            runtime: Final = _boto3_runtime(candidate, key)

            converse: Final = runtime.converse(modelId=router_name, messages=_MESSAGES, requestMetadata=forged)
            converse_stream: Final = [
                next(iter(event))
                for event in runtime.converse_stream(modelId=router_name, messages=_MESSAGES, requestMetadata=forged)[
                    "stream"
                ]
            ]
            invoke: Final = json.loads(
                runtime.invoke_model(modelId=router_name, body=json.dumps(_INVOKE_BODY))["body"].read()
            )
            invoke_stream: Final = [
                json.loads(event["chunk"]["bytes"])
                for event in runtime.invoke_model_with_response_stream(
                    modelId=router_name, body=json.dumps(_INVOKE_BODY)
                )["body"]
            ]
            assert converse["output"] == _CONVERSE_REPLY["output"], converse
            assert converse_stream == ["messageStart", "messageStop", "metadata"], converse_stream
            assert invoke == _INVOKE_REPLY, invoke
            assert invoke_stream == [_INVOKE_REPLY], invoke_stream

            received: Final = wire.drain()
            assert [request.target for request in received] == [
                f"/model/{_MODEL_ID}/converse",
                f"/model/{_MODEL_ID}/converse-stream",
                f"/model/{_MODEL_ID}/invoke",
                f"/model/{_MODEL_ID}/invoke-with-response-stream",
            ], received
            converse_requests: Final = received[:2]
            invoke_requests: Final = received[2:]
            assert [json.loads(request.body) for request in converse_requests] == [
                {"messages": _MESSAGES, "requestMetadata": {**identity, "ticket": "T-42"}},
                {"messages": _MESSAGES, "requestMetadata": {**identity, "ticket": "T-42"}},
            ], [request.body for request in converse_requests]
            assert [_METADATA_HEADER in request.headers for request in converse_requests] == [False, False]
            assert [json.loads(request.body) for request in invoke_requests] == [_INVOKE_BODY, _INVOKE_BODY]
            assert [json.loads(request.headers[_METADATA_HEADER]) for request in invoke_requests] == [
                identity,
                identity,
            ], [request.headers for request in invoke_requests]
            assert all(_METADATA_HEADER in _signed_header_names(request) for request in invoke_requests), [
                request.headers["authorization"] for request in invoke_requests
            ]
