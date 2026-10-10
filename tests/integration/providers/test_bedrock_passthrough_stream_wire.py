import base64
import json
import uuid
from pathlib import Path
from typing import Final
from urllib.parse import quote

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
_REQUEST_BODY: Final = {"messages": [{"role": "user", "content": [{"text": "synthetic passthrough stream"}]}]}
_EVENTS: Final = (
    ("messageStart", {"role": "assistant"}),
    ("contentBlockDelta", {"delta": {"text": "bedrock stream control"}, "contentBlockIndex": 0}),
    ("messageStop", {"stopReason": "end_turn"}),
    ("metadata", {"usage": {"inputTokens": 11, "outputTokens": 4, "totalTokens": 15}}),
)
_STREAM_BYTES: Final = b"".join(_aws_event_frame(kind, payload, "sc", "u") for kind, payload in _EVENTS)


def event_stream_peer(request: Request) -> Reply:
    assert request.method == "POST"
    assert request.target == f"/model/{_MODEL_ID}/converse-stream"
    assert json.loads(request.body)["messages"] == _REQUEST_BODY["messages"]
    return Reply(body=_STREAM_BYTES, content_type=_EVENT_STREAM)


@pytest.mark.covers("other.provider_wire.bedrock.passthrough_stream_keeps_event_stream_content_type")
def test_bedrock_passthrough_converse_stream_response_carries_event_stream_content_type(gateway: Gateway) -> None:
    with wire_server(event_stream_peer) as wire, gateway.scenario() as scenario:
        deployment: Final = scenario.model(
            model=f"bedrock/{_MODEL_ID}",
            api_base=wire.url,
            aws_access_key_id="AKIASCRIPTEDPROVIDER",
            aws_secret_access_key="scripted-secret",
            aws_region_name="us-east-1",
        )
        response: Final = gateway.request("POST", f"/bedrock/model/{deployment}/converse-stream", _REQUEST_BODY)
        assert response.status_code == 200, response.text
        assert len(wire.drain()) == 1, response.text
        assert response.headers.get("content-type") == _EVENT_STREAM, dict(response.headers)
        assert response.content == _STREAM_BYTES, response.text


_ACCOUNT: Final = "123456789012"
_CONVERSE_BODY: Final[dict[str, JsonValue]] = {
    "messages": [{"role": "user", "content": [{"text": "full body"}]}],
    "system": [{"text": "be terse"}],
    "inferenceConfig": {"maxTokens": 64, "temperature": 0.2, "stopSequences": ["END"]},
    "toolConfig": {
        "tools": [
            {
                "toolSpec": {
                    "name": "lookup",
                    "description": "look a value up",
                    "inputSchema": {"json": {"type": "object", "properties": {"q": {"type": "string"}}}},
                }
            }
        ],
        "toolChoice": {"auto": {}},
    },
}
_INVOKE_BODY: Final[dict[str, JsonValue]] = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 64,
    "system": "be terse",
    "messages": [{"role": "user", "content": "full body"}],
}
_CONVERSE_REPLY: Final[dict[str, JsonValue]] = {
    "output": {"message": {"role": "assistant", "content": [{"text": "bedrock converse"}]}},
    "stopReason": "end_turn",
    "usage": {"inputTokens": 5, "outputTokens": 2, "totalTokens": 7},
    "metrics": {"latencyMs": 12},
}
_INVOKE_REPLY: Final[dict[str, JsonValue]] = {
    "id": "msg_bedrock",
    "type": "message",
    "role": "assistant",
    "model": "claude",
    "content": [{"type": "text", "text": "bedrock invoke"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 5, "output_tokens": 2},
}


_CONVERSE_STREAM_FRAMES: Final = tuple(_aws_event_frame(kind, payload, "sc", "u") for kind, payload in _EVENTS)
_INVOKE_STREAM_FRAMES: Final = (
    _aws_event_frame(
        "chunk",
        {"bytes": base64.b64encode(json.dumps(_INVOKE_REPLY).encode()).decode()},
        "sc",
        "u",
    ),
)


def _full_body_peer(request: Request) -> Reply:
    if request.target.endswith("/converse-stream"):
        return Reply(chunks=_CONVERSE_STREAM_FRAMES, content_type=_EVENT_STREAM)
    if request.target.endswith("/invoke-with-response-stream"):
        return Reply(chunks=_INVOKE_STREAM_FRAMES, content_type=_EVENT_STREAM)
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


def _assert_signed_with_the_deployment_key(request: Request, key: str) -> None:
    authorization: Final = request.headers["authorization"]
    assert authorization.startswith("AWS4-HMAC-SHA256 Credential=AKIASCRIPTEDPROVIDER/"), request.headers
    assert "/us-east-1/bedrock/aws4_request" in authorization, request.headers
    assert {name: value for name, value in request.headers.items() if key in value} == {}, request.headers
    assert key not in request.target, request.target


def test_boto3_converse_invoke_and_stream_methods_forward_full_bodies_for_deployments_ids_and_profile_arns(
    gateway: Gateway, tmp_path: Path
) -> None:
    suffix: Final = uuid.uuid4().hex[:12]
    raw_id: Final = f"anthropic.claude-scripted-{suffix}-v1:0"
    profile_arn: Final = f"arn:aws:bedrock:us-east-1:{_ACCOUNT}:application-inference-profile/{suffix}"
    router_name: Final = f"bedrock-router-{suffix}"
    base_config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    with wire_server(_full_body_peer) as wire:
        deployment_params: Final = {
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
                {
                    "model_name": router_name,
                    "litellm_params": {"model": f"bedrock/{_MODEL_ID}", **deployment_params},
                },
                {
                    "model_name": raw_id,
                    "litellm_params": {"model": f"bedrock/{raw_id}", **deployment_params},
                },
                {
                    "model_name": profile_arn,
                    "litellm_params": {"model": f"bedrock/{profile_arn}", **deployment_params},
                },
            ],
        }
        config_path: Final = tmp_path / "bedrock-passthrough.yaml"
        config_path.write_text(yaml.safe_dump(config))
        with (
            owned_proxy(gateway, tmp_path, {}, config=config_path, workers=1) as candidate,
            candidate.scenario() as scenario,
        ):
            key: Final = scenario.key()
            runtime: Final = _boto3_runtime(candidate, key)
            router_converse: Final = runtime.converse(modelId=router_name, **_CONVERSE_BODY)
            profile_converse: Final = runtime.converse(modelId=profile_arn, **_CONVERSE_BODY)
            raw_invoke: Final = runtime.invoke_model(modelId=raw_id, body=json.dumps(_INVOKE_BODY))
            profile_invoke: Final = runtime.invoke_model(modelId=profile_arn, body=json.dumps(_INVOKE_BODY))
            raw_converse_stream: Final = runtime.converse_stream(modelId=raw_id, **_CONVERSE_BODY)
            profile_converse_stream: Final = runtime.converse_stream(modelId=profile_arn, **_CONVERSE_BODY)
            raw_invoke_stream: Final = runtime.invoke_model_with_response_stream(
                modelId=raw_id, body=json.dumps(_INVOKE_BODY)
            )
            profile_invoke_stream: Final = runtime.invoke_model_with_response_stream(
                modelId=profile_arn, body=json.dumps(_INVOKE_BODY)
            )
            assert tuple(
                {name: reply[name] for name in _CONVERSE_REPLY} for reply in (router_converse, profile_converse)
            ) == (_CONVERSE_REPLY, _CONVERSE_REPLY), (router_converse, profile_converse)
            assert tuple(json.loads(response["body"].read()) for response in (raw_invoke, profile_invoke)) == (
                _INVOKE_REPLY,
                _INVOKE_REPLY,
            ), (raw_invoke, profile_invoke)
            expected_converse_events: Final = [{kind: payload} for kind, payload in _EVENTS]
            assert [list(response["stream"]) for response in (raw_converse_stream, profile_converse_stream)] == [
                expected_converse_events,
                expected_converse_events,
            ], (raw_converse_stream, profile_converse_stream)
            assert [
                [json.loads(event["chunk"]["bytes"]) for event in response["body"]]
                for response in (raw_invoke_stream, profile_invoke_stream)
            ] == [[_INVOKE_REPLY], [_INVOKE_REPLY]], (raw_invoke_stream, profile_invoke_stream)

            received: Final = wire.drain()
            encoded_arn: Final = quote(profile_arn, safe=":")
            assert [(request.method, request.target) for request in received] == [
                ("POST", f"/model/{_MODEL_ID}/converse"),
                ("POST", f"/model/{encoded_arn}/converse"),
                ("POST", f"/model/{raw_id}/invoke"),
                ("POST", f"/model/{encoded_arn}/invoke"),
                ("POST", f"/model/{raw_id}/converse-stream"),
                ("POST", f"/model/{encoded_arn}/converse-stream"),
                ("POST", f"/model/{raw_id}/invoke-with-response-stream"),
                ("POST", f"/model/{encoded_arn}/invoke-with-response-stream"),
            ], received
            assert tuple(json.loads(request.body) for request in received) == (
                _CONVERSE_BODY,
                _CONVERSE_BODY,
                _INVOKE_BODY,
                _INVOKE_BODY,
                _CONVERSE_BODY,
                _CONVERSE_BODY,
                _INVOKE_BODY,
                _INVOKE_BODY,
            ), tuple(request.body for request in received)
            for request in received:
                _assert_signed_with_the_deployment_key(request, key)


def test_boto3_direct_raw_model_ids_forward_full_bodies(gateway: Gateway, tmp_path: Path) -> None:
    direct_raw_id: Final = f"anthropic.claude-direct-{uuid.uuid4().hex[:12]}-v1:0"
    with wire_server(_full_body_peer) as wire:
        config_path: Final = tmp_path / "bedrock-direct.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "model_list": [{"model_name": "bedrock/*", "litellm_params": {"model": "bedrock/*"}}],
                    "general_settings": {"supported_db_objects": []},
                }
            )
        )
        aws_environment: Final = {
            "AWS_ACCESS_KEY_ID": "AKIASCRIPTEDPROVIDER",
            "AWS_SECRET_ACCESS_KEY": "scripted-secret",
            "AWS_REGION_NAME": "us-east-1",
            "AWS_BEDROCK_RUNTIME_ENDPOINT": wire.url,
        }
        with (
            owned_proxy(
                gateway,
                tmp_path,
                aws_environment,
                config=config_path,
                remove_environment=(
                    "AWS_ACCESS_KEY_ID",
                    "AWS_SECRET_ACCESS_KEY",
                    "AWS_SESSION_TOKEN",
                    "AWS_SECURITY_TOKEN",
                    "AWS_BEARER_TOKEN_BEDROCK",
                    "AWS_PROFILE",
                    "AWS_DEFAULT_PROFILE",
                    "AWS_SHARED_CREDENTIALS_FILE",
                    "AWS_CONFIG_FILE",
                    "AWS_WEB_IDENTITY_TOKEN_FILE",
                    "AWS_ROLE_ARN",
                    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
                    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
                    "AWS_REGION",
                    "AWS_DEFAULT_REGION",
                ),
                workers=1,
            ) as candidate,
            candidate.scenario() as scenario,
        ):
            key: Final = scenario.key()
            runtime: Final = _boto3_runtime(candidate, key)
            direct_converse: Final = runtime.converse(modelId=direct_raw_id, **_CONVERSE_BODY)
            direct_invoke: Final = runtime.invoke_model(modelId=direct_raw_id, body=json.dumps(_INVOKE_BODY))
            assert {name: direct_converse[name] for name in _CONVERSE_REPLY} == _CONVERSE_REPLY, direct_converse
            assert json.loads(direct_invoke["body"].read()) == _INVOKE_REPLY, direct_invoke

            received: Final = wire.drain()
            assert [(request.method, request.target) for request in received] == [
                ("POST", f"/model/{direct_raw_id}/converse"),
                ("POST", f"/model/{direct_raw_id}/invoke"),
            ], received
            assert tuple(json.loads(request.body) for request in received) == (_CONVERSE_BODY, _INVOKE_BODY), tuple(
                request.body for request in received
            )
            for request in received:
                _assert_signed_with_the_deployment_key(request, key)
