import json
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Final
from urllib.parse import quote

import boto3
import pytest
from botocore.awsrequest import AWSPreparedRequest
from botocore.config import Config
from integration._support.client import Gateway, Scenario
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


def _full_body_peer(request: Request) -> Reply:
    if request.target.endswith("/converse"):
        return Reply(body=json.dumps(_CONVERSE_REPLY).encode())
    return Reply(body=json.dumps(_INVOKE_REPLY).encode())


def _bedrock_deployment(scenario: Scenario, name: str, model_id: str, api_base: str) -> None:
    created: Final = scenario.gateway.post(
        "/model/new",
        {
            "model_name": name,
            "litellm_params": {
                "model": f"bedrock/{model_id}",
                "api_base": api_base,
                "aws_access_key_id": "AKIASCRIPTEDPROVIDER",
                "aws_secret_access_key": "scripted-secret",
                "aws_region_name": "us-east-1",
            },
        },
    )
    model_info: Final = created["model_info"]
    assert isinstance(model_info, Mapping), created
    scenario.cleanups.callback(scenario.delete_model, str(model_info["id"]))


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


def test_boto3_converse_and_invoke_reach_upstream_with_the_full_body_for_names_ids_and_profile_arns(
    gateway: Gateway, tmp_path: Path
) -> None:
    suffix: Final = uuid.uuid4().hex[:12]
    raw_id: Final = f"anthropic.claude-scripted-{suffix}-v1:0"
    profile_arn: Final = f"arn:aws:bedrock:us-east-1:{_ACCOUNT}:application-inference-profile/{suffix}"
    with (
        wire_server(_full_body_peer) as wire,
        owned_proxy(gateway, tmp_path, {}, workers=2) as candidate,
        candidate.scenario() as scenario,
    ):
        router_name: Final = scenario.model(
            model=f"bedrock/{_MODEL_ID}",
            api_base=wire.url,
            aws_access_key_id="AKIASCRIPTEDPROVIDER",
            aws_secret_access_key="scripted-secret",
            aws_region_name="us-east-1",
        )
        _bedrock_deployment(scenario, raw_id, raw_id, wire.url)
        _bedrock_deployment(scenario, profile_arn, profile_arn, wire.url)
        key: Final = scenario.key()
        runtime: Final = _boto3_runtime(candidate, key)
        replies: Final = [
            runtime.converse(modelId=router_name, **_CONVERSE_BODY),
            runtime.converse(modelId=profile_arn, **_CONVERSE_BODY),
        ]
        invoked: Final = [
            json.loads(runtime.invoke_model(modelId=model, body=json.dumps(_INVOKE_BODY))["body"].read())
            for model in (raw_id, profile_arn)
        ]
        assert [{name: reply[name] for name in _CONVERSE_REPLY} for reply in replies] == [_CONVERSE_REPLY] * 2, replies
        assert invoked == [_INVOKE_REPLY] * 2, invoked
        received: Final = wire.drain()
        encoded_arn: Final = quote(profile_arn, safe=":")
        assert [(request.method, request.target) for request in received] == [
            ("POST", f"/model/{_MODEL_ID}/converse"),
            ("POST", f"/model/{encoded_arn}/converse"),
            ("POST", f"/model/{raw_id}/invoke"),
            ("POST", f"/model/{encoded_arn}/invoke"),
        ], received
        assert [json.loads(request.body) for request in received] == [_CONVERSE_BODY] * 2 + [_INVOKE_BODY] * 2, [
            request.body for request in received
        ]
        for request in received:
            _assert_signed_with_the_deployment_key(request, key)
