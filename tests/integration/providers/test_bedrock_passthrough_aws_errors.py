"""Errors on the Bedrock passthrough reach the AWS SDK with their message (LIT-9617).

boto3 reads an error's code from the ``x-amzn-errortype`` header and its message from the body's ``message``. A
guardrail block on ``/bedrock/model/<model>/...`` that happens before any frame was sent reaches the caller as a
``ValidationException`` carrying the guardrail's message, and an error Bedrock itself returned, such as a 429
``ThrottlingException``, reaches the caller with Bedrock's status, ``x-amzn-errortype`` and body."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import boto3
import httpx
import pytest
import yaml
from botocore.config import Config
from botocore.exceptions import ClientError
from integration._support.budget import case_timeout_seconds
from integration._support.client import Gateway, JsonValue
from integration._support.process import owned_proxy_process
from integration._support.upstream import aws_event_stream_frame
from integration._support.wire import Reply, Request, Wire, wire_server

MODEL_ID: Final = "anthropic.claude-sonnet-5-v1:0"
ROUTER_MODEL: Final = "aws-errors"
EVENT_STREAM: Final = "application/vnd.amazon.eventstream"
POST_CALL: Final = "aws-errors-post-call"
PRE_CALL: Final = "aws-errors-pre-call"
BEDROCK: Final = "aws-errors-bedrock"
UNGUARDED: Final = "unguarded"
BLOCKED_WORD: Final = "FORBIDDEN"
BLOCK_REASON: Final = "synthetic policy says no"
BEDROCK_BLOCK_MESSAGE: Final = "Violated guardrail policy"
THROTTLE_MESSAGE: Final = "Too many requests, please wait before trying again."
THROTTLE_TYPE: Final = "ThrottlingException:http://internal.amazon.com/coral/com.amazon.bedrock/"
INVALID_MESSAGE: Final = "The provided model identifier is invalid."
INVALID_TYPE: Final = "ValidationException:http://internal.amazon.com/coral/com.amazon.bedrock/"
ANTHROPIC_THROTTLE: Final = b'{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}'


@dataclass(frozen=True, slots=True)
class UpstreamError:
    status: int
    errortype: str
    message: str

    @property
    def body(self) -> bytes:
        return json.dumps({"message": self.message}).encode()

    @property
    def code(self) -> str:
        return self.errortype.split(":", 1)[0]


THROTTLED: Final = UpstreamError(429, THROTTLE_TYPE, THROTTLE_MESSAGE)
INVALID: Final = UpstreamError(400, INVALID_TYPE, INVALID_MESSAGE)


def event_frame(event_type: str, payload: Mapping[str, JsonValue]) -> bytes:
    return aws_event_stream_frame(
        {":event-type": event_type, ":content-type": "application/json", ":message-type": "event"},
        json.dumps(payload, separators=(",", ":")).encode(),
    )


def converse_frames(text: str) -> tuple[bytes, ...]:
    return (
        event_frame("messageStart", {"role": "assistant"}),
        event_frame("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": text}}),
        event_frame("contentBlockStop", {"contentBlockIndex": 0}),
        event_frame("messageStop", {"stopReason": "end_turn"}),
        event_frame("metadata", {"usage": {"inputTokens": 9, "outputTokens": 4, "totalTokens": 13}, "metrics": {}}),
    )


def converse_reply(text: str) -> bytes:
    return json.dumps(
        {
            "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 9, "outputTokens": 4, "totalTokens": 13},
            "metrics": {"latencyMs": 1},
        }
    ).encode()


def marker_of(request: Request) -> str:
    return str(json.loads(request.body)["messages"][0]["content"][0]["text"])


def bedrock_texts(request: Request) -> tuple[str, ...]:
    return tuple(str(item["text"]["text"]) for item in json.loads(request.body)["content"])


@dataclass(frozen=True, slots=True)
class Rig:
    gateway: Gateway
    provider: Wire
    vendor: Wire
    errors: dict[str, UpstreamError]
    keys: Mapping[str, str]

    def provider_calls(self, marker: str) -> int:
        return sum(1 for request in self.provider.drain() if marker.encode() in request.body)

    def boto3(self, guardrail: str, attempts: int = 1):  # noqa: ANN201 - boto3 clients are untyped
        client: Final = boto3.client(
            "bedrock-runtime",
            region_name="us-east-1",
            endpoint_url=str(self.gateway.client.base_url).rstrip("/") + "/bedrock",
            aws_access_key_id="AKIASYNTHETICCALLER",
            aws_secret_access_key="synthetic-caller-secret",
            config=Config(
                read_timeout=case_timeout_seconds(),
                retries={"total_max_attempts": attempts, "mode": "standard"},
            ),
        )
        key: Final = self.keys[guardrail]

        def authenticate(request: object, **_: object) -> None:
            getattr(request, "headers")["x-litellm-api-key"] = key

        client.meta.events.register("before-send.bedrock-runtime.*", authenticate)
        return client

    def call(self, guardrail: str, action: str, marker: str, attempts: int = 1) -> ClientError | None:
        client: Final = self.boto3(guardrail, attempts)
        messages: Final = [{"role": "user", "content": [{"text": marker}]}]
        try:
            if action == "converse-stream":
                for _ in client.converse_stream(modelId=ROUTER_MODEL, messages=messages)["stream"]:
                    pass
            else:
                client.converse(modelId=ROUTER_MODEL, messages=messages)
        except ClientError as error:
            return error
        return None

    def raw(self, guardrail: str, action: str, marker: str) -> httpx.Response:
        return self.gateway.client.post(
            f"/bedrock/model/{ROUTER_MODEL}/{action}",
            json={"messages": [{"role": "user", "content": [{"text": marker}]}]},
            headers={"x-litellm-api-key": self.keys[guardrail]},
            timeout=case_timeout_seconds(),
        )


def _generic_guardrail(name: str, mode: str, vendor: Wire, **settings: JsonValue) -> dict[str, JsonValue]:
    return {
        "guardrail_name": name,
        "litellm_params": {
            "guardrail": "generic_guardrail_api",
            "mode": mode,
            "default_on": False,
            "api_base": f"{vendor.url}/{name}",
            "api_key": "synthetic-guardrail-key",
            **settings,
        },
    }


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    errors: Final[dict[str, UpstreamError]] = {}

    def provider(request: Request) -> Reply:
        if request.target == "/v1/messages":
            return Reply(status=429, body=ANTHROPIC_THROTTLE)
        assert request.target in (f"/model/{MODEL_ID}/converse-stream", f"/model/{MODEL_ID}/converse"), request.target
        marker: Final = marker_of(request)
        error: Final = errors.get(marker)
        if error is not None:
            return Reply(status=error.status, body=error.body, headers={"x-amzn-errortype": error.errortype})
        text: Final = f"The answer to {marker} is {BLOCKED_WORD}"
        if request.target.endswith("/converse-stream"):
            return Reply(content_type=EVENT_STREAM, chunks=converse_frames(text))
        return Reply(body=converse_reply(text))

    def vendor(request: Request) -> Reply:
        if request.target.startswith(f"/{BEDROCK}/"):
            if not any(BLOCKED_WORD in text for text in bedrock_texts(request)):
                return Reply(body=json.dumps({"action": "NONE", "outputs": [], "assessments": []}).encode())
            return Reply(
                body=json.dumps(
                    {
                        "action": "GUARDRAIL_INTERVENED",
                        "outputs": [{"text": "Sorry, the model cannot answer this."}],
                        "assessments": [
                            {"wordPolicy": {"customWords": [{"match": BLOCKED_WORD, "action": "BLOCKED"}]}}
                        ],
                    }
                ).encode()
            )
        texts: Final = [str(text) for text in json.loads(request.body)["texts"]]
        if any(BLOCKED_WORD in text for text in texts):
            return Reply(body=json.dumps({"action": "BLOCKED", "blocked_reason": BLOCK_REASON}).encode())
        return Reply(body=json.dumps({"action": "NONE"}).encode())

    with httpx.Client(base_url=os.environ["INTEGRATION_PROXY_URL"], timeout=30, trust_env=False) as root_client:
        root: Final = Gateway(
            root_client,
            os.environ.get("INTEGRATION_MASTER_KEY", "sk-integration-master"),
            os.environ["INTEGRATION_UPSTREAM_URL"],
        )
        directory: Final = tmp_path_factory.mktemp("bedrock-aws-errors")
        with wire_server(provider) as upstream, wire_server(vendor) as sink:
            config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
            config["guardrails"] = [
                _generic_guardrail(POST_CALL, "post_call", sink, streaming_buffer_until_moderated=True),
                _generic_guardrail(PRE_CALL, "pre_call", sink),
                {
                    "guardrail_name": BEDROCK,
                    "litellm_params": {
                        "guardrail": "bedrock",
                        "mode": "post_call",
                        "default_on": False,
                        "guardrailIdentifier": "aws-errors",
                        "guardrailVersion": "1",
                        "aws_access_key_id": "AKIASYNTHETICGUARD",
                        "aws_secret_access_key": "synthetic-guardrail-secret",
                        "aws_region_name": "us-east-1",
                        "aws_bedrock_runtime_endpoint": f"{sink.url}/{BEDROCK}",
                    },
                },
            ]
            config["model_list"] = [
                {
                    "model_name": ROUTER_MODEL,
                    "litellm_params": {
                        "model": f"bedrock/{MODEL_ID}",
                        "api_base": upstream.url,
                        "aws_access_key_id": "AKIASYNTHETICPROVIDER",
                        "aws_secret_access_key": "synthetic-provider-secret",
                        "aws_region_name": "us-east-1",
                    },
                }
            ]
            config["router_settings"] = {**config.get("router_settings", {}), "num_retries": 0}
            config_path: Final = directory / "bedrock-aws-errors.yaml"
            config_path.write_text(yaml.safe_dump(config))
            overrides: Final = {"ANTHROPIC_API_BASE": upstream.url, "ANTHROPIC_API_KEY": "sk-ant-synthetic"}
            with owned_proxy_process(root, directory, overrides, config=config_path, workers=2) as owned:
                with owned.gateway.scenario() as scenario:
                    yield Rig(
                        owned.gateway,
                        upstream,
                        sink,
                        errors,
                        {
                            UNGUARDED: scenario.key(),
                            **{name: scenario.key(guardrails=[name]) for name in (POST_CALL, PRE_CALL, BEDROCK)},
                        },
                    )


BLOCKS: Final = (
    pytest.param(POST_CALL, "converse-stream", BLOCK_REASON, 1, id="post-call-held-converse-stream"),
    pytest.param(POST_CALL, "converse", BLOCK_REASON, 1, id="post-call-converse"),
    pytest.param(PRE_CALL, "converse-stream", BLOCK_REASON, 0, id="pre-call-converse-stream"),
    pytest.param(PRE_CALL, "converse", BLOCK_REASON, 0, id="pre-call-converse"),
    pytest.param(BEDROCK, "converse-stream", BEDROCK_BLOCK_MESSAGE, 1, id="bedrock-guardrail-converse-stream"),
    pytest.param(BEDROCK, "converse", BEDROCK_BLOCK_MESSAGE, 1, id="bedrock-guardrail-converse"),
)


@pytest.mark.parametrize(("guardrail", "action", "message", "provider_calls"), BLOCKS)
def test_guardrail_block_reaches_boto3_as_a_validation_exception_with_the_guardrail_message(
    rig: Rig, guardrail: str, action: str, message: str, provider_calls: int
) -> None:
    marker: Final = f"{BLOCKED_WORD}-{uuid.uuid4().hex}"
    error: Final = rig.call(guardrail, action, marker)
    assert error is not None, f"{guardrail} {action} was not blocked"
    assert error.response["ResponseMetadata"]["HTTPStatusCode"] == 400, error.response
    assert (error.response["Error"]["Code"], error.response["Error"]["Message"]) == (
        "ValidationException",
        message,
    ), error.response
    assert rig.provider_calls(marker) == provider_calls


@pytest.mark.parametrize("action", ("converse-stream", "converse"))
def test_guardrail_block_body_adds_the_aws_message_and_keeps_the_litellm_error(rig: Rig, action: str) -> None:
    marker: Final = f"{BLOCKED_WORD}-{uuid.uuid4().hex}"
    response: Final = rig.raw(POST_CALL, action, marker)
    assert response.status_code == 400, response.text
    assert response.headers.get("x-amzn-errortype") == "ValidationException", (dict(response.headers), response.text)
    assert response.headers.get("content-type", "").startswith("application/json"), dict(response.headers)
    body: Final = response.json()
    assert body["message"] == BLOCK_REASON, body
    assert body["error"]["message"] == BLOCK_REASON, body
    assert response.headers.get("x-litellm-call-id"), dict(response.headers)


@pytest.mark.parametrize("upstream", (pytest.param(THROTTLED, id="throttled"), pytest.param(INVALID, id="invalid")))
@pytest.mark.parametrize("action", ("converse-stream", "converse"))
@pytest.mark.parametrize("guardrail", (UNGUARDED, POST_CALL))
def test_bedrock_error_reaches_the_caller_as_bedrock_sent_it(
    rig: Rig, upstream: UpstreamError, action: str, guardrail: str
) -> None:
    marker: Final = f"upstream-{uuid.uuid4().hex}"
    rig.errors[marker] = upstream
    response: Final = rig.raw(guardrail, action, marker)
    assert response.status_code == upstream.status, response.text
    assert response.headers.get("x-amzn-errortype") == upstream.errortype, (dict(response.headers), response.text)
    assert response.content == upstream.body, response.text
    assert response.headers.get("x-litellm-call-id"), dict(response.headers)
    assert rig.provider_calls(marker) == 1


@pytest.mark.parametrize("action", ("converse-stream", "converse"))
@pytest.mark.parametrize("guardrail", (UNGUARDED, POST_CALL))
def test_boto3_sees_bedrock_throttling_and_retries_it_as_against_bedrock(
    rig: Rig, action: str, guardrail: str
) -> None:
    marker: Final = f"upstream-{uuid.uuid4().hex}"
    rig.errors[marker] = THROTTLED
    error: Final = rig.call(guardrail, action, marker, attempts=3)
    assert error is not None, f"{action} was not throttled"
    assert (error.response["Error"]["Code"], error.response["Error"]["Message"]) == (
        THROTTLED.code,
        THROTTLE_MESSAGE,
    ), error.response
    assert error.response["ResponseMetadata"]["HTTPStatusCode"] == 429, error.response
    assert error.response["ResponseMetadata"]["RetryAttempts"] == 2, error.response
    assert rig.provider_calls(marker) == 3


def test_other_provider_passthrough_errors_carry_no_aws_error_type(rig: Rig) -> None:
    response: Final = rig.gateway.client.post(
        "/anthropic/v1/messages",
        json={"model": "claude-sonnet-5", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]},
        headers={"x-litellm-api-key": rig.keys[UNGUARDED]},
        timeout=case_timeout_seconds(),
    )
    assert response.status_code == 429, response.text
    assert "x-amzn-errortype" not in response.headers, dict(response.headers)
    assert response.content == ANTHROPIC_THROTTLE, response.text
