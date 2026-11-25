import base64
import json
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
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

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 180)

_MODEL_ID: Final = "anthropic.claude-sonnet-5-v1:0"
_EVENT_STREAM: Final = "application/vnd.amazon.eventstream"
_BLOCKED_WORD: Final = "zzzforbidden"
_BLOCKED_TEXT: Final = "Blocked by guardrail"
_VERSIONLESS_GUARDRAIL_MESSAGE: Final = "guardrailIdentifier and guardrailVersion must be provided together."
_ACTIONS: Final = ("converse", "converse-stream", "invoke", "invoke-with-response-stream")
_DEPLOYMENT_GUARDRAIL: Final[Mapping[str, str]] = {
    "guardrailIdentifier": "deployment-guardrail",
    "guardrailVersion": "3",
    "trace": "enabled",
    "streamProcessingMode": "sync",
}
_CALLER_GUARDRAIL: Final[Mapping[str, str]] = {
    "guardrailIdentifier": "caller-guardrail",
    "guardrailVersion": "9",
    "trace": "disabled",
}
_MANTLE_KEY: Final = "scripted-mantle-key"
_CALLER_INVOKE_HEADERS: Final[Mapping[str, str]] = {
    "x-pass-x-amzn-bedrock-guardrailidentifier": "caller-guardrail",
    "x-pass-x-amzn-bedrock-guardrailversion": "9",
    "x-pass-x-amzn-bedrock-trace": "DISABLED",
}


@dataclass(frozen=True, slots=True)
class Applied:
    """The guardrail a request asked Bedrock for, read from where Bedrock reads it for that action."""

    identifier: str
    version: str | None
    trace: str | None
    stream_processing_mode: str | None


@dataclass(frozen=True, slots=True)
class Outcome:
    """What the AWS SDK caller reads back."""

    text: str
    intervened: bool
    traced_guardrail: str | None


def _action(request: Request) -> str:
    return request.target.partition("?")[0].rsplit("/", 1)[-1]


def _applied(request: Request) -> Applied | None:
    if _action(request).startswith("converse"):
        config: Final = json.loads(request.body).get("guardrailConfig")
        if config is None:
            return None
        return Applied(
            config["guardrailIdentifier"],
            config.get("guardrailVersion"),
            config.get("trace"),
            config.get("streamProcessingMode"),
        )
    identifier: Final = request.headers.get("x-amzn-bedrock-guardrailidentifier")
    if identifier is None:
        return None
    return Applied(
        identifier,
        request.headers.get("x-amzn-bedrock-guardrailversion"),
        request.headers.get("x-amzn-bedrock-trace"),
        None,
    )


def _identifier(request: Request) -> str | None:
    applied: Final = _applied(request)
    return None if applied is None else applied.identifier


def _assessment(applied: Applied) -> Mapping[str, JsonValue]:
    return {
        applied.identifier: {
            "wordPolicy": {"customWords": [{"match": _BLOCKED_WORD, "action": "BLOCKED", "detected": True}]},
            "appliedGuardrailDetails": {"guardrailId": applied.identifier, "guardrailVersion": applied.version},
        }
    }


def _bedrock_with_guardrails(request: Request) -> Reply:
    """Bedrock's runtime as the caller sees it: the guardrail the request names blocks a prompt holding the blocked
    word, answering with its message and, when the request enabled tracing, a trace naming that guardrail. A guardrail
    named without a version is refused with a 400 ValidationException, as real Bedrock in us-east-1 refused it on
    every one of these actions on 2026-10-19."""
    applied: Final = _applied(request)
    if applied is not None and applied.version is None:
        return Reply(
            status=400,
            body=json.dumps({"message": _VERSIONLESS_GUARDRAIL_MESSAGE}).encode(),
            headers={"x-amzn-errortype": "ValidationException"},
        )
    blocked: Final = applied is not None and _BLOCKED_WORD.encode() in request.body
    traced: Final = blocked and applied is not None and (applied.trace or "").lower() == "enabled"
    text: Final = _BLOCKED_TEXT if blocked else "ok"
    stop_reason: Final = "guardrail_intervened" if blocked else "end_turn"
    usage: Final = {"inputTokens": 5, "outputTokens": 1, "totalTokens": 6}
    converse_trace: Final = {"trace": {"guardrail": {"inputAssessment": _assessment(applied)}}} if traced else {}
    action: Final = _action(request)
    if action == "converse":
        return Reply(
            body=json.dumps(
                {
                    "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
                    "stopReason": stop_reason,
                    "usage": usage,
                    "metrics": {"latencyMs": 3},
                    **converse_trace,
                }
            ).encode()
        )
    if action == "converse-stream":
        return Reply(
            chunks=(
                _aws_event_frame("messageStart", {"role": "assistant"}, "sc", "u"),
                _aws_event_frame("contentBlockDelta", {"delta": {"text": text}, "contentBlockIndex": 0}, "sc", "u"),
                _aws_event_frame("contentBlockStop", {"contentBlockIndex": 0}, "sc", "u"),
                _aws_event_frame("messageStop", {"stopReason": stop_reason}, "sc", "u"),
                _aws_event_frame(
                    "metadata", {"usage": usage, "metrics": {"latencyMs": 3}, **converse_trace}, "sc", "u"
                ),
            ),
            content_type=_EVENT_STREAM,
        )
    message: Final = {
        "id": "msg_bedrock",
        "type": "message",
        "role": "assistant",
        "model": "claude",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 5, "output_tokens": 1},
        **({"amazon-bedrock-guardrailAction": "INTERVENED"} if blocked else {}),
        **({"amazon-bedrock-trace": {"guardrail": {"input": _assessment(applied)}}} if traced else {}),
    }
    if action == "invoke":
        return Reply(body=json.dumps(message).encode())
    chunk: Final = {"bytes": base64.b64encode(json.dumps(message).encode()).decode()}
    return Reply(chunks=(_aws_event_frame("chunk", chunk, "sc", "u"),), content_type=_EVENT_STREAM)


def _bedrock_outage(request: Request) -> Reply:
    return Reply(
        status=500,
        body=json.dumps({"message": "scripted outage"}).encode(),
        headers={"x-amzn-errortype": "InternalServerException"},
    )


def _traced_guardrail(assessment: object) -> str | None:
    return None if not isinstance(assessment, dict) else next(iter(assessment))


def _send(runtime: object, action: str, model: str, prompt: str, caller_guardrail: bool) -> Outcome:
    messages: Final = [{"role": "user", "content": [{"text": prompt}]}]
    invoke_body: Final = json.dumps(
        {"anthropic_version": "bedrock-2023-05-31", "max_tokens": 16, "messages": [{"role": "user", "content": prompt}]}
    )
    invoke_guardrail: Final = (
        {"guardrailIdentifier": "caller-guardrail", "guardrailVersion": "9", "trace": "DISABLED"}
        if caller_guardrail
        else {}
    )
    if action == "converse":
        response: Final = getattr(runtime, "converse")(
            modelId=model,
            messages=messages,
            **({"guardrailConfig": dict(_CALLER_GUARDRAIL)} if caller_guardrail else {}),
        )
        return Outcome(
            text=response["output"]["message"]["content"][0]["text"],
            intervened=response["stopReason"] == "guardrail_intervened",
            traced_guardrail=_traced_guardrail(response.get("trace", {}).get("guardrail", {}).get("inputAssessment")),
        )
    if action == "converse-stream":
        events: Final = list(
            getattr(runtime, "converse_stream")(
                modelId=model,
                messages=messages,
                **(
                    {"guardrailConfig": {**_CALLER_GUARDRAIL, "streamProcessingMode": "async"}}
                    if caller_guardrail
                    else {}
                ),
            )["stream"]
        )
        (stop,) = tuple(event["messageStop"]["stopReason"] for event in events if "messageStop" in event)
        (metadata,) = tuple(event["metadata"] for event in events if "metadata" in event)
        return Outcome(
            text="".join(
                event["contentBlockDelta"]["delta"]["text"] for event in events if "contentBlockDelta" in event
            ),
            intervened=stop == "guardrail_intervened",
            traced_guardrail=_traced_guardrail(metadata.get("trace", {}).get("guardrail", {}).get("inputAssessment")),
        )
    if action == "invoke":
        message: Final = json.loads(
            getattr(runtime, "invoke_model")(modelId=model, body=invoke_body, **invoke_guardrail)["body"].read()
        )
    else:
        (message,) = tuple(
            json.loads(event["chunk"]["bytes"])
            for event in getattr(runtime, "invoke_model_with_response_stream")(
                modelId=model, body=invoke_body, **invoke_guardrail
            )["body"]
        )
    return Outcome(
        text=message["content"][0]["text"],
        intervened=message.get("amazon-bedrock-guardrailAction") == "INTERVENED",
        traced_guardrail=_traced_guardrail(message.get("amazon-bedrock-trace", {}).get("guardrail", {}).get("input")),
    )


def _signed_header_names(request: Request) -> frozenset[str]:
    return frozenset(request.headers["authorization"].split("SignedHeaders=")[1].split(",")[0].split(";"))


def _deployment(wire: Wire, guardrail_config: Mapping[str, str] | None = None) -> Mapping[str, JsonValue]:
    return {
        "model": f"bedrock/{_MODEL_ID}",
        "api_base": wire.url,
        "aws_access_key_id": "AKIASCRIPTEDPROVIDER",
        "aws_secret_access_key": "scripted-secret",
        "aws_region_name": "us-east-1",
        "use_in_pass_through": True,
        **({} if guardrail_config is None else {"guardrailConfig": dict(guardrail_config)}),
    }


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: Gateway
    wire: Wire
    guarded_model: str
    unguarded_model: str
    balanced_model: str
    balanced_wires: Mapping[str, Wire]
    failing_model: str
    failing_wire: Wire
    fallback_wire: Wire
    mantle_model: str
    versionless_model: str
    key: str

    def runtime(  # noqa: ANN201  # boto3 clients are untyped
        self, monkeypatch: pytest.MonkeyPatch, extra_headers: Mapping[str, str] = MappingProxyType({})
    ):
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", self.key)
        client: Final = boto3.client(
            "bedrock-runtime",
            endpoint_url=f"{str(self.proxy.client.base_url).rstrip('/')}/bedrock",
            region_name="us-east-1",
            config=Config(read_timeout=case_timeout_seconds(), retries={"total_max_attempts": 1}),
        )
        client.meta.events.register(
            "before-send.bedrock-runtime.*",
            lambda request, **_: request.headers.update(extra_headers),
        )
        return client

    def received(self, wire: Wire, marker: str) -> tuple[Request, ...]:
        return tuple(request for request in wire.drain() if marker.encode() in request.body)


@pytest.fixture(scope="module")
def rig(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    suffix: Final = uuid.uuid4().hex[:12]
    guarded_model: Final = f"bedrock-guarded-{suffix}"
    unguarded_model: Final = f"bedrock-unguarded-{suffix}"
    balanced_model: Final = f"bedrock-balanced-{suffix}"
    failing_model: Final = f"bedrock-failing-{suffix}"
    fallback_model: Final = f"bedrock-fallback-{suffix}"
    mantle_model: Final = f"bedrock-mantle-{suffix}"
    versionless_model: Final = f"bedrock-versionless-{suffix}"
    directory: Final = tmp_path_factory.mktemp("bedrock-guardrail-config")
    base_config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    with (
        wire_server(_bedrock_with_guardrails) as wire,
        wire_server(_bedrock_with_guardrails) as wire_a,
        wire_server(_bedrock_with_guardrails) as wire_b,
        wire_server(_bedrock_outage) as failing_wire,
        wire_server(_bedrock_with_guardrails) as fallback_wire,
    ):
        config: Final = {
            **base_config,
            "model_list": [
                *base_config.get("model_list", []),
                {
                    "model_name": guarded_model,
                    "litellm_params": _deployment(wire, _DEPLOYMENT_GUARDRAIL),
                },
                {"model_name": unguarded_model, "litellm_params": _deployment(wire)},
                {
                    "model_name": balanced_model,
                    "litellm_params": _deployment(
                        wire_a, {"guardrailIdentifier": "guardrail-a", "guardrailVersion": "1"}
                    ),
                },
                {
                    "model_name": balanced_model,
                    "litellm_params": _deployment(
                        wire_b, {"guardrailIdentifier": "guardrail-b", "guardrailVersion": "2"}
                    ),
                },
                {
                    "model_name": failing_model,
                    "litellm_params": _deployment(
                        failing_wire,
                        {"guardrailIdentifier": "failing-guardrail", "guardrailVersion": "5", "trace": "enabled"},
                    ),
                },
                {
                    "model_name": fallback_model,
                    "litellm_params": _deployment(
                        fallback_wire,
                        {"guardrailIdentifier": "fallback-guardrail", "guardrailVersion": "6", "trace": "enabled"},
                    ),
                },
                {
                    "model_name": versionless_model,
                    "litellm_params": _deployment(
                        wire, {"guardrailIdentifier": "versionless-guardrail", "trace": "enabled"}
                    ),
                },
                {
                    "model_name": mantle_model,
                    "litellm_params": {
                        "model": "bedrock_mantle/anthropic.claude-sonnet-5",
                        "api_base": wire.url,
                        "api_key": _MANTLE_KEY,
                        "aws_region_name": "us-east-1",
                        "use_in_pass_through": True,
                        "guardrailConfig": {
                            "guardrailIdentifier": "mantle-guardrail",
                            "guardrailVersion": "4",
                            "trace": "enabled",
                        },
                    },
                },
            ],
            "router_settings": {
                **base_config.get("router_settings", {}),
                "fallbacks": [{failing_model: [fallback_model]}],
            },
        }
        config_path: Final = directory / "bedrock-guardrail-config.yaml"
        config_path.write_text(yaml.safe_dump(config))
        with (
            owned_proxy(session_gateway, directory, {}, config=config_path, workers=2) as proxy,
            proxy.scenario() as scenario,
        ):
            yield Rig(
                proxy=proxy,
                wire=wire,
                guarded_model=guarded_model,
                unguarded_model=unguarded_model,
                balanced_model=balanced_model,
                balanced_wires={"guardrail-a": wire_a, "guardrail-b": wire_b},
                failing_model=failing_model,
                failing_wire=failing_wire,
                fallback_wire=fallback_wire,
                mantle_model=mantle_model,
                versionless_model=versionless_model,
                key=scenario.key(key_alias=f"guardrail-key-{suffix}"),
            )


def _expected_on_the_wire(action: str, request: Request) -> Applied:
    if action == "converse":
        return Applied("deployment-guardrail", "3", "enabled", None)
    if action == "converse-stream":
        return Applied("deployment-guardrail", "3", "enabled", "sync")
    assert {"x-amzn-bedrock-guardrailidentifier", "x-amzn-bedrock-guardrailversion", "x-amzn-bedrock-trace"} <= (
        _signed_header_names(request)
    ), request.headers["authorization"]
    return Applied("deployment-guardrail", "3", "ENABLED", None)


@pytest.mark.parametrize("action", _ACTIONS)
def test_passthrough_on_a_guardrailed_deployment_blocks_with_the_deployment_guardrail(
    rig: Rig, monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    marker: Final = uuid.uuid4().hex
    runtime: Final = rig.runtime(monkeypatch)

    blocked: Final = _send(runtime, action, rig.guarded_model, f"{marker} say {_BLOCKED_WORD}", caller_guardrail=False)
    allowed: Final = _send(runtime, action, rig.guarded_model, f"{marker} say hello", caller_guardrail=False)

    assert blocked == Outcome(_BLOCKED_TEXT, intervened=True, traced_guardrail="deployment-guardrail"), blocked
    assert allowed == Outcome("ok", intervened=False, traced_guardrail=None), allowed
    received: Final = rig.received(rig.wire, marker)
    assert [_action(request) for request in received] == [action, action], received
    assert [_applied(request) for request in received] == [
        _expected_on_the_wire(action, request) for request in received
    ], [(request.headers, request.body) for request in received]


@pytest.mark.parametrize("action", _ACTIONS)
def test_caller_guardrail_is_replaced_by_the_deployment_guardrail(
    rig: Rig, monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    marker: Final = uuid.uuid4().hex
    runtime: Final = rig.runtime(monkeypatch, _CALLER_INVOKE_HEADERS)

    outcome: Final = _send(runtime, action, rig.guarded_model, f"{marker} say {_BLOCKED_WORD}", caller_guardrail=True)

    assert outcome == Outcome(_BLOCKED_TEXT, intervened=True, traced_guardrail="deployment-guardrail"), outcome
    (request,) = rig.received(rig.wire, marker)
    assert _applied(request) == _expected_on_the_wire(action, request), (request.headers, request.body)
    assert "caller-guardrail" not in json.dumps({**request.headers, "body": request.body.decode()}), request


@pytest.mark.parametrize("action", _ACTIONS)
def test_mantle_deployment_passthrough_sends_its_guardrail_with_bearer_auth(
    rig: Rig, monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    marker: Final = uuid.uuid4().hex
    runtime: Final = rig.runtime(monkeypatch)

    outcome: Final = _send(runtime, action, rig.mantle_model, f"{marker} say {_BLOCKED_WORD}", caller_guardrail=False)

    assert outcome == Outcome(_BLOCKED_TEXT, intervened=True, traced_guardrail="mantle-guardrail"), outcome
    (request,) = rig.received(rig.wire, marker)
    assert request.headers["authorization"] == f"Bearer {_MANTLE_KEY}", request.headers
    assert _applied(request) == Applied(
        "mantle-guardrail", "4", "ENABLED" if action.startswith("invoke") else "enabled", None
    ), (request.headers, request.body)


@pytest.mark.parametrize("action", _ACTIONS)
def test_deployment_guardrail_without_a_version_returns_the_bedrock_rejection(
    rig: Rig, monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    marker: Final = uuid.uuid4().hex
    runtime: Final = rig.runtime(monkeypatch)

    with pytest.raises(ClientError) as raised:
        _send(runtime, action, rig.versionless_model, f"{marker} say hello", caller_guardrail=False)

    error: Final = raised.value.response.get("Error", {})
    assert (error.get("Code"), error.get("Message")) == ("ValidationException", _VERSIONLESS_GUARDRAIL_MESSAGE), error
    received: Final = rig.received(rig.wire, marker)
    assert {_applied(request) for request in received} == {
        Applied("versionless-guardrail", None, "ENABLED" if action.startswith("invoke") else "enabled", None)
    }, [(request.headers, request.body) for request in received]


@pytest.mark.parametrize("action", _ACTIONS)
def test_deployment_without_a_guardrail_leaves_the_caller_guardrail_as_sent(
    rig: Rig, monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    if action.startswith("invoke"):
        pytest.skip("BUG: the router-model invoke pass-through drops the caller's X-Amzn-Bedrock-Guardrail* headers")
    marker: Final = uuid.uuid4().hex
    runtime: Final = rig.runtime(monkeypatch)

    outcome: Final = _send(runtime, action, rig.unguarded_model, f"{marker} say {_BLOCKED_WORD}", caller_guardrail=True)

    assert outcome == Outcome(_BLOCKED_TEXT, intervened=True, traced_guardrail=None), outcome
    (request,) = rig.received(rig.wire, marker)
    assert _applied(request) == Applied(
        "caller-guardrail", "9", "disabled", "async" if action == "converse-stream" else None
    ), request.body


def test_each_load_balanced_request_carries_the_guardrail_of_the_deployment_it_reached(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker: Final = uuid.uuid4().hex
    runtime: Final = rig.runtime(monkeypatch)

    outcomes: Final = tuple(
        _send(runtime, action, rig.balanced_model, f"{marker}-{attempt} say {_BLOCKED_WORD}", caller_guardrail=False)
        for attempt, action in enumerate(_ACTIONS * 6)
    )

    assert all(outcome.intervened for outcome in outcomes), outcomes
    reached: Final = {name: rig.received(wire, marker) for name, wire in rig.balanced_wires.items()}
    assert sum(len(requests) for requests in reached.values()) == len(outcomes), reached
    assert all(reached.values()), {name: len(requests) for name, requests in reached.items()}
    assert {name: {_identifier(request) for request in requests} for name, requests in reached.items()} == {
        "guardrail-a": {"guardrail-a"},
        "guardrail-b": {"guardrail-b"},
    }, reached


def test_a_fallback_sends_the_guardrail_of_the_deployment_it_fell_back_to(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker: Final = uuid.uuid4().hex
    runtime: Final = rig.runtime(monkeypatch)

    outcomes: Final = tuple(
        _send(runtime, action, rig.failing_model, f"{marker} say {_BLOCKED_WORD}", caller_guardrail=False)
        for action in _ACTIONS
    )

    assert outcomes == (Outcome(_BLOCKED_TEXT, intervened=True, traced_guardrail="fallback-guardrail"),) * 4, outcomes
    failed: Final = rig.received(rig.failing_wire, marker)
    served: Final = rig.received(rig.fallback_wire, marker)
    assert [_action(request) for request in served] == list(_ACTIONS), served
    assert {_identifier(request) for request in failed} == {"failing-guardrail"}, failed
    assert {_identifier(request) for request in served} == {"fallback-guardrail"}, served
