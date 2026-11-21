"""A provider key the caller forwards in ``x-api-key`` belongs to the deployment it was sent for (#45309).

The proxy runs with ``general_settings.forward_llm_provider_auth_headers`` on, so a caller can bring its own
Anthropic key. When the Anthropic group fails and the router falls back to a Bedrock group that signs with
its own IAM keys, Bedrock must get a SigV4 request and never the caller's Anthropic key. The scripted Bedrock
peers accept SigV4 from the configured access key or a bearer token that is not an Anthropic key, as real
Bedrock does for its own API keys, and answer 403 to anything else.

Tests named ``..._regression`` fail before the fix. Tests named ``..._guard`` pin behavior the fix keeps.
"""

from __future__ import annotations

import base64
import json
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final
from urllib.parse import unquote

import pytest
import yaml
from integration._support.anthropic_sse import status_reply
from integration._support.client import JSON_OBJECT, Gateway, gateway_from_environment, object_value
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.upstream import aws_event_stream_frame
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.security._canary import Canary, canary, find_canary
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 180)

ACCESS_KEY: Final = "AKIAFALLBACKSCOPE0001"
SECRET_KEY: Final = "synthetic-fallback-scope-secret"
REGION: Final = "us-east-1"
SIGV4: Final = f"AWS4-HMAC-SHA256 Credential={ACCESS_KEY}/"
ANTHROPIC_MODEL: Final = "anthropic/claude-haiku-5-5"
BEDROCK_BACKEND: Final = "us.anthropic.claude-haiku-5-5"
MANTLE_MODEL: Final = "bedrock_mantle/anthropic.claude-haiku-5-5"
ANTHROPIC_DEPLOYMENT_KEY: Final = "synthetic-anthropic-deployment-key"
BEDROCK_DEPLOYMENT_TOKEN: Final = "synthetic-bedrock-deployment-bearer"
BEDROCK_TEXT: Final = "served by the bedrock fallback"
ANTHROPIC_TEXT: Final = "served by anthropic"
CONTEXT_BETA: Final = "context-1m-2025-08-07"
APP_HEADER: Final = "x-app"
EVENT_STREAM: Final = "application/vnd.amazon.eventstream"
REMOVED_ENVIRONMENT: Final = ("AWS_BEARER_TOKEN_BEDROCK", "BEDROCK_MANTLE_API_KEY")


@dataclass(frozen=True, slots=True)
class Groups:
    primary: str
    overloaded: str
    to_mantle: str
    forwarding: str
    to_bearer: str
    to_anthropic: str
    keyless: str
    bedrock: str
    mantle: str
    bearer: str
    anthropic: str


@dataclass(frozen=True, slots=True)
class Wires:
    rejecting: Wire
    overloaded: Wire
    answering: Wire
    bedrock: Wire
    mantle: Wire


@dataclass(frozen=True, slots=True)
class Rig:
    gateway: Gateway
    groups: Groups
    wires: Wires


@dataclass(frozen=True, slots=True)
class Answer:
    status: int
    text: str


@dataclass(frozen=True, slots=True)
class Route:
    path: str
    stream: bool

    def body(self, model: str, prompt: str) -> dict[str, JsonValue]:
        if self.path == "/v1/responses":
            return {"model": model, "input": prompt, "stream": self.stream}
        return {
            "model": model,
            "max_tokens": 32,
            "messages": [{"role": "user", "content": prompt}],
            "stream": self.stream,
        }


MESSAGES: Final = Route("/v1/messages", stream=False)
CHAT: Final = Route("/v1/chat/completions", stream=False)
ROUTES: Final = (
    pytest.param(MESSAGES, id="messages"),
    pytest.param(Route("/v1/messages", stream=True), id="messages_stream"),
    pytest.param(CHAT, id="chat"),
    pytest.param(Route("/v1/chat/completions", stream=True), id="chat_stream"),
    pytest.param(Route("/v1/responses", stream=False), id="responses"),
)


def _anthropic_message(model: str, text: str) -> dict[str, JsonValue]:
    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 9, "output_tokens": 5},
    }


def _anthropic_events(model: str, text: str) -> tuple[dict[str, JsonValue], ...]:
    message: Final = _anthropic_message(model, text)
    return (
        {"type": "message_start", "message": {**message, "content": [], "stop_reason": None}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}},
        {"type": "message_stop"},
    )


def _streaming(request: Request) -> bool:
    return JSON_OBJECT.validate_json(request.body or b"{}").get("stream") is True


def _anthropic_reply(request: Request, model: str, text: str) -> Reply:
    if not _streaming(request):
        return Reply(body=json.dumps(_anthropic_message(model, text)).encode())
    return Reply(
        content_type="text/event-stream",
        chunks=tuple(
            f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()
            for event in _anthropic_events(model, text)
        ),
    )


def _answering_anthropic(request: Request) -> Reply:
    return _anthropic_reply(request, "claude-haiku-5-5", ANTHROPIC_TEXT)


def _failing_anthropic(status: int) -> Callable[[Request], Reply]:
    def respond(_request: Request) -> Reply:
        return status_reply(status)

    return respond


def _aws_event(event_type: str, payload: Mapping[str, JsonValue]) -> bytes:
    return aws_event_stream_frame(
        {":event-type": event_type, ":content-type": "application/json", ":message-type": "event"},
        json.dumps(payload, separators=(",", ":")).encode(),
    )


def _invoke_chunk(event: Mapping[str, JsonValue]) -> bytes:
    return _aws_event("chunk", {"bytes": base64.b64encode(json.dumps(event).encode()).decode()})


def _converse_reply() -> Reply:
    return Reply(
        body=json.dumps(
            {
                "output": {"message": {"role": "assistant", "content": [{"text": BEDROCK_TEXT}]}},
                "stopReason": "end_turn",
                "usage": {"inputTokens": 9, "outputTokens": 5, "totalTokens": 14},
                "metrics": {"latencyMs": 1},
            }
        ).encode()
    )


def _converse_stream_reply() -> Reply:
    events: Final[tuple[tuple[str, Mapping[str, JsonValue]], ...]] = (
        ("messageStart", {"role": "assistant"}),
        ("contentBlockDelta", {"delta": {"text": BEDROCK_TEXT}, "contentBlockIndex": 0}),
        ("contentBlockStop", {"contentBlockIndex": 0}),
        ("messageStop", {"stopReason": "end_turn"}),
        ("metadata", {"usage": {"inputTokens": 9, "outputTokens": 5, "totalTokens": 14}, "metrics": {"latencyMs": 1}}),
    )
    return Reply(content_type=EVENT_STREAM, chunks=tuple(_aws_event(kind, payload) for kind, payload in events))


def _bedrock_rejection(request: Request) -> Reply | None:
    authorization: Final = request.headers.get("authorization", "")
    if authorization.startswith(SIGV4):
        return None
    if authorization.startswith("Bearer ") and not authorization.startswith("Bearer sk-ant-"):
        return None
    return Reply(status=403, body=b'{"message": "The security token included in the request is invalid."}')


def _bedrock_runtime(request: Request) -> Reply:
    rejection: Final = _bedrock_rejection(request)
    if rejection is not None:
        return rejection
    target: Final = unquote(request.target)
    if target.endswith("/converse"):
        return _converse_reply()
    if target.endswith("/converse-stream"):
        return _converse_stream_reply()
    if target.endswith("/invoke"):
        return Reply(body=json.dumps(_anthropic_message(BEDROCK_BACKEND, BEDROCK_TEXT)).encode())
    if target.endswith("/invoke-with-response-stream"):
        return Reply(
            content_type=EVENT_STREAM,
            chunks=tuple(_invoke_chunk(event) for event in _anthropic_events(BEDROCK_BACKEND, BEDROCK_TEXT)),
        )
    return Reply(status=404, body=json.dumps({"message": f"no scripted bedrock route {target}"}).encode())


def _bedrock_mantle(request: Request) -> Reply:
    rejection: Final = _bedrock_rejection(request)
    if rejection is not None:
        return rejection
    if unquote(request.target) != "/anthropic/v1/messages":
        return Reply(status=404, body=json.dumps({"message": f"no scripted mantle route {request.target}"}).encode())
    return _anthropic_reply(request, "anthropic.claude-haiku-5-5", BEDROCK_TEXT)


def _anthropic(group: str, wire: Wire, api_key: str | None = ANTHROPIC_DEPLOYMENT_KEY) -> dict[str, JsonValue]:
    params: Final[dict[str, JsonValue]] = {"model": ANTHROPIC_MODEL, "api_base": wire.url}
    return {"model_name": group, "litellm_params": params if api_key is None else {**params, "api_key": api_key}}


def _bedrock(group: str, model: str, wire: Wire, **auth: JsonValue) -> dict[str, JsonValue]:
    endpoint: Final[dict[str, JsonValue]] = (
        {"api_base": wire.url} if model.startswith("bedrock_mantle/") else {"aws_bedrock_runtime_endpoint": wire.url}
    )
    return {"model_name": group, "litellm_params": {"model": model, "aws_region_name": REGION, **endpoint, **auth}}


def _iam() -> dict[str, JsonValue]:
    return {"aws_access_key_id": ACCESS_KEY, "aws_secret_access_key": SECRET_KEY}


def _config(directory: Path, groups: Groups, wires: Wires) -> Path:
    base: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    model_list: Final[tuple[JsonValue, ...]] = (
        _anthropic(groups.primary, wires.rejecting),
        _anthropic(groups.overloaded, wires.overloaded),
        _anthropic(groups.to_mantle, wires.rejecting),
        _anthropic(groups.forwarding, wires.rejecting),
        _anthropic(groups.to_bearer, wires.rejecting),
        _anthropic(groups.to_anthropic, wires.rejecting),
        _anthropic(groups.keyless, wires.answering, api_key=None),
        _anthropic(groups.anthropic, wires.answering),
        _bedrock(groups.bedrock, f"bedrock/{BEDROCK_BACKEND}", wires.bedrock, **_iam()),
        _bedrock(groups.mantle, MANTLE_MODEL, wires.mantle, **_iam()),
        _bedrock(groups.bearer, f"bedrock/{BEDROCK_BACKEND}", wires.bedrock, api_key=BEDROCK_DEPLOYMENT_TOKEN),
    )
    fallbacks: Final[tuple[JsonValue, ...]] = (
        {groups.primary: [groups.bedrock]},
        {groups.overloaded: [groups.bedrock]},
        {groups.to_mantle: [groups.mantle]},
        {groups.forwarding: [groups.bedrock]},
        {groups.to_bearer: [groups.bearer]},
        {groups.to_anthropic: [groups.anthropic]},
    )
    config: Final = {
        **base,
        "model_list": model_list,
        "general_settings": {**object_value(base["general_settings"]), "forward_llm_provider_auth_headers": True},
        "litellm_settings": {
            **object_value(base["litellm_settings"]),
            "model_group_settings": {"forward_client_headers_to_llm_api": [groups.forwarding]},
        },
        "router_settings": {**object_value(base["router_settings"]), "num_retries": 0, "fallbacks": fallbacks},
    }
    path: Final = directory / "fallback-caller-credential-scope.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _groups() -> Groups:
    suffix: Final = uuid.uuid4().hex[:10]
    return Groups(
        primary=f"scope-anthropic-{suffix}",
        overloaded=f"scope-anthropic-overloaded-{suffix}",
        to_mantle=f"scope-anthropic-to-mantle-{suffix}",
        forwarding=f"scope-anthropic-forwarding-{suffix}",
        to_bearer=f"scope-anthropic-to-bearer-{suffix}",
        to_anthropic=f"scope-anthropic-to-anthropic-{suffix}",
        keyless=f"scope-anthropic-keyless-{suffix}",
        bedrock=f"scope-bedrock-iam-{suffix}",
        mantle=f"scope-mantle-iam-{suffix}",
        bearer=f"scope-bedrock-bearer-{suffix}",
        anthropic=f"scope-anthropic-second-{suffix}",
    )


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    directory: Final = tmp_path_factory.mktemp("fallback-caller-credential-scope")
    groups: Final = _groups()
    with ExitStack() as stack:
        gateway: Final = stack.enter_context(gateway_from_environment())
        wires: Final = Wires(
            rejecting=stack.enter_context(wire_server(_failing_anthropic(401))),
            overloaded=stack.enter_context(wire_server(_failing_anthropic(529))),
            answering=stack.enter_context(wire_server(_answering_anthropic)),
            bedrock=stack.enter_context(wire_server(_bedrock_runtime)),
            mantle=stack.enter_context(wire_server(_bedrock_mantle)),
        )
        owned: Final = stack.enter_context(
            owned_proxy_process(
                gateway,
                directory,
                {},
                config=_config(directory, groups, wires),
                remove_environment=REMOVED_ENVIRONMENT,
                workers=2,
            )
        )
        yield Rig(owned.gateway, groups, wires)


def _anthropic_key() -> Canary:
    issued: Final = canary("D2")
    return replace(issued, value=f"sk-ant-api03-{issued.value}")


def _prompt(label: str) -> str:
    return f"{label} {uuid.uuid4().hex}"


def _send(rig: Rig, route: Route, model: str, prompt: str, headers: Mapping[str, str]) -> Answer:
    with rig.gateway.client.stream("POST", route.path, json=route.body(model, prompt), headers=dict(headers)) as reply:
        text: Final = b"".join(reply.iter_bytes()).decode()
    return Answer(reply.status_code, text)


def _litellm_key_header(rig: Rig) -> dict[str, str]:
    return {"x-litellm-api-key": rig.gateway.key}


def _bearer_litellm_key(rig: Rig) -> dict[str, str]:
    return {"Authorization": f"Bearer {rig.gateway.key}"}


def _carrying(wire: Wire, prompt: str) -> tuple[Request, ...]:
    return tuple(request for request in wire.drain() if prompt.encode() in request.body)


def _seen(request: Request) -> bytes:
    lines: Final = (request.target, *(f"{name}: {value}" for name, value in request.header_lines))
    return "\n".join(lines).encode() + b"\n" + request.body


def _only(requests: tuple[Request, ...], where: str) -> Request:
    assert len(requests) == 1, f"Positive control: expected one {where} call, got {len(requests)}"
    return requests[0]


def _assert_forwarded_to(wire: Wire, prompt: str, key: str) -> None:
    sent: Final = tuple(request.headers.get("x-api-key") for request in _carrying(wire, prompt))
    assert sent and all(value == key for value in sent), f"Positive control: the Anthropic upstream saw {sent}"


def _assert_iam_signed_without(request: Request, canaries: tuple[Canary, ...]) -> None:
    assert find_canary(_seen(request), canaries) == (), (
        f"caller credential reached the fallback: {request.target} {dict(request.headers)}"
    )
    assert request.headers.get("authorization", "").startswith(SIGV4), dict(request.headers)


def _assert_served(answer: Answer, text: str) -> None:
    assert answer.status == 200, answer.text
    assert text in answer.text, answer.text


@pytest.mark.parametrize("route", ROUTES)
def test_client_anthropic_key_stays_off_the_bedrock_fallback_regression(rig: Rig, route: Route) -> None:
    key: Final = _anthropic_key()
    prompt: Final = _prompt("bedrock fallback")
    answer: Final = _send(rig, route, rig.groups.primary, prompt, {**_litellm_key_header(rig), "x-api-key": key.value})
    _assert_forwarded_to(rig.wires.rejecting, prompt, key.value)
    _assert_iam_signed_without(_only(_carrying(rig.wires.bedrock, prompt), "Bedrock"), (key,))
    _assert_served(answer, BEDROCK_TEXT)


def test_overloaded_anthropic_falls_back_to_bedrock_without_the_client_key_regression(rig: Rig) -> None:
    key: Final = _anthropic_key()
    prompt: Final = _prompt("overloaded fallback")
    answer: Final = _send(
        rig, MESSAGES, rig.groups.overloaded, prompt, {**_bearer_litellm_key(rig), "x-api-key": key.value}
    )
    _assert_forwarded_to(rig.wires.overloaded, prompt, key.value)
    _assert_iam_signed_without(_only(_carrying(rig.wires.bedrock, prompt), "Bedrock"), (key,))
    _assert_served(answer, BEDROCK_TEXT)


def test_client_anthropic_key_stays_off_the_bedrock_mantle_fallback_regression(rig: Rig) -> None:
    key: Final = _anthropic_key()
    prompt: Final = _prompt("mantle fallback")
    answer: Final = _send(
        rig, MESSAGES, rig.groups.to_mantle, prompt, {**_bearer_litellm_key(rig), "x-api-key": key.value}
    )
    _assert_forwarded_to(rig.wires.rejecting, prompt, key.value)
    _assert_iam_signed_without(_only(_carrying(rig.wires.mantle, prompt), "Bedrock Mantle"), (key,))
    _assert_served(answer, BEDROCK_TEXT)


def test_forwarded_client_headers_stay_off_the_bedrock_fallback_regression(rig: Rig) -> None:
    key: Final = _anthropic_key()
    app: Final = canary("D3")
    prompt: Final = _prompt("forwarded headers fallback")
    answer: Final = _send(
        rig,
        CHAT,
        rig.groups.forwarding,
        prompt,
        {**_litellm_key_header(rig), "x-api-key": key.value, APP_HEADER: app.value},
    )
    primary: Final = _carrying(rig.wires.rejecting, prompt)
    assert primary and all(request.headers.get(APP_HEADER) == app.value for request in primary), (
        f"Positive control: the Anthropic upstream saw {[dict(request.headers) for request in primary]}"
    )
    fallback: Final = _only(_carrying(rig.wires.bedrock, prompt), "Bedrock")
    assert APP_HEADER not in fallback.headers, dict(fallback.headers)
    assert "x-api-key" not in fallback.headers, dict(fallback.headers)
    _assert_iam_signed_without(fallback, (key, app))
    _assert_served(answer, BEDROCK_TEXT)


def test_bearer_bedrock_fallback_sends_its_own_key_not_the_client_key_regression(rig: Rig) -> None:
    key: Final = _anthropic_key()
    prompt: Final = _prompt("bearer bedrock fallback")
    answer: Final = _send(
        rig, MESSAGES, rig.groups.to_bearer, prompt, {**_litellm_key_header(rig), "x-api-key": key.value}
    )
    _assert_forwarded_to(rig.wires.rejecting, prompt, key.value)
    fallback: Final = _only(_carrying(rig.wires.bedrock, prompt), "Bedrock")
    assert find_canary(_seen(fallback), (key,)) == (), dict(fallback.headers)
    assert fallback.headers.get("authorization") == f"Bearer {BEDROCK_DEPLOYMENT_TOKEN}", dict(fallback.headers)
    _assert_served(answer, BEDROCK_TEXT)


def test_anthropic_fallback_receives_the_client_key_guard(rig: Rig) -> None:
    key: Final = _anthropic_key()
    prompt: Final = _prompt("anthropic fallback")
    answer: Final = _send(
        rig, MESSAGES, rig.groups.to_anthropic, prompt, {**_litellm_key_header(rig), "x-api-key": key.value}
    )
    _assert_forwarded_to(rig.wires.rejecting, prompt, key.value)
    fallback: Final = _only(_carrying(rig.wires.answering, prompt), "second Anthropic")
    assert fallback.headers.get("x-api-key") == key.value, dict(fallback.headers)
    _assert_served(answer, ANTHROPIC_TEXT)


def test_keyless_anthropic_deployment_receives_the_client_key_guard(rig: Rig) -> None:
    key: Final = _anthropic_key()
    prompt: Final = _prompt("keyless anthropic")
    answer: Final = _send(
        rig, MESSAGES, rig.groups.keyless, prompt, {**_litellm_key_header(rig), "x-api-key": key.value}
    )
    upstream: Final = _only(_carrying(rig.wires.answering, prompt), "Anthropic")
    assert upstream.headers.get("x-api-key") == key.value, dict(upstream.headers)
    _assert_served(answer, ANTHROPIC_TEXT)


def test_client_key_sent_straight_to_bedrock_is_its_bearer_token_guard(rig: Rig) -> None:
    token: Final = canary("D2")
    prompt: Final = _prompt("direct bedrock")
    answer: Final = _send(
        rig, MESSAGES, rig.groups.bedrock, prompt, {**_litellm_key_header(rig), "x-api-key": token.value}
    )
    upstream: Final = _only(_carrying(rig.wires.bedrock, prompt), "Bedrock")
    assert upstream.headers.get("authorization") == f"Bearer {token.value}", dict(upstream.headers)
    _assert_served(answer, BEDROCK_TEXT)


def test_caller_without_a_provider_key_falls_back_to_bedrock_iam_guard(rig: Rig) -> None:
    prompt: Final = _prompt("no provider key fallback")
    answer: Final = _send(rig, MESSAGES, rig.groups.primary, prompt, _litellm_key_header(rig))
    _assert_forwarded_to(rig.wires.rejecting, prompt, ANTHROPIC_DEPLOYMENT_KEY)
    fallback: Final = _only(_carrying(rig.wires.bedrock, prompt), "Bedrock")
    assert fallback.headers.get("authorization", "").startswith(SIGV4), dict(fallback.headers)
    _assert_served(answer, BEDROCK_TEXT)


def _sent_betas(request: Request) -> tuple[str, ...]:
    betas: Final = JSON_OBJECT.validate_json(request.body).get("anthropic_beta")
    assert isinstance(betas, list), request.body
    return tuple(sorted(str(beta) for beta in betas))


def test_anthropic_beta_reaches_the_bedrock_fallback_like_a_direct_call_guard(rig: Rig) -> None:
    headers: Final = {**_litellm_key_header(rig), "anthropic-beta": CONTEXT_BETA}
    direct_prompt: Final = _prompt("direct beta")
    direct: Final = _send(rig, MESSAGES, rig.groups.bedrock, direct_prompt, headers)
    _assert_served(direct, BEDROCK_TEXT)
    direct_betas: Final = _sent_betas(_only(_carrying(rig.wires.bedrock, direct_prompt), "direct Bedrock"))
    assert CONTEXT_BETA in direct_betas, direct_betas

    prompt: Final = _prompt("fallback beta")
    answer: Final = _send(rig, MESSAGES, rig.groups.primary, prompt, headers)
    _assert_served(answer, BEDROCK_TEXT)
    assert _sent_betas(_only(_carrying(rig.wires.bedrock, prompt), "Bedrock fallback")) == direct_betas
