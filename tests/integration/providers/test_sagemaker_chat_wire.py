import asyncio
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

import httpx
import pytest
from integration._support.client import Gateway, Scenario, eventually, list_value, object_value, string_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers._count_tokens_system_lift import (
    anthropic_client,
    async_anthropic_client,
    async_openai_client,
    openai_client,
)
from integration.providers._sagemaker_chat_container import (
    ACCESS_KEY,
    CONTENT,
    ENDPOINT,
    JSON_OBJECT,
    NO_CACHE,
    PROMPT,
    REGION,
    SECRET_KEY,
    SERVED_MODEL,
    answered_choices,
    assert_signed,
    by_marker,
    container_body,
    container_reply,
    marked_prompt,
    marker,
    only_request,
)
from pydantic import JsonValue

_INFERENCE_COMPONENT: Final = "integration-vllm-component"
_SESSION_TOKEN: Final = "synthetic-session-token-for-testing"
_CALLER_ACCESS_KEY: Final = "AKIAINTEGRATIONCALLER"
_CALLER_SECRET_KEY: Final = "synthetic-caller-secret-key-for-testing"
_DEPLOYMENT_AUTH_KEYS: Final = frozenset({"aws_access_key_id", "aws_secret_access_key", "aws_region_name"})

Endpoint: TypeAlias = Literal["chat", "messages", "responses"]

_ENDPOINTS: Final[tuple[Endpoint, ...]] = ("chat", "messages", "responses")
_PATHS: Final[Mapping[Endpoint, str]] = MappingProxyType(
    {"chat": "/v1/chat/completions", "messages": "/v1/messages", "responses": "/v1/responses"}
)


def _deployment(
    scenario: Scenario, wire: Wire, extra: Mapping[str, JsonValue], *, provider: str = "sagemaker_chat"
) -> str:
    return scenario.model(
        model_info=None,
        model=f"{provider}/{ENDPOINT}",
        api_key=None,
        api_base=None,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        aws_region_name=REGION,
        sagemaker_base_url=wire.url,
        **extra,
    )


def _chat(gateway: Gateway, model: str, prompt: str, caller: Mapping[str, JsonValue]) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 16, **NO_CACHE, **caller},
    )


def _spend_row(call_id: str) -> Mapping[str, JsonValue]:
    (row,) = eventually(
        lambda: read_rows(
            'SELECT status, model_group, row_to_json(s)::text AS stored FROM "LiteLLM_SpendLogs" s '
            "WHERE litellm_call_id = %s",
            (call_id,),
        ),
        lambda rows: len(rows) == 1,
        seconds=70,
    )
    return row


@dataclass(frozen=True, slots=True)
class _Answer:
    call_id: str
    text: str


def _chat_openai_sync(gateway: Gateway, model: str, prompt: str) -> _Answer:
    with openai_client(gateway) as client:
        raw: Final = client.chat.completions.with_raw_response.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=16,
            extra_body=dict(NO_CACHE),
        )
        completion: Final = raw.parse()
    return _Answer(raw.headers["x-litellm-call-id"], completion.choices[0].message.content or "")


async def _chat_openai_async_stream(gateway: Gateway, model: str, prompt: str) -> _Answer:
    async with async_openai_client(gateway) as client:
        stream: Final = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=16,
            stream=True,
            stream_options={"include_usage": True},
            extra_body=dict(NO_CACHE),
        )
        text: Final = "".join([chunk.choices[0].delta.content or "" async for chunk in stream if chunk.choices])
    return _Answer(stream.response.headers["x-litellm-call-id"], text)


def _messages_anthropic_sync(gateway: Gateway, model: str, prompt: str) -> _Answer:
    with anthropic_client(gateway) as client:
        raw: Final = client.messages.with_raw_response.create(
            model=model,
            max_tokens=16,
            messages=[{"role": "user", "content": prompt}],
            extra_body=dict(NO_CACHE),
        )
        message: Final = raw.parse()
    return _Answer(
        raw.headers["x-litellm-call-id"], "".join(block.text for block in message.content if block.type == "text")
    )


async def _messages_anthropic_async_stream(gateway: Gateway, model: str, prompt: str) -> _Answer:
    async with async_anthropic_client(gateway) as client:
        stream: Final = await client.messages.create(
            model=model,
            max_tokens=16,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            extra_body=dict(NO_CACHE),
        )
        deltas: Final = [event.delta async for event in stream if event.type == "content_block_delta"]
    return _Answer(
        stream.response.headers["x-litellm-call-id"],
        "".join(delta.text for delta in deltas if delta.type == "text_delta"),
    )


def _responses_httpx(gateway: Gateway, model: str, prompt: str) -> _Answer:
    response: Final = gateway.request(
        "POST", "/v1/responses", {"model": model, "input": prompt, "max_output_tokens": 16, **NO_CACHE}
    )
    assert response.status_code == 200, response.text
    output: Final = tuple(
        object_value(item) for item in list_value(JSON_OBJECT.validate_json(response.content)["output"])
    )
    (message,) = (item for item in output if item["type"] == "message")
    text: Final = "".join(string_value(object_value(part)["text"]) for part in list_value(message["content"]))
    return _Answer(response.headers["x-litellm-call-id"], text)


def _responses_httpx_stream(gateway: Gateway, model: str, prompt: str) -> _Answer:
    with gateway.client.stream(
        "POST",
        "/v1/responses",
        json={"model": model, "input": prompt, "max_output_tokens": 16, "stream": True, **NO_CACHE},
        headers={"Authorization": f"Bearer {gateway.key}"},
    ) as response:
        assert response.status_code == 200, response.read()
        lines: Final = tuple(response.iter_lines())
    events: Final = tuple(
        JSON_OBJECT.validate_json(line.removeprefix("data: ")) for line in lines if line.startswith("data: {")
    )
    assert [event["type"] for event in events if event["type"] == "response.completed"] == ["response.completed"], lines
    text: Final = "".join(
        string_value(event["delta"]) for event in events if event["type"] == "response.output_text.delta"
    )
    return _Answer(response.headers["x-litellm-call-id"], text)


def _chat_openai_stream(gateway: Gateway, model: str, prompt: str) -> _Answer:
    return asyncio.run(_chat_openai_async_stream(gateway, model, prompt))


def _messages_anthropic_stream(gateway: Gateway, model: str, prompt: str) -> _Answer:
    return asyncio.run(_messages_anthropic_async_stream(gateway, model, prompt))


@dataclass(frozen=True, slots=True)
class _Driver:
    run: Callable[[Gateway, str, str], _Answer]
    stream: bool
    forwarded: Mapping[str, JsonValue]


_STREAM_WITH_USAGE: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {"stream": True, "stream_options": {"include_usage": True}}
)
_DRIVERS: Final[Mapping[str, _Driver]] = MappingProxyType(
    {
        "chat-openai-sync": _Driver(_chat_openai_sync, False, {"stream": False}),
        "chat-openai-async-stream": _Driver(_chat_openai_stream, True, _STREAM_WITH_USAGE),
        "messages-anthropic-sync": _Driver(_messages_anthropic_sync, False, {}),
        "messages-anthropic-async-stream": _Driver(_messages_anthropic_stream, True, _STREAM_WITH_USAGE),
        "responses-httpx": _Driver(_responses_httpx, False, {}),
        "responses-httpx-stream": _Driver(_responses_httpx_stream, True, _STREAM_WITH_USAGE),
    }
)


@pytest.mark.covers(
    "providers.sagemaker_chat_wire.inference_component_header_is_signed_and_hf_model_name_is_the_body_model"
)
def test_sagemaker_chat_signs_the_inference_component_header_and_sends_hf_model_name_as_the_body_model(
    gateway: Gateway,
) -> None:
    identity: Final = f"sagemaker-chat-{uuid.uuid4().hex}"

    with wire_server(lambda _request: container_reply(identity, False)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, {"model_id": _INFERENCE_COMPONENT, "hf_model_name": SERVED_MODEL})
        response: Final = _chat(gateway, model, PROMPT, {})
        request: Final = only_request(wire)
    assert response.status_code == 200, response.text
    payload: Final = JSON_OBJECT.validate_json(response.content)
    assert payload["id"] == identity, response.text
    assert payload["choices"] == answered_choices(), response.text
    assert request.headers["x-amzn-sagemaker-inference-component"] == _INFERENCE_COMPONENT, dict(request.headers)
    assert_signed(request, ACCESS_KEY, REGION, None)
    signed_headers: Final = next(
        part for part in request.headers["authorization"].split(", ") if part.startswith("SignedHeaders=")
    )
    assert "x-amzn-sagemaker-inference-component" in signed_headers.removeprefix("SignedHeaders=").split(";")
    assert JSON_OBJECT.validate_json(request.body) == {
        **container_body(wire.url, PROMPT, stream=False, model_id=_INFERENCE_COMPONENT),
        "model": SERVED_MODEL,
    }, request.body


@pytest.mark.parametrize("driver", tuple(_DRIVERS))
def test_sagemaker_chat_keeps_the_deployment_aws_credentials_out_of_the_container_body_on_every_endpoint(
    gateway: Gateway, driver: str
) -> None:
    chosen: Final = _DRIVERS[driver]
    identity: Final = f"sagemaker-chat-{uuid.uuid4().hex}"
    prompt: Final = marked_prompt()

    with wire_server(lambda _request: container_reply(identity, chosen.stream)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, {})
        answer: Final = chosen.run(gateway, model, prompt)
        request: Final = only_request(wire)
    assert answer.text == "".join(CONTENT), answer
    assert JSON_OBJECT.validate_json(request.body) == container_body(wire.url, prompt, **chosen.forwarded), request.body
    assert_signed(request, ACCESS_KEY, REGION, None)
    row: Final = _spend_row(answer.call_id)
    assert (row["status"], row["model_group"]) == ("success", model), row
    assert SECRET_KEY not in string_value(row["stored"]), row


@dataclass(frozen=True, slots=True)
class _Shape:
    provider: str
    deployment: Mapping[str, JsonValue]
    caller: Mapping[str, JsonValue]
    forwarded: Mapping[str, JsonValue]
    signer: str
    region: str


def _caller_flag(value: JsonValue) -> _Shape:
    return _Shape("sagemaker_chat", {}, {"aws_custom_flag": value}, {"aws_custom_flag": value}, ACCESS_KEY, REGION)


_SHAPES: Final[Mapping[str, _Shape]] = MappingProxyType(
    {
        "deployment-session-token": _Shape(
            "sagemaker_chat", {"aws_session_token": _SESSION_TOKEN}, {}, {}, ACCESS_KEY, REGION
        ),
        "nova-deployment": _Shape("sagemaker_nova", {}, {}, {}, ACCESS_KEY, REGION),
        "caller-top-k": _Shape("sagemaker_chat", {}, {"top_k": 5}, {"top_k": 5}, ACCESS_KEY, REGION),
        "caller-aws-custom-flag": _caller_flag("keep"),
        "caller-aws-custom-flag-int": _caller_flag(7),
        "caller-aws-custom-flag-list": _caller_flag(["a", 1]),
        "caller-aws-custom-flag-empty": _caller_flag(""),
        "caller-aws-custom-flag-5kb": _caller_flag("x" * 5120),
        "caller-aws-region-name": _Shape(
            "sagemaker_chat", {}, {"aws_region_name": "us-west-2"}, {}, ACCESS_KEY, "us-west-2"
        ),
        "caller-own-credentials": _Shape(
            "sagemaker_chat",
            {},
            {"aws_access_key_id": _CALLER_ACCESS_KEY, "aws_secret_access_key": _CALLER_SECRET_KEY},
            {},
            _CALLER_ACCESS_KEY,
            REGION,
        ),
    }
)


@pytest.mark.parametrize("shape", tuple(_SHAPES))
def test_sagemaker_chat_forwards_only_non_credential_params_for_each_deployment_and_caller_shape(
    gateway: Gateway, shape: str
) -> None:
    case: Final = _SHAPES[shape]
    identity: Final = f"sagemaker-chat-{uuid.uuid4().hex}"
    prompt: Final = marked_prompt()

    with wire_server(lambda _request: container_reply(identity, False)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, case.deployment, provider=case.provider)
        response: Final = _chat(gateway, model, prompt, case.caller)
        request: Final = only_request(wire)
    assert response.status_code == 200, response.text
    assert JSON_OBJECT.validate_json(response.content)["id"] == identity, response.text
    expected: Final = container_body(wire.url, prompt, stream=False, **case.forwarded)
    assert JSON_OBJECT.validate_json(request.body) == (
        expected if case.provider == "sagemaker_chat" else {k: v for k, v in expected.items() if k != "model"}
    ), request.body
    session_token: Final = case.deployment.get("aws_session_token")
    assert_signed(request, case.signer, case.region, session_token if isinstance(session_token, str) else None)


@pytest.mark.parametrize("caller", ("proxy-admin", "model-scoped-key"))
def test_sagemaker_chat_failed_health_check_shows_no_deployment_aws_credentials(gateway: Gateway, caller: str) -> None:
    failure: Final = Reply(status=500, body=b'{"message":"synthetic container failure"}')

    with wire_server(lambda _request: failure) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, {})
        key: Final = gateway.key if caller == "proxy-admin" else scenario.key(models=[model])
        response: Final = gateway.request("GET", "/health", params={"model": model}, key=key)
        probes: Final = wire.drain()
    assert response.status_code == 503, response.text
    assert SECRET_KEY not in response.text
    payload: Final = JSON_OBJECT.validate_json(response.content)
    assert payload["healthy_count"] == 0, response.text
    (unhealthy,) = list_value(payload["unhealthy_endpoints"])
    endpoint: Final = object_value(unhealthy)
    assert "synthetic container failure" in string_value(endpoint["error"]), response.text
    shown_body: Final = object_value(object_value(endpoint["raw_request_typed_dict"])["raw_request_body"])
    assert shown_body["model"] == ENDPOINT, response.text
    assert not _DEPLOYMENT_AUTH_KEYS & set(shown_body), response.text
    assert [JSON_OBJECT.validate_json(probe.body) for probe in probes] == [shown_body], probes


def test_transform_request_preview_of_a_sagemaker_chat_call_shows_no_aws_credentials(gateway: Gateway) -> None:
    prompt: Final = marked_prompt()
    response: Final = gateway.request(
        "POST",
        "/utils/transform_request",
        {
            "call_type": "completion",
            "request_body": {
                "model": f"sagemaker_chat/{ENDPOINT}",
                "messages": [{"role": "user", "content": prompt}],
                "aws_access_key_id": _CALLER_ACCESS_KEY,
                "aws_secret_access_key": _CALLER_SECRET_KEY,
                "aws_region_name": REGION,
            },
        },
    )
    assert response.status_code == 200, response.text
    assert _CALLER_SECRET_KEY not in response.text
    payload: Final = JSON_OBJECT.validate_json(response.content)
    assert payload["raw_request_body"] == {"model": ENDPOINT, "messages": [{"role": "user", "content": prompt}]}, (
        response.text
    )
    assert string_value(payload["raw_request_api_base"]).endswith(f"/endpoints/{ENDPOINT}/invocations"), response.text


def test_sagemaker_chat_container_error_reaches_the_caller_after_one_attempt_without_credentials(
    gateway: Gateway,
) -> None:
    prompt: Final = marked_prompt()
    failure: Final = Reply(
        status=424, body=b'{"ErrorCode":"CLIENT_ERROR_FROM_MODEL","Message":"synthetic model error"}'
    )

    with wire_server(lambda _request: failure) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, {})
        response: Final = _chat(gateway, model, prompt, {})
        request: Final = only_request(wire)
    assert response.status_code == 424, response.text
    assert "synthetic model error" in response.text, response.text
    assert JSON_OBJECT.validate_json(request.body) == container_body(wire.url, prompt, stream=False), request.body
    assert_signed(request, ACCESS_KEY, REGION, None)


@dataclass(frozen=True, slots=True)
class _Call:
    endpoint: Endpoint
    stream: bool
    marker: str


@dataclass(frozen=True, slots=True)
class _Served:
    call: _Call
    status: int
    text: str
    call_id: str


def _calls(count: int) -> tuple[_Call, ...]:
    return tuple(
        _Call(endpoint=_ENDPOINTS[index % len(_ENDPOINTS)], stream=index % 2 == 1, marker=marker(marked_prompt()))
        for index in range(count)
    )


def _request_body(model: str, call: _Call) -> Mapping[str, JsonValue]:
    prompt: Final = f"{PROMPT} marker-{call.marker}"
    common: Final[Mapping[str, JsonValue]] = {"model": model, "stream": call.stream, **NO_CACHE}
    match call.endpoint:
        case "chat" | "messages":
            return {**common, "max_tokens": 16, "messages": [{"role": "user", "content": prompt}]}
        case "responses":
            return {**common, "max_output_tokens": 16, "input": prompt}


async def _send(client: httpx.AsyncClient, key: str, model: str, call: _Call) -> _Served:
    async with client.stream(
        "POST",
        _PATHS[call.endpoint],
        json=_request_body(model, call),
        headers={"Authorization": f"Bearer {key}", "anthropic-version": "2023-06-01"},
    ) as response:
        raw: Final = await response.aread()
    return _Served(call, response.status_code, raw.decode(), response.headers["x-litellm-call-id"])


async def _burst(gateway: Gateway, model: str, calls: tuple[_Call, ...]) -> tuple[_Served, ...]:
    async with httpx.AsyncClient(base_url=str(gateway.client.base_url), timeout=60, trust_env=False) as client:
        return tuple(await asyncio.gather(*(_send(client, gateway.key, model, call) for call in calls)))


def _assert_answered(served: _Served, down: frozenset[str]) -> None:
    if served.call.marker in down:
        assert served.status == 503, (served.call, served.text)
        assert "synthetic container outage" in served.text, (served.call, served.text)
        return
    assert served.status == 200, (served.call, served.text)
    assert all(part in served.text for part in CONTENT), (served.call, served.text)


def _assert_clean(request: Request) -> None:
    assert not _DEPLOYMENT_AUTH_KEYS & set(JSON_OBJECT.validate_json(request.body)), request.body
    assert SECRET_KEY.encode() not in request.body, request.body


async def test_sagemaker_chat_container_outage_mid_burst_answers_every_call_once_without_credentials(
    gateway: Gateway,
) -> None:
    burst: Final = _calls(24)
    recovery: Final = _calls(12)
    down: Final = frozenset(call.marker for index, call in enumerate(burst) if (index // 2) % 2 == 0)
    streams: Final = MappingProxyType({call.marker: call.stream for call in (*burst, *recovery)})

    def respond(request: Request) -> Reply:
        found: Final = marker(request.body.decode())
        if found in down:
            return Reply(status=503, body=b'{"message":"synthetic container outage"}')
        return container_reply(f"sagemaker-chat-{found}", streams[found])

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, {})
        answered: Final = (*await _burst(gateway, model, burst), *await _burst(gateway, model, recovery))
        forwarded: Final = by_marker(wire.drain())
    assert set(forwarded) == set(streams), sorted(forwarded)
    for request in forwarded.values():
        _assert_clean(request)
    for served in answered:
        _assert_answered(served, down)
    rows: Final = eventually(
        lambda: read_rows('SELECT litellm_call_id, status FROM "LiteLLM_SpendLogs" WHERE model_group = %s', (model,)),
        lambda found: len(found) >= len(answered),
        seconds=70,
    )
    statuses: Final = {string_value(row["litellm_call_id"]): row["status"] for row in rows}
    assert len(statuses) == len(rows) == len(answered), rows
    assert statuses == {
        served.call_id: "failure" if served.call.marker in down else "success" for served in answered
    }, rows
