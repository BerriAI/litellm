import asyncio
import base64
import json
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from types import MappingProxyType
from typing import Final, Literal, NamedTuple
from urllib.parse import unquote

import anthropic
import httpx
import openai
import pytest
from integration._support.client import Gateway, Scenario, eventually, object_value
from integration._support.database import read_rows
from integration._support.responses_stream import (
    created,
    delta,
    error_event,
    failed,
    frame,
    healthy_stream,
    openai_error_event,
    spend_row,
)
from integration._support.upstream import _aws_event_frame, aws_event_stream_frame
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

import litellm
from litellm.exceptions import MidStreamFallbackError
from litellm.litellm_core_utils.litellm_logging import _get_provider_request_id

_MODEL: Final = "bedrock/converse/anthropic.claude-sonnet-4-5-20250929-v1:0"
_BEDROCK_RESPONSES_MODEL: Final = "bedrock/us.openai.gpt-5.6-sol"
_TOKEN: Final = "synthetic-bedrock-bearer"
_SHAPES: Final = ("nested-error", "top-level-error", "response.failed", "error-before-output")
_ERROR_CODES: Final = ("server_error", "rate_limit_exceeded", "invalid_request_error", "content_policy_violation")
_INVOKE_MODEL_ID: Final = "anthropic.claude-3-haiku-20240307-v1:0"
_BEDROCK_AWS: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {
        "aws_access_key_id": "AKIASCRIPTEDPROVIDER",
        "aws_secret_access_key": "scripted-secret",
        "aws_region_name": "us-east-1",
    }
)
_EVENT_STREAM: Final = "application/vnd.amazon.eventstream"
_BEDROCK_STREAM_TARGETS: Final = frozenset(
    (
        "/model/anthropic.claude-sonnet-4-5-20250929-v1:0/converse-stream",
        f"/model/{_INVOKE_MODEL_ID}/invoke-with-response-stream",
    )
)

BedrockFailure = Literal["exception-event", "exception-message", "no-known-event", "truncated-frame"]


class _FailureShape(NamedTuple):
    bedrock_status: int
    converse_class: str
    after_output: tuple[bool, ...]


BedrockEndpoint = Literal["chat", "responses", "messages", "messages-invoke"]
_BEDROCK_ENDPOINTS: Final[tuple[BedrockEndpoint, ...]] = ("chat", "responses", "messages", "messages-invoke")
_BEDROCK_FAILURES: Final[Mapping[BedrockFailure, _FailureShape]] = MappingProxyType(
    {
        "exception-event": _FailureShape(500, "ServiceUnavailableError", (False, True)),
        "exception-message": _FailureShape(429, "RateLimitError", (False, True)),
        "no-known-event": _FailureShape(502, "BadGatewayError", (False,)),
        "truncated-frame": _FailureShape(502, "BadGatewayError", (True,)),
    }
)
_BEDROCK_CELLS: Final = tuple(
    pytest.param(
        endpoint, failure, after_output, id=f"{endpoint}-{failure}-{'after' if after_output else 'before'}-output"
    )
    for endpoint in _BEDROCK_ENDPOINTS
    for failure, shape in _BEDROCK_FAILURES.items()
    for after_output in shape.after_output
)


def _stream(identity: str, shape: str, error_code: str, message: str) -> tuple[bytes, ...]:
    opening: Final = (frame(created(identity)), frame(delta(identity, "partial output")))
    nested: Final = frame(error_event({"type": error_code, "code": error_code, "message": message}))
    match shape:
        case "nested-error":
            return (*opening, nested)
        case "top-level-error":
            return (*opening, frame(openai_error_event(error_code, message)))
        case "response.failed":
            return (*opening, frame(failed(identity, error_code, message)))
        case "error-before-output":
            return (nested,)
    raise AssertionError(shape)


def _bedrock_frame(event_type: str, payload: dict[str, JsonValue]) -> bytes:
    return _aws_event_frame(event_type, payload, "sc", "u")


def _invoke_chunk(event: dict[str, JsonValue]) -> bytes:
    return _bedrock_frame("chunk", {"bytes": base64.b64encode(json.dumps(event).encode()).decode()})


def _bedrock_output(invoke: bool, text: str = "partial output") -> tuple[bytes, ...]:
    if not invoke:
        return (
            _bedrock_frame("messageStart", {"role": "assistant"}),
            _bedrock_frame("contentBlockDelta", {"delta": {"text": text}, "contentBlockIndex": 0}),
        )
    started: Final[dict[str, JsonValue]] = {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "content": [],
        "model": _INVOKE_MODEL_ID,
        "stop_reason": None,
        "usage": {"input_tokens": 3, "output_tokens": 1},
    }
    return (
        _invoke_chunk({"type": "message_start", "message": started}),
        _invoke_chunk({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        _invoke_chunk({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}}),
    )


def _bedrock_failure_frame(failure: BedrockFailure, message: str) -> bytes:
    match failure:
        case "exception-event":
            return _bedrock_frame("internalServerException", {"message": message})
        case "exception-message":
            headers: Final = {
                ":exception-type": "throttlingException",
                ":content-type": "application/json",
                ":message-type": "exception",
            }
            return aws_event_stream_frame(headers, json.dumps({"message": message}).encode())
        case "no-known-event":
            return _bedrock_frame("somethingBedrockAddedLater", {"message": message})
        case "truncated-frame":
            return _bedrock_frame("contentBlockDelta", {"delta": {"text": message}, "contentBlockIndex": 0})[:-7]


def _bedrock_failure_text(failure: BedrockFailure, message: str) -> str:
    return "undecoded bytes after" if failure == "truncated-frame" else message


def _bedrock_stream_request(
    endpoint: BedrockEndpoint, model: str, prompt: str, *, stream: bool = True
) -> tuple[str, dict[str, JsonValue]]:
    extra: Final[dict[str, JsonValue]] = {"stream": stream, "num_retries": 0, "cache": {"no-cache": True}}
    turn: Final[list[JsonValue]] = [{"role": "user", "content": prompt}]
    match endpoint:
        case "chat":
            return "/v1/chat/completions", {"model": model, "messages": turn, **extra}
        case "responses":
            return "/v1/responses", {"model": model, "input": prompt, **extra}
        case "messages" | "messages-invoke":
            return "/v1/messages", {"model": model, "messages": turn, "max_tokens": 16, **extra}


def _failure_information(call_id: str) -> dict[str, JsonValue]:
    row: Final = spend_row(call_id)
    assert row["status"] == "failure", row
    metadata: Final = row["metadata"]
    parsed: Final = json.loads(metadata) if isinstance(metadata, str) else object_value(metadata)
    return object_value(parsed["error_information"])


def _streamed_response(gateway: Gateway, model: str, prompt: str) -> httpx.Response:
    response: Final = gateway.request(
        "POST",
        "/v1/responses",
        {"model": model, "input": prompt, "stream": True, "num_retries": 0, "cache": {"no-cache": True}},
    )
    assert response.status_code == 200, response.text
    return response


async def _consume_sdk_stream(asynchronous: bool, api_base: str, identity: str) -> None:
    if asynchronous:
        async_stream: Final = await litellm.aresponses(
            model="openai/gpt-6",
            input=identity,
            stream=True,
            api_base=api_base,
            api_key="synthetic-provider-key",
            num_retries=0,
        )
        async for _ in async_stream:
            pass
        return
    tuple(
        litellm.responses(
            model="openai/gpt-6",
            input=identity,
            stream=True,
            api_base=api_base,
            api_key="synthetic-provider-key",
            num_retries=0,
        )
    )


def test_bedrock_500_keeps_amzn_request_id_on_error_headers_and_failure_log(gateway: Gateway) -> None:
    identity: Final = f"bedrock-request-id-{uuid.uuid4().hex}"
    amzn_request_id: Final = str(uuid.uuid4())
    prompt: Final = f"failure probe {identity}"

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/model/anthropic.claude-sonnet-4-5-20250929-v1%3A0/converse", request.target
        return Reply(
            status=500,
            headers={"x-amzn-RequestId": amzn_request_id},
            body=b'{"message":"synthetic bedrock failure"}',
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=_MODEL,
            api_key=_TOKEN,
            aws_region_name="us-east-1",
            aws_bedrock_runtime_endpoint=wire.url,
            num_retries=0,
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": prompt}]},
        )
        assert response.status_code >= 400, response.text
        assert response.headers.get("llm_provider-x-amzn-requestid") == amzn_request_id, dict(response.headers)
        call_id: Final = response.headers["x-litellm-call-id"]
        assert len(wire.drain()) == 1
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT status, metadata FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (call_id,)
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        row: Final = rows[0]
        assert row["status"] == "failure", row
        metadata: Final = row["metadata"]
        parsed: Final = json.loads(metadata) if isinstance(metadata, str) else object_value(metadata)
        error_information: Final = object_value(parsed["error_information"])
        assert error_information["error_provider_request_id"] == amzn_request_id, error_information


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("error_code", _ERROR_CODES)
def test_responses_stream_failure_log_retains_provider_request_id(
    gateway: Gateway, shape: str, error_code: str
) -> None:
    identity: Final = f"resp_{uuid.uuid4().hex}"
    provider_request_id: Final = str(uuid.uuid4())
    message: Final = f"stream failure {identity}"

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/responses", request.target
        assert json.loads(request.body)["stream"] is True
        return Reply(
            headers={"x-amzn-RequestId": provider_request_id},
            content_type="text/event-stream",
            chunks=_stream(identity, shape, error_code, message),
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/gpt-6", api_base=wire.url, num_retries=0)
        response: Final = _streamed_response(gateway, model, identity)
        assert message in response.text, response.text
        assert ("partial output" in response.text) is (shape != "error-before-output"), response.text
        assert response.headers.get("llm_provider-x-amzn-requestid") == provider_request_id, dict(response.headers)
        assert len(wire.drain()) == 1
        information: Final = _failure_information(response.headers["x-litellm-call-id"])
        assert information["error_provider_request_id"] == provider_request_id, information


@pytest.mark.parametrize("shape", _SHAPES)
def test_responses_stream_failure_without_provider_request_id_logs_none(gateway: Gateway, shape: str) -> None:
    identity: Final = f"resp_{uuid.uuid4().hex}"
    message: Final = f"stream failure {identity}"

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/responses", request.target
        return Reply(content_type="text/event-stream", chunks=_stream(identity, shape, "server_error", message))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/gpt-6", api_base=wire.url, num_retries=0)
        response: Final = _streamed_response(gateway, model, identity)
        assert message in response.text, response.text
        assert "llm_provider-x-amzn-requestid" not in response.headers, dict(response.headers)
        assert len(wire.drain()) == 1
        information: Final = _failure_information(response.headers["x-litellm-call-id"])
        assert information["error_class"] == "InternalServerError", information
        assert information["error_provider_request_id"] is None, information


@pytest.mark.parametrize("shape", ("nested-error", "response.failed"))
def test_bedrock_responses_stream_server_error_logs_amzn_request_id(gateway: Gateway, shape: str) -> None:
    identity: Final = f"resp_{uuid.uuid4().hex}"
    amzn_request_id: Final = str(uuid.uuid4())
    message: Final = f"bedrock stream failure {identity}"

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/openai/v1/responses", request.target
        assert request.headers["authorization"] == f"Bearer {_TOKEN}", request.headers
        assert json.loads(request.body)["stream"] is True
        return Reply(
            headers={"x-amzn-RequestId": amzn_request_id},
            content_type="text/event-stream",
            chunks=_stream(identity, shape, "server_error", message),
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=_BEDROCK_RESPONSES_MODEL,
            api_key=_TOKEN,
            api_base=wire.url,
            aws_region_name="us-east-1",
            num_retries=0,
        )
        response: Final = _streamed_response(gateway, model, identity)
        assert "partial output" in response.text and message in response.text, response.text
        assert response.headers.get("llm_provider-x-amzn-requestid") == amzn_request_id, dict(response.headers)
        assert len(wire.drain()) == 1
        information: Final = _failure_information(response.headers["x-litellm-call-id"])
        assert information["llm_provider"] == "bedrock", information
        assert information["error_provider_request_id"] == amzn_request_id, information


def test_responses_stream_failure_log_reads_openai_x_request_id(gateway: Gateway) -> None:
    identity: Final = f"resp_{uuid.uuid4().hex}"
    provider_request_id: Final = f"req_{uuid.uuid4().hex}"
    message: Final = f"stream failure {identity}"

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/responses", request.target
        return Reply(
            headers={"x-request-id": provider_request_id},
            content_type="text/event-stream",
            chunks=_stream(identity, "top-level-error", "server_error", message),
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/gpt-6", api_base=wire.url, num_retries=0)
        response: Final = _streamed_response(gateway, model, identity)
        assert message in response.text, response.text
        assert len(wire.drain()) == 1
        information: Final = _failure_information(response.headers["x-litellm-call-id"])
        assert information["error_provider_request_id"] == provider_request_id, information


@pytest.mark.parametrize("asynchronous", (False, True), ids=("sync", "async"))
@pytest.mark.parametrize("event_type", ("error", "response.failed"))
async def test_responses_sdk_stream_error_retains_provider_request_id(asynchronous: bool, event_type: str) -> None:
    identity: Final = f"resp_{uuid.uuid4().hex}"
    provider_request_id: Final = str(uuid.uuid4())
    shape: Final = "nested-error" if event_type == "error" else "response.failed"

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/responses", request.target
        return Reply(
            headers={"x-amzn-RequestId": provider_request_id, "retry-after": "7"},
            content_type="text/event-stream",
            chunks=_stream(identity, shape, "server_error", identity),
        )

    with wire_server(respond) as wire:
        with pytest.raises(MidStreamFallbackError) as raised:
            await _consume_sdk_stream(asynchronous, wire.url, identity)
        assert raised.value.status_code == 500
        assert raised.value.response.status_code == 500
        assert str(raised.value.response.request.url) == "https://openai.com/v1/"
        assert raised.value.generated_content == "partial output"
        assert dict(raised.value.response.headers) == {"x-amzn-requestid": provider_request_id}
        assert _get_provider_request_id(raised.value) == provider_request_id
        assert raised.value.original_exception is not None
        assert _get_provider_request_id(raised.value.original_exception) == provider_request_id
        assert len(wire.drain()) == 1


def _concurrent_reply(request: Request, failing: tuple[str, ...], recovery: str) -> Reply:
    prompt: Final = json.loads(request.body)["input"]
    assert prompt in (*failing, recovery)
    chunks: Final = (
        healthy_stream(f"resp_{prompt}", "recovered")
        if prompt == recovery
        else _stream(f"resp_{prompt}", "nested-error", "server_error", prompt)
    )
    return Reply(content_type="text/event-stream", headers={"x-amzn-requestid": prompt}, chunks=chunks)


def test_concurrent_responses_failures_keep_distinct_request_ids_and_recover(gateway: Gateway) -> None:
    identity: Final = uuid.uuid4().hex
    prompts: Final = tuple(f"{identity}-{index}" for index in range(4))
    recovery: Final = f"{identity}-recovery"

    with (
        wire_server(lambda request: _concurrent_reply(request, prompts, recovery)) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(model="openai/gpt-6", api_base=wire.url, num_retries=0)

        def call(prompt: str) -> str:
            response: Final = _streamed_response(gateway, model, prompt)
            assert (prompt if prompt != recovery else "recovered") in response.text, response.text
            return response.headers["x-litellm-call-id"]

        with ThreadPoolExecutor(max_workers=4) as workers:
            call_ids: Final = tuple(workers.map(call, prompts))
        recovered_call_id: Final = call(recovery)
        assert len(wire.drain()) == 5
        for prompt, call_id in zip(prompts, call_ids, strict=True):
            assert _failure_information(call_id)["error_provider_request_id"] == prompt
        assert spend_row(recovered_call_id)["status"] == "success"


def _bedrock_logged_error(endpoint: BedrockEndpoint, failure: BedrockFailure) -> tuple[str, str]:
    shape: Final = _BEDROCK_FAILURES[failure]
    if endpoint == "messages-invoke":
        return "BedrockError", str(shape.bedrock_status)
    return shape.converse_class, "503" if failure == "exception-event" else str(shape.bedrock_status)


def _bedrock_client_status(endpoint: BedrockEndpoint, failure: BedrockFailure, after_output: bool) -> int:
    logged_code: Final = int(_bedrock_logged_error(endpoint, failure)[1])
    if after_output or endpoint in ("responses", "messages"):
        return 200
    if endpoint == "messages-invoke" and failure == "no-known-event":
        return 200
    return logged_code


@pytest.mark.parametrize(("endpoint", "failure", "after_output"), _BEDROCK_CELLS)
def test_bedrock_event_stream_failure_log_retains_amzn_request_id(
    gateway: Gateway, endpoint: BedrockEndpoint, failure: BedrockFailure, after_output: bool
) -> None:
    amzn_request_id: Final = str(uuid.uuid4())
    message: Final = f"bedrock stream failure {amzn_request_id}"
    invoke: Final = endpoint == "messages-invoke"
    frames: Final = (*(_bedrock_output(invoke) if after_output else ()), _bedrock_failure_frame(failure, message))

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and unquote(request.target) in _BEDROCK_STREAM_TARGETS, request.target
        return Reply(
            headers={"x-amzn-RequestId": amzn_request_id, "retry-after": "7"},
            content_type=_EVENT_STREAM,
            body=b"".join(frames),
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"bedrock/invoke/{_INVOKE_MODEL_ID}" if invoke else _MODEL,
            api_base=wire.url,
            num_retries=0,
            **_BEDROCK_AWS,
        )
        path, body = _bedrock_stream_request(endpoint, model, message)
        with gateway.client.stream(
            "POST", path, json=body, headers={"Authorization": f"Bearer {gateway.key}"}
        ) as response:
            text: Final = "\n".join(line for line in response.iter_lines() if line)
            status: Final = response.status_code
            response_headers: Final = dict(response.headers)
        assert status == _bedrock_client_status(endpoint, failure, after_output), text
        assert _bedrock_failure_text(failure, message) in text and ("partial output" in text) is after_output, text
        assert len(wire.drain()) == 1
        information: Final = _failure_information(response_headers["x-litellm-call-id"])
        assert (information["error_class"], information["error_code"]) == _bedrock_logged_error(endpoint, failure)
        assert information["error_provider_request_id"] == amzn_request_id, information
        if status != 200 and not invoke:
            assert response_headers.get("llm_provider-x-amzn-requestid") == amzn_request_id, response_headers
            assert "llm_provider-retry-after" not in response_headers, response_headers


def _bedrock_model(scenario: Scenario, endpoint: BedrockEndpoint, api_base: str) -> str:
    return scenario.model(
        model=f"bedrock/invoke/{_INVOKE_MODEL_ID}" if endpoint == "messages-invoke" else _MODEL,
        api_base=api_base,
        num_retries=0,
        **_BEDROCK_AWS,
    )


def _bedrock_stream(
    gateway: Gateway, endpoint: BedrockEndpoint, model: str, prompt: str
) -> tuple[int, str, dict[str, str]]:
    path, body = _bedrock_stream_request(endpoint, model, prompt)
    with gateway.client.stream("POST", path, json=body, headers={"Authorization": f"Bearer {gateway.key}"}) as response:
        text: Final = "\n".join(line for line in response.iter_lines() if line)
        return response.status_code, text, dict(response.headers)


@pytest.mark.parametrize("after_output", (False, True), ids=("before-output", "after-output"))
@pytest.mark.parametrize("endpoint", _BEDROCK_ENDPOINTS)
def test_bedrock_event_stream_failure_without_amzn_request_id_logs_none(
    gateway: Gateway, endpoint: BedrockEndpoint, after_output: bool
) -> None:
    message: Final = f"bedrock stream failure {uuid.uuid4()}"
    frames: Final = (
        *(_bedrock_output(endpoint == "messages-invoke") if after_output else ()),
        _bedrock_failure_frame("exception-event", message),
    )

    def respond(request: Request) -> Reply:
        assert unquote(request.target) in _BEDROCK_STREAM_TARGETS, request.target
        return Reply(headers={"retry-after": "7"}, content_type=_EVENT_STREAM, body=b"".join(frames))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        status, text, response_headers = _bedrock_stream(
            gateway, endpoint, _bedrock_model(scenario, endpoint, wire.url), message
        )
        assert status == _bedrock_client_status(endpoint, "exception-event", after_output), text
        assert message in text, text
        assert not any("amzn" in name for name in response_headers), response_headers
        assert len(wire.drain()) == 1
        information: Final = _failure_information(response_headers["x-litellm-call-id"])
        assert (information["error_class"], information["error_code"]) == _bedrock_logged_error(
            endpoint, "exception-event"
        )
        assert information["error_provider_request_id"] is None, information


@pytest.mark.parametrize("endpoint", ("responses", "messages", "messages-invoke"))
def test_bedrock_non_stream_500_logs_amzn_request_id_on_every_endpoint(
    gateway: Gateway, endpoint: BedrockEndpoint
) -> None:
    amzn_request_id: Final = str(uuid.uuid4())
    targets: Final = frozenset(
        (
            "/model/anthropic.claude-sonnet-4-5-20250929-v1:0/converse",
            f"/model/{_INVOKE_MODEL_ID}/invoke",
        )
    )

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and unquote(request.target) in targets, request.target
        return Reply(
            status=500,
            headers={"x-amzn-RequestId": amzn_request_id},
            body=b'{"message":"synthetic bedrock failure"}',
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        path, body = _bedrock_stream_request(
            endpoint, _bedrock_model(scenario, endpoint, wire.url), amzn_request_id, stream=False
        )
        response: Final = gateway.request("POST", path, body)
        assert response.status_code == 503, response.text
        if endpoint == "responses":
            assert response.headers.get("llm_provider-x-amzn-requestid") == amzn_request_id, dict(response.headers)
        assert len(wire.drain()) == 1
        information: Final = _failure_information(response.headers["x-litellm-call-id"])
        assert information["error_provider_request_id"] == amzn_request_id, information


@pytest.mark.parametrize("amzn_request_id", ("", "a" * 5000, "dup-1, dup-1"), ids=("empty", "5kb", "joined-duplicate"))
def test_bedrock_stream_failure_logs_odd_request_ids_like_the_non_stream_error(
    gateway: Gateway, amzn_request_id: str
) -> None:
    stream_body: Final = b"".join((*_bedrock_output(False), _bedrock_failure_frame("exception-event", "odd id")))

    def respond(request: Request) -> Reply:
        if unquote(request.target) in _BEDROCK_STREAM_TARGETS:
            return Reply(headers={"x-amzn-RequestId": amzn_request_id}, content_type=_EVENT_STREAM, body=stream_body)
        return Reply(status=500, headers={"x-amzn-RequestId": amzn_request_id}, body=b'{"message":"odd id"}')

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _bedrock_model(scenario, "chat", wire.url)
        status, text, stream_headers = _bedrock_stream(gateway, "chat", model, "odd id")
        path, body = _bedrock_stream_request("chat", model, "odd id", stream=False)
        non_stream: Final = gateway.request("POST", path, body)
        assert status == 200 and "odd id" in text, text
        assert non_stream.status_code == 503, non_stream.text
        assert len(wire.drain()) == 2
        streamed: Final = _failure_information(stream_headers["x-litellm-call-id"])
        unstreamed: Final = _failure_information(non_stream.headers["x-litellm-call-id"])
        assert streamed["error_provider_request_id"] == unstreamed["error_provider_request_id"], (streamed, unstreamed)


_BURST_CALLS: Final = tuple(
    (endpoint, failure, after_output)
    for endpoint in _BEDROCK_ENDPOINTS
    for failure, shape in _BEDROCK_FAILURES.items()
    for after_output in shape.after_output
)


def _healthy_bedrock_stream(invoke: bool) -> bytes:
    closing: Final = (
        (
            _invoke_chunk({"type": "content_block_stop", "index": 0}),
            _invoke_chunk(
                {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}}
            ),
            _invoke_chunk({"type": "message_stop"}),
        )
        if invoke
        else (
            _bedrock_frame("contentBlockStop", {"contentBlockIndex": 0}),
            _bedrock_frame("messageStop", {"stopReason": "end_turn"}),
            _bedrock_frame("metadata", {"usage": {"inputTokens": 3, "outputTokens": 1, "totalTokens": 4}}),
        )
    )
    return b"".join((*_bedrock_output(invoke, "recovered"), *closing))


def test_concurrent_bedrock_stream_failures_on_every_endpoint_keep_distinct_request_ids(gateway: Gateway) -> None:
    identity: Final = uuid.uuid4().hex
    failing: Final = {f"{identity}-{index:02d}": call for index, call in enumerate(_BURST_CALLS)}
    recoveries: Final = {f"{identity}-r{index}": endpoint for index, endpoint in enumerate(_BEDROCK_ENDPOINTS)}

    def respond(request: Request) -> Reply:
        (token,) = (token for token in (*failing, *recoveries) if token in request.body.decode())
        if token in recoveries:
            body = _healthy_bedrock_stream(recoveries[token] == "messages-invoke")
        else:
            endpoint, failure, after_output = failing[token]
            opening = _bedrock_output(endpoint == "messages-invoke") if after_output else ()
            body = b"".join((*opening, _bedrock_failure_frame(failure, token)))
        return Reply(headers={"x-amzn-RequestId": token, "retry-after": "7"}, content_type=_EVENT_STREAM, body=body)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        models: Final = {endpoint: _bedrock_model(scenario, endpoint, wire.url) for endpoint in _BEDROCK_ENDPOINTS}

        def call(token: str, endpoint: BedrockEndpoint) -> tuple[int, str, dict[str, str]]:
            return _bedrock_stream(gateway, endpoint, models[endpoint], token)

        with ThreadPoolExecutor(max_workers=8) as workers:
            outcomes: Final = dict(
                zip(failing, workers.map(call, failing, (endpoint for endpoint, _, _ in failing.values())), strict=True)
            )
        recovered: Final = {token: call(token, endpoint) for token, endpoint in recoveries.items()}
        assert len(wire.drain()) == len(failing) + len(recoveries)
        for token, (endpoint, failure, after_output) in failing.items():
            status, text, response_headers = outcomes[token]
            assert status == _bedrock_client_status(endpoint, failure, after_output), (token, text)
            assert _bedrock_failure_text(failure, token) in text, (token, text)
            information = _failure_information(response_headers["x-litellm-call-id"])
            assert information["error_provider_request_id"] == token, (token, information)
        for token, (status, text, response_headers) in recovered.items():
            assert status == 200 and "recovered" in text, (token, text)
            assert spend_row(response_headers["x-litellm-call-id"])["status"] == "success"


def _sdk_stream_text(
    client_kind: str, asynchronous: bool, gateway: Gateway, model: str, prompt: str
) -> tuple[str, str]:
    extra_body: Final[dict[str, JsonValue]] = {"num_retries": 0, "cache": {"no-cache": True}}
    turn: Final = [{"role": "user", "content": prompt}]

    def open_stream(client: openai.OpenAI | openai.AsyncOpenAI | anthropic.Anthropic | anthropic.AsyncAnthropic):
        match client_kind:
            case "openai-chat":
                return client.chat.completions.with_raw_response.create(
                    model=model, messages=turn, stream=True, extra_body=extra_body
                )
            case "openai-responses":
                return client.responses.with_raw_response.create(
                    model=model, input=prompt, stream=True, extra_body=extra_body
                )
            case "anthropic-messages":
                return client.messages.with_raw_response.create(
                    model=model, messages=turn, max_tokens=16, stream=True, extra_body=extra_body
                )
        raise AssertionError(client_kind)

    seen: Final[list[str]] = []
    base_url: Final = str(gateway.client.base_url).rstrip("/")
    if not asynchronous:
        with httpx.Client(timeout=15, trust_env=False) as transport:
            client = (
                anthropic.Anthropic(api_key=gateway.key, base_url=base_url, max_retries=0, http_client=transport)
                if client_kind == "anthropic-messages"
                else openai.OpenAI(api_key=gateway.key, base_url=f"{base_url}/v1", max_retries=0, http_client=transport)
            )
            raw = open_stream(client)
            try:
                seen.extend(repr(event) for event in raw.parse())
            except (openai.APIError, anthropic.APIError) as error:
                seen.append(str(error))
            return raw.headers["x-litellm-call-id"], "\n".join(seen)

    async def run() -> str:
        async with httpx.AsyncClient(timeout=15, trust_env=False) as transport:
            client = (
                anthropic.AsyncAnthropic(api_key=gateway.key, base_url=base_url, max_retries=0, http_client=transport)
                if client_kind == "anthropic-messages"
                else openai.AsyncOpenAI(
                    api_key=gateway.key, base_url=f"{base_url}/v1", max_retries=0, http_client=transport
                )
            )
            raw = await open_stream(client)
            try:
                async for event in raw.parse():
                    seen.append(repr(event))
            except (openai.APIError, anthropic.APIError) as error:
                seen.append(str(error))
            return raw.headers["x-litellm-call-id"]

    return asyncio.run(run()), "\n".join(seen)


@pytest.mark.parametrize("asynchronous", (False, True), ids=("sync", "async"))
@pytest.mark.parametrize("client_kind", ("openai-chat", "openai-responses", "anthropic-messages"))
def test_sdk_clients_streaming_bedrock_failure_through_proxy_log_amzn_request_id(
    gateway: Gateway, client_kind: str, asynchronous: bool
) -> None:
    amzn_request_id: Final = str(uuid.uuid4())
    message: Final = f"bedrock stream failure {amzn_request_id}"
    frames: Final = (*_bedrock_output(False), _bedrock_failure_frame("exception-event", message))

    def respond(request: Request) -> Reply:
        assert unquote(request.target) in _BEDROCK_STREAM_TARGETS, request.target
        return Reply(headers={"x-amzn-RequestId": amzn_request_id}, content_type=_EVENT_STREAM, body=b"".join(frames))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        call_id, seen = _sdk_stream_text(
            client_kind, asynchronous, gateway, _bedrock_model(scenario, "chat", wire.url), message
        )
        assert "partial output" in seen and message in seen, seen
        assert len(wire.drain()) == 1
        information: Final = _failure_information(call_id)
        assert information["error_provider_request_id"] == amzn_request_id, information


async def _consume_bedrock_sdk_stream(api: Literal["chat", "responses"], asynchronous: bool, api_base: str) -> str:
    common: Final[dict[str, JsonValue]] = {
        "model": _MODEL,
        "stream": True,
        "api_base": api_base,
        "num_retries": 0,
        **_BEDROCK_AWS,
    }
    turn: Final[list[JsonValue]] = [{"role": "user", "content": "probe"}]
    match api, asynchronous:
        case "chat", True:
            return "".join([repr(chunk) async for chunk in await litellm.acompletion(messages=turn, **common)])
        case "chat", False:
            return "".join(repr(chunk) for chunk in litellm.completion(messages=turn, **common))
        case "responses", True:
            return "".join([repr(event) async for event in await litellm.aresponses(input="probe", **common)])
        case "responses", False:
            return "".join(repr(event) for event in litellm.responses(input="probe", **common))


@pytest.mark.parametrize("asynchronous", (False, True), ids=("sync", "async"))
@pytest.mark.parametrize("api", ("chat", "responses"))
async def test_bedrock_sdk_stream_exception_carries_amzn_request_id(
    api: Literal["chat", "responses"], asynchronous: bool
) -> None:
    amzn_request_id: Final = str(uuid.uuid4())
    message: Final = f"bedrock stream failure {uuid.uuid4()}"
    frames: Final = (*_bedrock_output(False), _bedrock_failure_frame("exception-event", message))

    def respond(request: Request) -> Reply:
        assert unquote(request.target) in _BEDROCK_STREAM_TARGETS, request.target
        return Reply(
            headers={"x-amzn-RequestId": amzn_request_id, "retry-after": "7"},
            content_type=_EVENT_STREAM,
            body=b"".join(frames),
        )

    with wire_server(respond) as wire:
        with pytest.raises(MidStreamFallbackError) as raised:
            await _consume_bedrock_sdk_stream(api, asynchronous, wire.url)
        assert len(wire.drain()) == 1
    provider_error: Final = raised.value.original_exception
    assert isinstance(provider_error, litellm.ServiceUnavailableError), provider_error
    assert raised.value.generated_content == "partial output"
    assert message in str(provider_error)
    assert _get_provider_request_id(provider_error) == amzn_request_id
    assert "retry-after" not in provider_error.response.headers, provider_error.response.headers


@pytest.mark.parametrize("api", ("chat", "responses"))
@pytest.mark.parametrize("healthy", (True, False), ids=("healthy", "failing"))
async def test_sync_bedrock_sdk_stream_with_non_ascii_request_id_header_keeps_streaming(
    api: Literal["chat", "responses"], healthy: bool
) -> None:
    body: Final = (
        _healthy_bedrock_stream(False)
        if healthy
        else b"".join((*_bedrock_output(False), _bedrock_failure_frame("exception-event", "non-ascii id failure")))
    )

    def respond(request: Request) -> Reply:
        assert unquote(request.target) in _BEDROCK_STREAM_TARGETS, request.target
        return Reply(headers={"x-amzn-RequestId": "req-\u00fc-1"}, content_type=_EVENT_STREAM, body=body)

    with wire_server(respond) as wire:
        if healthy:
            assert "recovered" in await _consume_bedrock_sdk_stream(api, False, wire.url)
        else:
            with pytest.raises(MidStreamFallbackError) as raised:
                await _consume_bedrock_sdk_stream(api, False, wire.url)
            assert "non-ascii id failure" in str(raised.value.original_exception)
            assert _get_provider_request_id(raised.value.original_exception) is None
        assert len(wire.drain()) == 1
