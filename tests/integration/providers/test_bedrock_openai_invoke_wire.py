import json
import re
import uuid
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final
from urllib.parse import unquote

import httpx
import openai
from integration._support.bedrock_runtime_peer import answer, body_of, marker_of
from integration._support.client import Gateway, Scenario
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

INFERENCE_PROFILE: Final = "global.openai.gpt-6-luna"
PROFILE_ARN: Final = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123def456"
IMPORTED_MODEL_ARN: Final = "arn:aws:bedrock:us-east-1:123456789012:imported-model/m4gc1mrfuddy"
TOKEN: Final = "synthetic-bedrock-bearer"
VALIDATION_ERROR_TYPE: Final = "ValidationException:http://internal.amazon.com/coral/com.amazon.bedrock/"
_INVOKE: Final = re.compile(r"^/model/([^/]+)/(invoke|invoke-with-response-stream)$")
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


class _Message(BaseModel):
    model_config = ConfigDict(frozen=True)
    content: str


class _Choice(BaseModel):
    model_config = ConfigDict(frozen=True)
    finish_reason: str
    message: _Message


class _Completion(BaseModel):
    model_config = ConfigDict(frozen=True)
    choices: tuple[_Choice, ...]


_CHOICES: Final = TypeAdapter(_Completion)


def _validation_error(message: str) -> Reply:
    return Reply(
        status=400,
        body=json.dumps({"message": message}).encode(),
        headers=MappingProxyType({"x-amzn-errortype": VALIDATION_ERROR_TYPE}),
    )


def _completion(marker: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": f"chatcmpl-{marker}",
                "object": "chat.completion",
                "created": 1,
                "model": INFERENCE_PROFILE,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": answer(marker)}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 9, "completion_tokens": 5, "total_tokens": 14},
            }
        ).encode()
    )


def _chunk(marker: str, delta: Mapping[str, JsonValue], finish_reason: str | None) -> bytes:
    frame: Final = {
        "id": f"chatcmpl-{marker}",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": INFERENCE_PROFILE,
        "choices": [{"index": 0, "delta": dict(delta), "finish_reason": finish_reason}],
    }
    return b"data: " + json.dumps(frame).encode() + b"\n\n"


def _completion_stream(marker: str) -> Reply:
    return Reply(
        content_type="text/event-stream",
        chunks=(
            _chunk(marker, {"role": "assistant", "content": "answer "}, None),
            _chunk(marker, {"content": f"marker-{marker}"}, None),
            _chunk(marker, {}, "stop"),
            b"data: [DONE]\n\n",
        ),
    )


def bedrock_runtime(request: Request) -> Reply:
    route: Final = _INVOKE.match(request.target)
    if request.method != "POST" or route is None:
        return Reply(status=404, body=json.dumps({"message": f"unknown route {request.target}"}).encode())
    body: Final = body_of(request)
    if "model_id" in body:
        return _validation_error("Unknown parameter: 'model_id'")
    if "model" in body and body["model"] != unquote(route.group(1)):
        return _validation_error("The model in the request body does not match the model in the URL")
    marker: Final = marker_of(request)
    return _completion_stream(marker) if route.group(2) == "invoke-with-response-stream" else _completion(marker)


def rejecting_bedrock_runtime(request: Request) -> Reply:
    return _validation_error(f"Malformed request rejected marker-{marker_of(request)}")


def _deployment(scenario: Scenario, wire: Wire, model: str, **overrides: JsonValue) -> str:
    return scenario.model(model=model, api_base=wire.url, api_key=TOKEN, aws_region_name="us-east-1", **overrides)


def _prompt(marker: str) -> str:
    return f"synthetic invoke request marker-{marker}"


def _chat(gateway: Gateway, model: str, marker: str) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": _prompt(marker)}], "cache": {"no-cache": True}},
    )


def _stream(gateway: Gateway, model: str, marker: str) -> None:
    client: Final = openai.OpenAI(base_url=str(gateway.client.base_url) + "/v1", api_key=gateway.key, max_retries=0)
    stream: Final = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": _prompt(marker)}],
        stream=True,
        extra_body={"cache": {"no-cache": True}},
    )
    for _ in stream:
        pass


def _assert_answered(response: httpx.Response, marker: str) -> None:
    assert response.status_code == 200, response.text
    choice: Final = _CHOICES.validate_json(response.content).choices[0]
    assert (choice.finish_reason, choice.message.content) == ("stop", answer(marker)), response.text


def _only_request(wire: Wire) -> Request:
    received: Final = wire.drain()
    assert len(received) == 1, [(request.method, request.target) for request in received]
    return received[0]


def _invoked_body(request: Request, model_id: str, route: str) -> dict[str, JsonValue]:
    segments: Final = request.target.split("/")
    assert request.method == "POST", request.target
    assert len(segments) == 4 and segments[:2] == ["", "model"], request.target
    assert (unquote(segments[2]), segments[3]) == (model_id, route), request.target
    return _JSON_OBJECT.validate_json(request.body)


def _assert_targets_profile(body: Mapping[str, JsonValue]) -> None:
    assert "model_id" not in body, body
    assert body.get("model", PROFILE_ARN) == PROFILE_ARN, body


def test_model_id_inference_profile_is_invoked_at_the_profile_arn(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(bedrock_runtime) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, f"bedrock/openai/{INFERENCE_PROFILE}", model_id=PROFILE_ARN)
        response: Final = _chat(gateway, model, marker)
        _assert_targets_profile(_invoked_body(_only_request(wire), PROFILE_ARN, "invoke"))
        _assert_answered(response, marker)


def test_model_id_inference_profile_streams_from_the_profile_arn(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(bedrock_runtime) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, f"bedrock/openai/{INFERENCE_PROFILE}", model_id=PROFILE_ARN)
        _stream(gateway, model, marker)
        _assert_targets_profile(_invoked_body(_only_request(wire), PROFILE_ARN, "invoke-with-response-stream"))


def test_inference_profile_without_model_id_is_invoked_at_its_own_id(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(bedrock_runtime) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, f"bedrock/openai/{INFERENCE_PROFILE}")
        response: Final = _chat(gateway, model, marker)
        request: Final = _only_request(wire)
        assert request.target == f"/model/{INFERENCE_PROFILE}/invoke", request.target
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == INFERENCE_PROFILE and "model_id" not in body, body
        _assert_answered(response, marker)


def test_inference_profile_without_model_id_streams_from_its_own_id(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(bedrock_runtime) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, f"bedrock/openai/{INFERENCE_PROFILE}")
        _stream(gateway, model, marker)
        request: Final = _only_request(wire)
        assert request.target == f"/model/{INFERENCE_PROFILE}/invoke-with-response-stream", request.target
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == INFERENCE_PROFILE and "model_id" not in body, body


def test_imported_model_arn_is_invoked_with_its_slash_encoded(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(bedrock_runtime) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, f"bedrock/openai/{IMPORTED_MODEL_ARN}")
        response: Final = _chat(gateway, model, marker)
        assert "model_id" not in _invoked_body(_only_request(wire), IMPORTED_MODEL_ARN, "invoke")
        _assert_answered(response, marker)


def test_bedrock_validation_error_on_a_model_id_deployment_reaches_the_caller_as_400(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(rejecting_bedrock_runtime) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, f"bedrock/openai/{INFERENCE_PROFILE}", model_id=PROFILE_ARN)
        response: Final = _chat(gateway, model, marker)
        _only_request(wire)
        assert response.status_code == 400, response.text
        error: Final = _JSON_OBJECT.validate_json(response.content)["error"]
        assert isinstance(error, dict), response.text
        assert f"Malformed request rejected marker-{marker}" in str(error["message"]), response.text
