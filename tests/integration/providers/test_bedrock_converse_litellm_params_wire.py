import json
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Final, Literal

import httpx
import pytest
from integration._support.client import (
    JSON_OBJECT,
    Gateway,
    gateway_from_environment,
    list_value,
    object_value,
    string_value,
)
from integration._support.ports import allocate_port
from integration._support.responses_stream import data_frames, sse_events
from integration._support.upstream import aws_event_stream_frame
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

Endpoint = Literal["/v1/chat/completions", "/v1/responses", "/v1/messages"]

MODEL_ID: Final = "anthropic.claude-haiku-5-5"
CONVERSE_TARGET: Final = f"/model/{MODEL_ID}/converse"
STREAM_TARGET: Final = f"{CONVERSE_TARGET}-stream"
TOKEN: Final = "synthetic-bedrock-bearer"
ANSWER: Final = "bedrock litellm_params control"
ANTHROPIC_BETA: Final[list[JsonValue]] = ["interleaved-thinking-2025-05-14"]
REJECTION: Final = "unknown_parameter: synthetic rejection before streaming"
ENDPOINTS: Final = (
    pytest.param("/v1/chat/completions", id="chat_completions"),
    pytest.param("/v1/responses", id="responses"),
    pytest.param("/v1/messages", id="messages"),
)
STREAMS: Final = (pytest.param(False, id="non_stream"), pytest.param(True, id="stream"))
CONVERSE_RESPONSE: Final = json.dumps(
    {
        "output": {"message": {"role": "assistant", "content": [{"text": ANSWER}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 11, "outputTokens": 4, "totalTokens": 15},
        "metrics": {"latencyMs": 1},
    }
).encode()
STREAM_EVENTS: Final[tuple[tuple[str, Mapping[str, JsonValue]], ...]] = (
    ("messageStart", {"role": "assistant"}),
    ("contentBlockDelta", {"delta": {"text": ANSWER}, "contentBlockIndex": 0}),
    ("contentBlockStop", {"contentBlockIndex": 0}),
    ("messageStop", {"stopReason": "end_turn"}),
    ("metadata", {"usage": {"inputTokens": 11, "outputTokens": 4, "totalTokens": 15}, "metrics": {"latencyMs": 1}}),
)
STREAM_REPLY: Final = Reply(
    chunks=tuple(
        aws_event_stream_frame(
            {":event-type": event_type, ":content-type": "application/json", ":message-type": "event"},
            json.dumps(payload, separators=(",", ":")).encode(),
        )
        for event_type, payload in STREAM_EVENTS
    ),
    content_type="application/vnd.amazon.eventstream",
)


@dataclass(frozen=True, slots=True)
class Deployment:
    gateway: Gateway
    model: str
    port: int


@pytest.fixture(scope="module")
def deployment() -> Iterator[Deployment]:
    port: Final = allocate_port()
    with gateway_from_environment() as gateway, gateway.scenario() as scenario:
        yield Deployment(
            gateway,
            scenario.model(
                model=f"bedrock/converse/{MODEL_ID}",
                api_key=TOKEN,
                aws_region_name="us-east-1",
                aws_bedrock_runtime_endpoint=f"http://127.0.0.1:{port}",
            ),
            port,
        )


def converse_peer(request: Request) -> Reply:
    assert request.method == "POST" and request.target in (CONVERSE_TARGET, STREAM_TARGET), request.target
    body: Final = JSON_OBJECT.validate_json(request.body)
    assert "litellm_params" not in body, body
    assert "litellm_params" not in object_value(body.get("additionalModelRequestFields", {})), body
    return STREAM_REPLY if request.target == STREAM_TARGET else Reply(body=CONVERSE_RESPONSE)


def rejecting_peer(request: Request) -> Reply:
    assert (request.method, request.target) == ("POST", STREAM_TARGET), request.target
    return Reply(
        status=400,
        body=json.dumps({"message": REJECTION}).encode(),
        headers={"x-amzn-errortype": "ValidationException"},
    )


def _prompt_fields(endpoint: Endpoint, prompt: str) -> dict[str, JsonValue]:
    match endpoint:
        case "/v1/responses":
            return {"input": prompt, "max_output_tokens": 16}
        case "/v1/chat/completions" | "/v1/messages":
            return {"messages": [{"role": "user", "content": prompt}], "max_tokens": 16}


def _payload(
    model: str, endpoint: Endpoint, stream: bool, prompt: str, extra: Mapping[str, JsonValue]
) -> dict[str, JsonValue]:
    return {"model": model, **_prompt_fields(endpoint, prompt), "stream": stream, "cache": {"no-cache": True}, **extra}


def _chunk_text(frame: Mapping[str, JsonValue]) -> str:
    return "".join(
        string_value(object_value(object_value(choice)["delta"]).get("content") or "")
        for choice in list_value(frame["choices"])
    )


def _streamed_text(endpoint: Endpoint, text: str) -> str:
    match endpoint:
        case "/v1/chat/completions":
            return "".join(_chunk_text(frame) for frame in data_frames(text))
        case "/v1/responses":
            events: Final = sse_events(text)
            assert events[-1][0] == "response.completed", events[-1]
            return "".join(
                string_value(data["delta"]) for event, data in events if event == "response.output_text.delta"
            )
        case "/v1/messages":
            return "".join(
                string_value(object_value(data["delta"]).get("text") or "")
                for event, data in sse_events(text)
                if event == "content_block_delta"
            )


def _answer(endpoint: Endpoint, body: Mapping[str, JsonValue]) -> JsonValue:
    match endpoint:
        case "/v1/chat/completions":
            return object_value(object_value(list_value(body["choices"])[0])["message"])["content"]
        case "/v1/responses":
            return object_value(list_value(object_value(list_value(body["output"])[0])["content"])[0])["text"]
        case "/v1/messages":
            return object_value(list_value(body["content"])[0])["text"]


def _assert_answered(endpoint: Endpoint, stream: bool, response: httpx.Response) -> None:
    assert response.status_code == 200, response.text
    if stream:
        assert _streamed_text(endpoint, response.text) == ANSWER, response.text
        return
    assert _answer(endpoint, JSON_OBJECT.validate_json(response.content)) == ANSWER, response.text


def _sent_body(
    deployment: Deployment, wire: Wire, endpoint: Endpoint, stream: bool, prompt: str, extra: Mapping[str, JsonValue]
) -> dict[str, JsonValue]:
    response: Final = deployment.gateway.request(
        "POST", endpoint, _payload(deployment.model, endpoint, stream, prompt, extra)
    )
    _assert_answered(endpoint, stream, response)
    received: Final = wire.drain()
    assert len(received) == 1, received
    return JSON_OBJECT.validate_json(received[0].body)


@pytest.mark.parametrize("value", (pytest.param({"foo": 1}, id="dict"), pytest.param("x", id="string")))
@pytest.mark.parametrize("stream", STREAMS)
@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_a_top_level_litellm_params_leaves_the_converse_body_identical_to_omitting_it(
    deployment: Deployment, endpoint: Endpoint, stream: bool, value: JsonValue
) -> None:
    prompt: Final = uuid.uuid4().hex
    with wire_server(converse_peer, port=deployment.port) as wire:
        without: Final = _sent_body(deployment, wire, endpoint, stream, prompt, {})
        with_field: Final = _sent_body(deployment, wire, endpoint, stream, prompt, {"litellm_params": value})

    assert with_field == without


@pytest.mark.parametrize("stream", STREAMS)
@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_anthropic_beta_still_lands_in_additional_model_request_fields(
    deployment: Deployment, endpoint: Endpoint, stream: bool
) -> None:
    with wire_server(converse_peer, port=deployment.port) as wire:
        body: Final = _sent_body(
            deployment, wire, endpoint, stream, uuid.uuid4().hex, {"anthropic_beta": ANTHROPIC_BETA}
        )

    assert body["additionalModelRequestFields"] == {"anthropic_beta": ANTHROPIC_BETA}, body


def test_a_streamed_responses_request_rejected_before_streaming_ends_with_a_prompt_error(
    deployment: Deployment,
) -> None:
    with wire_server(rejecting_peer, port=deployment.port) as wire:
        with deployment.gateway.client.stream(
            "POST",
            "/v1/responses",
            json=_payload(deployment.model, "/v1/responses", True, uuid.uuid4().hex, {}),
            headers={"Authorization": f"Bearer {deployment.gateway.key}"},
            timeout=httpx.Timeout(20),
        ) as response:
            text: Final = response.read().decode()
        received: Final = wire.drain()

    assert len(received) == 1, received
    assert response.status_code == 400, text
    assert REJECTION in string_value(object_value(JSON_OBJECT.validate_json(text)["error"])["message"]), text
