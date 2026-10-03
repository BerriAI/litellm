import asyncio
import base64
import copy
import json
import uuid
from collections.abc import Callable, Iterator, Mapping
from itertools import count
from pathlib import Path
from typing import Final

import anthropic
import httpx
import pytest
import yaml
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from integration._support.upstream import _aws_event_frame
from integration._support.wire import Reply, Request, wire_server
from integration.streaming.test_streamed_usage_cost_burst import _wait_for_proxy_models
from openai import AsyncOpenAI, OpenAI
from pydantic import JsonValue, TypeAdapter

import litellm

FREE: Final = {"input_cost_per_token": 0.0, "output_cost_per_token": 0.0}
PRICED: Final = {"input_cost_per_token": 0.001, "output_cost_per_token": 0.002}
UNPRICED: Final[Mapping[str, float]] = {}
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


def _chat_frame(identity: str, body: Mapping[str, JsonValue]) -> bytes:
    return (
        b"data: "
        + json.dumps({"id": identity, "object": "chat.completion.chunk", "created": 1, **body}).encode()
        + b"\n\n"
    )


def _chat_stream(
    identity: str,
    model: str,
    usage: Mapping[str, JsonValue] | None = None,
    *,
    choice_count: int = 1,
    include_usage: bool = True,
) -> tuple[bytes, ...]:
    response_usage: Final = (
        dict(usage) if usage is not None else {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15}
    )
    first_choices: Final = tuple(
        {"index": index, "delta": {"role": "assistant", "content": "Hello there"}, "finish_reason": None}
        for index in range(choice_count)
    )
    finished_choices: Final = tuple(
        {"index": index, "delta": {}, "finish_reason": "stop"} for index in range(choice_count)
    )
    usage_frame: Final = (
        (_chat_frame(identity, {"model": model, "choices": [], "usage": response_usage}),) if include_usage else ()
    )
    return (
        _chat_frame(
            identity,
            {"model": model, "choices": first_choices},
        ),
        _chat_frame(identity, {"model": model, "choices": finished_choices}),
        *usage_frame,
        b"data: [DONE]\n\n",
    )


def _ollama_stream(model: str) -> tuple[bytes, ...]:
    return (
        json.dumps(
            {
                "model": model,
                "created_at": "2025-01-01T00:00:00Z",
                "message": {"role": "assistant", "content": "Hello there"},
                "done": False,
            }
        ).encode()
        + b"\n",
        json.dumps(
            {
                "model": model,
                "created_at": "2025-01-01T00:00:00Z",
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "prompt_eval_count": 11,
                "eval_count": 4,
            }
        ).encode()
        + b"\n",
    )


def _responses_frame(event: str, payload: Mapping[str, JsonValue]) -> bytes:
    return f"event: {event}\ndata: {json.dumps({'type': event, **payload})}\n\n".encode()


def _responses_stream(
    identity: str,
    model: str,
    usage: Mapping[str, JsonValue] | None = None,
) -> tuple[bytes, ...]:
    message: Final = {
        "type": "message",
        "id": "msg_1",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "Hello there", "annotations": []}],
    }
    response: Final = {"id": identity, "object": "response", "created_at": 1, "model": model, "output": []}
    return (
        _responses_frame("response.created", {"response": {**response, "status": "in_progress"}}),
        _responses_frame(
            "response.output_text.delta",
            {"item_id": "msg_1", "output_index": 0, "content_index": 0, "delta": "Hello there"},
        ),
        _responses_frame("response.output_item.done", {"output_index": 0, "item": message}),
        _responses_frame(
            "response.completed",
            {
                "response": {
                    **response,
                    "status": "completed",
                    "output": [message],
                    "usage": dict(usage)
                    if usage is not None
                    else {"input_tokens": 11, "output_tokens": 4, "total_tokens": 15},
                }
            },
        ),
    )


def _sse_events(text: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        _JSON_OBJECT.validate_json(line.removeprefix("data: "))
        for line in text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    )


def _cost_field(usage: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {key: value for key, value in usage.items() if key == "cost"}


def _responses_spend_request_id(model_id: str, response_id: str) -> str:
    identity: Final = (f"litellm:custom_llm_provider:openai;model_id:{model_id};response_id:{response_id}").encode()
    return "resp_" + base64.b64encode(identity).decode()


def _usage_events(text: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        _JSON_OBJECT.validate_python(event["usage"]) for event in _sse_events(text) if event.get("usage") is not None
    )


def _proxy_stream(
    gateway: Gateway,
    endpoint: str,
    body: dict[str, JsonValue],
    headers: Mapping[str, str] | None = None,
) -> tuple[int, dict[str, str], str]:
    request_headers: Final = {"Authorization": f"Bearer {gateway.key}", **dict(headers or {})}
    with gateway.client.stream("POST", endpoint, json=body, headers=request_headers) as response:
        status: Final = response.status_code
        response_headers: Final = dict(response.headers)
        text: Final = response.read().decode()
    return status, response_headers, text


def _read_stream_text(response: httpx.Response) -> str:
    try:
        return response.read().decode()
    except httpx.HTTPError as error:
        return f"{type(error).__name__}: {error}"


def _record_observed(record_property: Callable[[str, object], None], name: str, value: object) -> None:
    record_property(name, json.dumps(value, sort_keys=True))


def _completion_stream(identity: str, model: str) -> tuple[bytes, ...]:
    return (
        b"data: "
        + json.dumps(
            {
                "id": identity,
                "object": "text_completion",
                "created": 1,
                "model": model,
                "choices": [{"text": "Hello there", "index": 0, "logprobs": None, "finish_reason": None}],
            }
        ).encode()
        + b"\n\n",
        b"data: "
        + json.dumps(
            {
                "id": identity,
                "object": "text_completion",
                "created": 1,
                "model": model,
                "choices": [{"text": "", "index": 0, "logprobs": None, "finish_reason": "stop"}],
            }
        ).encode()
        + b"\n\n",
        b"data: "
        + json.dumps(
            {
                "id": identity,
                "object": "text_completion",
                "created": 1,
                "model": model,
                "choices": [],
                "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
            }
        ).encode()
        + b"\n\n",
        b"data: [DONE]\n\n",
    )


def _anthropic_stream(identity: str, model: str) -> tuple[bytes, ...]:
    events: Final = (
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": identity,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 11, "output_tokens": 0},
                },
            },
        ),
        (
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello there"}},
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 4},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    )
    return tuple(f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode() for event, payload in events)


def _model_list_reply(request: Request) -> Reply | None:
    if request.method != "GET" or request.target != "/v1/models":
        return None
    return Reply(
        body=b'{"object":"list","data":[{"id":"gpt-4o-mini","object":"model","created":1,"owned_by":"litellm"},'
        b'{"id":"litellm-unpriced-9097","object":"model","created":1,"owned_by":"litellm"}]}',
        content_type="application/json",
    )


@pytest.mark.parametrize(
    ("deployment_model", "rates", "expected_cost_field"),
    (
        pytest.param("openai/gpt-4o-mini", FREE, {"cost": 0.0}, id="A1"),
        pytest.param("openai/gpt-4o-mini", PRICED, {"cost": pytest.approx(0.019)}, id="A2"),
        pytest.param("openai/litellm-unpriced-9097", UNPRICED, {}, id="A3"),
    ),
)
def test_chat_completions_stream_final_usage_reports_the_computed_cost(
    gateway: Gateway, deployment_model: str, rates: Mapping[str, float], expected_cost_field: dict[str, JsonValue]
) -> None:
    identity: Final = "chat-usage-cost-" + uuid.uuid4().hex
    upstream_model: Final = deployment_model.removeprefix("openai/")

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/chat/completions", request.target
        outbound: Final = _JSON_OBJECT.validate_json(request.body)
        assert outbound["model"] == upstream_model and outbound["stream"] is True, outbound
        assert outbound["stream_options"] == {"include_usage": True}, outbound
        return Reply(content_type="text/event-stream", chunks=_chat_stream(identity, upstream_model))

    with (
        wire_server(respond) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(model=deployment_model, api_base=wire.url + "/v1", **rates)
        with gateway.client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": identity}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            headers={"Authorization": f"Bearer {gateway.key}"},
        ) as response:
            text: Final = response.read().decode()
    assert response.status_code == 200, text
    usages: Final = tuple(event["usage"] for event in _sse_events(text) if event.get("usage") is not None)
    assert len(usages) == 1, text
    usage: Final = usages[0]
    assert isinstance(usage, dict), text
    assert (usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"]) == (11, 4, 15), text
    assert _cost_field(usage) == expected_cost_field, text
    requests: Final = wire.drain()
    assert len(requests) == 1 and requests[0].target == "/v1/chat/completions", text


@pytest.mark.parametrize(
    ("deployment_model", "rates", "expected_cost_field"),
    (
        pytest.param("openai/gpt-4o-mini", FREE, {"cost": 0.0}, id="A7"),
        pytest.param("openai/gpt-4o-mini", PRICED, {"cost": pytest.approx(0.019)}, id="A8"),
        pytest.param("openai/litellm-unpriced-9097", UNPRICED, {}, id="A9"),
    ),
)
def test_responses_stream_completed_usage_reports_the_computed_cost(
    gateway: Gateway, deployment_model: str, rates: Mapping[str, float], expected_cost_field: dict[str, JsonValue]
) -> None:
    identity: Final = "resp_usage_cost_" + uuid.uuid4().hex
    upstream_model: Final = deployment_model.removeprefix("openai/")

    def respond(request: Request) -> Reply:
        assert request.target == "/v1/responses", request.target
        return Reply(content_type="text/event-stream", chunks=_responses_stream(identity, upstream_model))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=deployment_model, api_base=wire.url + "/v1", **rates)
        with gateway.client.stream(
            "POST",
            "/v1/responses",
            json={"model": model, "input": identity, "stream": True},
            headers={"Authorization": f"Bearer {gateway.key}"},
        ) as response:
            text: Final = response.read().decode()
    assert response.status_code == 200, text
    completed: Final = tuple(event for event in _sse_events(text) if event["type"] == "response.completed")
    assert len(completed) == 1, text
    completed_response: Final = completed[0]["response"]
    assert isinstance(completed_response, dict), text
    usage: Final = completed_response["usage"]
    assert isinstance(usage, dict), text
    assert (usage["input_tokens"], usage["output_tokens"], usage["total_tokens"]) == (11, 4, 15), text
    assert _cost_field(usage) == expected_cost_field, text
    requests: Final = wire.drain()
    assert len(requests) == 1 and requests[0].target == "/v1/responses", text


@pytest.mark.parametrize("expected_cost_field", (pytest.param({"cost": 0.0}, id="A4"),))
def test_openai_sdk_sync_chat_stream_reports_free_cost(
    gateway: Gateway, expected_cost_field: dict[str, JsonValue]
) -> None:
    identity: Final = "sdk-chat-sync-free-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/chat/completions", request.target
        outbound: Final = _JSON_OBJECT.validate_json(request.body)
        assert outbound["stream_options"] == {"include_usage": True}, outbound
        return Reply(content_type="text/event-stream", chunks=_chat_stream(identity, "gpt-4o-mini"))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=wire.url + "/v1", **FREE)
        with OpenAI(
            base_url=f"{str(gateway.client.base_url).rstrip('/')}/v1",
            api_key=gateway.key,
            max_retries=0,
            timeout=15,
        ) as client:
            with client.chat.completions.with_streaming_response.create(
                model=model,
                messages=[{"role": "user", "content": identity}],
                stream=True,
                stream_options={"include_usage": True},
            ) as response:
                status: Final = response.status_code
                text: Final = response.read().decode()
    assert status == 200, text
    usages: Final = _usage_events(text)
    assert len(usages) == 1, text
    usage: Final = usages[0]
    assert (usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"]) == (11, 4, 15), text
    assert _cost_field(usage) == expected_cost_field, text
    requests: Final = wire.drain()
    assert len(requests) == 1 and requests[0].target == "/v1/chat/completions", text


@pytest.mark.parametrize(
    ("rates", "expected_cost_field"),
    (
        pytest.param(FREE, {"cost": 0.0}, id="A5"),
        pytest.param(PRICED, {"cost": pytest.approx(0.019)}, id="A6"),
    ),
)
async def test_openai_sdk_async_chat_stream_reports_cost(
    gateway: Gateway, rates: Mapping[str, float], expected_cost_field: dict[str, JsonValue]
) -> None:
    identity: Final = "sdk-chat-async-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/chat/completions", request.target
        outbound: Final = _JSON_OBJECT.validate_json(request.body)
        assert outbound["stream_options"] == {"include_usage": True}, outbound
        return Reply(content_type="text/event-stream", chunks=_chat_stream(identity, "gpt-4o-mini"))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=wire.url + "/v1", **rates)
        async with AsyncOpenAI(
            base_url=f"{str(gateway.client.base_url).rstrip('/')}/v1",
            api_key=gateway.key,
            max_retries=0,
            timeout=15,
        ) as client:
            async with client.chat.completions.with_streaming_response.create(
                model=model,
                messages=[{"role": "user", "content": identity}],
                stream=True,
                stream_options={"include_usage": True},
            ) as response:
                status: Final = response.status_code
                text: Final = (await response.read()).decode()
    assert status == 200, text
    usages: Final = _usage_events(text)
    assert len(usages) == 1, text
    usage: Final = usages[0]
    assert (usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"]) == (11, 4, 15), text
    assert _cost_field(usage) == expected_cost_field, text
    requests: Final = wire.drain()
    assert len(requests) == 1 and requests[0].target == "/v1/chat/completions", text


@pytest.mark.parametrize("expected_cost_field", (pytest.param({"cost": 0.0}, id="A10"),))
def test_openai_sdk_sync_responses_stream_reports_free_cost(
    gateway: Gateway, expected_cost_field: dict[str, JsonValue]
) -> None:
    identity: Final = "sdk-responses-sync-free-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/responses", request.target
        outbound: Final = _JSON_OBJECT.validate_json(request.body)
        assert outbound["stream"] is True, outbound
        return Reply(content_type="text/event-stream", chunks=_responses_stream(identity, "gpt-4o-mini"))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=wire.url + "/v1", **FREE)
        with OpenAI(
            base_url=f"{str(gateway.client.base_url).rstrip('/')}/v1",
            api_key=gateway.key,
            max_retries=0,
            timeout=15,
        ) as client:
            with client.responses.with_streaming_response.create(model=model, input=identity, stream=True) as response:
                status: Final = response.status_code
                text: Final = response.read().decode()
    assert status == 200, text
    events: Final = _sse_events(text)
    completed: Final = tuple(event for event in events if event["type"] == "response.completed")
    assert len(completed) == 1, events
    response_body: Final = _JSON_OBJECT.validate_python(completed[0]["response"])
    usage: Final = _JSON_OBJECT.validate_python(response_body["usage"])
    assert (usage["input_tokens"], usage["output_tokens"], usage["total_tokens"]) == (11, 4, 15), text
    assert _cost_field(usage) == expected_cost_field, text
    requests: Final = wire.drain()
    assert len(requests) == 1 and requests[0].target == "/v1/responses", text


@pytest.mark.parametrize(
    ("rates", "expected_cost_field"),
    (
        pytest.param(FREE, {"cost": 0.0}, id="A11"),
        pytest.param(PRICED, {"cost": pytest.approx(0.019)}, id="A12"),
    ),
)
async def test_openai_sdk_async_responses_stream_reports_cost(
    gateway: Gateway, rates: Mapping[str, float], expected_cost_field: dict[str, JsonValue]
) -> None:
    identity: Final = "sdk-responses-async-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/responses", request.target
        outbound: Final = _JSON_OBJECT.validate_json(request.body)
        assert outbound["stream"] is True, outbound
        return Reply(content_type="text/event-stream", chunks=_responses_stream(identity, "gpt-4o-mini"))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=wire.url + "/v1", **rates)
        async with AsyncOpenAI(
            base_url=f"{str(gateway.client.base_url).rstrip('/')}/v1",
            api_key=gateway.key,
            max_retries=0,
            timeout=15,
        ) as client:
            async with client.responses.with_streaming_response.create(
                model=model, input=identity, stream=True
            ) as response:
                status: Final = response.status_code
                text: Final = (await response.read()).decode()
    assert status == 200, text
    events: Final = _sse_events(text)
    completed: Final = tuple(event for event in events if event["type"] == "response.completed")
    assert len(completed) == 1, events
    response_body: Final = _JSON_OBJECT.validate_python(completed[0]["response"])
    usage: Final = _JSON_OBJECT.validate_python(response_body["usage"])
    assert (usage["input_tokens"], usage["output_tokens"], usage["total_tokens"]) == (11, 4, 15), text
    assert _cost_field(usage) == expected_cost_field, text
    requests: Final = wire.drain()
    assert len(requests) == 1 and requests[0].target == "/v1/responses", text


@pytest.mark.parametrize(
    "expected_cost_field",
    (pytest.param({"cost": pytest.approx(4.05e-6)}, id="A13-model-info-rates-do-not-drive-stream-cost"),),
)
def test_model_info_rates_do_not_drive_stream_cost(
    gateway: Gateway,
    expected_cost_field: dict[str, JsonValue],
    record_property: Callable[[str, object], None],
) -> None:
    identity: Final = "model-info-chat-free-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/chat/completions", request.target
        return Reply(content_type="text/event-stream", chunks=_chat_stream(identity, "gpt-4o-mini"))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/gpt-4o-mini",
            api_base=wire.url + "/v1",
            model_info={"input_cost_per_token": 0.0, "output_cost_per_token": 0.0},
        )
        status, _, text = _proxy_stream(
            gateway,
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": identity}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
    assert status == 200, text
    usages: Final = _usage_events(text)
    assert len(usages) == 1, text
    observed_cost_field: Final = _cost_field(usages[0])
    _record_observed(record_property, "observed_cost_field", observed_cost_field)
    assert observed_cost_field == expected_cost_field, text
    requests: Final = wire.drain()
    assert len(requests) == 1 and requests[0].target == "/v1/chat/completions", text


@pytest.mark.parametrize(
    ("deployment_model", "rates", "expected_cost_field", "expected_upstream_target"),
    (
        pytest.param("openai/gpt-4o-mini", FREE, {"cost": 0.0}, "/v1/chat/completions", id="A14-free"),
        pytest.param(
            "openai/gpt-4o-mini",
            PRICED,
            {"cost": pytest.approx(0.019)},
            "/v1/chat/completions",
            id="A14-priced",
        ),
        pytest.param("openai/litellm-unpriced-9097", UNPRICED, {}, "/v1/completions", id="A14-unpriced"),
    ),
)
def test_openai_sdk_completion_stream_reports_cost(
    gateway: Gateway,
    deployment_model: str,
    rates: Mapping[str, float],
    expected_cost_field: dict[str, JsonValue],
    expected_upstream_target: str,
    record_property: Callable[[str, object], None],
) -> None:
    identity: Final = "completion-usage-cost-" + uuid.uuid4().hex
    upstream_model: Final = deployment_model.removeprefix("openai/")

    def respond(request: Request) -> Reply:
        model_list: Final = _model_list_reply(request)
        if model_list is not None:
            return model_list
        assert request.method == "POST" and request.target == expected_upstream_target, request.target
        outbound: Final = _JSON_OBJECT.validate_json(request.body)
        assert outbound["stream"] is True, outbound
        assert outbound["model"] == upstream_model, outbound
        if expected_upstream_target == "/v1/completions":
            assert outbound["prompt"] == identity, outbound
            chunks: Final = _completion_stream(identity, upstream_model)
        else:
            assert outbound["messages"] == [{"role": "user", "content": identity}], outbound
            chunks = _chat_stream(identity, upstream_model)
        return Reply(content_type="text/event-stream", chunks=chunks)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=deployment_model, api_base=wire.url + "/v1", **rates)
        with OpenAI(
            base_url=f"{str(gateway.client.base_url).rstrip('/')}/v1",
            api_key=gateway.key,
            max_retries=0,
            timeout=15,
        ) as client:
            with client.completions.with_streaming_response.create(
                model=model,
                prompt=identity,
                max_tokens=4,
                stream=True,
                stream_options={"include_usage": True},
            ) as response:
                status: Final = response.status_code
                text: Final = response.read().decode()
    assert status == 200, text
    frames: Final = _sse_events(text)
    assert len(frames) == 3, text
    assert all(frame["object"] == "text_completion" for frame in frames), text
    assert frames[0]["choices"] == [{"index": 0, "text": "Hello there"}], text
    assert frames[1]["choices"] == [{"finish_reason": "stop", "index": 0}], text
    assert frames[2]["choices"] == [{"index": 0}], text
    usage: Final = _JSON_OBJECT.validate_python(frames[2]["usage"])
    assert (usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"]) == (11, 4, 15), text
    observed: Final = _cost_field(usage)
    _record_observed(record_property, "observed_raw_usage", usage)
    _record_observed(record_property, "observed_cost_field", observed)
    assert observed == expected_cost_field, text
    requests: Final = wire.drain()
    posts: Final = tuple(request for request in requests if request.method == "POST")
    gets: Final = tuple(request for request in requests if request.method == "GET")
    assert len(posts) == 1 and posts[0].target == expected_upstream_target, requests
    assert all(request.target == "/v1/models" for request in gets), requests


@pytest.mark.parametrize(
    ("deployment_model", "rates", "expected_cost_field"),
    (
        pytest.param("anthropic/litellm-responses-bridge-9097", FREE, {"cost": 0.0}, id="A15-free"),
        pytest.param(
            "anthropic/litellm-responses-bridge-9097", PRICED, {"cost": pytest.approx(0.019)}, id="A15-priced"
        ),
        pytest.param("anthropic/litellm-responses-bridge-9097", UNPRICED, {}, id="A15-unpriced"),
    ),
)
def test_responses_stream_through_chat_completions_bridge_reports_cost(
    gateway: Gateway,
    deployment_model: str,
    rates: Mapping[str, float],
    expected_cost_field: dict[str, JsonValue],
    record_property: Callable[[str, object], None],
) -> None:
    identity: Final = "responses-bridge-" + uuid.uuid4().hex
    upstream_model: Final = "litellm-responses-bridge-9097"

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/messages", request.target
        outbound: Final = _JSON_OBJECT.validate_json(request.body)
        assert outbound["stream"] is True, outbound
        assert request.headers["x-api-key"] == "synthetic-anthropic-key"
        return Reply(content_type="text/event-stream", chunks=_anthropic_stream(identity, upstream_model))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=deployment_model,
            api_base=wire.url,
            api_key="synthetic-anthropic-key",
            **rates,
        )
        status, _, text = _proxy_stream(gateway, "/v1/responses", {"model": model, "input": identity, "stream": True})
    assert status == 200, text
    completed: Final = tuple(event for event in _sse_events(text) if event["type"] == "response.completed")
    assert len(completed) == 1, text
    response_body: Final = _JSON_OBJECT.validate_python(completed[0]["response"])
    usage: Final = _JSON_OBJECT.validate_python(response_body["usage"])
    assert (usage["input_tokens"], usage["output_tokens"], usage["total_tokens"]) == (11, 4, 15), text
    observed: Final = _cost_field(usage)
    _record_observed(record_property, "observed_cost_field", observed)
    assert observed == expected_cost_field, text
    requests: Final = wire.drain()
    assert len(requests) == 1 and requests[0].target == "/v1/messages", text


@pytest.mark.parametrize(
    ("rates", "expected_cost_field"),
    (
        pytest.param(FREE, {}, id="A16-free"),
        pytest.param(PRICED, {}, id="A16-priced"),
    ),
)
def test_anthropic_sdk_sync_messages_stream_preserves_usage_shape(
    gateway: Gateway,
    rates: Mapping[str, float],
    expected_cost_field: dict[str, JsonValue],
    record_property: Callable[[str, object], None],
) -> None:
    identity: Final = "anthropic-sync-" + uuid.uuid4().hex
    model_id: Final = "claude-3-5-sonnet-20241022"

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/messages", request.target
        assert request.headers["x-api-key"] == "synthetic-anthropic-key"
        return Reply(content_type="text/event-stream", chunks=_anthropic_stream(identity, model_id))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"anthropic/{model_id}", api_base=wire.url, api_key="synthetic-anthropic-key", **rates
        )
        with anthropic.Anthropic(
            base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0, timeout=15
        ) as client:
            with client.messages.with_streaming_response.create(
                model=model, max_tokens=16, messages=[{"role": "user", "content": identity}], stream=True
            ) as response:
                status: Final = response.status_code
                text: Final = response.read().decode()
    assert status == 200, text
    events: Final = _sse_events(text)
    starts: Final = tuple(event for event in events if event["type"] == "message_start")
    deltas: Final = tuple(event for event in events if event["type"] == "message_delta")
    assert len(starts) == len(deltas) == 1, text
    message_start: Final = _JSON_OBJECT.validate_python(starts[0]["message"])
    start_usage: Final = _JSON_OBJECT.validate_python(message_start["usage"])
    delta_usage: Final = _JSON_OBJECT.validate_python(deltas[0]["usage"])
    raw_usage: Final = (start_usage, delta_usage)
    assert (start_usage["input_tokens"], start_usage["output_tokens"], delta_usage["output_tokens"]) == (11, 0, 4), text
    observed: Final = tuple(_cost_field(usage) for usage in raw_usage)
    _record_observed(record_property, "observed_raw_usage_cost_fields", observed)
    assert observed == (expected_cost_field, expected_cost_field), text
    requests: Final = wire.drain()
    assert len(requests) == 1 and requests[0].target == "/v1/messages", text


@pytest.mark.parametrize(
    ("rates", "expected_cost_field"),
    (
        pytest.param(FREE, {}, id="A17-free"),
        pytest.param(PRICED, {}, id="A17-priced"),
    ),
)
async def test_anthropic_sdk_async_messages_stream_through_chat_bridge(
    gateway: Gateway,
    rates: Mapping[str, float],
    expected_cost_field: dict[str, JsonValue],
    record_property: Callable[[str, object], None],
) -> None:
    identity: Final = "anthropic-async-bridge-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/responses", request.target
        return Reply(content_type="text/event-stream", chunks=_responses_stream(identity, "gpt-4o-mini"))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=wire.url + "/v1", **rates)
        async with anthropic.AsyncAnthropic(
            base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0, timeout=15
        ) as client:
            async with client.messages.with_streaming_response.create(
                model=model,
                max_tokens=16,
                messages=[{"role": "user", "content": identity}],
                stream=True,
            ) as response:
                status: Final = response.status_code
                text: Final = (await response.read()).decode()
    assert status == 200, text
    events: Final = _sse_events(text)
    starts: Final = tuple(event for event in events if event["type"] == "message_start")
    deltas: Final = tuple(event for event in events if event["type"] == "message_delta")
    assert len(starts) == len(deltas) == 1, text
    message_start: Final = _JSON_OBJECT.validate_python(starts[0]["message"])
    start_usage: Final = _JSON_OBJECT.validate_python(message_start["usage"])
    delta_usage: Final = _JSON_OBJECT.validate_python(deltas[0]["usage"])
    raw_usage: Final = (start_usage, delta_usage)
    assert (start_usage["input_tokens"], start_usage["output_tokens"], delta_usage["output_tokens"]) == (0, 0, 4), text
    observed: Final = tuple(_cost_field(usage) for usage in raw_usage)
    _record_observed(record_property, "observed_raw_usage_cost_fields", observed)
    assert observed == (expected_cost_field, expected_cost_field), text
    requests: Final = wire.drain()
    assert len(requests) == 1 and requests[0].target == "/v1/responses", text


@pytest.mark.parametrize(
    ("endpoint", "deployment_model", "rates", "expected_cost_field", "expected_response_cost_header"),
    (
        pytest.param("/v1/chat/completions", "openai/gpt-4o-mini", FREE, {}, None, id="B1-chat-free"),
        pytest.param(
            "/v1/chat/completions",
            "openai/gpt-4o-mini",
            PRICED,
            {},
            "0.019",
            id="B1-chat-priced",
        ),
        pytest.param(
            "/v1/chat/completions",
            "openai/litellm-unpriced-9097",
            UNPRICED,
            {},
            None,
            id="B1-chat-unpriced",
        ),
        pytest.param("/v1/responses", "openai/gpt-4o-mini", FREE, {"cost": None}, None, id="B2-responses-free"),
        pytest.param(
            "/v1/responses",
            "openai/gpt-4o-mini",
            PRICED,
            {"cost": None},
            "0.019",
            id="B2-responses-priced",
        ),
        pytest.param(
            "/v1/responses",
            "openai/litellm-unpriced-9097",
            UNPRICED,
            {"cost": None},
            None,
            id="B2-responses-unpriced",
        ),
    ),
)
def test_openai_sdk_nonstreaming_usage_cost_control(
    gateway: Gateway,
    endpoint: str,
    deployment_model: str,
    rates: Mapping[str, float],
    expected_cost_field: dict[str, JsonValue],
    expected_response_cost_header: str | None,
    record_property: Callable[[str, object], None],
) -> None:
    identity: Final = "nonstream-cost-" + uuid.uuid4().hex
    upstream_model: Final = deployment_model.removeprefix("openai/")

    def respond(request: Request) -> Reply:
        model_list: Final = _model_list_reply(request)
        if model_list is not None:
            return model_list
        expected_target: Final = "/v1/responses" if endpoint.endswith("responses") else "/v1/chat/completions"
        assert request.method == "POST" and request.target == expected_target, request.target
        if endpoint.endswith("responses"):
            return Reply(
                body=json.dumps(
                    {
                        "id": identity,
                        "object": "response",
                        "created_at": 1,
                        "status": "completed",
                        "model": upstream_model,
                        "output": [
                            {
                                "type": "message",
                                "id": "msg_" + identity,
                                "status": "completed",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": "Hello there", "annotations": []}],
                            }
                        ],
                        "usage": {
                            "input_tokens": 11,
                            "output_tokens": 4,
                            "total_tokens": 15,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens_details": {"reasoning_tokens": 0},
                        },
                    }
                ).encode()
            )
        return Reply(
            body=json.dumps(
                {
                    "id": identity,
                    "object": "chat.completion",
                    "created": 1,
                    "model": upstream_model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Hello there"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
                }
            ).encode()
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=deployment_model, api_base=wire.url + "/v1", **rates)
        with OpenAI(
            base_url=f"{str(gateway.client.base_url).rstrip('/')}/v1",
            api_key=gateway.key,
            max_retries=0,
            timeout=15,
        ) as client:
            if endpoint.endswith("responses"):
                result: Final = client.responses.with_raw_response.create(model=model, input=identity)
            else:
                result = client.chat.completions.with_raw_response.create(
                    model=model, messages=[{"role": "user", "content": identity}]
                )
            status: Final = result.status_code
            text: Final = result.http_response.text
            response_cost_header: Final = result.http_response.headers.get("x-litellm-response-cost")
    assert status == 200, text
    response_body: Final = _JSON_OBJECT.validate_json(text)
    usage: Final = _JSON_OBJECT.validate_python(response_body["usage"])
    input_tokens: Final = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_tokens: Final = usage.get("completion_tokens", usage.get("output_tokens"))
    assert (input_tokens, output_tokens) == (11, 4), text
    observed: Final = _cost_field(usage)
    _record_observed(record_property, "observed_raw_usage", usage)
    _record_observed(record_property, "observed_response_cost_header", response_cost_header)
    assert observed == expected_cost_field, text
    assert response_cost_header == expected_response_cost_header, text
    requests: Final = wire.drain()
    expected_target: Final = "/v1/responses" if endpoint.endswith("responses") else "/v1/chat/completions"
    posts: Final = tuple(request for request in requests if request.method == "POST")
    gets: Final = tuple(request for request in requests if request.method == "GET")
    assert len(posts) == 1 and posts[0].target == expected_target, text
    assert all(request.target == "/v1/models" for request in gets), requests


@pytest.mark.usefixtures("restore_model_cost_after_sdk_call")
@pytest.mark.parametrize("expected_cost_field", (pytest.param({"cost": 0.0}, id="D7"),))
def test_sdk_ollama_stream_reports_dynamic_zero_cost(
    expected_cost_field: dict[str, JsonValue],
    record_property: Callable[[str, object], None],
) -> None:
    model: Final = "ollama_chat/litellm-unmapped-9097"
    identity: Final = "sdk-ollama-usage-cost-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        if request.target == "/api/show":
            return Reply(status=404)
        assert request.target == "/api/chat", request.target
        return Reply(content_type="application/x-ndjson", chunks=_ollama_stream("litellm-unmapped-9097"))

    litellm.get_model_info.cache_clear()
    try:
        with wire_server(respond) as wire:
            stream: Final = litellm.completion(
                model=model,
                api_base=wire.url,
                messages=[{"role": "user", "content": identity}],
                stream=True,
                stream_options={"include_usage": True},
                num_retries=0,
            )
            try:
                chunks: Final = tuple(stream)
            finally:
                asyncio.run(stream.aclose())
    finally:
        litellm.get_model_info.cache_clear()
    requests: Final = wire.drain()
    chat_requests: Final = tuple(request for request in requests if request.target == "/api/chat")
    show_requests: Final = tuple(request for request in requests if request.target == "/api/show")
    _record_observed(record_property, "observed_api_show_count", len(show_requests))
    assert len(chat_requests) == 1, requests
    assert all(request.target == "/api/show" for request in requests if request.target != "/api/chat"), requests
    usages: Final = tuple(chunk.usage for chunk in chunks if getattr(chunk, "usage", None) is not None)
    assert len(usages) == 1, chunks
    usage: Final = usages[0]
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (11, 4, 15), chunks
    usage_dump: Final = _JSON_OBJECT.validate_python(usage.model_dump())
    assert _cost_field(usage_dump) == expected_cost_field, chunks


@pytest.fixture
def restore_model_cost_after_sdk_call() -> Iterator[None]:
    original: Final = copy.deepcopy(litellm.model_cost)
    litellm.get_model_info.cache_clear()
    litellm.utils._invalidate_model_cost_lowercase_map()
    yield
    litellm.model_cost.clear()
    litellm.model_cost.update(original)
    litellm.utils._invalidate_model_cost_lowercase_map()
    litellm.get_model_info.cache_clear()


@pytest.mark.usefixtures("restore_model_cost_after_sdk_call")
@pytest.mark.parametrize(
    ("model", "rates", "expected_cost_field"),
    (
        pytest.param("openai/gpt-4o-mini", FREE, {"cost": 0.0}, id="D1"),
        pytest.param("openai/gpt-4o-mini", PRICED, {"cost": pytest.approx(0.019)}, id="D2"),
        pytest.param("openai/litellm-unpriced-9097", UNPRICED, {}, id="D3"),
    ),
)
def test_sdk_completion_stream_final_usage_reports_the_computed_cost(
    model: str, rates: Mapping[str, float], expected_cost_field: dict[str, JsonValue]
) -> None:
    identity: Final = "sdk-usage-cost-" + uuid.uuid4().hex
    upstream_model: Final = model.removeprefix("openai/")

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/chat/completions", request.target
        outbound: Final = _JSON_OBJECT.validate_json(request.body)
        assert outbound["model"] == upstream_model and outbound["stream"] is True, outbound
        return Reply(content_type="text/event-stream", chunks=_chat_stream(identity, upstream_model))

    with wire_server(respond) as wire:
        stream: Final = litellm.completion(
            model=model,
            api_base=wire.url + "/v1",
            api_key="synthetic-stream-key",
            messages=[{"role": "user", "content": identity}],
            stream=True,
            stream_options={"include_usage": True},
            num_retries=0,
            **rates,
        )
        try:
            chunks: Final = tuple(stream)
        finally:
            asyncio.run(stream.aclose())
    usages: Final = tuple(chunk.usage for chunk in chunks if getattr(chunk, "usage", None) is not None)
    assert len(usages) == 1, chunks
    usage: Final = usages[0]
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (11, 4, 15), chunks
    usage_dump: Final = _JSON_OBJECT.validate_python(usage.model_dump())
    assert _cost_field(usage_dump) == expected_cost_field, chunks
    requests: Final = wire.drain()
    assert len(requests) == 1 and requests[0].target == "/v1/chat/completions", requests


@pytest.mark.usefixtures("restore_model_cost_after_sdk_call")
@pytest.mark.parametrize("expected_cost_field", (pytest.param({"cost": 0.0}, id="D4"),))
async def test_direct_async_sdk_completion_stream_reports_free_cost(
    expected_cost_field: dict[str, JsonValue],
) -> None:
    identity: Final = "sdk-async-free-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/chat/completions", request.target
        return Reply(content_type="text/event-stream", chunks=_chat_stream(identity, "gpt-4o-mini"))

    with wire_server(respond) as wire:
        stream: Final = await litellm.acompletion(
            model="openai/gpt-4o-mini",
            api_base=wire.url + "/v1",
            api_key="synthetic-stream-key",
            messages=[{"role": "user", "content": identity}],
            stream=True,
            stream_options={"include_usage": True},
            num_retries=0,
            **FREE,
        )
        chunks: Final = tuple([chunk async for chunk in stream])
    usages: Final = tuple(chunk.usage for chunk in chunks if getattr(chunk, "usage", None) is not None)
    assert len(usages) == 1, chunks
    usage: Final = usages[0]
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (11, 4, 15), chunks
    usage_dump: Final = _JSON_OBJECT.validate_python(usage.model_dump())
    assert _cost_field(usage_dump) == expected_cost_field, chunks
    requests: Final = wire.drain()
    assert len(requests) == 1 and requests[0].target == "/v1/chat/completions", requests


@pytest.mark.usefixtures("restore_model_cost_after_sdk_call")
@pytest.mark.parametrize(
    ("name", "cost_entry", "expected_cost_field"),
    (
        pytest.param(
            "litellm-audit-raw-zero-9097",
            {
                "litellm_provider": "openai",
                "mode": "chat",
                "input_cost_per_token": 0.0,
                "output_cost_per_token": 0.0,
            },
            {"cost": 0.0},
            id="D5",
        ),
        pytest.param(
            "litellm-audit-input-zero-only-9097",
            {"litellm_provider": "openai", "mode": "chat", "input_cost_per_token": 0.0},
            {},
            id="D6",
        ),
    ),
)
def test_direct_sdk_raw_cost_map_zero_requires_both_rates(
    name: str,
    cost_entry: Mapping[str, JsonValue],
    expected_cost_field: dict[str, JsonValue],
) -> None:
    identity: Final = "sdk-raw-zero-" + uuid.uuid4().hex
    litellm.register_model(model_cost={name: dict(cost_entry)})

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/chat/completions", request.target
        return Reply(content_type="text/event-stream", chunks=_chat_stream(identity, name))

    with wire_server(respond) as wire:
        stream: Final = litellm.completion(
            model=f"openai/{name}",
            api_base=wire.url + "/v1",
            api_key="synthetic-stream-key",
            messages=[{"role": "user", "content": identity}],
            stream=True,
            stream_options={"include_usage": True},
            num_retries=0,
        )
        try:
            chunks: Final = tuple(stream)
        finally:
            asyncio.run(stream.aclose())
    usages: Final = tuple(chunk.usage for chunk in chunks if getattr(chunk, "usage", None) is not None)
    assert len(usages) == 1, chunks
    usage: Final = usages[0]
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (11, 4, 15), chunks
    usage_dump: Final = _JSON_OBJECT.validate_python(usage.model_dump())
    assert _cost_field(usage_dump) == expected_cost_field, chunks
    requests: Final = wire.drain()
    assert len(requests) == 1 and requests[0].target == "/v1/chat/completions", requests


@pytest.mark.usefixtures("restore_model_cost_after_sdk_call")
@pytest.mark.parametrize("expected_cost_field", (pytest.param({"cost": 0.0}, id="D8"),))
async def test_direct_async_ollama_stream_reports_dynamic_zero_cost(
    expected_cost_field: dict[str, JsonValue],
    record_property: Callable[[str, object], None],
) -> None:
    identity: Final = "sdk-async-ollama-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        if request.target == "/api/show":
            return Reply(status=404)
        assert request.target == "/api/chat", request.target
        return Reply(content_type="application/x-ndjson", chunks=_ollama_stream("litellm-unmapped-9097"))

    with wire_server(respond) as wire:
        stream: Final = await litellm.acompletion(
            model="ollama_chat/litellm-unmapped-9097",
            api_base=wire.url,
            messages=[{"role": "user", "content": identity}],
            stream=True,
            stream_options={"include_usage": True},
            num_retries=0,
        )
        chunks: Final = tuple([chunk async for chunk in stream])
        requests: Final = wire.drain()
    chat_requests: Final = tuple(request for request in requests if request.target == "/api/chat")
    show_requests: Final = tuple(request for request in requests if request.target == "/api/show")
    _record_observed(record_property, "observed_api_show_count", len(show_requests))
    assert len(chat_requests) == 1, requests
    assert all(request.target == "/api/show" for request in requests if request.target != "/api/chat"), requests
    usages: Final = tuple(chunk.usage for chunk in chunks if getattr(chunk, "usage", None) is not None)
    assert len(usages) == 1, chunks
    usage: Final = usages[0]
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (11, 4, 15), chunks
    usage_dump: Final = _JSON_OBJECT.validate_python(usage.model_dump())
    assert _cost_field(usage_dump) == expected_cost_field, chunks


@pytest.mark.usefixtures("restore_model_cost_after_sdk_call")
@pytest.mark.parametrize("expected_cost_field", (pytest.param({"cost": 0.0}, id="D9"),))
async def test_direct_async_responses_stream_reports_free_cost(
    expected_cost_field: dict[str, JsonValue],
) -> None:
    identity: Final = "sdk-async-responses-free-" + uuid.uuid4().hex
    model: Final = "litellm-free-responses-9097"
    litellm.register_model(
        model_cost={
            model: {
                "litellm_provider": "openai",
                "mode": "responses",
                "input_cost_per_token": 0.0,
                "output_cost_per_token": 0.0,
            }
        },
        persist_across_reloads=False,
    )

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/responses", request.target
        return Reply(content_type="text/event-stream", chunks=_responses_stream(identity, model))

    with wire_server(respond) as wire:
        stream: Final = await litellm.aresponses(
            model=f"openai/{model}",
            api_base=wire.url + "/v1",
            api_key="synthetic-stream-key",
            input=identity,
            stream=True,
            num_retries=0,
        )
        events: Final = tuple([event async for event in stream])
    completed: Final = tuple(event for event in events if event.type == "response.completed")
    assert len(completed) == 1, events
    usage: Final = completed[0].response.usage
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (11, 4, 15), events
    usage_dump: Final = _JSON_OBJECT.validate_python(usage.model_dump(exclude_none=True))
    assert _cost_field(usage_dump) == expected_cost_field, events
    requests: Final = wire.drain()
    assert len(requests) == 1 and requests[0].target == "/v1/responses", requests


@pytest.mark.usefixtures("restore_model_cost_after_sdk_call")
@pytest.mark.parametrize("expected_cost_field", (pytest.param({}, id="D10"),))
def test_direct_sdk_unmapped_bedrock_style_provider_omits_cost(
    expected_cost_field: dict[str, JsonValue],
) -> None:
    identity: Final = "sdk-bedrock-unmapped-" + uuid.uuid4().hex
    model: Final = "bedrock/converse/litellm-unmapped-9097"
    events: Final = (
        ("messageStart", {"role": "assistant"}),
        ("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": "Hello there"}}),
        ("messageStop", {"stopReason": "end_turn"}),
        ("metadata", {"usage": {"inputTokens": 11, "outputTokens": 4, "totalTokens": 15}}),
    )

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/model/litellm-unmapped-9097/converse-stream", (
            request.target
        )
        return Reply(
            content_type="application/vnd.amazon.eventstream",
            body=b"".join(_aws_event_frame(kind, payload, identity, "chunk-1") for kind, payload in events),
        )

    with wire_server(respond) as wire:
        stream: Final = litellm.completion(
            model=model,
            api_base=wire.url,
            api_key="synthetic-bedrock-key",
            aws_access_key_id="synthetic-access-key",
            aws_secret_access_key="synthetic-secret-key",
            aws_region_name="us-east-1",
            messages=[{"role": "user", "content": identity}],
            stream=True,
            stream_options={"include_usage": True},
            num_retries=0,
        )
        try:
            chunks: Final = tuple(stream)
        finally:
            asyncio.run(stream.aclose())
    usages: Final = tuple(chunk.usage for chunk in chunks if getattr(chunk, "usage", None) is not None)
    assert len(usages) == 1, chunks
    usage: Final = usages[0]
    usage_dump: Final = _JSON_OBJECT.validate_python(usage.model_dump())
    assert _cost_field(usage_dump) == expected_cost_field, chunks
    requests: Final = wire.drain()
    assert len(requests) == 1 and requests[0].target == "/model/litellm-unmapped-9097/converse-stream", requests


@pytest.mark.parametrize(
    ("endpoint", "deployment_model", "rates", "expected_cost", "row_id"),
    (
        pytest.param(
            "/v1/chat/completions",
            "openai/gpt-4o-mini",
            FREE,
            0.0,
            "C1-chat",
            id="C1-chat",
        ),
        pytest.param("/v1/responses", "openai/gpt-4o-mini", FREE, 0.0, "C1-responses", id="C1-responses"),
        pytest.param(
            "/v1/chat/completions",
            "openai/gpt-4o-mini",
            PRICED,
            0.019,
            "C2-chat",
            id="C2-chat",
        ),
        pytest.param("/v1/responses", "openai/gpt-4o-mini", PRICED, 0.019, "C2-responses", id="C2-responses"),
        pytest.param(
            "/v1/chat/completions",
            "openai/litellm-unpriced-9097",
            UNPRICED,
            0.0,
            "C3-chat",
            id="C3-chat",
        ),
    ),
)
def test_streamed_usage_cost_is_persisted_by_response_id(
    gateway: Gateway,
    endpoint: str,
    deployment_model: str,
    rates: Mapping[str, float],
    expected_cost: float,
    row_id: str,
    record_property: Callable[[str, object], None],
) -> None:
    identity: Final = f"spend-{row_id}-" + uuid.uuid4().hex
    upstream_model: Final = deployment_model.removeprefix("openai/")
    model_id: Final = str(uuid.uuid4())

    def respond(request: Request) -> Reply:
        model_list: Final = _model_list_reply(request)
        if model_list is not None:
            return model_list
        if endpoint.endswith("responses"):
            assert request.method == "POST" and request.target == "/v1/responses", request.target
            return Reply(content_type="text/event-stream", chunks=_responses_stream(identity, upstream_model))
        assert request.method == "POST" and request.target == "/v1/chat/completions", request.target
        return Reply(content_type="text/event-stream", chunks=_chat_stream(identity, upstream_model))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=deployment_model,
            api_base=wire.url + "/v1",
            model_info={"id": model_id},
            **rates,
        )
        body: Final = (
            {"model": model, "input": identity, "stream": True}
            if endpoint.endswith("responses")
            else {
                "model": model,
                "messages": [{"role": "user", "content": identity}],
                "stream": True,
                "stream_options": {"include_usage": True},
            }
        )
        status, headers, text = _proxy_stream(gateway, endpoint, body)
    assert status == 200, text
    call_id: Final = headers.get("x-litellm-call-id")
    assert call_id, headers
    events: Final = _sse_events(text)
    response_ids: Final = (
        tuple(str(event["id"]) for event in events if isinstance(event.get("id"), str))
        if endpoint.endswith("chat/completions")
        else tuple(
            str(_JSON_OBJECT.validate_python(event["response"])["id"])
            for event in events
            if isinstance(event.get("response"), dict)
        )
    )
    assert response_ids, text
    if endpoint.endswith("chat/completions"):
        assert identity in response_ids, text
    spend_request_id: Final = (
        response_ids[0] if endpoint.endswith("chat/completions") else _responses_spend_request_id(model_id, identity)
    )
    _record_observed(record_property, "spend_row_query_request_id", spend_request_id)
    _record_observed(record_property, "spend_row_call_id", call_id)
    _record_observed(record_property, "observed_response_body_ids", response_ids)
    _record_observed(record_property, "observed_model_id", model_id)
    _record_observed(record_property, "upstream_response_id", identity)
    rows: Final = eventually(
        lambda: read_rows(
            "SELECT request_id, litellm_call_id, spend, prompt_tokens, completion_tokens "
            'FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
            (spend_request_id,),
        ),
        lambda values: len(values) == 1,
        seconds=70,
    )
    assert str(rows[0]["request_id"]) == spend_request_id, text
    assert str(rows[0]["litellm_call_id"]) == call_id, text
    assert (rows[0]["prompt_tokens"], rows[0]["completion_tokens"]) == (11, 4), text
    assert float(rows[0]["spend"]) == pytest.approx(expected_cost), text
    requests: Final = wire.drain()
    expected_target: Final = "/v1/responses" if endpoint.endswith("responses") else "/v1/chat/completions"
    posts: Final = tuple(request for request in requests if request.method == "POST")
    gets: Final = tuple(request for request in requests if request.method == "GET")
    assert len(posts) == 1 and posts[0].target == expected_target, text
    assert all(request.target == "/v1/models" for request in gets), requests


@pytest.mark.parametrize(
    ("deployment_model", "rates", "expected_header"),
    (
        pytest.param("openai/gpt-4o-mini", FREE, None, id="C4-free"),
        pytest.param("openai/gpt-4o-mini", PRICED, None, id="C4-priced"),
        pytest.param("openai/litellm-unpriced-9097", UNPRICED, None, id="C4-unpriced"),
    ),
)
def test_streamed_response_cost_header_remains_stable(
    gateway: Gateway,
    deployment_model: str,
    rates: Mapping[str, float],
    expected_header: str | None,
    record_property: Callable[[str, object], None],
) -> None:
    identity: Final = "cost-header-" + uuid.uuid4().hex
    upstream_model: Final = deployment_model.removeprefix("openai/")

    def respond(request: Request) -> Reply:
        assert request.target == "/v1/chat/completions", request.target
        return Reply(content_type="text/event-stream", chunks=_chat_stream(identity, upstream_model))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=deployment_model, api_base=wire.url + "/v1", **rates)
        status, headers, text = _proxy_stream(
            gateway,
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": identity}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
    assert status == 200, text
    observed: Final = headers.get("x-litellm-response-cost")
    _record_observed(record_property, "observed_response_cost_header", observed)
    assert observed == expected_header, text
    requests: Final = wire.drain()
    assert len(requests) == 1 and requests[0].target == "/v1/chat/completions", text


@pytest.mark.parametrize(
    ("usage", "expected_cost_field"),
    (pytest.param({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, {}, id="E3"),),
)
def test_zero_token_usage_does_not_claim_a_zero_rate_was_used(
    gateway: Gateway, usage: Mapping[str, JsonValue], expected_cost_field: dict[str, JsonValue]
) -> None:
    identity: Final = "zero-token-usage-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        assert request.target == "/v1/chat/completions", request.target
        return Reply(content_type="text/event-stream", chunks=_chat_stream(identity, "gpt-4o-mini", usage))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=wire.url + "/v1", **FREE)
        status, _, text = _proxy_stream(
            gateway,
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": identity}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
    assert status == 200, text
    usages: Final = _usage_events(text)
    assert len(usages) == 1, text
    assert (usages[0]["prompt_tokens"], usages[0]["completion_tokens"], usages[0]["total_tokens"]) == (0, 0, 0), text
    assert _cost_field(usages[0]) == expected_cost_field, text
    requests: Final = wire.drain()
    assert len(requests) == 1 and requests[0].target == "/v1/chat/completions", text


@pytest.mark.parametrize(
    ("rates", "expected_cost_field"),
    (
        pytest.param({"input_cost_per_token": 0.0}, {}, id="E4"),
        pytest.param(
            {"input_cost_per_token": 0.0, "output_cost_per_token": 0.002},
            {"cost": pytest.approx(0.008)},
            id="E5",
        ),
    ),
)
def test_input_zero_output_rate_stream_cost(
    gateway: Gateway, rates: Mapping[str, float], expected_cost_field: dict[str, JsonValue]
) -> None:
    identity: Final = "input-zero-output-rate-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        assert request.target == "/v1/chat/completions", request.target
        return Reply(content_type="text/event-stream", chunks=_chat_stream(identity, "litellm-unpriced-9097"))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/litellm-unpriced-9097",
            api_base=wire.url + "/v1",
            **rates,
        )
        status, _, text = _proxy_stream(
            gateway,
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": identity}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
    assert status == 200, text
    usages: Final = _usage_events(text)
    assert len(usages) == 1, text
    assert _cost_field(usages[0]) == expected_cost_field, text
    requests: Final = wire.drain()
    assert len(requests) == 1 and requests[0].target == "/v1/chat/completions", text


def test_e1_free_stream_recovers_after_upstream_500(gateway: Gateway) -> None:
    identity: Final = "free-500-recovery-" + uuid.uuid4().hex
    requests_seen: Final = count()

    def respond(request: Request) -> Reply:
        index: Final = next(requests_seen)
        assert request.target == "/v1/chat/completions", request.target
        if index == 0:
            return Reply(status=500, body=b'{"error":{"message":"controlled upstream failure"}}')
        return Reply(content_type="text/event-stream", chunks=_chat_stream(identity, "gpt-4o-mini"))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=wire.url + "/v1", **FREE)
        first_status, _, first_text = _proxy_stream(
            gateway,
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": identity}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        second_status, _, second_text = _proxy_stream(
            gateway,
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": identity + "-recovery"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
    assert first_status >= 500 and "controlled upstream failure" in first_text, first_text
    assert _usage_events(first_text) == (), first_text
    assert second_status == 200, second_text
    second_usages: Final = _usage_events(second_text)
    assert len(second_usages) == 1, second_text
    assert (
        second_usages[0]["prompt_tokens"],
        second_usages[0]["completion_tokens"],
        second_usages[0]["total_tokens"],
    ) == (11, 4, 15), second_text
    health: Final = gateway.client.get("/health/liveliness")
    assert health.status_code == 200, health.text
    requests: Final = wire.drain()
    assert len(requests) == 2 and all(request.target == "/v1/chat/completions" for request in requests), requests


def test_truncated_free_stream_does_not_fabricate_usage_cost(gateway: Gateway) -> None:
    identity: Final = "truncated-free-stream-" + uuid.uuid4().hex
    requests_seen: Final = count()
    first_frame: Final = _chat_frame(
        identity,
        {
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello"}, "finish_reason": None}],
        },
    )

    def respond(request: Request) -> Reply:
        model_list: Final = _model_list_reply(request)
        if model_list is not None:
            return model_list
        assert request.method == "POST" and request.target == "/v1/chat/completions", request.target
        if next(requests_seen) == 0:
            return Reply(content_type="text/event-stream", chunks=(first_frame, b"unused"), abort_after=1)
        return Reply(content_type="text/event-stream", chunks=_chat_stream(identity, "gpt-4o-mini"))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=wire.url + "/v1", **FREE)
        with gateway.client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": identity}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            headers={"Authorization": f"Bearer {gateway.key}"},
        ) as response:
            status: Final = response.status_code
            text: Final = _read_stream_text(response)
        assert status == 200 or status >= 400, text
        if status == 200:
            assert "[DONE]" not in text, text
        frame_usage: Final = tuple(
            _JSON_OBJECT.validate_python(event["usage"])
            for event in _sse_events(text)
            if isinstance(event.get("usage"), dict)
        )
        body: Final = _JSON_OBJECT.validate_json(text) if text.lstrip().startswith("{") else {}
        body_usage: Final = body.get("usage")
        received_usage: Final = frame_usage + (
            (_JSON_OBJECT.validate_python(body_usage),) if isinstance(body_usage, dict) else ()
        )
        assert all(_cost_field(usage) == {} for usage in received_usage), text
        health: Final = gateway.client.get("/health/liveliness")
        assert health.status_code == 200, health.text
        next_status, _, next_text = _proxy_stream(
            gateway,
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": identity + "-recovery"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        assert next_status == 200, next_text
        next_usages: Final = _usage_events(next_text)
        assert len(next_usages) == 1, next_text
        assert (
            next_usages[0]["prompt_tokens"],
            next_usages[0]["completion_tokens"],
            next_usages[0]["total_tokens"],
        ) == (11, 4, 15), next_text
        requests: Final = wire.drain()
        posts: Final = tuple(request for request in requests if request.method == "POST")
        gets: Final = tuple(request for request in requests if request.method == "GET")
        assert len(posts) == 2 and all(request.target == "/v1/chat/completions" for request in posts), next_text
        assert all(request.target == "/v1/models" for request in gets), requests


def test_free_stream_without_authorization_never_reaches_upstream(gateway: Gateway) -> None:
    identity: Final = "no-auth-stream-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        return Reply(content_type="text/event-stream", chunks=_chat_stream(identity, "gpt-4o-mini"))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=wire.url + "/v1", **FREE)
        response: Final = gateway.client.post(
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": identity}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
    assert response.status_code == 401, response.text
    requests: Final = wire.drain()
    assert requests == (), requests


def test_unknown_model_stream_never_reaches_upstream(gateway: Gateway) -> None:
    identity: Final = "unknown-model-stream-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        return Reply(content_type="text/event-stream", chunks=_chat_stream(identity, "unreachable"))

    with wire_server(respond) as wire:
        response: Final = gateway.client.post(
            "/v1/chat/completions",
            json={
                "model": "litellm-audit-unknown-9097",
                "messages": [{"role": "user", "content": identity}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            headers={"Authorization": f"Bearer {gateway.key}"},
        )
    assert response.status_code == 400, response.text
    requests: Final = wire.drain()
    assert requests == (), requests


@pytest.mark.parametrize(
    ("endpoint", "expected_cost_field"),
    (
        pytest.param("/v1/chat/completions", {"cost": pytest.approx(0.25)}, id="E8-chat"),
        pytest.param("/v1/responses", {"cost": pytest.approx(0.25)}, id="E8-responses"),
    ),
)
def test_provider_reported_usage_cost_is_preserved(
    gateway: Gateway,
    endpoint: str,
    expected_cost_field: dict[str, JsonValue],
    record_property: Callable[[str, object], None],
) -> None:
    identity: Final = "provider-reported-cost-" + uuid.uuid4().hex
    provider_usage: Final = (
        {"input_tokens": 11, "output_tokens": 4, "total_tokens": 15, "cost": 0.25}
        if endpoint.endswith("responses")
        else {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15, "cost": 0.25}
    )

    def respond(request: Request) -> Reply:
        if endpoint.endswith("responses"):
            assert request.target == "/v1/responses", request.target
            return Reply(
                content_type="text/event-stream", chunks=_responses_stream(identity, "gpt-4o-mini", provider_usage)
            )
        assert request.target == "/v1/chat/completions", request.target
        return Reply(content_type="text/event-stream", chunks=_chat_stream(identity, "gpt-4o-mini", provider_usage))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=wire.url + "/v1", **PRICED)
        body: Final = (
            {"model": model, "input": identity, "stream": True}
            if endpoint.endswith("responses")
            else {
                "model": model,
                "messages": [{"role": "user", "content": identity}],
                "stream": True,
                "stream_options": {"include_usage": True},
            }
        )
        status, _, text = _proxy_stream(gateway, endpoint, body)
    assert status == 200, text
    if endpoint.endswith("responses"):
        completed: Final = tuple(event for event in _sse_events(text) if event["type"] == "response.completed")
        assert len(completed) == 1, text
        usage: Final = _JSON_OBJECT.validate_python(completed[0]["response"]["usage"])
    else:
        usages: Final = _usage_events(text)
        assert len(usages) == 1, text
        usage = usages[0]
    observed: Final = _cost_field(usage)
    _record_observed(record_property, "observed_cost_field", observed)
    assert observed == expected_cost_field, text
    requests: Final = wire.drain()
    expected_target: Final = "/v1/responses" if endpoint.endswith("responses") else "/v1/chat/completions"
    assert len(requests) == 1 and requests[0].target == expected_target, text


@pytest.mark.parametrize(
    ("stream_options", "row_id"),
    (
        pytest.param({"include_usage": False}, "false", id="F1-false"),
        pytest.param(None, "omitted", id="F1-omitted"),
    ),
)
def test_free_chat_stream_without_include_usage_keeps_base_shape(
    gateway: Gateway,
    stream_options: Mapping[str, JsonValue] | None,
    row_id: str,
    record_property: Callable[[str, object], None],
) -> None:
    identity: Final = f"no-include-usage-{row_id}-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        assert request.target == "/v1/chat/completions", request.target
        return Reply(
            content_type="text/event-stream",
            chunks=_chat_stream(identity, "gpt-4o-mini", include_usage=False),
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=wire.url + "/v1", **FREE)
        request_body: Final = {
            "model": model,
            "messages": [{"role": "user", "content": identity}],
            "stream": True,
            **({"stream_options": dict(stream_options)} if stream_options is not None else {}),
        }
        status, _, text = _proxy_stream(gateway, "/v1/chat/completions", request_body)
    assert status == 200, text
    observed: Final = tuple(_cost_field(usage) for usage in _usage_events(text))
    _record_observed(record_property, "observed_usage_cost_fields", observed)
    assert observed == (), text
    requests: Final = wire.drain()
    assert len(requests) == 1 and requests[0].target == "/v1/chat/completions", text


def test_free_chat_stream_with_n_two_reports_combined_usage_cost(gateway: Gateway) -> None:
    identity: Final = "n-two-free-stream-" + uuid.uuid4().hex
    usage: Final = {"prompt_tokens": 11, "completion_tokens": 8, "total_tokens": 19}

    def respond(request: Request) -> Reply:
        assert request.target == "/v1/chat/completions", request.target
        outbound: Final = _JSON_OBJECT.validate_json(request.body)
        assert outbound["n"] == 2, outbound
        return Reply(
            content_type="text/event-stream",
            chunks=_chat_stream(identity, "gpt-4o-mini", usage, choice_count=2),
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=wire.url + "/v1", **FREE)
        status, _, text = _proxy_stream(
            gateway,
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": identity}],
                "n": 2,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
    assert status == 200, text
    usages: Final = _usage_events(text)
    assert len(usages) == 1, text
    assert (usages[0]["prompt_tokens"], usages[0]["completion_tokens"], usages[0]["total_tokens"]) == (11, 8, 19), text
    assert _cost_field(usages[0]) == {"cost": 0.0}, text
    requests: Final = wire.drain()
    assert len(requests) == 1 and requests[0].target == "/v1/chat/completions", text


@pytest.mark.parametrize(
    ("rates", "expected_cost_field", "row_id"),
    (
        pytest.param(FREE, {"cost": 0.0}, "free", id="F4"),
        pytest.param(PRICED, {"cost": pytest.approx(0.019)}, "priced", id="F5"),
    ),
)
def test_cached_chat_stream_hit_replays_usage_cost(
    gateway: Gateway,
    rates: Mapping[str, float],
    expected_cost_field: dict[str, JsonValue],
    row_id: str,
    record_property: Callable[[str, object], None],
) -> None:
    identity: Final = f"cached-chat-{row_id}-" + uuid.uuid4().hex
    upstream_model: Final = "gpt-4o-mini"

    def respond(request: Request) -> Reply:
        model_list: Final = _model_list_reply(request)
        if model_list is not None:
            return model_list
        assert request.method == "POST" and request.target == "/v1/chat/completions", request.target
        return Reply(content_type="text/event-stream", chunks=_chat_stream(identity, upstream_model))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=wire.url + "/v1", **rates)
        body: Final = {
            "model": model,
            "messages": [{"role": "user", "content": identity}],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        first_status, _, first_text = _proxy_stream(gateway, "/v1/chat/completions", body)
        second_status, _, second_text = _proxy_stream(gateway, "/v1/chat/completions", body)
    assert first_status == second_status == 200, f"{first_text}\n{second_text}"
    first_usages: Final = _usage_events(first_text)
    second_usages: Final = _usage_events(second_text)
    assert len(first_usages) == len(second_usages) == 1, f"{first_text}\n{second_text}"
    first_cost: Final = _cost_field(first_usages[0])
    second_cost: Final = _cost_field(second_usages[0])
    _record_observed(record_property, "observed_first_raw_usage", first_usages[0])
    _record_observed(record_property, "observed_cached_raw_usage", second_usages[0])
    _record_observed(record_property, "observed_first_cost_field", first_cost)
    _record_observed(record_property, "observed_cached_cost_field", second_cost)
    assert first_cost == expected_cost_field, first_text
    assert second_cost == expected_cost_field, second_text
    requests: Final = wire.drain()
    posts: Final = tuple(request for request in requests if request.method == "POST")
    gets: Final = tuple(request for request in requests if request.method == "GET")
    assert len(posts) == 1, f"cache upstream POSTs={len(posts)}\n{second_text}"
    assert posts[0].target == "/v1/chat/completions", posts
    assert all(request.target == "/v1/models" for request in gets), requests


@pytest.mark.parametrize(
    ("expected_cost_field", "row_id"),
    (pytest.param({"cost": 0.0}, "responses", id="F6"),),
)
def test_cached_responses_stream_hit_replays_free_usage_cost(
    gateway: Gateway,
    expected_cost_field: dict[str, JsonValue],
    row_id: str,
    record_property: Callable[[str, object], None],
) -> None:
    identity: Final = f"cached-responses-{row_id}-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        model_list: Final = _model_list_reply(request)
        if model_list is not None:
            return model_list
        assert request.method == "POST" and request.target == "/v1/responses", request.target
        return Reply(content_type="text/event-stream", chunks=_responses_stream(identity, "gpt-4o-mini"))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=wire.url + "/v1", **FREE)
        body: Final = {"model": model, "input": identity, "stream": True}
        first_status, _, first_text = _proxy_stream(gateway, "/v1/responses", body)
        second_status, _, second_text = _proxy_stream(gateway, "/v1/responses", body)
    assert first_status == second_status == 200, f"{first_text}\n{second_text}"
    first_completed: Final = tuple(event for event in _sse_events(first_text) if event["type"] == "response.completed")
    second_completed: Final = tuple(
        event for event in _sse_events(second_text) if event["type"] == "response.completed"
    )
    assert len(first_completed) == len(second_completed) == 1, f"{first_text}\n{second_text}"
    first_usage: Final = _JSON_OBJECT.validate_python(first_completed[0]["response"]["usage"])
    second_usage: Final = _JSON_OBJECT.validate_python(second_completed[0]["response"]["usage"])
    first_cost: Final = _cost_field(first_usage)
    second_cost: Final = _cost_field(second_usage)
    _record_observed(record_property, "observed_first_raw_usage", first_usage)
    _record_observed(record_property, "observed_cached_raw_usage", second_usage)
    _record_observed(record_property, "observed_first_cost_field", first_cost)
    _record_observed(record_property, "observed_cached_cost_field", second_cost)
    assert first_cost == expected_cost_field, first_text
    assert second_cost == expected_cost_field, second_text
    requests: Final = wire.drain()
    posts: Final = tuple(request for request in requests if request.method == "POST")
    gets: Final = tuple(request for request in requests if request.method == "GET")
    assert len(posts) == 1, f"cache upstream POSTs={len(posts)}\n{second_text}"
    assert posts[0].target == "/v1/responses", posts
    assert all(request.target == "/v1/models" for request in gets), requests


def test_identical_free_streams_with_cache_off_create_separate_spend_rows(gateway: Gateway) -> None:
    identity: Final = "duplicate-free-stream-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        model_list: Final = _model_list_reply(request)
        if model_list is not None:
            return model_list
        assert request.method == "POST" and request.target == "/v1/chat/completions", request.target
        return Reply(
            content_type="text/event-stream",
            chunks=_chat_stream(uuid.uuid4().hex, "gpt-4o-mini"),
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=wire.url + "/v1", **FREE)
        body: Final = {
            "model": model,
            "messages": [{"role": "user", "content": identity}],
            "stream": True,
            "stream_options": {"include_usage": True},
            "cache": {"no-cache": True},
        }
        first_status, first_headers, first_text = _proxy_stream(gateway, "/v1/chat/completions", body)
        second_status, second_headers, second_text = _proxy_stream(gateway, "/v1/chat/completions", body)
    assert first_status == second_status == 200, f"{first_text}\n{second_text}"
    first_usages: Final = _usage_events(first_text)
    second_usages: Final = _usage_events(second_text)
    assert len(first_usages) == len(second_usages) == 1, f"{first_text}\n{second_text}"
    assert _cost_field(first_usages[0]) == {"cost": 0.0}, first_text
    assert _cost_field(second_usages[0]) == {"cost": 0.0}, second_text
    first_events: Final = _sse_events(first_text)
    second_events: Final = _sse_events(second_text)
    assert first_events and second_events, f"{first_text}\n{second_text}"
    first_response_id: Final = first_events[0].get("id")
    second_response_id: Final = second_events[0].get("id")
    assert isinstance(first_response_id, str) and isinstance(second_response_id, str), f"{first_text}\n{second_text}"
    call_ids: Final = (
        first_headers.get("x-litellm-call-id"),
        second_headers.get("x-litellm-call-id"),
    )
    assert all(call_id is not None for call_id in call_ids), (first_headers, second_headers)
    spend_row_ids: Final = (first_response_id, second_response_id)
    assert len(set(spend_row_ids)) == 2, spend_row_ids
    validated_call_ids: Final = tuple(call_id for call_id in call_ids if call_id is not None)
    expected_call_ids: Final = dict(zip(spend_row_ids, validated_call_ids, strict=True))
    placeholders: Final = ",".join("%s" for _ in spend_row_ids)
    rows: Final = eventually(
        lambda: read_rows(
            f'SELECT request_id, litellm_call_id, spend FROM "LiteLLM_SpendLogs" WHERE request_id IN ({placeholders})',
            spend_row_ids,
        ),
        lambda values: len(values) == 2,
        seconds=70,
    )
    assert {str(row["request_id"]) for row in rows} == set(spend_row_ids), rows
    observed_call_ids: Final = {str(row["request_id"]): str(row["litellm_call_id"]) for row in rows}
    assert observed_call_ids == expected_call_ids, rows
    assert all(float(row["spend"]) == pytest.approx(0.0) for row in rows), rows
    requests: Final = wire.drain()
    posts: Final = tuple(request for request in requests if request.method == "POST")
    gets: Final = tuple(request for request in requests if request.method == "GET")
    assert len(posts) == 2 and all(request.target == "/v1/chat/completions" for request in posts), requests
    assert all(request.target == "/v1/models" for request in gets), requests


def test_shared_backend_deployments_keep_cost_bound_to_served_model_id(
    gateway: Gateway, tmp_path: Path, record_property: Callable[[str, object], None]
) -> None:
    base_config: Final = _JSON_OBJECT.validate_python(
        yaml.safe_load((Path(__file__).resolve().parents[1] / "proxy_config.yaml").read_text())
    )
    existing_router: Final = _JSON_OBJECT.validate_python(base_config.get("router_settings", {}))
    configuration: Final = {
        **base_config,
        "model_list": [
            {
                "model_name": "audit-shared-gpt-mini-9097",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_base": "http://127.0.0.1",
                    "api_key": "synthetic-f7-key",
                    **FREE,
                },
                "model_info": {"id": "audit-free-9097"},
            },
            {
                "model_name": "audit-shared-gpt-mini-9097",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_base": "http://127.0.0.1",
                    "api_key": "synthetic-f7-key",
                    **PRICED,
                },
                "model_info": {"id": "audit-priced-9097"},
            },
        ],
        "router_settings": {
            **existing_router,
            "routing_strategy": "simple-shuffle",
            "num_retries": 0,
            "disable_cooldowns": True,
        },
    }
    config_path: Final = tmp_path / "shared-backend-cost.yaml"
    config_path.write_text(yaml.safe_dump(configuration))

    def respond(request: Request) -> Reply:
        model_list: Final = _model_list_reply(request)
        if model_list is not None:
            return model_list
        assert request.method == "POST" and request.target == "/v1/chat/completions", request.target
        outbound: Final = _JSON_OBJECT.validate_json(request.body)
        assert outbound["model"] == "gpt-4o-mini", outbound
        prompt: Final = _JSON_OBJECT.validate_python(outbound["messages"][0])
        identity: Final = str(prompt["content"])
        return Reply(content_type="text/event-stream", chunks=_chat_stream(identity, "gpt-4o-mini"))

    with wire_server(respond) as wire:
        model_entries: Final = [
            {
                **_JSON_OBJECT.validate_python(entry),
                "litellm_params": {
                    **_JSON_OBJECT.validate_python(_JSON_OBJECT.validate_python(entry)["litellm_params"]),
                    "api_base": wire.url + "/v1",
                },
            }
            for entry in configuration["model_list"]
        ]
        running_configuration: Final = {**configuration, "model_list": model_entries}
        config_path.write_text(yaml.safe_dump(running_configuration))
        with owned_proxy(gateway, tmp_path, {}, config=config_path) as candidate:
            _wait_for_proxy_models(candidate, ("audit-shared-gpt-mini-9097",))
            results: Final = tuple(_send_shared_model_request(candidate, index) for index in range(20))
    free_results: Final = tuple(result for result in results if result[0] == "audit-free-9097")
    priced_results: Final = tuple(result for result in results if result[0] == "audit-priced-9097")
    assert free_results and priced_results, tuple(result[0] for result in results)
    _record_observed(record_property, "observed_free_cost_fields", tuple(result[1] for result in free_results))
    _record_observed(record_property, "observed_priced_cost_fields", tuple(result[1] for result in priced_results))
    assert all(result[1] == {"cost": 0.0} for result in free_results), free_results
    assert all(result[1] == {"cost": pytest.approx(0.019)} for result in priced_results), priced_results
    requests: Final = wire.drain()
    posts: Final = tuple(request for request in requests if request.method == "POST")
    gets: Final = tuple(request for request in requests if request.method == "GET")
    assert len(posts) == 20 and all(request.target == "/v1/chat/completions" for request in posts), requests
    assert all(request.target == "/v1/models" for request in gets), requests


def _send_shared_model_request(candidate: Gateway, index: int) -> tuple[str, dict[str, JsonValue]]:
    identity: Final = f"shared-backend-{index}-" + uuid.uuid4().hex
    response: Final = candidate.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": "audit-shared-gpt-mini-9097",
            "messages": [{"role": "user", "content": identity}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    assert response.status_code == 200, response.text
    model_id: Final = response.headers.get("x-litellm-model-id")
    assert model_id in {"audit-free-9097", "audit-priced-9097"}, response.text
    usages: Final = _usage_events(response.text)
    assert len(usages) == 1, response.text
    return model_id, _cost_field(usages[0])


@pytest.mark.parametrize(
    ("deployment_model", "rates", "expected_cost_field"),
    (
        pytest.param("openai/gpt-4o-mini", FREE, {"cost": 0.0}, id="F8-free"),
        pytest.param("openai/gpt-4o-mini", PRICED, {"cost": pytest.approx(0.019)}, id="F8-priced"),
        pytest.param("openai/litellm-unpriced-9097", UNPRICED, {}, id="F8-unpriced"),
    ),
)
def test_owned_proxy_injects_cost_into_streaming_usage_when_configured(
    gateway: Gateway,
    tmp_path: Path,
    deployment_model: str,
    rates: Mapping[str, float],
    expected_cost_field: dict[str, JsonValue],
    record_property: Callable[[str, object], None],
) -> None:
    identity: Final = "include-cost-setting-" + uuid.uuid4().hex
    upstream_model: Final = deployment_model.removeprefix("openai/")
    base_config: Final = _JSON_OBJECT.validate_python(
        yaml.safe_load((Path(__file__).resolve().parents[1] / "proxy_config.yaml").read_text())
    )
    existing_settings: Final = _JSON_OBJECT.validate_python(base_config.get("litellm_settings", {}))
    existing_router: Final = _JSON_OBJECT.validate_python(base_config.get("router_settings", {}))
    configuration: Final = {
        **base_config,
        "litellm_settings": {**existing_settings, "include_cost_in_streaming_usage": True},
        "router_settings": {**existing_router, "num_retries": 0},
    }
    config_path: Final = tmp_path / "include-stream-cost.yaml"
    config_path.write_text(yaml.safe_dump(configuration))

    def respond(request: Request) -> Reply:
        model_list: Final = _model_list_reply(request)
        if model_list is not None:
            return model_list
        assert request.method == "POST" and request.target == "/v1/chat/completions", request.target
        return Reply(content_type="text/event-stream", chunks=_chat_stream(identity, upstream_model))

    with wire_server(respond) as wire, owned_proxy(gateway, tmp_path, {}, config=config_path) as candidate:
        with candidate.scenario() as scenario:
            model: Final = scenario.model(model=deployment_model, api_base=wire.url + "/v1", **rates)
            _wait_for_proxy_models(candidate, (model,))
            status, _, text = _proxy_stream(
                candidate,
                "/v1/chat/completions",
                {
                    "model": model,
                    "messages": [{"role": "user", "content": identity}],
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
            )
    assert status == 200, text
    usages: Final = _usage_events(text)
    assert len(usages) == 1, text
    observed: Final = _cost_field(usages[0])
    _record_observed(record_property, "observed_raw_usage", usages[0])
    _record_observed(record_property, "observed_cost_field", observed)
    assert observed == expected_cost_field, text
    requests: Final = wire.drain()
    posts: Final = tuple(request for request in requests if request.method == "POST")
    gets: Final = tuple(request for request in requests if request.method == "GET")
    assert len(posts) == 1 and posts[0].target == "/v1/chat/completions", text
    assert all(request.target == "/v1/models" for request in gets), requests


def test_priced_primary_failure_falls_back_to_free_stream(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "priced-primary-free-fallback-" + uuid.uuid4().hex
    base_config: Final = _JSON_OBJECT.validate_python(
        yaml.safe_load((Path(__file__).resolve().parents[1] / "proxy_config.yaml").read_text())
    )
    existing_router: Final = _JSON_OBJECT.validate_python(base_config.get("router_settings", {}))
    configuration: Final = {
        **base_config,
        "model_list": [
            {
                "model_name": "audit-priced-primary-9097",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_base": "http://127.0.0.1",
                    "api_key": "synthetic-f9-key",
                    **PRICED,
                },
                "model_info": {"id": "audit-priced-primary-9097"},
            },
            {
                "model_name": "audit-free-fallback-9097",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_base": "http://127.0.0.1",
                    "api_key": "synthetic-f9-key",
                    **FREE,
                },
                "model_info": {"id": "audit-free-fallback-9097"},
            },
        ],
        "router_settings": {
            **existing_router,
            "num_retries": 0,
            "disable_cooldowns": True,
            "fallbacks": [{"audit-priced-primary-9097": ["audit-free-fallback-9097"]}],
        },
    }
    config_path: Final = tmp_path / "priced-to-free-fallback.yaml"
    config_path.write_text(yaml.safe_dump(configuration))

    def fail(request: Request) -> Reply:
        model_list: Final = _model_list_reply(request)
        if model_list is not None:
            return model_list
        assert request.target == "/v1/chat/completions", request.target
        return Reply(status=500, body=b'{"error":{"message":"controlled primary failure"}}')

    def fallback(request: Request) -> Reply:
        model_list: Final = _model_list_reply(request)
        if model_list is not None:
            return model_list
        assert request.target == "/v1/chat/completions", request.target
        outbound: Final = _JSON_OBJECT.validate_json(request.body)
        assert outbound["model"] == "gpt-4o-mini", outbound
        return Reply(content_type="text/event-stream", chunks=_chat_stream(identity, "gpt-4o-mini"))

    with (
        wire_server(fail) as primary,
        wire_server(fallback) as fallback_wire,
    ):
        model_entries: Final = [
            {
                **_JSON_OBJECT.validate_python(entry),
                "litellm_params": {
                    **_JSON_OBJECT.validate_python(_JSON_OBJECT.validate_python(entry)["litellm_params"]),
                    "api_base": (
                        primary.url + "/v1"
                        if entry["model_name"] == "audit-priced-primary-9097"
                        else fallback_wire.url + "/v1"
                    ),
                },
            }
            for entry in configuration["model_list"]
        ]
        config_path.write_text(yaml.safe_dump({**configuration, "model_list": model_entries}))
        with owned_proxy(gateway, tmp_path, {}, config=config_path) as candidate:
            _wait_for_proxy_models(
                candidate,
                ("audit-priced-primary-9097", "audit-free-fallback-9097"),
            )
            response: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                {
                    "model": "audit-priced-primary-9097",
                    "messages": [{"role": "user", "content": identity}],
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
            )
    assert response.status_code == 200, response.text
    assert response.headers["x-litellm-model-id"] == "audit-free-fallback-9097", response.headers
    usages: Final = _usage_events(response.text)
    assert len(usages) == 1, response.text
    assert _cost_field(usages[0]) == {"cost": 0.0}, response.text
    primary_requests: Final = primary.drain()
    fallback_requests: Final = fallback_wire.drain()
    primary_posts: Final = tuple(request for request in primary_requests if request.method == "POST")
    fallback_posts: Final = tuple(request for request in fallback_requests if request.method == "POST")
    primary_gets: Final = tuple(request for request in primary_requests if request.method == "GET")
    fallback_gets: Final = tuple(request for request in fallback_requests if request.method == "GET")
    assert len(primary_posts) == len(fallback_posts) == 1, (primary_requests, fallback_requests)
    assert primary_posts[0].target == fallback_posts[0].target == "/v1/chat/completions"
    assert all(request.target == "/v1/models" for request in (*primary_gets, *fallback_gets))
