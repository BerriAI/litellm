import asyncio
import copy
import json
import uuid
from collections.abc import Iterator, Mapping
from typing import Final

import litellm
import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

FREE: Final = {"input_cost_per_token": 0.0, "output_cost_per_token": 0.0}
PRICED: Final = {"input_cost_per_token": 0.001, "output_cost_per_token": 0.002}
UNPRICED: Final[Mapping[str, float]] = {}


def _chat_frame(identity: str, body: Mapping[str, JsonValue]) -> bytes:
    return (
        b"data: "
        + json.dumps({"id": identity, "object": "chat.completion.chunk", "created": 1, **body}).encode()
        + b"\n\n"
    )


def _chat_stream(identity: str, model: str) -> tuple[bytes, ...]:
    return (
        _chat_frame(
            identity,
            {
                "model": model,
                "choices": [
                    {"index": 0, "delta": {"role": "assistant", "content": "Hello there"}, "finish_reason": None}
                ],
            },
        ),
        _chat_frame(identity, {"model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        _chat_frame(
            identity,
            {"model": model, "choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15}},
        ),
        b"data: [DONE]\n\n",
    )


def _responses_frame(event: str, payload: Mapping[str, JsonValue]) -> bytes:
    return f"event: {event}\ndata: {json.dumps({'type': event, **payload})}\n\n".encode()


def _responses_stream(identity: str, model: str) -> tuple[bytes, ...]:
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
                    "usage": {"input_tokens": 11, "output_tokens": 4, "total_tokens": 15},
                }
            },
        ),
    )


def _sse_events(text: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        json.loads(line.removeprefix("data: "))
        for line in text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    )


def _cost_field(usage: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {key: value for key, value in usage.items() if key == "cost"}


@pytest.mark.parametrize(
    ("deployment_model", "rates", "expected_cost_field"),
    (
        pytest.param("openai/gpt-4o-mini", FREE, {"cost": 0.0}, id="free-model-reports-zero"),
        pytest.param("openai/gpt-4o-mini", PRICED, {"cost": pytest.approx(0.019)}, id="priced-model"),
        pytest.param("openai/Qwen/Qwen3-8B", UNPRICED, {}, id="unpriced-model-omits-cost"),
    ),
)
def test_chat_completions_stream_final_usage_reports_the_computed_cost(
    gateway: Gateway, deployment_model: str, rates: Mapping[str, float], expected_cost_field: dict[str, JsonValue]
) -> None:
    identity: Final = "chat-usage-cost-" + uuid.uuid4().hex
    upstream_model: Final = deployment_model.removeprefix("openai/")
    with (
        wire_server(
            lambda request: Reply(content_type="text/event-stream", chunks=_chat_stream(identity, upstream_model))
        ) as wire,
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


@pytest.mark.parametrize(
    ("deployment_model", "rates", "expected_cost_field"),
    (
        pytest.param("openai/gpt-4o-mini", FREE, {"cost": 0.0}, id="free-model-reports-zero"),
        pytest.param("openai/gpt-4o-mini", PRICED, {"cost": pytest.approx(0.019)}, id="priced-model"),
        pytest.param("openai/Qwen/Qwen3-8B", UNPRICED, {}, id="unpriced-model-omits-cost"),
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


@pytest.fixture
def restore_model_cost_after_sdk_call() -> Iterator[None]:
    original: Final = copy.deepcopy(litellm.model_cost)
    yield
    litellm.model_cost.clear()
    litellm.model_cost.update(original)


@pytest.mark.usefixtures("restore_model_cost_after_sdk_call")
@pytest.mark.parametrize(
    ("model", "rates", "expected_cost_field"),
    (
        pytest.param("openai/gpt-4o-mini", FREE, {"cost": 0.0}, id="free-model-reports-zero"),
        pytest.param("openai/gpt-4o-mini", PRICED, {"cost": pytest.approx(0.019)}, id="priced-model"),
        pytest.param("openai/Qwen/Qwen3-8B", UNPRICED, {}, id="unpriced-model-omits-cost"),
    ),
)
def test_sdk_completion_stream_final_usage_reports_the_computed_cost(
    model: str, rates: Mapping[str, float], expected_cost_field: dict[str, JsonValue]
) -> None:
    identity: Final = "sdk-usage-cost-" + uuid.uuid4().hex
    upstream_model: Final = model.removeprefix("openai/")
    with wire_server(
        lambda request: Reply(content_type="text/event-stream", chunks=_chat_stream(identity, upstream_model))
    ) as wire:
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
    usage: Final = usages[0].model_dump()
    assert (usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"]) == (11, 4, 15), usage
    assert _cost_field(usage) == expected_cost_field, usage
