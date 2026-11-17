import json
import threading
import uuid
from collections.abc import Callable, Iterable, Iterator
from hashlib import sha256
from typing import Final

import httpx
from integration._support.client import Gateway, Scenario, eventually, held
from integration._support.wire import Reply, Request, Wire
from openai.types.chat import ChatCompletionChunk
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
BACKEND: Final = "integration-partial-stream"
PIECE: Final = "partial stream billed tokens "
DELIVERED_PIECES: Final = 12
_FAILURE_ROWS: Final = 'SELECT spend FROM "LiteLLM_SpendLogs" WHERE api_key=%s AND status=%s ORDER BY "startTime"'
ROWS: Final = 'SELECT status, spend FROM "LiteLLM_SpendLogs" WHERE api_key=%s ORDER BY "startTime"'


class _SpendRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    spend: float


class StatusRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: str
    spend: float


class _BudgetError(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: str


class _ErrorResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    error: _BudgetError


def _chunk(stream_id: str, delta: dict[str, JsonValue], finish_reason: str | None = None) -> bytes:
    body: Final = {
        "id": stream_id,
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": BACKEND,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(body)}\n\n".encode()


def completion() -> bytes:
    return json.dumps(
        {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": 1700000000,
            "model": BACKEND,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    ).encode()


def peer(delivered: int | None, gate: threading.Event | None = None) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.method != "POST":
            return Reply(status=404)
        if JSON_OBJECT.validate_json(request.body).get("stream") is True:
            stream_id: Final = f"chatcmpl-{uuid.uuid4().hex}"
            return Reply(
                content_type="text/event-stream",
                chunks=(
                    *(_chunk(stream_id, {"role": "assistant", "content": PIECE}) for _ in range(DELIVERED_PIECES)),
                    _chunk(stream_id, {}, "stop"),
                    b"data: [DONE]\n\n",
                ),
                abort_after=delivered,
                pause_between_chunks=0.02,
                gate_before_first=gate,
            )
        return Reply(body=completion())

    return respond


def priced_model(scenario: Scenario, wire: Wire) -> str:
    return scenario.model(
        model=f"openai/{BACKEND}",
        api_base=f"{wire.url}/v1",
        api_key="synthetic-openai-key",
        input_cost_per_token=0.001,
        output_cost_per_token=0.002,
    )


def stream(gateway: Gateway, model: str, key: str, **fields: JsonValue) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": model,
            "max_tokens": 5,
            "stream": True,
            "messages": [{"role": "user", "content": "partial stream"}],
            **fields,
        },
        key=key,
    )


def broken_stream(gateway: Gateway, model: str, key: str, **fields: JsonValue) -> None:
    response: Final = stream(gateway, model, key, **fields)
    assert response.status_code == 200, response.text
    events: Final = tuple(
        line.removeprefix("data: ")
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    )
    assert "error" in JSON_OBJECT.validate_json(events[-1]), response.text
    delivered: Final = "".join(_streamed_text(ChatCompletionChunk.model_validate_json(event) for event in events[:-1]))
    assert delivered == PIECE * DELIVERED_PIECES, response.text


def _streamed_text(chunks: Iterable[ChatCompletionChunk]) -> Iterator[str]:
    for chunk in chunks:
        yield from (choice.delta.content or "" for choice in chunk.choices)


def chat(gateway: Gateway, model: str, key: str, **fields: JsonValue) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": model,
            "max_tokens": 5,
            "messages": [{"role": "user", "content": f"after {uuid.uuid4().hex}"}],
            **fields,
        },
        key=key,
    )


def failure_spend(gateway: Gateway, key: str, rows: int = 1) -> tuple[float, ...]:
    digest: Final = sha256(key.encode()).hexdigest()
    eventually(lambda: gateway.rows(_FAILURE_ROWS, (digest, "failure")), lambda found: len(found) >= rows, seconds=70)
    found: Final = held(
        lambda: gateway.rows(_FAILURE_ROWS, (digest, "failure")), lambda latest: len(latest) == rows, holding=2
    )
    return tuple(_SpendRow.model_validate(row).spend for row in found)


def chats_sent(wire: Wire) -> int:
    return sum(1 for request in wire.drain() if request.method == "POST")


def recovered_cost(gateway: Gateway, scenario: Scenario, model: str) -> float:
    key: Final = scenario.key(models=[model], max_budget=100)
    broken_stream(gateway, model, key)
    (spend,) = failure_spend(gateway, key)
    assert spend > 0
    return spend


def assert_budget_refusal(response: httpx.Response) -> None:
    assert response.status_code == 422, response.text
    assert _ErrorResponse.model_validate_json(response.content).error.type == "budget_exceeded", response.text
