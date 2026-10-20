import json
import uuid
from collections.abc import Callable, Iterable, Iterator
from hashlib import sha256
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, Scenario, eventually, held
from integration._support.wire import Reply, Request, Wire, wire_server
from openai.types.chat import ChatCompletionChunk
from pydantic import BaseModel, JsonValue, TypeAdapter

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_BACKEND: Final = "integration-partial-stream"
_PIECE: Final = "partial stream billed tokens "
_DELIVERED_PIECES: Final = 12
_FAILURE_ROWS: Final = 'SELECT spend FROM "LiteLLM_SpendLogs" WHERE api_key=%s AND status=%s ORDER BY "startTime"'


class _SpendRow(BaseModel):
    spend: float


class _BudgetError(BaseModel):
    type: str


class _ErrorResponse(BaseModel):
    error: _BudgetError


def _chunk(delta: dict[str, JsonValue], finish_reason: str | None = None) -> bytes:
    body: Final = {
        "id": "chatcmpl-partial",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": _BACKEND,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(body)}\n\n".encode()


def _completion() -> bytes:
    return json.dumps(
        {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": 1700000000,
            "model": _BACKEND,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    ).encode()


def _peer(delivered: int) -> Callable[[Request], Reply]:
    frames: Final = (
        *(_chunk({"role": "assistant", "content": _PIECE}) for _ in range(_DELIVERED_PIECES)),
        _chunk({}, "stop"),
        b"data: [DONE]\n\n",
    )

    def respond(request: Request) -> Reply:
        if request.method != "POST":
            return Reply(status=404)
        if _JSON_OBJECT.validate_json(request.body).get("stream") is True:
            return Reply(
                content_type="text/event-stream", chunks=frames, abort_after=delivered, pause_between_chunks=0.02
            )
        return Reply(body=_completion())

    return respond


def _priced_model(scenario: Scenario, wire: Wire) -> str:
    return scenario.model(
        model=f"openai/{_BACKEND}",
        api_base=f"{wire.url}/v1",
        api_key="synthetic-openai-key",
        input_cost_per_token=0.001,
        output_cost_per_token=0.002,
    )


def _stream(gateway: Gateway, model: str, key: str) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": model,
            "max_tokens": 5,
            "stream": True,
            "messages": [{"role": "user", "content": "partial stream"}],
        },
        key=key,
    )


def _broken_stream(gateway: Gateway, model: str, key: str) -> str:
    response: Final = _stream(gateway, model, key)
    assert response.status_code == 200, response.text
    events: Final = tuple(
        line.removeprefix("data: ")
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    )
    assert "error" in _JSON_OBJECT.validate_json(events[-1]), response.text
    return "".join(_streamed_text(ChatCompletionChunk.model_validate_json(event) for event in events[:-1]))


def _streamed_text(chunks: Iterable[ChatCompletionChunk]) -> Iterator[str]:
    for chunk in chunks:
        yield from (choice.delta.content or "" for choice in chunk.choices)


def _chat(gateway: Gateway, model: str, key: str) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "max_tokens": 5, "messages": [{"role": "user", "content": f"after {uuid.uuid4().hex}"}]},
        key=key,
    )


def _failure_spend(gateway: Gateway, key: str, rows: int = 1) -> tuple[float, ...]:
    digest: Final = sha256(key.encode()).hexdigest()
    eventually(lambda: gateway.rows(_FAILURE_ROWS, (digest, "failure")), lambda found: len(found) >= rows, seconds=70)
    found: Final = held(
        lambda: gateway.rows(_FAILURE_ROWS, (digest, "failure")), lambda latest: len(latest) == rows, holding=2
    )
    return tuple(_SpendRow.model_validate(row).spend for row in found)


def _chats_sent(wire: Wire) -> int:
    return sum(1 for request in wire.drain() if request.method == "POST")


def _recovered_cost(gateway: Gateway, scenario: Scenario, model: str) -> float:
    key: Final = scenario.key(models=[model], max_budget=100)
    assert _broken_stream(gateway, model, key) == _PIECE * _DELIVERED_PIECES
    (spend,) = _failure_spend(gateway, key)
    assert spend > 0
    return spend


def _assert_budget_refusal(response: httpx.Response) -> None:
    assert response.status_code == 422, response.text
    assert _ErrorResponse.model_validate_json(response.content).error.type == "budget_exceeded", response.text


@pytest.mark.timeout(240)
def test_a_broken_stream_that_spends_past_the_budget_blocks_the_next_request(gateway: Gateway) -> None:
    with wire_server(_peer(_DELIVERED_PIECES)) as wire, gateway.scenario() as scenario:
        model: Final = _priced_model(scenario, wire)
        cost: Final = _recovered_cost(gateway, scenario, model)
        key: Final = scenario.key(models=[model], max_budget=cost / 2)
        _chats_sent(wire)

        assert _broken_stream(gateway, model, key) == _PIECE * _DELIVERED_PIECES
        after: Final = _chat(gateway, model, key)

        _assert_budget_refusal(after)
        assert _failure_spend(gateway, key, rows=2) == pytest.approx((cost, 0.0))
        assert _chats_sent(wire) == 1


@pytest.mark.timeout(240)
def test_broken_stream_spend_counts_once_toward_the_budget(gateway: Gateway) -> None:
    with wire_server(_peer(_DELIVERED_PIECES)) as wire, gateway.scenario() as scenario:
        model: Final = _priced_model(scenario, wire)
        cost: Final = _recovered_cost(gateway, scenario, model)
        key: Final = scenario.key(models=[model], max_budget=cost * 1.5)

        assert _broken_stream(gateway, model, key) == _PIECE * _DELIVERED_PIECES
        assert _broken_stream(gateway, model, key) == _PIECE * _DELIVERED_PIECES
        after: Final = _chat(gateway, model, key)

        _assert_budget_refusal(after)
        assert _failure_spend(gateway, key, rows=3) == pytest.approx((cost, cost, 0.0))


@pytest.mark.timeout(240)
def test_a_stream_that_breaks_before_any_chunk_leaves_the_budget_untouched(gateway: Gateway) -> None:
    with wire_server(_peer(0)) as wire, gateway.scenario() as scenario:
        model: Final = _priced_model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=0.05)

        refused: Final = _stream(gateway, model, key)
        after: Final = _chat(gateway, model, key)

        assert refused.status_code == 500, refused.text

        assert after.status_code == 200, after.text
        assert _failure_spend(gateway, key) == (0.0,)
