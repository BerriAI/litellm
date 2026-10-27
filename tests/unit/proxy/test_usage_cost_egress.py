from __future__ import annotations

from typing import Final

import pytest

import litellm
from litellm.proxy.usage_cost_egress import with_client_stream_usage_cost, with_client_usage_cost
from litellm.types.llms.openai import (
    ResponseAPIUsage,
    ResponseCompletedEvent,
    ResponseIncompleteEvent,
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
)
from litellm.types.utils import Choices, Message, ModelResponse, ModelResponseStream, StreamingChoices, Usage

UPSTREAM_COST: Final = 0.037
OPERATOR_BILL: Final = 100 * 2e-6 + 20 * 6e-6


def _chat_reply(upstream_cost: float | None) -> ModelResponse:
    reply: Final = ModelResponse(
        id="chatcmpl-egress",
        model="omni-model",
        choices=[Choices(index=0, message=Message(role="assistant", content="hi"), finish_reason="stop")],
        usage=Usage(prompt_tokens=100, completion_tokens=20, total_tokens=120, cost=upstream_cost),
    )
    reply._hidden_params["response_cost"] = OPERATOR_BILL
    return reply


def _responses_reply(upstream_cost: float | None) -> ResponsesAPIResponse:
    return ResponsesAPIResponse(
        id="resp_egress",
        created_at=1,
        status="completed",
        model="omni-model",
        object="response",
        output=[],
        usage=ResponseAPIUsage(input_tokens=100, output_tokens=20, total_tokens=120, cost=upstream_cost),
    )


def _usage_chunk(upstream_cost: float | None) -> ModelResponseStream:
    return ModelResponseStream(
        id="chatcmpl-egress",
        model="omni-model",
        choices=[],
        usage=Usage(prompt_tokens=100, completion_tokens=20, total_tokens=120, cost=upstream_cost),
    )


def _bill(_reply: object) -> float | None:
    return OPERATOR_BILL


def _unpriceable(_reply: object) -> float | None:
    return None


@pytest.mark.parametrize("reply", [_chat_reply(UPSTREAM_COST), _responses_reply(UPSTREAM_COST)])
def test_a_reply_whose_upstream_reported_a_cost_tells_the_client_what_litellm_billed(reply: object) -> None:
    billed: Final = with_client_usage_cost(reply, OPERATOR_BILL)

    assert billed is not reply, "the pipeline's own object keeps the upstream figure"
    assert getattr(getattr(billed, "usage", None), "cost", None) == pytest.approx(OPERATOR_BILL)
    assert getattr(getattr(reply, "usage", None), "cost", None) == UPSTREAM_COST


def test_the_billed_copy_of_a_chat_reply_keeps_its_hidden_params() -> None:
    reply: Final = _chat_reply(UPSTREAM_COST)

    billed: Final = with_client_usage_cost(reply, OPERATOR_BILL)

    assert isinstance(billed, ModelResponse)
    assert billed._hidden_params["response_cost"] == pytest.approx(OPERATOR_BILL)
    assert billed.model_dump()["usage"]["cost"] == pytest.approx(OPERATOR_BILL)


@pytest.mark.parametrize("reply", [_chat_reply(None), _responses_reply(None)])
def test_a_reply_without_an_upstream_cost_goes_out_as_the_upstream_sent_it(
    monkeypatch: pytest.MonkeyPatch, reply: object
) -> None:
    monkeypatch.setattr(litellm, "include_cost_in_streaming_usage", False)

    assert with_client_usage_cost(reply, OPERATOR_BILL) is reply


@pytest.mark.parametrize("reply", [_chat_reply(None), _responses_reply(None)])
def test_include_cost_in_streaming_usage_puts_the_bill_on_a_reply_without_an_upstream_cost(
    monkeypatch: pytest.MonkeyPatch, reply: object
) -> None:
    monkeypatch.setattr(litellm, "include_cost_in_streaming_usage", True)

    billed: Final = with_client_usage_cost(reply, OPERATOR_BILL)

    assert getattr(getattr(billed, "usage", None), "cost", None) == pytest.approx(OPERATOR_BILL)


def test_a_zero_bill_replaces_the_upstream_cost() -> None:
    billed: Final = with_client_usage_cost(_chat_reply(UPSTREAM_COST), 0.0)

    assert getattr(getattr(billed, "usage", None), "cost", None) == 0.0


@pytest.mark.parametrize("bill", [None, -0.001, float("nan"), float("inf"), True, "0.1"])
def test_an_unpriceable_reply_keeps_the_upstream_cost(bill: object) -> None:
    reply: Final = _chat_reply(UPSTREAM_COST)

    assert with_client_usage_cost(reply, bill) is reply
    assert reply.usage is not None and reply.usage.cost == UPSTREAM_COST


def test_a_reply_that_is_neither_chat_nor_responses_passes_through() -> None:
    reply: Final = {"usage": {"cost": UPSTREAM_COST}}

    assert with_client_usage_cost(reply, OPERATOR_BILL) is reply


def test_the_streamed_usage_chunk_tells_the_client_what_litellm_billed() -> None:
    chunk: Final = _usage_chunk(UPSTREAM_COST)

    billed: Final = with_client_stream_usage_cost(chunk, _bill)

    assert isinstance(billed, ModelResponseStream) and billed is not chunk
    assert billed.usage is not None and billed.usage.cost == pytest.approx(OPERATOR_BILL)
    assert chunk.usage is not None and chunk.usage.cost == UPSTREAM_COST


def test_a_content_chunk_is_not_priced() -> None:
    chunk: Final = ModelResponseStream(
        id="chatcmpl-egress", model="omni-model", choices=[StreamingChoices(index=0, delta={"content": "hi"})]
    )
    priced: list[object] = []

    def price(reply: object) -> float | None:
        priced.append(reply)
        return OPERATOR_BILL

    assert with_client_stream_usage_cost(chunk, price) is chunk
    assert priced == []


def test_a_streamed_usage_chunk_without_an_upstream_cost_goes_out_as_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "include_cost_in_streaming_usage", False)
    chunk: Final = _usage_chunk(None)

    assert with_client_stream_usage_cost(chunk, _bill) is chunk


def test_include_cost_in_streaming_usage_puts_the_bill_on_the_streamed_usage_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "include_cost_in_streaming_usage", True)

    billed: Final = with_client_stream_usage_cost(_usage_chunk(None), _bill)

    assert isinstance(billed, ModelResponseStream)
    assert billed.usage is not None and billed.usage.cost == pytest.approx(OPERATOR_BILL)


def test_the_stream_pricer_sees_the_upstream_cost_so_the_bill_matches_the_spend_row() -> None:
    seen: list[object] = []

    def price(reply: object) -> float | None:
        seen.append(getattr(getattr(reply, "usage", None), "cost", None))
        return OPERATOR_BILL

    with_client_stream_usage_cost(_usage_chunk(UPSTREAM_COST), price)

    assert seen == [UPSTREAM_COST]


@pytest.mark.parametrize("upstream_cost", [UPSTREAM_COST, None])
@pytest.mark.parametrize(
    "event_type", [ResponsesAPIStreamEvents.RESPONSE_COMPLETED, ResponsesAPIStreamEvents.RESPONSE_INCOMPLETE]
)
def test_a_terminal_responses_event_carries_the_bill_with_or_without_an_upstream_cost(
    monkeypatch: pytest.MonkeyPatch, upstream_cost: float | None, event_type: ResponsesAPIStreamEvents
) -> None:
    monkeypatch.setattr(litellm, "include_cost_in_streaming_usage", False)
    event_cls: Final = (
        ResponseCompletedEvent if event_type == ResponsesAPIStreamEvents.RESPONSE_COMPLETED else ResponseIncompleteEvent
    )
    event: Final = event_cls(type=event_type, response=_responses_reply(upstream_cost))

    billed: Final = with_client_stream_usage_cost(event, _bill)

    assert billed is not event
    assert getattr(billed, "response").usage.cost == pytest.approx(OPERATOR_BILL)
    assert event.response.usage is not None and event.response.usage.cost == upstream_cost


def test_an_unpriceable_terminal_responses_event_goes_out_as_sent() -> None:
    event: Final = ResponseCompletedEvent(
        type=ResponsesAPIStreamEvents.RESPONSE_COMPLETED, response=_responses_reply(None)
    )

    assert with_client_stream_usage_cost(event, _unpriceable) is event
