from __future__ import annotations

from typing import Final

import pytest

from litellm.proxy.usage_cost_egress import with_client_stream_usage_cost
from litellm.types.llms.openai import (
    ResponseAPIUsage,
    ResponseCompletedEvent,
    ResponseIncompleteEvent,
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
)
from litellm.types.utils import ModelResponseStream, StreamingChoices, Usage

UPSTREAM_COST: Final = 0.037
OPERATOR_BILL: Final = 100 * 2e-6 + 20 * 6e-6


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


def _usage_chunk(upstream_cost: float | None, service_tier: str | None = None) -> ModelResponseStream:
    return ModelResponseStream(
        id="chatcmpl-egress",
        model="omni-model",
        choices=[],
        service_tier=service_tier,
        usage=Usage(prompt_tokens=100, completion_tokens=20, total_tokens=120, cost=upstream_cost),
    )


def _bill(_reply: object) -> float | None:
    return OPERATOR_BILL


def _unpriceable(_reply: object) -> float | None:
    return None


def test_the_streamed_usage_chunk_tells_the_client_what_litellm_billed() -> None:
    chunk: Final = _usage_chunk(UPSTREAM_COST)

    billed: Final = with_client_stream_usage_cost(chunk, _bill)

    assert isinstance(billed, ModelResponseStream) and billed is not chunk
    assert billed.usage is not None and billed.usage.cost == pytest.approx(OPERATOR_BILL)
    assert chunk.usage is not None and chunk.usage.cost == UPSTREAM_COST


def test_a_zero_bill_replaces_the_upstream_cost() -> None:
    billed: Final = with_client_stream_usage_cost(_usage_chunk(UPSTREAM_COST), lambda _reply: 0.0)

    assert isinstance(billed, ModelResponseStream)
    assert billed.usage is not None and billed.usage.cost == 0.0


@pytest.mark.parametrize("bill", [None, -0.001, float("nan"), float("inf")])
def test_an_unpriceable_stream_keeps_the_upstream_cost(bill: float | None) -> None:
    chunk: Final = _usage_chunk(UPSTREAM_COST)

    assert with_client_stream_usage_cost(chunk, lambda _reply: bill) is chunk
    assert chunk.usage is not None and chunk.usage.cost == UPSTREAM_COST


def test_a_priced_stream_without_an_upstream_cost_still_tells_the_client_the_bill() -> None:
    chunk: Final = _usage_chunk(None)

    billed: Final = with_client_stream_usage_cost(chunk, _bill)

    assert isinstance(billed, ModelResponseStream) and billed is not chunk
    assert billed.usage is not None and billed.usage.cost == pytest.approx(OPERATOR_BILL)


@pytest.mark.parametrize("bill", [None, 0.0])
def test_an_unpriced_stream_without_an_upstream_cost_goes_out_as_sent(bill: float | None) -> None:
    chunk: Final = _usage_chunk(None)

    assert with_client_stream_usage_cost(chunk, lambda _reply: bill) is chunk


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


def test_the_stream_pricer_sees_the_upstream_cost_and_the_served_tier_so_the_bill_matches_the_spend_row() -> None:
    seen: list[tuple[object, object]] = []

    def price(reply: object) -> float | None:
        seen.append((getattr(getattr(reply, "usage", None), "cost", None), getattr(reply, "service_tier", None)))
        return OPERATOR_BILL

    with_client_stream_usage_cost(_usage_chunk(UPSTREAM_COST, service_tier="default"), price)

    assert seen == [(UPSTREAM_COST, "default")]


@pytest.mark.parametrize("upstream_cost", [UPSTREAM_COST, None])
@pytest.mark.parametrize(
    "event_type", [ResponsesAPIStreamEvents.RESPONSE_COMPLETED, ResponsesAPIStreamEvents.RESPONSE_INCOMPLETE]
)
def test_a_terminal_responses_event_carries_the_bill_with_or_without_an_upstream_cost(
    upstream_cost: float | None, event_type: ResponsesAPIStreamEvents
) -> None:
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
