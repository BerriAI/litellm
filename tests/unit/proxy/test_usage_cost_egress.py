from __future__ import annotations

from typing import Final

import pytest

from litellm.litellm_core_utils.hidden_params import get_hidden_params
from litellm.proxy.usage_cost_egress import with_client_stream_usage_cost
from litellm.types.llms.openai import (
    ResponseAPIUsage,
    ResponseCompletedEvent,
    ResponseIncompleteEvent,
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
)
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices, Usage

UPSTREAM_COST: Final = 0.037
OPERATOR_BILL: Final = 100 * 2e-6 + 20 * 6e-6


def _responses_reply(upstream_cost: float | None, bill: object) -> ResponsesAPIResponse:
    reply: Final = ResponsesAPIResponse(
        id="resp_egress",
        created_at=1,
        status="completed",
        model="omni-model",
        object="response",
        output=[],
        usage=ResponseAPIUsage(input_tokens=100, output_tokens=20, total_tokens=120, cost=upstream_cost),
    )
    hidden: Final = get_hidden_params(reply)
    assert hidden is not None
    hidden["response_cost"] = bill
    return reply


def _usage_chunk(upstream_cost: float | None, bill: object) -> ModelResponseStream:
    chunk: Final = ModelResponseStream(
        id="chatcmpl-egress",
        model="omni-model",
        choices=[],
        usage=Usage(prompt_tokens=100, completion_tokens=20, total_tokens=120, cost=upstream_cost),
    )
    chunk.hidden_params = {"response_cost": bill}
    return chunk


def test_the_streamed_usage_chunk_tells_the_client_what_litellm_billed() -> None:
    chunk: Final = _usage_chunk(UPSTREAM_COST, OPERATOR_BILL)

    billed: Final = with_client_stream_usage_cost(chunk)

    assert isinstance(billed, ModelResponseStream) and billed is not chunk
    assert billed.usage is not None and billed.usage.cost == OPERATOR_BILL
    assert chunk.usage is not None and chunk.usage.cost == UPSTREAM_COST


def test_a_zero_bill_replaces_the_upstream_cost() -> None:
    billed: Final = with_client_stream_usage_cost(_usage_chunk(UPSTREAM_COST, 0.0))

    assert isinstance(billed, ModelResponseStream)
    assert billed.usage is not None and billed.usage.cost == 0.0


@pytest.mark.parametrize("bill", [None, -0.001, float("nan"), float("inf"), True])
def test_an_unpriced_stream_keeps_the_upstream_cost(bill: object) -> None:
    chunk: Final = _usage_chunk(UPSTREAM_COST, bill)

    assert with_client_stream_usage_cost(chunk) is chunk
    assert chunk.usage is not None and chunk.usage.cost == UPSTREAM_COST


def test_a_priced_stream_without_an_upstream_cost_still_tells_the_client_the_bill() -> None:
    billed: Final = with_client_stream_usage_cost(_usage_chunk(None, OPERATOR_BILL))

    assert isinstance(billed, ModelResponseStream)
    assert billed.usage is not None and billed.usage.cost == OPERATOR_BILL


@pytest.mark.parametrize("bill", [None, 0.0])
def test_an_unpriced_stream_without_an_upstream_cost_goes_out_as_sent(bill: float | None) -> None:
    chunk: Final = _usage_chunk(None, bill)

    assert with_client_stream_usage_cost(chunk) is chunk


def test_a_content_chunk_is_not_touched() -> None:
    chunk: Final = ModelResponseStream(
        id="chatcmpl-egress",
        model="omni-model",
        choices=[StreamingChoices(index=0, delta=Delta(content="hi"), finish_reason=None)],
    )
    chunk.hidden_params = {"response_cost": OPERATOR_BILL}

    assert with_client_stream_usage_cost(chunk) is chunk


@pytest.mark.parametrize("upstream_cost", [UPSTREAM_COST, None])
@pytest.mark.parametrize(
    "event_type", [ResponsesAPIStreamEvents.RESPONSE_COMPLETED, ResponsesAPIStreamEvents.RESPONSE_INCOMPLETE]
)
def test_a_terminal_responses_event_carries_the_bill_with_or_without_an_upstream_cost(
    upstream_cost: float | None, event_type: ResponsesAPIStreamEvents
) -> None:
    reply: Final = _responses_reply(upstream_cost, OPERATOR_BILL)
    event: Final = (
        ResponseCompletedEvent(type=ResponsesAPIStreamEvents.RESPONSE_COMPLETED, response=reply)
        if event_type is ResponsesAPIStreamEvents.RESPONSE_COMPLETED
        else ResponseIncompleteEvent(type=ResponsesAPIStreamEvents.RESPONSE_INCOMPLETE, response=reply)
    )

    billed: Final = with_client_stream_usage_cost(event)

    assert billed is not event
    assert isinstance(billed, (ResponseCompletedEvent, ResponseIncompleteEvent))
    assert billed.response.usage is not None and billed.response.usage.cost == OPERATOR_BILL
    assert reply.usage is not None and reply.usage.cost == upstream_cost


def test_an_unpriced_terminal_responses_event_goes_out_as_sent() -> None:
    event: Final = ResponseCompletedEvent(
        type=ResponsesAPIStreamEvents.RESPONSE_COMPLETED, response=_responses_reply(UPSTREAM_COST, None)
    )

    assert with_client_stream_usage_cost(event) is event
