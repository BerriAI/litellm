from collections.abc import Sequence
from datetime import datetime
from types import MappingProxyType
from typing import Final
from unittest.mock import Mock

import pytest

from litellm.exceptions import MidStreamFallbackError, RateLimitError
from litellm.litellm_core_utils.hidden_params import get_hidden_params
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.responses.streaming_iterator import (
    BaseResponsesAPIStreamingIterator,
    partial_responses_incomplete_event_on_disconnect,
)
from litellm.router_rust.invoke.streaming import FallbackResponsesStreamWrapper
from litellm.types.llms.openai import ResponseIncompleteEvent, ResponsesAPIResponse, ResponsesAPIStreamEvents
from tests.unit.test_router.core.scripted_host import ScriptedHost, deployment, rate_limit, rust_core
from tests.unit.test_router.core.streaming_helpers import Event, ScriptedResponsesStream


def _incomplete_event(
    response_id: str, input_tokens: int, cached_tokens: int, response_cost: float | None
) -> ResponseIncompleteEvent:
    response: Final = ResponsesAPIResponse.model_validate(
        MappingProxyType(
            {
                "id": response_id,
                "object": "response",
                "created_at": 1,
                "status": "incomplete",
                "model": "claude-haiku-5-5",
                "output": (),
                "usage": MappingProxyType(
                    {
                        "input_tokens": input_tokens,
                        "input_tokens_details": MappingProxyType({"cached_tokens": cached_tokens}),
                        "output_tokens": 0,
                        "total_tokens": input_tokens,
                        "cost": response_cost,
                    }
                ),
            }
        )
    )
    if response_cost is not None:
        response.hidden_params["response_cost"] = response_cost
    return ResponseIncompleteEvent(type=ResponsesAPIStreamEvents.RESPONSE_INCOMPLETE, response=response)


def _cut_stream(response_id: str, input_tokens: int, cached_tokens: int, response_cost: float | None) -> Mock:
    stream: Final = Mock(spec=BaseResponsesAPIStreamingIterator)
    stream.partial_response_on_disconnect = Mock(
        return_value=_incomplete_event(response_id, input_tokens, cached_tokens, response_cost)
    )
    return stream


_FIXED_MARGIN: Final = 0.001


def _priced_cut_stream(response_id: str, response_cost: float, logging_obj: LiteLLMLoggingObj) -> Mock:
    stream: Final = _cut_stream(response_id, 25, 0, response_cost)
    event: Final = stream.partial_response_on_disconnect.return_value

    def priced_partial() -> ResponseIncompleteEvent:
        logging_obj.set_cost_breakdown(
            input_cost=response_cost - _FIXED_MARGIN,
            output_cost=0.0,
            total_cost=response_cost,
            cost_for_built_in_tools_cost_usd_dollar=None,
            margin_fixed_amount=_FIXED_MARGIN,
            margin_total_amount=_FIXED_MARGIN,
        )
        return event

    stream.partial_response_on_disconnect = Mock(side_effect=priced_partial)
    stream.logging_obj = logging_obj
    return stream


def _logging_obj(call_id: str) -> LiteLLMLoggingObj:
    return LiteLLMLoggingObj(
        model="claude-haiku-5-5",
        messages=[],
        stream=True,
        call_type="aresponses",
        start_time=datetime.now(),
        litellm_call_id=call_id,
        function_id=call_id,
    )


@pytest.mark.parametrize("deployments_share_one_logging_object", [True, False])
def test_a_client_cut_after_a_fallback_charges_the_fixed_margin_once(deployments_share_one_logging_object: bool):
    primary_logging: Final = _logging_obj("primary")
    fallback_logging: Final = primary_logging if deployments_share_one_logging_object else _logging_obj("fallback")
    wrapper: Final = FallbackResponsesStreamWrapper(
        async_generator=Mock(),
        attempt=Mock(),
        source_iterator=_priced_cut_stream("resp_primary", 0.00105, primary_logging),
    )
    wrapper.adopt_fallback_headers(_priced_cut_stream("resp_fallback", 0.0010325, fallback_logging))

    billed: Final = wrapper.partial_response_on_disconnect()

    assert billed is not None
    billed_cost: Final = (get_hidden_params(billed.response) or {}).get("response_cost")
    assert billed_cost == pytest.approx(0.00105 + 0.0010325 - _FIXED_MARGIN)
    assert billed.response.usage is not None
    assert billed.response.usage.cost == pytest.approx(billed_cost)
    assert primary_logging.cost_breakdown is not None
    assert primary_logging.cost_breakdown.get("total_cost") == pytest.approx(billed_cost)
    assert primary_logging.cost_breakdown.get("margin_total_amount") == pytest.approx(_FIXED_MARGIN)


def test_a_client_cut_after_a_fallback_bills_the_tokens_cache_reads_and_cost_of_both_deployments():
    wrapper: Final = FallbackResponsesStreamWrapper(
        async_generator=Mock(), attempt=Mock(), source_iterator=_cut_stream("resp_primary", 25, 10, 0.0000025)
    )
    wrapper.adopt_fallback_headers(_cut_stream("resp_fallback", 30, 20, 0.000004))

    billed: Final = wrapper.partial_response_on_disconnect()

    assert billed is not None
    assert billed.response.id == "resp_fallback"
    assert billed.response.usage is not None
    assert billed.response.usage.input_tokens == 55
    assert billed.response.usage.input_tokens_details is not None
    assert billed.response.usage.input_tokens_details.cached_tokens == 30
    assert (get_hidden_params(billed.response) or {}).get("response_cost") == pytest.approx(0.0000065)


def _billed_cost(event: ResponseIncompleteEvent | None) -> object:
    assert event is not None
    return (get_hidden_params(event.response) or {}).get("response_cost")


@pytest.mark.parametrize("unpriced_side", ["primary", "fallback"])
def test_a_client_cut_after_a_fallback_bills_an_unpriced_deployment_nothing_and_keeps_the_priced_breakdown(
    unpriced_side: str,
):
    primary_logging: Final = _logging_obj("primary")
    fallback_logging: Final = _logging_obj("fallback")
    stale: Final = 0.5
    unpriced_logging: Final = primary_logging if unpriced_side == "primary" else fallback_logging
    unpriced_logging.set_cost_breakdown(
        input_cost=stale, output_cost=0.0, total_cost=stale, cost_for_built_in_tools_cost_usd_dollar=None
    )
    unpriced: Final = _cut_stream(f"resp_{unpriced_side}", 25, 0, None)
    unpriced.logging_obj = unpriced_logging
    priced_logging: Final = fallback_logging if unpriced_side == "primary" else primary_logging
    priced: Final = _priced_cut_stream("resp_priced", 0.00105, priced_logging)
    wrapper: Final = FallbackResponsesStreamWrapper(
        async_generator=Mock(),
        attempt=Mock(),
        source_iterator=unpriced if unpriced_side == "primary" else priced,
    )
    wrapper.adopt_fallback_headers(priced if unpriced_side == "primary" else unpriced)

    billed: Final = wrapper.partial_response_on_disconnect()

    assert billed is not None
    assert billed.response.usage is not None
    assert billed.response.usage.input_tokens == 50
    assert _billed_cost(billed) == pytest.approx(0.00105)
    assert primary_logging.cost_breakdown is not None
    assert primary_logging.cost_breakdown.get("total_cost") == pytest.approx(0.00105)


def test_a_client_cut_after_a_fallback_with_no_partial_bills_an_unpriced_first_deployment_nothing():
    wrapper: Final = FallbackResponsesStreamWrapper(
        async_generator=Mock(), attempt=Mock(), source_iterator=_cut_stream("resp_primary", 25, 0, None)
    )
    fallback: Final = Mock(spec=BaseResponsesAPIStreamingIterator)
    fallback.partial_response_on_disconnect = Mock(return_value=None)
    wrapper.adopt_fallback_headers(fallback)

    billed: Final = wrapper.partial_response_on_disconnect()

    assert billed is not None
    assert billed.response.usage is not None
    assert billed.response.usage.input_tokens == 25
    assert _billed_cost(billed) == 0.0


_PRICE_PER_INPUT_TOKEN: Final = MappingProxyType({"primary-deployment": 1e-5, "fallback-deployment": 3e-6})
_CREATED: Final = "response.created"
_TEXT_DELTA: Final = "response.output_text.delta"


class _PricedOnTheSharedLogging(ScriptedResponsesStream):
    def __init__(
        self,
        events: Sequence[object],
        error: BaseException | None,
        logging_obj: LiteLLMLoggingObj,
        response_id: str,
        input_tokens: int | None,
    ) -> None:
        super().__init__(events, error, "d1", "anthropic", None)
        self.logging_obj = logging_obj
        self._response_id: Final = response_id
        self._input_tokens: Final = input_tokens

    def partial_response_on_disconnect(self) -> ResponseIncompleteEvent | None:
        if self._input_tokens is None:
            return None
        cost: Final = self._input_tokens * _PRICE_PER_INPUT_TOKEN[self.logging_obj.model]
        self.logging_obj.set_cost_breakdown(
            input_cost=cost, output_cost=0.0, total_cost=cost, cost_for_built_in_tools_cost_usd_dollar=None
        )
        return _incomplete_event(self._response_id, self._input_tokens, 0, cost)


async def _read_until_the_fallbacks_text(stream: object) -> None:
    deltas: Final = (event async for event in stream if getattr(event, "type", None) == _TEXT_DELTA)  # pyright: ignore[reportGeneralTypeIssues, reportUnknownVariableType]  # an async stream
    _ = await anext(deltas)
    _ = await anext(deltas)


def _shared_logging() -> LiteLLMLoggingObj:
    logging_obj: Final = _logging_obj("shared")
    logging_obj.model = "primary-deployment"
    return logging_obj


def _primary_cut_after_text(shared_logging: LiteLLMLoggingObj) -> object:
    cut: Final = MidStreamFallbackError(
        message="stream cut",
        model="d1",
        llm_provider="anthropic",
        original_exception=rate_limit(message="stream cut"),
        generated_content="hel",
        is_pre_first_chunk=False,
    )

    def primary(**_request: object) -> ScriptedResponsesStream:  # kwargs-ok: the host passes the request kwargs
        return _PricedOnTheSharedLogging(
            (Event(_CREATED), Event(_TEXT_DELTA, delta="hel")), cut, shared_logging, "resp_primary", 25
        )

    return primary


def _core_falling_back(host: ScriptedHost) -> object:
    return rust_core(
        host, model_list=[deployment("a", "a"), deployment("b", "b")], fallbacks=[{"a": ["b"]}], num_retries=0
    )


@pytest.mark.parametrize("fallback_input_tokens", [30, None])
async def test_a_client_cut_after_a_fallback_bills_the_first_deployment_at_its_own_price(
    fallback_input_tokens: int | None,
):
    shared_logging: Final = _shared_logging()
    host: Final = ScriptedHost()

    def fallback(**_request: object) -> ScriptedResponsesStream:  # kwargs-ok: the host passes the request kwargs
        shared_logging.model = "fallback-deployment"
        return _PricedOnTheSharedLogging(
            (Event(_CREATED), Event(_TEXT_DELTA, delta="lo"), Event(_TEXT_DELTA, delta=" there")),
            None,
            shared_logging,
            "resp_fallback",
            fallback_input_tokens,
        )

    host.answer("openai/a", _primary_cut_after_text(shared_logging))
    host.answer("openai/b", fallback)
    stream: Final = await _core_falling_back(host).aresponses(model="a", input="hi", stream=True)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]  # endpoint method
    await _read_until_the_fallbacks_text(stream)

    billed: Final = partial_responses_incomplete_event_on_disconnect(stream)

    expected: Final = 25 * _PRICE_PER_INPUT_TOKEN["primary-deployment"] + (
        0.0 if fallback_input_tokens is None else fallback_input_tokens * _PRICE_PER_INPUT_TOKEN["fallback-deployment"]
    )
    assert _billed_cost(billed) == pytest.approx(expected)
    assert shared_logging.cost_breakdown is not None
    assert shared_logging.cost_breakdown.get("total_cost") == pytest.approx(expected)
    await stream.aclose()  # pyright: ignore[reportUnknownMemberType]  # an async stream


def test_a_client_cut_while_the_fallback_connects_bills_the_first_deployment_at_its_own_price():
    shared_logging: Final = _shared_logging()
    wrapper: Final = FallbackResponsesStreamWrapper(
        async_generator=Mock(),
        attempt=Mock(),
        source_iterator=_PricedOnTheSharedLogging((), None, shared_logging, "resp_primary", 25),
    )
    wrapper.hold_primary_partial()
    shared_logging.model = "fallback-deployment"

    billed: Final = wrapper.partial_response_on_disconnect()

    expected: Final = 25 * _PRICE_PER_INPUT_TOKEN["primary-deployment"]
    assert _billed_cost(billed) == pytest.approx(expected)
    assert shared_logging.cost_breakdown is not None
    assert shared_logging.cost_breakdown.get("total_cost") == pytest.approx(expected)


async def test_a_fallback_that_fails_after_a_cut_leaves_no_breakdown_from_the_first_deployments_partial():
    shared_logging: Final = _shared_logging()
    host: Final = ScriptedHost()
    host.answer("openai/a", _primary_cut_after_text(shared_logging))
    host.answer("openai/b", rate_limit(message="fallback down"))
    stream: Final = await _core_falling_back(host).aresponses(model="a", input="hi", stream=True)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]  # endpoint method

    with pytest.raises(RateLimitError, match="stream cut"):
        async for _ in stream:  # pyright: ignore[reportUnknownVariableType]  # an async stream
            pass

    assert shared_logging.cost_breakdown is None


class _StreamThatOutlivesItsConsumer:
    def __init__(self) -> None:
        self.kept_open = False
        self.settled_with_drain: tuple[bool, ...] = ()

    def outlive_consumer_until_settled(self) -> None:
        self.kept_open = True

    async def settle_after_consumer_left(self, *, drain: bool) -> bool:
        self.settled_with_drain = (*self.settled_with_drain, drain)
        return drain


@pytest.mark.asyncio
async def test_a_cut_stream_served_by_its_primary_deployment_is_drained_to_the_provider_usage():
    source: Final = _StreamThatOutlivesItsConsumer()
    wrapper: Final = FallbackResponsesStreamWrapper(async_generator=Mock(), attempt=Mock(), source_iterator=source)

    wrapper.outlive_consumer_until_settled()
    drained: Final = await wrapper.settle_after_consumer_left(drain=True)

    assert source.kept_open is True
    assert (drained, source.settled_with_drain) == (True, (True,))


@pytest.mark.asyncio
async def test_a_cut_stream_that_fell_back_is_closed_so_both_deployments_keep_their_partial_bill():
    source: Final = _StreamThatOutlivesItsConsumer()
    wrapper: Final = FallbackResponsesStreamWrapper(async_generator=Mock(), attempt=Mock(), source_iterator=source)
    wrapper.adopt_fallback_headers(_cut_stream("resp_fallback", 30, 20, 0.000004))

    drained: Final = await wrapper.settle_after_consumer_left(drain=True)

    assert (drained, source.settled_with_drain) == (False, (False,))
