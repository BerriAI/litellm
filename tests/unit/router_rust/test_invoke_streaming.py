from datetime import datetime
from types import MappingProxyType
from typing import Final
from unittest.mock import Mock

import pytest

from litellm.litellm_core_utils.hidden_params import get_hidden_params
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
from litellm.router_rust.invoke.streaming import FallbackResponsesStreamWrapper
from litellm.types.llms.openai import ResponseIncompleteEvent, ResponsesAPIResponse, ResponsesAPIStreamEvents


def _cut_stream(response_id: str, input_tokens: int, cached_tokens: int, response_cost: float) -> Mock:
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
    response.hidden_params["response_cost"] = response_cost
    stream: Final = Mock(spec=BaseResponsesAPIStreamingIterator)
    stream.partial_response_on_disconnect = Mock(
        return_value=ResponseIncompleteEvent(type=ResponsesAPIStreamEvents.RESPONSE_INCOMPLETE, response=response)
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
