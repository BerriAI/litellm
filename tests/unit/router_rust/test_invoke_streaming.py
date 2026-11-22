from types import MappingProxyType
from typing import Final
from unittest.mock import Mock

import pytest

from litellm.litellm_core_utils.hidden_params import get_hidden_params
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
