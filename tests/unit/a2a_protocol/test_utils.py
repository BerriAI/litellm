"""Tests for litellm/a2a_protocol/utils.py token/usage extraction."""

from types import SimpleNamespace
from typing import Final

import pytest

pytest.importorskip("a2a.compat.v0_3.types")

from a2a.compat.v0_3.types import MessageSendParams, SendMessageRequest

from litellm.a2a_protocol.utils import A2ARequestUtils
from tests.unit.litellm_core_utils.attribute_forwarder import AttributeForwarder


def _request(user_text: str) -> SendMessageRequest:
    return SendMessageRequest(
        id="r1",
        params=MessageSendParams(
            message={
                "messageId": "m1",
                "role": "user",
                "parts": [{"kind": "text", "text": user_text}],
            }
        ),
    )


def test_calculate_usage_counts_input_tokens_from_request_object():
    """Regression: request-side Part is a RootModel; input tokens must be counted."""
    request = _request("count these input tokens please")
    response_dict = {
        "result": {
            "kind": "message",
            "parts": [{"kind": "text", "text": "ok"}],
        }
    }

    prompt_tokens, completion_tokens, total_tokens = A2ARequestUtils.calculate_usage_from_request_response(
        request=request, response_dict=response_dict
    )

    assert prompt_tokens > 0
    assert completion_tokens > 0
    assert total_tokens == prompt_tokens + completion_tokens


def test_calculate_usage_counts_input_tokens_from_a_message_that_forwards_its_fields():
    request: Final = _request("count these input tokens please")
    forwarded_request: Final = SimpleNamespace(params=SimpleNamespace(message=AttributeForwarder(request.params.message)))
    response_dict: Final = {"result": {"kind": "message", "parts": [{"kind": "text", "text": "ok"}]}}

    assert A2ARequestUtils.calculate_usage_from_request_response(
        request=forwarded_request, response_dict=response_dict
    ) == A2ARequestUtils.calculate_usage_from_request_response(request=request, response_dict=response_dict)
