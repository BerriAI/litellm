"""
Unit tests for Presidio PII masking of tool-call arguments.

Covers two channels that were previously unmasked:
1. Pre-call: tool_calls[*].function.arguments and function_call.arguments
2. Anthropic response: tool_use block input values
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../../../.."))

from unittest.mock import MagicMock

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.presidio import (
    _OPTIONAL_PresidioPIIMasking,
)
from litellm.types.guardrails import PiiAction, PiiEntityType


@pytest.fixture
def presidio_guardrail():
    return _OPTIONAL_PresidioPIIMasking(
        mock_testing=True,
        output_parse_pii=False,
        pii_entities_config={
            PiiEntityType.US_SSN: PiiAction.MASK,
            PiiEntityType.EMAIL_ADDRESS: PiiAction.MASK,
        },
    )


@pytest.fixture
def mock_user_api_key():
    return UserAPIKeyAuth(api_key="test_key", user_id="test_user")


@pytest.fixture
def mock_cache():
    return MagicMock()


@pytest.mark.asyncio
async def test_precall_masks_pii_in_tool_call_arguments(
    presidio_guardrail, mock_user_api_key, mock_cache
):
    """
    async_pre_call_hook must mask PII in tool_calls[*].function.arguments
    and function_call.arguments, not just message content.
    """
    test_data = {
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "save_record",
                            "arguments": '{"ssn": "123-45-6789", "name": "Alice"}',
                        },
                    }
                ],
            },
            {
                "role": "assistant",
                "content": None,
                "function_call": {
                    "name": "save_record",
                    "arguments": '{"ssn": "123-45-6789"}',
                },
            },
        ],
        "model": "gpt-4",
    }

    async def mock_check_pii(text, output_parse_pii, presidio_config, request_data):
        return text.replace("123-45-6789", "<US_SSN>")

    presidio_guardrail.check_pii = mock_check_pii

    result = await presidio_guardrail.async_pre_call_hook(
        user_api_key_dict=mock_user_api_key,
        cache=mock_cache,
        data=test_data,
        call_type="completion",
    )

    tc_args = result["messages"][0]["tool_calls"][0]["function"]["arguments"]
    assert "123-45-6789" not in tc_args, "tool_calls arguments should be masked"
    assert "<US_SSN>" in tc_args

    fc_args = result["messages"][1]["function_call"]["arguments"]
    assert "123-45-6789" not in fc_args, "function_call arguments should be masked"
    assert "<US_SSN>" in fc_args


@pytest.mark.asyncio
async def test_anthropic_response_masks_pii_in_tool_use_input(presidio_guardrail):
    """
    _process_anthropic_response_for_pii must mask PII in tool_use block
    input values, not just text blocks.  The method takes a raw Anthropic
    message dict (type=="message"), not a ModelResponse.
    """
    response = {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Saving SSN 078-05-1120 now."},
            {
                "type": "tool_use",
                "id": "tu_1",
                "name": "save_record",
                "input": {"ssn": "078-05-1120", "note": "customer"},
            },
        ],
        "model": "claude-sonnet-4-20250514",
        "stop_reason": "end_turn",
    }

    async def mock_check_pii(text, output_parse_pii, presidio_config, request_data):
        return text.replace("078-05-1120", "<US_SSN>")

    presidio_guardrail.check_pii = mock_check_pii

    result = await presidio_guardrail._process_anthropic_response_for_pii(
        response=response,
        request_data={},
        mode="mask",
    )

    text_block = result["content"][0]
    tool_block = result["content"][1]

    assert "078-05-1120" not in text_block["text"], "text block should be masked"
    assert "<US_SSN>" in text_block["text"]

    assert (
        "078-05-1120" not in tool_block["input"]["ssn"]
    ), "tool_use input should be masked"
    assert tool_block["input"]["ssn"] == "<US_SSN>"
    assert tool_block["input"]["note"] == "customer", "non-PII values unchanged"
