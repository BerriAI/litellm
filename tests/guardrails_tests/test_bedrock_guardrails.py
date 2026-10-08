import io, asyncio
import pytest

import litellm
from litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails import (
    BedrockGuardrail,
    _redact_pii_matches,
)
from litellm.proxy._types import UserAPIKeyAuth
from unittest.mock import MagicMock, AsyncMock, patch


@pytest.mark.asyncio
async def test_bedrock_guardrails_pii_masking():
    # Create proper mock objects
    mock_user_api_key_dict = UserAPIKeyAuth()

    guardrail = BedrockGuardrail(
        guardrailIdentifier="wf0hkdb5x07f",
        guardrailVersion="DRAFT",
    )

    request_data = {
        "model": "gpt-5.5",
        "messages": [
            {"role": "user", "content": "Hello, my phone number is +1 412 555 1212"},
            {"role": "assistant", "content": "Hello, how can I help you today?"},
            {"role": "user", "content": "I need to cancel my order"},
            {
                "role": "user",
                "content": "ok, my credit card number is 1234-5678-9012-3456",
            },
        ],
    }

    response = await guardrail.async_moderation_hook(
        data=request_data,
        user_api_key_dict=mock_user_api_key_dict,
        call_type="completion",
    )
    print("response after moderation hook", response)

    if response:  # Only assert if response is not None
        assert response["messages"][0]["content"] == "Hello, my phone number is {PHONE}"
        assert response["messages"][1]["content"] == "Hello, how can I help you today?"
        assert response["messages"][2]["content"] == "I need to cancel my order"
        assert (
            response["messages"][3]["content"]
            == "ok, my credit card number is {CREDIT_DEBIT_CARD_NUMBER}"
        )


@pytest.mark.asyncio
async def test_bedrock_guardrails_pii_masking_content_list():
    # Create proper mock objects
    mock_user_api_key_dict = UserAPIKeyAuth()

    guardrail = BedrockGuardrail(
        guardrailIdentifier="wf0hkdb5x07f",
        guardrailVersion="DRAFT",
    )

    request_data = {
        "model": "gpt-5.5",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Hello, my phone number is +1 412 555 1212",
                    },
                    {"type": "text", "text": "what time is it?"},
                ],
            },
            {"role": "assistant", "content": "Hello, how can I help you today?"},
            {"role": "user", "content": "who is the president of the united states?"},
        ],
    }

    response = await guardrail.async_moderation_hook(
        data=request_data,
        user_api_key_dict=mock_user_api_key_dict,
        call_type="completion",
    )
    print(response)

    if response:  # Only assert if response is not None
        # Verify that the list content is properly masked
        assert isinstance(response["messages"][0]["content"], list)
        assert (
            response["messages"][0]["content"][0]["text"]
            == "Hello, my phone number is {PHONE}"
        )
        assert response["messages"][0]["content"][1]["text"] == "what time is it?"
        assert response["messages"][1]["content"] == "Hello, how can I help you today?"
        assert (
            response["messages"][2]["content"]
            == "who is the president of the united states?"
        )
