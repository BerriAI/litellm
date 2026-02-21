"""
Test DynamoAI Guardrails integration
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.proxy.guardrails.guardrail_hooks.dynamoai import DynamoAIGuardrails
from litellm.proxy._types import UserAPIKeyAuth
from litellm.caching.caching import DualCache
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_dynamoai_blocks_content_with_block_action():
    """
    Test that DynamoAI guardrail blocks content when finalAction is BLOCK.
    """
    # Create guardrail instance
    guardrail = DynamoAIGuardrails(
        guardrail_name="test-dynamoai",
        api_key="test-api-key",
        api_base="https://api.dynamo.ai",
    )

    # Mock the DynamoAI API response with BLOCK action
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "text": "This is harmful content",
        "textType": "MODEL_INPUT",
        "finalAction": "BLOCK",
        "appliedPolicies": [
            {
                "policy": {
                    "id": "policy-123",
                    "name": "Toxicity Policy",
                    "description": "Blocks toxic content",
                    "method": "TOXICITY",
                },
                "outputs": {
                    "action": "BLOCK",
                    "message": "Content contains toxic language"
                }
            }
        ]
    }
    mock_response.raise_for_status = MagicMock()
    with patch.object(guardrail.async_handler, "post", AsyncMock(return_value=mock_response)):
        request_data = {
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "This is harmful content"}
            ],
        }

        # Mock should_run_guardrail to return True
        guardrail.should_run_guardrail = MagicMock(return_value=True)

        # Test that the guardrail raises ValueError for blocked content
        with pytest.raises(ValueError) as exc_info:
            await guardrail.async_pre_call_hook(
                data=request_data,
                user_api_key_dict=UserAPIKeyAuth(),
                call_type="completion",
                cache=MagicMock(spec=DualCache),
            )
    
    # Verify the error message contains policy information
    error_message = str(exc_info.value)
    assert "Guardrail failed" in error_message
    assert "TOXICITY POLICY" in error_message.upper()
    assert "BLOCK" in error_message.upper()


@pytest.mark.asyncio
async def test_dynamoai_allows_content_with_none_action():
    """
    Test that DynamoAI guardrail allows content when finalAction is NONE.
    """
    # Create guardrail instance
    guardrail = DynamoAIGuardrails(
        guardrail_name="test-dynamoai",
        api_key="test-api-key",
        api_base="https://api.dynamo.ai",
    )

    # Mock the DynamoAI API response with NONE action (no violations)
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "text": "Hello, how are you?",
        "textType": "MODEL_INPUT",
        "finalAction": "NONE",
        "appliedPolicies": []
    }
    mock_response.raise_for_status = MagicMock()
    with patch.object(guardrail.async_handler, "post", AsyncMock(return_value=mock_response)):
        request_data = {
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "Hello, how are you?"}
            ],
        }

        # Mock should_run_guardrail to return True
        guardrail.should_run_guardrail = MagicMock(return_value=True)

        # Test that the guardrail allows the content (no exception raised)
        result = await guardrail.async_pre_call_hook(
            data=request_data,
            user_api_key_dict=UserAPIKeyAuth(),
            call_type="completion",
            cache=MagicMock(spec=DualCache),
        )
    
    # Should return the request data unchanged
    assert result == request_data




