"""
Test DynamoAI Guardrails integration
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.proxy.guardrails.guardrail_hooks.dynamoai import DynamoAIGuardrails
from litellm.proxy._types import UserAPIKeyAuth
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_dynamoai_guardrails_api_call():
    """
    Test that _call_dynamoai_guardrails makes the correct API call and processes response
    """
    # Create mock user api key dict
    mock_user_api_key_dict = UserAPIKeyAuth()
    
    # Create guardrail instance with optional model_id
    guardrail = DynamoAIGuardrails(
        guardrail_name="test-dynamoai",
        api_key="test-api-key",
        api_base="https://api.dynamo.ai",
        model_id="test-model-id"
    )

    # Verify API URL is correct
    assert guardrail.api_url == "https://api.dynamo.ai/v1/moderation/analyze/"
    
    # Test messages to analyze
    test_messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"}
    ]
    
    # Call the method
    response = await guardrail._call_dynamoai_guardrails(
        messages=test_messages,
        text_type="input",
        request_data={"model": "gpt-4"}
    )
    
    print(f"DynamoAI API call test response={response}")




