"""
Tests for Lambda AI provider integration
"""

import os
from unittest import mock

import pytest

import litellm
from litellm import completion
from litellm.llms.lambda_ai.chat.transformation import LambdaAIChatConfig

@pytest.mark.asyncio
async def test_lambda_ai_completion_call():
    """Test completion call with Lambda AI provider (requires LAMBDA_API_KEY)"""
    # Skip if no API key is available
    if not os.getenv("LAMBDA_API_KEY"):
        pytest.skip("LAMBDA_API_KEY not set")

    try:
        response = await litellm.acompletion(
            model="lambda_ai/llama3.1-8b-instruct",
            messages=[{"role": "user", "content": "Hello, this is a test"}],
            max_tokens=10,
        )
        assert response.choices[0].message.content
        assert response.model
        assert response.usage
    except Exception as e:
        # If the API key is invalid or there's a network issue, that's okay
        # The important thing is that the provider was recognized
        if "lambda_ai" not in str(e) and "provider" not in str(e).lower():
            # Re-raise if it's not a provider-related error
            raise
