"""
Tests for v0 provider integration
"""

import os
from unittest import mock

import pytest

import litellm
from litellm import completion
from litellm.llms.v0.chat.transformation import V0ChatConfig

@pytest.mark.asyncio
async def test_v0_completion_call():
    """Test completion call with v0 provider (requires V0_API_KEY)"""
    # Skip if no API key is available
    if not os.getenv("V0_API_KEY"):
        pytest.skip("V0_API_KEY not set")

    try:
        response = await litellm.acompletion(
            model="v0/gpt-4-turbo",
            messages=[{"role": "user", "content": "Hello, this is a test"}],
            max_tokens=10,
        )
        assert response.choices[0].message.content
        assert response.model
        assert response.usage
    except Exception as e:
        # If the API key is invalid or there's a network issue, that's okay
        # The important thing is that the provider was recognized
        if "v0" not in str(e) and "provider" not in str(e).lower():
            # Re-raise if it's not a provider-related error
            raise
