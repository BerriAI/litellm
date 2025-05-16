import sys
import os
import io, asyncio
import pytest
import time
from litellm import mock_completion
from unittest.mock import MagicMock, AsyncMock, patch
sys.path.insert(0, os.path.abspath("../.."))
import litellm
from litellm.proxy.guardrails.guardrail_hooks.lakera_ai_v2 import LakeraAIGuardrail
from litellm.types.guardrails import PiiEntityType, PiiAction
from litellm.proxy._types import UserAPIKeyAuth
from litellm.caching.caching import DualCache
from litellm.exceptions import BlockedPiiEntityError
from litellm.types.utils import CallTypes as LitellmCallTypes


@pytest.mark.asyncio
async def test_lakera_pre_call_hook_for_pii_masking():
    """Test for Lakera guardrail pre-call hook for PII masking"""
    # Setup the guardrail with specific entities config
    litellm._turn_on_debug()
    lakera_guardrail = LakeraAIGuardrail(
        api_key=os.environ.get("LAKERA_API_KEY"),
    )
    
    # Create a sample request with PII data
    data = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "My credit card is 4111-1111-1111-1111 and my email is test@example.com. My phone number is 555-123-4567"}
        ],
        "model": "gpt-3.5-turbo",
        "metadata": {}
    }
    
    # Mock objects needed for the pre-call hook
    user_api_key_dict = UserAPIKeyAuth(api_key="test_key")
    cache = DualCache()
    
    # Call the pre-call hook with the specified call type
    modified_data = await lakera_guardrail.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data=data,
        call_type="completion"
    )
    print(modified_data)
    
    # Verify the messages have been modified to mask PII
    assert modified_data["messages"][0]["content"] == "You are a helpful assistant."  # System prompt should be unchanged
    
    user_message = modified_data["messages"][1]["content"]
    assert "4111-1111-1111-1111" not in user_message
    assert "test@example.com" not in user_message

