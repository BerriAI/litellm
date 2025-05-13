import sys
import os
import io, asyncio
import pytest
sys.path.insert(0, os.path.abspath("../.."))
import litellm
from litellm.proxy.guardrails.guardrail_hooks.presidio import _OPTIONAL_PresidioPIIMasking, PresidioPerRequestConfig
from litellm.types.guardrails import PiiEntityType, PiiAction
from litellm.proxy._types import UserAPIKeyAuth
from litellm.caching.caching import DualCache



@pytest.mark.asyncio
async def test_presidio_with_entities_config():
    """Test for Presidio guardrail with entities config - requires actual Presidio API"""
    # Setup the guardrail with specific entities config
    litellm._turn_on_debug()
    pii_entities_config = {
        PiiEntityType.CREDIT_CARD: PiiAction.MASK,
        PiiEntityType.EMAIL_ADDRESS: PiiAction.MASK,
    }
    
    presidio_guardrail = _OPTIONAL_PresidioPIIMasking(
        pii_entities_config=pii_entities_config,
        presidio_analyzer_api_base=os.environ.get("PRESIDIO_ANALYZER_API_BASE"),
        presidio_anonymizer_api_base=os.environ.get("PRESIDIO_ANONYMIZER_API_BASE")
    )
    
    # Test text with different PII types
    test_text = "My credit card number is 4111-1111-1111-1111, my email is test@example.com, and my phone is 555-123-4567"
    
    # Test the analyze request configuration
    analyze_request = presidio_guardrail._get_presidio_analyze_request_payload(
        text=test_text,
        presidio_config=None,
        request_data={}
    )
    
    # Verify entities were passed correctly
    assert "entities" in analyze_request
    assert set(analyze_request["entities"]) == set(pii_entities_config.keys())
    
    # Test the check_pii method - this will call the actual Presidio API
    redacted_text = await presidio_guardrail.check_pii(
        text=test_text,
        output_parse_pii=True,
        presidio_config=None,
        request_data={}
    )
    
    # Verify PII has been masked/replaced/redacted in the result
    assert "4111-1111-1111-1111" not in redacted_text
    assert "test@example.com" not in redacted_text

    # Since this entity is not in the config, it should not be masked
    assert "555-123-4567" in redacted_text
    
    # The specific replacements will vary based on Presidio's implementation
    print(f"Redacted text: {redacted_text}")



@pytest.mark.asyncio
async def test_presidio_pre_call_hook_with_entities_config():
    """Test for Presidio guardrail pre-call hook with entities config on a chat completion request"""
    # Setup the guardrail with specific entities config
    pii_entities_config = {
        PiiEntityType.CREDIT_CARD: PiiAction.MASK,
        PiiEntityType.EMAIL_ADDRESS: PiiAction.MASK,
    }
    
    presidio_guardrail = _OPTIONAL_PresidioPIIMasking(
        pii_entities_config=pii_entities_config,
        presidio_analyzer_api_base=os.environ.get("PRESIDIO_ANALYZER_API_BASE"),
        presidio_anonymizer_api_base=os.environ.get("PRESIDIO_ANONYMIZER_API_BASE")
    )
    
    # Create a sample chat completion request with PII data
    data = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "My credit card is 4111-1111-1111-1111 and my email is test@example.com. My phone number is 555-123-4567"}
        ],
        "model": "gpt-3.5-turbo"
    }
    
    # Mock objects needed for the pre-call hook
    user_api_key_dict = UserAPIKeyAuth(api_key="test_key")
    cache = DualCache()
    
    # Call the pre-call hook
    modified_data = await presidio_guardrail.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data=data,
        call_type="completion"
    )
    
    # Verify the messages have been modified to mask PII
    assert modified_data["messages"][0]["content"] == "You are a helpful assistant."  # System prompt should be unchanged
    
    user_message = modified_data["messages"][1]["content"]
    assert "4111-1111-1111-1111" not in user_message
    assert "test@example.com" not in user_message

    # Since this entity is not in the config, it should not be masked
    assert "555-123-4567" in user_message
    
    print(f"Modified user message: {user_message}")


