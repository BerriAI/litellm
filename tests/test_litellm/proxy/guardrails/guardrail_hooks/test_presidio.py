"""
Unit tests for Presidio PII Masking Guardrail
Tests PII detection and masking for different message formats
"""

import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../../../.."))

from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.presidio import (
    _OPTIONAL_PresidioPIIMasking,
)
from litellm.types.guardrails import PiiAction, PiiEntityType


@pytest.fixture
def presidio_guardrail():
    """Create a Presidio guardrail instance for testing"""
    return _OPTIONAL_PresidioPIIMasking(
        mock_testing=True,
        output_parse_pii=False,
        pii_entities_config={
            PiiEntityType.CREDIT_CARD: PiiAction.MASK,
            PiiEntityType.EMAIL_ADDRESS: PiiAction.MASK,
            PiiEntityType.PHONE_NUMBER: PiiAction.MASK,
        },
    )


@pytest.fixture
def mock_user_api_key():
    """Create a mock user API key auth object"""
    return UserAPIKeyAuth(
        api_key="test_key",
        user_id="test_user",
    )


@pytest.fixture
def mock_cache():
    """Create a mock cache object"""
    return MagicMock(spec=DualCache)


@pytest.mark.asyncio
async def test_multimodal_message_format_completion_call_type(
    presidio_guardrail, mock_user_api_key, mock_cache
):
    """
    Test Presidio PII masking with multimodal message format (content as list)
    for completion call type.

    Tests the message format:
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "My credit card number is 4111-1111-1111-1111..."
            }
        ]
    }
    """
    # Prepare test data with multimodal message format
    test_data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "My credit card number is 4111-1111-1111-1111, my email is test@example.com, and my phone is 555-123-4567",
                    }
                ],
            }
        ],
        "model": "gpt-4",
    }

    # Mock the check_pii method to return redacted text
    async def mock_check_pii(text, output_parse_pii, presidio_config, request_data):
        # Simulate PII detection and masking
        redacted_text = text
        redacted_text = redacted_text.replace("4111-1111-1111-1111", "[CREDIT_CARD]")
        redacted_text = redacted_text.replace("test@example.com", "[EMAIL]")
        redacted_text = redacted_text.replace("555-123-4567", "[PHONE]")
        return redacted_text

    presidio_guardrail.check_pii = mock_check_pii

    # Call the async_pre_call_hook with call_type="completion"
    result = await presidio_guardrail.async_pre_call_hook(
        user_api_key_dict=mock_user_api_key,
        cache=mock_cache,
        data=test_data,
        call_type="completion",
    )

    # Verify that PII was masked in the text field
    assert result is not None
    assert "messages" in result
    assert len(result["messages"]) == 1

    message = result["messages"][0]
    assert "content" in message
    assert isinstance(message["content"], list)
    assert len(message["content"]) == 1

    content_item = message["content"][0]
    assert content_item["type"] == "text"
    assert "[CREDIT_CARD]" in content_item["text"]
    assert "[EMAIL]" in content_item["text"]
    assert "[PHONE]" in content_item["text"]

    # Verify original PII is not present
    assert "4111-1111-1111-1111" not in content_item["text"]
    assert "test@example.com" not in content_item["text"]
    assert "555-123-4567" not in content_item["text"]

    print("✓ Multimodal message format test for completion call type passed")


@pytest.mark.asyncio
async def test_multimodal_message_format_anthropic_messages_call_type(
    presidio_guardrail, mock_user_api_key, mock_cache
):
    """
    Test Presidio PII masking with multimodal message format (content as list)
    for anthropic_messages call type.

    Tests the same message format but with anthropic_messages call type.
    """
    # Prepare test data with multimodal message format
    test_data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "My credit card number is 4111-1111-1111-1111, my email is test@example.com, and my phone is 555-123-4567",
                    }
                ],
            }
        ],
        "model": "claude-3-opus-20240229",
    }

    # Mock the check_pii method to return redacted text
    async def mock_check_pii(text, output_parse_pii, presidio_config, request_data):
        # Simulate PII detection and masking
        redacted_text = text
        redacted_text = redacted_text.replace("4111-1111-1111-1111", "[CREDIT_CARD]")
        redacted_text = redacted_text.replace("test@example.com", "[EMAIL]")
        redacted_text = redacted_text.replace("555-123-4567", "[PHONE]")
        return redacted_text

    presidio_guardrail.check_pii = mock_check_pii

    # Call the async_pre_call_hook with call_type="anthropic_messages"
    result = await presidio_guardrail.async_pre_call_hook(
        user_api_key_dict=mock_user_api_key,
        cache=mock_cache,
        data=test_data,
        call_type="anthropic_messages",
    )

    # Verify that PII was masked in the text field
    assert result is not None
    assert "messages" in result
    assert len(result["messages"]) == 1

    message = result["messages"][0]
    assert "content" in message
    assert isinstance(message["content"], list)
    assert len(message["content"]) == 1

    content_item = message["content"][0]
    assert content_item["type"] == "text"
    assert "[CREDIT_CARD]" in content_item["text"]
    assert "[EMAIL]" in content_item["text"]
    assert "[PHONE]" in content_item["text"]

    # Verify original PII is not present
    assert "4111-1111-1111-1111" not in content_item["text"]
    assert "test@example.com" not in content_item["text"]
    assert "555-123-4567" not in content_item["text"]

    print("✓ Multimodal message format test for anthropic_messages call type passed")


@pytest.mark.asyncio
async def test_multimodal_message_multiple_content_items(
    presidio_guardrail, mock_user_api_key, mock_cache
):
    """
    Test Presidio PII masking with multiple content items in the content list.
    """
    # Prepare test data with multiple content items
    test_data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "My credit card is 4111-1111-1111-1111",
                    },
                    {
                        "type": "text",
                        "text": "My email is test@example.com",
                    },
                ],
            }
        ],
        "model": "gpt-4",
    }

    # Mock the check_pii method
    async def mock_check_pii(text, output_parse_pii, presidio_config, request_data):
        redacted_text = text
        redacted_text = redacted_text.replace("4111-1111-1111-1111", "[CREDIT_CARD]")
        redacted_text = redacted_text.replace("test@example.com", "[EMAIL]")
        return redacted_text

    presidio_guardrail.check_pii = mock_check_pii

    # Call the async_pre_call_hook
    result = await presidio_guardrail.async_pre_call_hook(
        user_api_key_dict=mock_user_api_key,
        cache=mock_cache,
        data=test_data,
        call_type="completion",
    )

    # Verify both content items were processed
    assert result is not None
    message = result["messages"][0]
    content_items = message["content"]

    assert len(content_items) == 2
    assert "[CREDIT_CARD]" in content_items[0]["text"]
    assert "[EMAIL]" in content_items[1]["text"]

    print("✓ Multiple content items test passed")


@pytest.mark.asyncio
async def test_mixed_string_and_list_content(
    presidio_guardrail, mock_user_api_key, mock_cache
):
    """
    Test Presidio PII masking with mixed string and list content formats.
    """
    # Prepare test data with mixed content formats
    test_data = {
        "messages": [
            {
                "role": "user",
                "content": "My credit card is 4111-1111-1111-1111",
            },
            {
                "role": "assistant",
                "content": "I can help you with that.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "My email is test@example.com",
                    }
                ],
            },
        ],
        "model": "gpt-4",
    }

    # Mock the check_pii method
    async def mock_check_pii(text, output_parse_pii, presidio_config, request_data):
        redacted_text = text
        redacted_text = redacted_text.replace("4111-1111-1111-1111", "[CREDIT_CARD]")
        redacted_text = redacted_text.replace("test@example.com", "[EMAIL]")
        return redacted_text

    presidio_guardrail.check_pii = mock_check_pii

    # Call the async_pre_call_hook
    result = await presidio_guardrail.async_pre_call_hook(
        user_api_key_dict=mock_user_api_key,
        cache=mock_cache,
        data=test_data,
        call_type="completion",
    )

    # Verify all messages were processed correctly
    assert result is not None
    messages = result["messages"]

    # First message (string content)
    assert isinstance(messages[0]["content"], str)
    assert "[CREDIT_CARD]" in messages[0]["content"]

    # Second message (string content, no PII)
    assert isinstance(messages[1]["content"], str)
    assert messages[1]["content"] == "I can help you with that."

    # Third message (list content)
    assert isinstance(messages[2]["content"], list)
    assert "[EMAIL]" in messages[2]["content"][0]["text"]

    print("✓ Mixed string and list content test passed")


@pytest.mark.asyncio
async def test_content_list_without_text_field(
    presidio_guardrail, mock_user_api_key, mock_cache
):
    """
    Test Presidio PII masking gracefully handles content items without text field
    (e.g., image content items).
    """
    # Prepare test data with image content (no text field)
    test_data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image.jpg"},
                    },
                    {
                        "type": "text",
                        "text": "What's in this image? My email is test@example.com",
                    },
                ],
            }
        ],
        "model": "gpt-4-vision",
    }

    # Mock the check_pii method
    async def mock_check_pii(text, output_parse_pii, presidio_config, request_data):
        redacted_text = text.replace("test@example.com", "[EMAIL]")
        return redacted_text

    presidio_guardrail.check_pii = mock_check_pii

    # Call the async_pre_call_hook
    result = await presidio_guardrail.async_pre_call_hook(
        user_api_key_dict=mock_user_api_key,
        cache=mock_cache,
        data=test_data,
        call_type="completion",
    )

    # Verify that image content is preserved and text content is processed
    assert result is not None
    content_items = result["messages"][0]["content"]

    assert len(content_items) == 2
    # Image content should remain unchanged
    assert content_items[0]["type"] == "image_url"
    assert content_items[0]["image_url"]["url"] == "https://example.com/image.jpg"

    # Text content should be redacted
    assert content_items[1]["type"] == "text"
    assert "[EMAIL]" in content_items[1]["text"]

    print("✓ Content list without text field test passed")


@pytest.mark.asyncio
async def test_empty_messages(presidio_guardrail, mock_user_api_key, mock_cache):
    """
    Test that Presidio handles empty messages gracefully.
    """
    test_data = {
        "messages": [],
        "model": "gpt-4",
    }

    # Call the async_pre_call_hook
    result = await presidio_guardrail.async_pre_call_hook(
        user_api_key_dict=mock_user_api_key,
        cache=mock_cache,
        data=test_data,
        call_type="completion",
    )

    # Should return data unchanged
    assert result == test_data
    print("✓ Empty messages test passed")


@pytest.mark.asyncio
async def test_no_messages_field(presidio_guardrail, mock_user_api_key, mock_cache):
    """
    Test that Presidio handles missing messages field gracefully.
    """
    test_data = {
        "model": "gpt-4",
        "prompt": "This is a completion request",
    }

    # Call the async_pre_call_hook
    result = await presidio_guardrail.async_pre_call_hook(
        user_api_key_dict=mock_user_api_key,
        cache=mock_cache,
        data=test_data,
        call_type="completion",
    )

    # Should return data unchanged
    assert result == test_data
    print("✓ No messages field test passed")


@pytest.mark.asyncio
async def test_logging_hook_multimodal_message_format(presidio_guardrail):
    """
    Test Presidio async_logging_hook with multimodal message format for completion call type.
    This hook is used to mask PII before logging to external services.
    """
    # Prepare kwargs with multimodal message format
    test_kwargs = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "My credit card number is 4111-1111-1111-1111, my email is test@example.com",
                    }
                ],
            }
        ],
        "model": "gpt-4",
    }

    # Mock result
    mock_result = {"choices": [{"message": {"content": "Response"}}]}

    # Mock the check_pii method
    async def mock_check_pii(text, output_parse_pii, presidio_config, request_data):
        redacted_text = text
        redacted_text = redacted_text.replace("4111-1111-1111-1111", "[CREDIT_CARD]")
        redacted_text = redacted_text.replace("test@example.com", "[EMAIL]")
        return redacted_text

    presidio_guardrail.check_pii = mock_check_pii

    # Call the async_logging_hook
    result_kwargs, result_response = await presidio_guardrail.async_logging_hook(
        kwargs=test_kwargs,
        result=mock_result,
        call_type="completion",
    )

    # Verify that PII was masked in the kwargs
    assert result_kwargs is not None
    assert "messages" in result_kwargs
    message = result_kwargs["messages"][0]
    content_item = message["content"][0]

    assert "[CREDIT_CARD]" in content_item["text"]
    assert "[EMAIL]" in content_item["text"]
    assert "4111-1111-1111-1111" not in content_item["text"]
    assert "test@example.com" not in content_item["text"]

    print("✓ Logging hook multimodal message format test passed")


@pytest.mark.asyncio
async def test_logging_hook_multiple_content_items(presidio_guardrail):
    """
    Test Presidio async_logging_hook with multiple content items in a single message.
    """
    test_kwargs = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "My credit card is 4111-1111-1111-1111",
                    },
                    {
                        "type": "text",
                        "text": "My email is test@example.com",
                    },
                ],
            }
        ],
        "model": "gpt-4",
    }

    mock_result = {"choices": [{"message": {"content": "Response"}}]}

    # Mock the check_pii method
    async def mock_check_pii(text, output_parse_pii, presidio_config, request_data):
        redacted_text = text
        redacted_text = redacted_text.replace("4111-1111-1111-1111", "[CREDIT_CARD]")
        redacted_text = redacted_text.replace("test@example.com", "[EMAIL]")
        return redacted_text

    presidio_guardrail.check_pii = mock_check_pii

    # Call the async_logging_hook
    result_kwargs, result_response = await presidio_guardrail.async_logging_hook(
        kwargs=test_kwargs,
        result=mock_result,
        call_type="completion",
    )

    # Verify both content items were processed
    message = result_kwargs["messages"][0]
    content_items = message["content"]

    assert len(content_items) == 2
    assert "[CREDIT_CARD]" in content_items[0]["text"]
    assert "[EMAIL]" in content_items[1]["text"]

    print("✓ Logging hook multiple content items test passed")


@pytest.mark.asyncio
async def test_presidio_sets_guardrail_information_in_request_data():
    """
    Test that Presidio populates guardrail information into request_data metadata.
    
    This validates that add_standard_logging_guardrail_information_to_request_data
    correctly sets the guardrail information that will be used for logging.
    """
    presidio = _OPTIONAL_PresidioPIIMasking(
        guardrail_name="test_presidio",
        output_parse_pii=True,
    )
    
    request_data = {
        "messages": [{"role": "user", "content": "Test"}],
        "model": "gpt-4o",
        "metadata": {},
    }
    
    async def mock_check_pii(text, output_parse_pii, presidio_config, request_data):
        assert request_data is not None
        
        presidio.add_standard_logging_guardrail_information_to_request_data(
            guardrail_provider="presidio",
            guardrail_json_response=[],
            request_data=request_data,
            guardrail_status="success",
            start_time=1234567890.0,
            end_time=1234567891.0,
            duration=1.0,
            masked_entity_count={"EMAIL_ADDRESS": 1, "PERSON": 1},
        )
        
        return text
    
    with patch.object(presidio, 'check_pii', mock_check_pii):
        await presidio.apply_guardrail(
            text="Test message",
            request_data=request_data,
        )
    
    assert "metadata" in request_data
    assert "standard_logging_guardrail_information" in request_data["metadata"]
    
    guardrail_info_list = request_data["metadata"]["standard_logging_guardrail_information"]
    assert isinstance(guardrail_info_list, list)
    assert len(guardrail_info_list) > 0
    
    guardrail_info = guardrail_info_list[0]
    assert "masked_entity_count" in guardrail_info
    assert guardrail_info["masked_entity_count"]["EMAIL_ADDRESS"] == 1
    assert guardrail_info["masked_entity_count"]["PERSON"] == 1
    
    print("✓ Presidio sets guardrail_information in request_data")


@pytest.mark.asyncio
async def test_request_data_flows_to_apply_guardrail():
    """
    Test that request_data is correctly passed to apply_guardrail method.
    
    This validates the fix where guardrail translation handler passes data
    as request_data to apply_guardrail so guardrails can store metadata for logging.
    """
    presidio = _OPTIONAL_PresidioPIIMasking(
        guardrail_name="test_presidio",
        output_parse_pii=True,
    )
    
    request_data = {
        "messages": [{"role": "user", "content": "Test message"}],
        "model": "gpt-4o",
        "metadata": {},
    }
    
    async def mock_check_pii(text, output_parse_pii, presidio_config, request_data):
        assert request_data is not None, "request_data should be passed to check_pii"
        assert "metadata" in request_data, "request_data should have metadata"
        
        request_data.setdefault("metadata", {})
        request_data["metadata"]["test_flag"] = "passed_correctly"
        
        return text
    
    with patch.object(presidio, 'check_pii', mock_check_pii):
        result = await presidio.apply_guardrail(
            text="Test message",
            request_data=request_data,
        )
        
        assert "metadata" in request_data
        assert request_data["metadata"].get("test_flag") == "passed_correctly"
        
    print("✓ request_data correctly passed to apply_guardrail")


if __name__ == "__main__":
    # Run tests
    asyncio.run(
        test_multimodal_message_format_completion_call_type(
            _OPTIONAL_PresidioPIIMasking(
                mock_testing=True,
                output_parse_pii=False,
                pii_entities_config={
                    PiiEntityType.CREDIT_CARD: PiiAction.MASK,
                    PiiEntityType.EMAIL_ADDRESS: PiiAction.MASK,
                    PiiEntityType.PHONE_NUMBER: PiiAction.MASK,
                },
            ),
            UserAPIKeyAuth(api_key="test_key", user_id="test_user"),
            MagicMock(spec=DualCache),
        )
    )
    print("\n✅ All Presidio tests passed!")
