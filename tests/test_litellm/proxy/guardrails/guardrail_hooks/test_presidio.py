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
from litellm.types.guardrails import LitellmParams, PiiAction, PiiEntityType
from litellm.types.utils import Choices, Message, ModelResponse
import litellm


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
        mock_testing=True,
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

    with patch.object(presidio, "check_pii", mock_check_pii):
        await presidio.apply_guardrail(
            inputs={"texts": ["Test message"]},
            request_data=request_data,
            input_type="request",
        )

    assert "metadata" in request_data
    assert "standard_logging_guardrail_information" in request_data["metadata"]

    guardrail_info_list = request_data["metadata"][
        "standard_logging_guardrail_information"
    ]
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
        mock_testing=True,
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

    with patch.object(presidio, "check_pii", mock_check_pii):
        result = await presidio.apply_guardrail(
            inputs={"texts": ["Test message"]},
            request_data=request_data,
            input_type="request",
        )

        assert "metadata" in request_data
        assert request_data["metadata"].get("test_flag") == "passed_correctly"

    print("✓ request_data correctly passed to apply_guardrail")


@pytest.mark.asyncio
async def test_output_masking_apply_to_output_only(mock_user_api_key):
    """
    Ensure output masking runs when apply_to_output is enabled.
    """

    presidio = _OPTIONAL_PresidioPIIMasking(
        mock_testing=True,
        apply_to_output=True,
        pii_entities_config={PiiEntityType.CREDIT_CARD: PiiAction.MASK},
    )

    async def mock_check_pii(text, output_parse_pii, presidio_config, request_data):
        return text.replace("4111-1111-1111-1111", "[CREDIT_CARD]")

    presidio.check_pii = mock_check_pii

    response = ModelResponse(
        id="1",
        object="chat.completion",
        created=0,
        model="gpt-test",
        choices=[
            Choices(
                message=Message(
                    role="assistant",
                    content="Card is 4111-1111-1111-1111",
                ),
                index=0,
                finish_reason="stop",
            )
        ],
    )

    result = await presidio.async_post_call_success_hook(
        data={},
        user_api_key_dict=mock_user_api_key,
        response=response,
    )

    assert "[CREDIT_CARD]" in result.choices[0].message.content
    assert "4111-1111-1111-1111" not in result.choices[0].message.content


@pytest.mark.asyncio
async def test_presidio_filter_scope_initializer(monkeypatch):
    """
    Ensure initializer respects presidio_filter_scope for input/output/both.
    """

    created = []

    class DummyGuardrail:
        def __init__(self, apply_to_output: bool = False, event_hook=None, **kwargs):
            self.apply_to_output = apply_to_output
            self.event_hook = event_hook
            created.append(self)

        def update_in_memory_litellm_params(self, litellm_params):
            pass

    class DummyManager:
        def __init__(self):
            self.added = []

        def add_litellm_callback(self, cb):
            self.added.append(cb)

    mgr = DummyManager()
    monkeypatch.setattr(litellm, "logging_callback_manager", mgr, raising=False)
    import litellm.proxy.guardrails.guardrail_initializers as gi
    import litellm.proxy.guardrails.guardrail_hooks.presidio as presidio_mod
    monkeypatch.setattr(
        presidio_mod, "_OPTIONAL_PresidioPIIMasking", DummyGuardrail, raising=False
    )
    monkeypatch.setattr(gi, "_OPTIONAL_PresidioPIIMasking", DummyGuardrail, raising=False)

    # input-only
    created.clear()
    from litellm.proxy.guardrails.guardrail_initializers import initialize_presidio

    params_input = LitellmParams(guardrail="presidio", mode="pre_call", presidio_filter_scope="input")
    guardrail_dict = {"guardrail_name": "g1"}
    cb = initialize_presidio(params_input, guardrail_dict)
    assert cb is created[0]
    assert created[0].apply_to_output is False

    # output-only
    created.clear()
    params_output = LitellmParams(guardrail="presidio", mode="pre_call", presidio_filter_scope="output")
    cb = initialize_presidio(params_output, guardrail_dict)
    assert len(created) == 1
    assert created[0].apply_to_output is True

    # both -> expect two callbacks (input + output)
    created.clear()
    params_both = LitellmParams(guardrail="presidio", mode="pre_call", presidio_filter_scope="both")
    cb = initialize_presidio(params_both, guardrail_dict)
    assert len(created) == 2
    assert any(not c.apply_to_output for c in created)
    assert any(c.apply_to_output for c in created)


@pytest.mark.asyncio
async def test_empty_content_handling(presidio_guardrail, mock_user_api_key, mock_cache):
    """
    Test that Presidio handles empty content gracefully.
    
    This is common in tool/function calling where assistant messages have
    empty content but include tool_calls.
    
    Bug fix: Previously crashed with:
    TypeError: argument after ** must be a mapping, not str
    """
    test_data = {
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": "",  # Empty content - common in tool calls
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "calculator", "arguments": '{"a":2,"b":2}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_123", "content": "4"},
        ],
        "model": "gpt-4",
    }

    # Mock check_pii to simulate PII processing without needing Presidio API
    async def mock_check_pii(text, output_parse_pii, presidio_config, request_data):
        # Empty text returns as-is (this is what our fix ensures)
        return text

    presidio_guardrail.check_pii = mock_check_pii

    # This should not raise an exception
    result = await presidio_guardrail.async_pre_call_hook(
        user_api_key_dict=mock_user_api_key,
        cache=mock_cache,
        data=test_data,
        call_type="completion",
    )

    assert result is not None
    assert "messages" in result
    # Verify messages are preserved
    assert len(result["messages"]) == 3

    print("✓ Empty content handling test passed")


@pytest.mark.asyncio
async def test_whitespace_only_content(presidio_guardrail, mock_user_api_key, mock_cache):
    """
    Test that Presidio handles whitespace-only content gracefully.
    
    Whitespace-only content should be treated the same as empty content.
    """
    test_data = {
        "messages": [
            {"role": "user", "content": "   "},  # Whitespace only
            {"role": "assistant", "content": "\n\t  "},  # Tabs and newlines
            {"role": "user", "content": "Real question here"},
        ],
        "model": "gpt-4",
    }

    # Mock check_pii to simulate PII processing
    async def mock_check_pii(text, output_parse_pii, presidio_config, request_data):
        return text

    presidio_guardrail.check_pii = mock_check_pii

    result = await presidio_guardrail.async_pre_call_hook(
        user_api_key_dict=mock_user_api_key,
        cache=mock_cache,
        data=test_data,
        call_type="completion",
    )

    assert result is not None
    assert len(result["messages"]) == 3

    print("✓ Whitespace-only content test passed")


@pytest.mark.asyncio
async def test_analyze_text_with_empty_string():
    """
    Test analyze_text method directly with empty string.
    
    Should return empty list without making API call to Presidio.
    """
    presidio = _OPTIONAL_PresidioPIIMasking(
        presidio_analyzer_api_base="http://test:5002/",
        presidio_anonymizer_api_base="http://test:5001/",
        output_parse_pii=False,
    )

    # Test with empty string - should return immediately without API call
    result = await presidio.analyze_text(
        text="",
        presidio_config=None,
        request_data={},
    )
    assert result == [], "Empty text should return empty list"

    # Test with whitespace only - should return immediately
    result = await presidio.analyze_text(
        text="   \n\t   ",
        presidio_config=None,
        request_data={},
    )
    assert result == [], "Whitespace-only text should return empty list"

    print("✓ analyze_text empty string test passed")


@pytest.mark.asyncio
async def test_analyze_text_error_dict_handling():
    """
    Test that analyze_text handles error dict responses from Presidio API.
    
    When Presidio returns {'error': 'No text provided'}, should handle gracefully
    instead of crashing with TypeError.
    """
    presidio = _OPTIONAL_PresidioPIIMasking(
        presidio_analyzer_api_base="http://mock-presidio:5002/",
        presidio_anonymizer_api_base="http://mock-presidio:5001/",
        output_parse_pii=False,
    )

    # Mock the HTTP response to return error dict
    class MockResponse:
        async def json(self):
            return {"error": "No text provided"}
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
    
    class MockSession:
        def post(self, *args, **kwargs):
            return MockResponse()
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass

    with patch("aiohttp.ClientSession", return_value=MockSession()):
        result = await presidio.analyze_text(
            text="some text",
            presidio_config=None,
            request_data={},
        )
        # Should return empty list when error dict is received
        assert result == [], "Error dict should be handled gracefully"

    print("✓ analyze_text error dict handling test passed")


@pytest.mark.asyncio
async def test_tool_calling_complete_scenario(presidio_guardrail, mock_user_api_key, mock_cache):
    """
    Test complete tool calling scenario with PII in user message.
    
    This tests the real-world scenario where:
    1. User provides a query with PII
    2. Assistant responds with empty content + tool_calls
    3. Tool provides response
    4. Assistant provides final answer
    """
    test_data = {
        "messages": [
            {
                "role": "user",
                "content": "My email is john.doe@example.com. Can you look up my account?",
            },
            {
                "role": "assistant",
                "content": "",  # Empty - tool call
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "lookup_account", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_abc", "content": "Account found"},
            {"role": "assistant", "content": "I found your account information."},
        ],
        "model": "gpt-4",
    }

    # Mock check_pii to simulate PII masking
    async def mock_check_pii(text, output_parse_pii, presidio_config, request_data):
        if "john.doe@example.com" in text:
            return text.replace("john.doe@example.com", "[EMAIL]")
        return text

    presidio_guardrail.check_pii = mock_check_pii

    result = await presidio_guardrail.async_pre_call_hook(
        user_api_key_dict=mock_user_api_key,
        cache=mock_cache,
        data=test_data,
        call_type="completion",
    )

    assert result is not None
    # Verify PII was masked in user message
    assert "[EMAIL]" in result["messages"][0]["content"]
    assert "john.doe@example.com" not in result["messages"][0]["content"]
    # Verify other messages preserved
    assert len(result["messages"]) == 4

    print("✓ Tool calling complete scenario test passed")


def test_filter_drops_low_score_detection():
    """
    Detections below the configured score threshold should be removed.
    """
    guardrail = _OPTIONAL_PresidioPIIMasking(
        mock_testing=True,
        presidio_score_thresholds={PiiEntityType.CREDIT_CARD: 0.8},
    )
    analyze_results = [
        {"entity_type": PiiEntityType.CREDIT_CARD, "score": 0.7, "start": 0, "end": 4}
    ]

    filtered = guardrail.filter_analyze_results_by_score(analyze_results)
    assert filtered == []


def test_filter_preserves_high_score_detection():
    """
    Detections meeting the score threshold should be preserved.
    """
    guardrail = _OPTIONAL_PresidioPIIMasking(
        mock_testing=True,
        presidio_score_thresholds={PiiEntityType.CREDIT_CARD: 0.8},
    )
    analyze_results = [
        {"entity_type": PiiEntityType.CREDIT_CARD, "score": 0.9, "start": 0, "end": 4}
    ]

    filtered = guardrail.filter_analyze_results_by_score(analyze_results)
    assert len(filtered) == 1
    assert filtered[0]["entity_type"] == PiiEntityType.CREDIT_CARD


def test_no_thresholds_returns_all():
    """
    With no thresholds configured, all detections are kept.
    """
    guardrail = _OPTIONAL_PresidioPIIMasking(mock_testing=True)
    analyze_results = [
        {"entity_type": PiiEntityType.CREDIT_CARD, "score": 0.1, "start": 0, "end": 4},
        {"entity_type": PiiEntityType.EMAIL_ADDRESS, "score": 0.2, "start": 5, "end": 9},
    ]

    filtered = guardrail.filter_analyze_results_by_score(analyze_results)
    assert len(filtered) == 2


def test_entity_specific_threshold_only_applies_to_that_entity():
    """
    Entity-specific thresholds do not affect other entity types.
    """
    guardrail = _OPTIONAL_PresidioPIIMasking(
        mock_testing=True,
        presidio_score_thresholds={PiiEntityType.CREDIT_CARD: 0.8},
    )
    analyze_results = [
        {"entity_type": PiiEntityType.CREDIT_CARD, "score": 0.7, "start": 0, "end": 4},
        {"entity_type": PiiEntityType.EMAIL_ADDRESS, "score": 0.1, "start": 5, "end": 9},
    ]

    filtered = guardrail.filter_analyze_results_by_score(analyze_results)
    # CREDIT_CARD is filtered, EMAIL_ADDRESS is kept because no threshold
    assert len(filtered) == 1
    assert filtered[0]["entity_type"] == PiiEntityType.EMAIL_ADDRESS


def test_filter_uses_default_all_threshold():
    """
    Default ALL threshold applies to any entity without a specific override.
    """
    guardrail = _OPTIONAL_PresidioPIIMasking(
        mock_testing=True,
        presidio_score_thresholds={"ALL": 0.75},
    )
    analyze_results = [
        {"entity_type": PiiEntityType.CREDIT_CARD, "score": 0.7, "start": 0, "end": 4},
        {"entity_type": PiiEntityType.EMAIL_ADDRESS, "score": 0.8, "start": 5, "end": 9},
    ]

    filtered = guardrail.filter_analyze_results_by_score(analyze_results)
    assert len(filtered) == 1
    assert filtered[0]["entity_type"] == PiiEntityType.EMAIL_ADDRESS


def test_entity_specific_overrides_default_threshold():
    """
    Entity-specific threshold should override the ALL default.
    """
    guardrail = _OPTIONAL_PresidioPIIMasking(
        mock_testing=True,
        presidio_score_thresholds={
            "ALL": 0.8,
            PiiEntityType.CREDIT_CARD: 0.6,
        },
    )
    analyze_results = [
        {"entity_type": PiiEntityType.CREDIT_CARD, "score": 0.65, "start": 0, "end": 4},
        {"entity_type": PiiEntityType.EMAIL_ADDRESS, "score": 0.75, "start": 5, "end": 9},
    ]

    filtered = guardrail.filter_analyze_results_by_score(analyze_results)
    # CREDIT_CARD passes due to override, EMAIL_ADDRESS dropped by ALL threshold
    assert len(filtered) == 1
    assert filtered[0]["entity_type"] == PiiEntityType.CREDIT_CARD


@pytest.mark.asyncio
async def test_anonymize_skips_when_no_detections_after_filter():
    """
    When all detections are filtered out, anonymize_text should return the original text.
    """
    guardrail = _OPTIONAL_PresidioPIIMasking(
        mock_testing=True,
        presidio_score_thresholds={PiiEntityType.CREDIT_CARD: 0.8},
    )
    masked_entity_count = {}
    text = "4111"

    filtered = guardrail.filter_analyze_results_by_score(
        [{"entity_type": PiiEntityType.CREDIT_CARD, "score": 0.7, "start": 0, "end": 4}]
    )

    result = await guardrail.anonymize_text(
        text=text,
        analyze_results=filtered,
        output_parse_pii=False,
        masked_entity_count=masked_entity_count,
    )

    assert result == text
    assert masked_entity_count == {}


def test_blocking_respects_threshold_filter():
    """
    Entities filtered out by score should not trigger blocking, but high-score detections should.
    """
    guardrail = _OPTIONAL_PresidioPIIMasking(
        mock_testing=True,
        pii_entities_config={PiiEntityType.CREDIT_CARD: PiiAction.BLOCK},
        presidio_score_thresholds={PiiEntityType.CREDIT_CARD: 0.9},
    )

    low_score_results = [
        {"entity_type": PiiEntityType.CREDIT_CARD, "score": 0.7, "start": 0, "end": 4}
    ]
    filtered = guardrail.filter_analyze_results_by_score(low_score_results)
    guardrail.raise_exception_if_blocked_entities_detected(filtered)

    high_score_results = [
        {"entity_type": PiiEntityType.CREDIT_CARD, "score": 0.95, "start": 0, "end": 4}
    ]
    filtered_high = guardrail.filter_analyze_results_by_score(high_score_results)
    with pytest.raises(Exception):
        guardrail.raise_exception_if_blocked_entities_detected(filtered_high)


def test_update_in_memory_applies_score_thresholds():
    """
    update_in_memory_litellm_params should refresh score thresholds.
    """
    guardrail = _OPTIONAL_PresidioPIIMasking(mock_testing=True)
    assert guardrail.presidio_score_thresholds == {}

    params = LitellmParams(
        guardrail="presidio",
        mode="pre_call",
        presidio_score_thresholds={PiiEntityType.CREDIT_CARD: 0.85},
    )
    guardrail.update_in_memory_litellm_params(params)

    assert guardrail.presidio_score_thresholds == {PiiEntityType.CREDIT_CARD: 0.85}
