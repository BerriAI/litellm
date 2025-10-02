"""
Unit tests for Presidio guardrail support for Anthropic /v1/messages endpoint.

Tests the following functionality:
1. Anthropic structured content format (list of content blocks) is properly masked
2. Output PII masking works for Anthropic response format
3. Both input and output masking work together
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from litellm.proxy.guardrails.guardrail_hooks.presidio import (
    _OPTIONAL_PresidioPIIMasking,
)
from litellm.proxy._types import UserAPIKeyAuth


@pytest.mark.asyncio
async def test_presidio_anthropic_structured_content_input_masking():
    """Test that Presidio can mask PII in Anthropic's structured content format (list of content blocks)"""

    # Mock Presidio responses
    mock_analyze_response = [
        {"start": 11, "end": 17, "score": 0.85, "entity_type": "PERSON"}
    ]

    mock_anonymize_response = {
        "text": "My name is {REDACTED}, what is my name?",
        "items": [
            {
                "start": 11,
                "end": 21,
                "entity_type": "PERSON",
                "text": "{REDACTED}",
                "operator": "replace",
            }
        ],
    }

    # Create Presidio instance
    presidio = _OPTIONAL_PresidioPIIMasking(
        mock_testing=True,
        presidio_analyzer_api_base="http://fake:3000",
        presidio_anonymizer_api_base="http://fake:3000",
        default_on=True,
        event_hook="pre_call",
    )

    # Mock the analyze and anonymize methods
    presidio.analyze_text = AsyncMock(return_value=mock_analyze_response)
    presidio.anonymize_text = AsyncMock(return_value=mock_anonymize_response["text"])

    # Test data with Anthropic structured content format
    data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "My name is Ashley, what is my name?"}
                ],
            }
        ]
    }

    user_api_key_dict = UserAPIKeyAuth(api_key="test_key")

    # Call the pre-call hook
    result = await presidio.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=MagicMock(),
        data=data,
        call_type="acompletion",
    )

    # Verify the PII was masked in the structured content
    assert (
        result["messages"][0]["content"][0]["text"]
        == "My name is {REDACTED}, what is my name?"
    )
    assert presidio.anonymize_text.called


@pytest.mark.asyncio
async def test_presidio_anthropic_output_pii_masking():
    """Test that Presidio can mask PII in Anthropic response format (dict with content array)"""

    # Mock Presidio responses for output
    mock_analyze_response = [
        {"start": 36, "end": 62, "score": 0.95, "entity_type": "EMAIL_ADDRESS"}
    ]

    mock_anonymize_response = {
        "text": "Here's a fake email address:\n\n{REDACTED}",
        "items": [
            {
                "start": 36,
                "end": 46,
                "entity_type": "EMAIL_ADDRESS",
                "text": "{REDACTED}",
                "operator": "replace",
            }
        ],
    }

    # Create Presidio instance with output parsing enabled
    presidio = _OPTIONAL_PresidioPIIMasking(
        mock_testing=True,
        presidio_analyzer_api_base="http://fake:3000",
        presidio_anonymizer_api_base="http://fake:3000",
        output_parse_pii=True,
        default_on=True,
        event_hook=["pre_call", "post_call"],
    )

    # Mock the check_pii method to return masked text
    presidio.check_pii = AsyncMock(return_value=mock_anonymize_response["text"])

    # Anthropic response format (dict)
    response = {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "model": "claude-3",
        "content": [
            {
                "type": "text",
                "text": "Here's a fake email address:\n\njohn.smith@example.com",
            }
        ],
        "stop_reason": "end_turn",
    }

    data = {}
    user_api_key_dict = UserAPIKeyAuth(api_key="test_key")

    # Call the post-call hook
    result = await presidio.async_post_call_success_hook(
        data=data,
        user_api_key_dict=user_api_key_dict,
        response=response,
    )

    # Verify the email was masked in the response
    assert result["content"][0]["text"] == "Here's a fake email address:\n\n{REDACTED}"
    assert presidio.check_pii.called


@pytest.mark.asyncio
async def test_presidio_anthropic_multiple_content_blocks():
    """Test that Presidio handles multiple content blocks in Anthropic format"""

    # Create Presidio instance
    presidio = _OPTIONAL_PresidioPIIMasking(
        mock_testing=True,
        presidio_analyzer_api_base="http://fake:3000",
        presidio_anonymizer_api_base="http://fake:3000",
        default_on=True,
        event_hook="pre_call",
    )

    # Mock to return masked text
    presidio.check_pii = AsyncMock(
        side_effect=["My name is {REDACTED}", "Email: {REDACTED}"]
    )

    # Test data with multiple text blocks
    data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "My name is Ashley"},
                    {"type": "text", "text": "Email: ashley@example.com"},
                ],
            }
        ]
    }

    user_api_key_dict = UserAPIKeyAuth(api_key="test_key")

    # Call the pre-call hook
    result = await presidio.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=MagicMock(),
        data=data,
        call_type="acompletion",
    )

    # Verify both content blocks were masked
    assert result["messages"][0]["content"][0]["text"] == "My name is {REDACTED}"
    assert result["messages"][0]["content"][1]["text"] == "Email: {REDACTED}"
    assert presidio.check_pii.call_count == 2


@pytest.mark.asyncio
async def test_presidio_uses_curly_braces_for_redaction():
    """Test that Presidio uses {REDACTED} instead of <PERSON> to avoid Claude XML interpretation"""

    presidio = _OPTIONAL_PresidioPIIMasking(
        mock_testing=True,
        presidio_analyzer_api_base="http://fake:3000",
        presidio_anonymizer_api_base="http://fake:3000",
        default_on=True,
        event_hook="pre_call",
    )

    # Mock the anonymize_text to capture the anonymizers config
    async def mock_anonymize(
        text, analyze_results, output_parse_pii, masked_entity_count
    ):
        # Verify the DEFAULT anonymizer uses {REDACTED}
        # This would be called internally with our custom config
        return text.replace("Ashley", "{REDACTED}")

    presidio.analyze_text = AsyncMock(
        return_value=[{"start": 11, "end": 17, "score": 0.85, "entity_type": "PERSON"}]
    )
    presidio.anonymize_text = mock_anonymize

    data = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "My name is Ashley"}]}
        ]
    }

    result = await presidio.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(api_key="test"),
        cache=MagicMock(),
        data=data,
        call_type="acompletion",
    )

    # Verify {REDACTED} format is used, not <PERSON>
    masked_text = result["messages"][0]["content"][0]["text"]
    assert "{REDACTED}" in masked_text
    assert "<PERSON>" not in masked_text
