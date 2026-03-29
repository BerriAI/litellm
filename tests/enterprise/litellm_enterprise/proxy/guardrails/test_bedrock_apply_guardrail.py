"""
Test the Bedrock guardrail apply_guardrail functionality
"""

import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))


from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails import BedrockGuardrail
from litellm.types.guardrails import ApplyGuardrailRequest, ApplyGuardrailResponse


@pytest.mark.asyncio
async def test_bedrock_apply_guardrail_success():
    """Test that Bedrock guardrail apply_guardrail method works correctly"""
    # Create a BedrockGuardrail instance
    guardrail = BedrockGuardrail(
        guardrail_name="test-bedrock-guard",
        guardrailIdentifier="test-guard-id",
        guardrailVersion="DRAFT",
    )

    # Mock the make_bedrock_api_request method
    with patch.object(
        guardrail, "make_bedrock_api_request", new_callable=AsyncMock
    ) as mock_api_request:
        # Mock a successful response from Bedrock
        mock_response = {
            "action": "ALLOWED",
            "output": [{"text": "This is a test message with some content"}],
        }
        mock_api_request.return_value = mock_response

        # Test the apply_guardrail method with new signature
        guardrailed_inputs = await guardrail.apply_guardrail(
            inputs={"texts": ["This is a test message with some content"]},
            request_data={},
            input_type="request",
        )
        result = guardrailed_inputs.get("texts", [])

        # Verify the result
        assert result == ["This is a test message with some content"]
        mock_api_request.assert_called_once()


@pytest.mark.asyncio
async def test_bedrock_apply_guardrail_blocked():
    """Test that apply_guardrail lets HTTPException propagate as-is for blocked content.

    Regression test for issue #20045: when disable_exception_on_block=False (default),
    make_bedrock_api_request raises HTTPException for BLOCKED content. apply_guardrail
    must NOT wrap it in a generic Exception, otherwise the proxy loses the HTTP 400
    status and fails to block the LLM call.
    """
    from fastapi import HTTPException

    guardrail = BedrockGuardrail(
        guardrail_name="test-bedrock-guard",
        guardrailIdentifier="test-guard-id",
        guardrailVersion="DRAFT",
    )

    with patch.object(
        guardrail, "make_bedrock_api_request", new_callable=AsyncMock
    ) as mock_api_request:
        mock_api_request.side_effect = HTTPException(
            status_code=400,
            detail={
                "error": "Violated guardrail policy",
                "bedrock_guardrail_response": "Content blocked",
            },
        )

        # Test the apply_guardrail method propagates HTTPException (AWS error) to the client
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs={"texts": ["This is blocked content"]},
                request_data={},
                input_type="request",
            )

        assert exc_info.value.status_code == 400
        assert "Violated guardrail policy" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_bedrock_apply_guardrail_with_masking():
    """Test that Bedrock guardrail apply_guardrail method handles content masking"""
    # Create a BedrockGuardrail instance
    guardrail = BedrockGuardrail(
        guardrail_name="test-bedrock-guard",
        guardrailIdentifier="test-guard-id",
        guardrailVersion="DRAFT",
    )

    # Mock the make_bedrock_api_request method
    with patch.object(
        guardrail, "make_bedrock_api_request", new_callable=AsyncMock
    ) as mock_api_request:
        # Mock a response with masked content
        mock_response = {
            "action": "ALLOWED",
            "outputs": [{"text": "This is a test message with [REDACTED] content"}],
        }
        mock_api_request.return_value = mock_response

        # Test the apply_guardrail method with new signature
        guardrailed_inputs = await guardrail.apply_guardrail(
            inputs={"texts": ["This is a test message with sensitive content"]},
            request_data={},
            input_type="request",
        )
        result = guardrailed_inputs.get("texts", [])

        # Verify the result contains the masked content
        assert result == ["This is a test message with [REDACTED] content"]
        mock_api_request.assert_called_once()


@pytest.mark.asyncio
async def test_bedrock_apply_guardrail_api_failure():
    """Test that Bedrock guardrail apply_guardrail method handles API failures"""
    # Create a BedrockGuardrail instance
    guardrail = BedrockGuardrail(
        guardrail_name="test-bedrock-guard",
        guardrailIdentifier="test-guard-id",
        guardrailVersion="DRAFT",
    )

    # Mock the make_bedrock_api_request method to raise an exception
    with patch.object(
        guardrail, "make_bedrock_api_request", new_callable=AsyncMock
    ) as mock_api_request:
        mock_api_request.side_effect = Exception("API connection failed")

        # Test the apply_guardrail method should raise an exception
        with pytest.raises(Exception) as exc_info:
            await guardrail.apply_guardrail(
                inputs={"texts": ["This is a test message"]},
                request_data={},
                input_type="request",
            )

        # The error message should contain the original exception
        assert "API connection failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_bedrock_apply_guardrail_endpoint_integration():
    """Test the full endpoint integration with Bedrock guardrail"""
    from litellm.proxy.guardrails.guardrail_endpoints import apply_guardrail

    # Create a real BedrockGuardrail instance
    guardrail = BedrockGuardrail(
        guardrail_name="test-bedrock-guard",
        guardrailIdentifier="test-guard-id",
        guardrailVersion="DRAFT",
    )

    # Mock the guardrail registry
    with patch(
        "litellm.proxy.guardrails.guardrail_endpoints.GUARDRAIL_REGISTRY"
    ) as mock_registry:
        # Mock the make_bedrock_api_request method
        with patch.object(
            guardrail, "make_bedrock_api_request", new_callable=AsyncMock
        ) as mock_api_request:
            # Mock a successful response from Bedrock
            mock_response = {
                "action": "ALLOWED",
                "outputs": [{"text": "This is a test message with processed content"}],
            }
            mock_api_request.return_value = mock_response

            # Configure the registry to return our guardrail
            mock_registry.get_initialized_guardrail_callback.return_value = guardrail

            # Create the request
            request = ApplyGuardrailRequest(
                guardrail_name="test-bedrock-guard",
                text="This is a test message with some content",
                language="en",
            )

            # Create a mock user API key
            user_api_key_dict = UserAPIKeyAuth(api_key="test-key")

            # Call the endpoint
            response = await apply_guardrail(
                request=request, user_api_key_dict=user_api_key_dict
            )

            # Verify the response
            assert isinstance(response, ApplyGuardrailResponse)
            assert (
                response.response_text
                == "This is a test message with processed content"
            )
            # Note: The endpoint now calls apply_guardrail which internally calls make_bedrock_api_request
            # The call count check has been removed as it may be called multiple times through the chain


@pytest.mark.asyncio
async def test_bedrock_apply_guardrail_filters_request_messages_when_flag_enabled():
    guardrail = BedrockGuardrail(
        guardrail_name="test-bedrock-guard",
        guardrailIdentifier="test-guard-id",
        guardrailVersion="DRAFT",
        experimental_use_latest_role_message_only=True,
    )

    request_messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "response"},
        {"role": "user", "content": "latest question"},
    ]

    request_data = {"messages": request_messages}

    with patch.object(
        guardrail, "make_bedrock_api_request", new_callable=AsyncMock
    ) as mock_api:
        mock_api.return_value = {
            "action": "ALLOWED",
            "output": [{"text": "latest question"}],
        }

        guardrailed_inputs = await guardrail.apply_guardrail(
            inputs={"texts": ["latest question"]},
            request_data=request_data,
            input_type="request",
        )
        result = guardrailed_inputs.get("texts", [])

        assert mock_api.called
        _, kwargs = mock_api.call_args
        assert kwargs["messages"] == [request_messages[-1]]
        assert result == ["latest question"]


@pytest.mark.asyncio
async def test_bedrock_apply_guardrail_filters_request_messages_when_flag_enabled_blocked():
    guardrail = BedrockGuardrail(
        guardrail_name="test-bedrock-guard",
        guardrailIdentifier="test-guard-id",
        guardrailVersion="DRAFT",
        experimental_use_latest_role_message_only=True,
    )

    request_messages = [
        {"role": "user", "content": "first"},
        {"role": "user", "content": "blocked"},
    ]

    request_data = {"messages": request_messages}

    with patch.object(
        guardrail, "make_bedrock_api_request", new_callable=AsyncMock
    ) as mock_api:
        # Mock the method to raise an HTTPException as it would for blocked content
        from fastapi import HTTPException

        mock_api.side_effect = HTTPException(
            status_code=400,
            detail={
                "error": "Violated guardrail policy",
                "bedrock_guardrail_response": "policy",
            },
        )

        with pytest.raises(HTTPException, match="policy") as exc_info:
            await guardrail.apply_guardrail(
                inputs={"texts": ["blocked"]},
                request_data=request_data,
                input_type="request",
            )

        assert mock_api.called
        _, kwargs = mock_api.call_args
        assert kwargs["messages"] == [request_messages[-1]]
        # HTTPException from guardrail is propagated so the client gets the AWS message
        assert exc_info.value.status_code == 400
        assert "policy" in str(exc_info.value.detail)


def test_bedrock_guardrail_filters_latest_user_message_when_enabled():
    guardrail = BedrockGuardrail(
        guardrail_name="test-bedrock-guard",
        guardrailIdentifier="test-guard-id",
        guardrailVersion="DRAFT",
        experimental_use_latest_role_message_only=True,
    )

    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "response"},
        {"role": "user", "content": "latest question"},
    ]

    filter_result = guardrail._prepare_guardrail_messages_for_role(messages=messages)

    assert filter_result.payload_messages is not None
    assert len(filter_result.payload_messages) == 1
    assert filter_result.payload_messages[0]["content"] == "latest question"
    assert filter_result.target_indices == [3]

    masked_messages = guardrail._merge_filtered_messages(
        original_messages=filter_result.original_messages,
        updated_target_messages=[{"role": "user", "content": "[MASKED]"}],
        target_indices=filter_result.target_indices,
    )
    assert masked_messages[3]["content"] == "[MASKED]"


def test_bedrock_guardrail_filters_excludes_tool_and_assistant_messages():
    """
    Regression test for issue #23476: when experimental_use_latest_role_message_only
    is True, tool and assistant messages should not be sent to the guardrail.
    Only the latest user-role message should be sent.
    """
    guardrail = BedrockGuardrail(
        guardrail_name="test-bedrock-guard",
        guardrailIdentifier="test-guard-id",
        guardrailVersion="DRAFT",
        experimental_use_latest_role_message_only=True,
    )

    # Scenario: conversation with tool calls where last message is a tool result
    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Calculate 2+2"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "function": {"name": "calc", "arguments": "{}"}}]},
        {"role": "tool", "content": "4", "tool_call_id": "call_1"},
    ]

    filter_result = guardrail._prepare_guardrail_messages_for_role(messages=messages)

    assert filter_result.payload_messages is not None
    assert len(filter_result.payload_messages) == 1
    assert filter_result.payload_messages[0]["role"] == "user"
    assert filter_result.payload_messages[0]["content"] == "Calculate 2+2"
    assert filter_result.target_indices == [1]


def test_bedrock_guardrail_filters_excludes_assistant_as_last_message():
    """
    Regression test for issue #23476: when the last message is an assistant
    message, only the latest user message should be sent to the guardrail.
    """
    guardrail = BedrockGuardrail(
        guardrail_name="test-bedrock-guard",
        guardrailIdentifier="test-guard-id",
        guardrailVersion="DRAFT",
        experimental_use_latest_role_message_only=True,
    )

    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there! How can I help?"},
    ]

    filter_result = guardrail._prepare_guardrail_messages_for_role(messages=messages)

    assert filter_result.payload_messages is not None
    assert len(filter_result.payload_messages) == 1
    assert filter_result.payload_messages[0]["role"] == "user"
    assert filter_result.payload_messages[0]["content"] == "Hello"
    assert filter_result.target_indices == [1]


def test_bedrock_input_content_request_skips_non_user_messages_with_flag():
    """
    Regression test for issue #23476: _create_bedrock_input_content_request
    should only include user-role message content when
    experimental_use_latest_role_message_only is True.
    """
    guardrail = BedrockGuardrail(
        guardrail_name="test-bedrock-guard",
        guardrailIdentifier="test-guard-id",
        guardrailVersion="DRAFT",
        experimental_use_latest_role_message_only=True,
    )

    # Even if non-user messages are somehow passed, they should be filtered out
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "The answer is 4"},
        {"role": "tool", "content": "calculator result: 4"},
    ]

    request = guardrail._create_bedrock_input_content_request(messages=messages)
    content_items = request.get("content", [])

    # Only the user message content should be included
    assert len(content_items) == 1
    assert content_items[0]["text"]["text"] == "What is 2+2?"


@pytest.mark.asyncio
async def test_bedrock_apply_guardrail_blocked_with_disable_exception_on_block():
    """
    Regression test for issue #20045: when disable_exception_on_block=True,
    make_bedrock_api_request raises GuardrailInterventionNormalStringError.
    apply_guardrail must let it propagate as-is so the proxy can handle it
    properly instead of wrapping it in a generic Exception.
    """
    from litellm.exceptions import GuardrailInterventionNormalStringError

    guardrail = BedrockGuardrail(
        guardrail_name="test-bedrock-guard",
        guardrailIdentifier="test-guard-id",
        guardrailVersion="DRAFT",
        disable_exception_on_block=True,
    )

    with patch.object(
        guardrail, "make_bedrock_api_request", new_callable=AsyncMock
    ) as mock_api:
        mock_api.side_effect = GuardrailInterventionNormalStringError(
            message="Sorry, your question in its current format is unable to be answered."
        )

        with pytest.raises(GuardrailInterventionNormalStringError) as exc_info:
            await guardrail.apply_guardrail(
                inputs={"texts": ["harmful prompt content"]},
                request_data={},
                input_type="request",
            )

        assert "unable to be answered" in str(exc_info.value.message)
