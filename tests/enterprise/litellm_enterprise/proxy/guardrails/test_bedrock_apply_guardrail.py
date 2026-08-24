"""
Test the Bedrock guardrail apply_guardrail functionality
"""

import os
import sys
from unittest.mock import AsyncMock, Mock, patch

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
async def test_bedrock_apply_guardrail_endpoint_integration(mock_proxy_logging_ctx):
    """Test the full endpoint integration with Bedrock guardrail"""
    from litellm.proxy.guardrails.guardrail_endpoints import apply_guardrail

    # Create a real BedrockGuardrail instance
    guardrail = BedrockGuardrail(
        guardrail_name="test-bedrock-guard",
        guardrailIdentifier="test-guard-id",
        guardrailVersion="DRAFT",
    )

    # Mock the guardrail registry
    with (
        patch(
            "litellm.proxy.guardrails.guardrail_endpoints.GUARDRAIL_REGISTRY"
        ) as mock_registry,
        mock_proxy_logging_ctx(),
    ):
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
                fastapi_request=Mock(),
                request=request,
                user_api_key_dict=user_api_key_dict,
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


@pytest.mark.asyncio
async def test_bedrock_apply_guardrail_does_not_leak_tool_content_with_flag():
    """
    Regression test for issue #23476: with
    experimental_use_latest_role_message_only enabled, a conversation ending in
    a tool message (agentic tool-calling loop) must scan the latest USER
    message — not the tool result.
    """
    guardrail = BedrockGuardrail(
        guardrail_name="test-bedrock-guard",
        guardrailIdentifier="test-guard-id",
        guardrailVersion="DRAFT",
        experimental_use_latest_role_message_only=True,
    )

    structured_messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "what is the weather?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "TOOL OUTPUT SECRET"},
    ]
    # Flat texts as built by the unified guardrail translation handler
    texts = ["rules", "what is the weather?", "TOOL OUTPUT SECRET"]

    with patch.object(
        guardrail, "make_bedrock_api_request", new_callable=AsyncMock
    ) as mock_api:
        mock_api.return_value = {"action": "NONE", "output": []}

        await guardrail.apply_guardrail(
            inputs={"texts": texts, "structured_messages": structured_messages},
            request_data={"messages": structured_messages},
            input_type="request",
        )

        assert mock_api.called
        _, kwargs = mock_api.call_args
        # The latest USER message is scanned — not the trailing tool message
        assert kwargs["messages"] == [structured_messages[1]]


@pytest.mark.asyncio
async def test_bedrock_apply_guardrail_skips_scan_when_no_user_message_with_flag():
    """With the flag on and no user-role message in the request, there is nothing
    to scan as INPUT — the guardrail must not be called."""
    guardrail = BedrockGuardrail(
        guardrail_name="test-bedrock-guard",
        guardrailIdentifier="test-guard-id",
        guardrailVersion="DRAFT",
        experimental_use_latest_role_message_only=True,
    )

    structured_messages = [
        {"role": "assistant", "content": "assistant text"},
        {"role": "tool", "tool_call_id": "call_1", "content": "tool text"},
    ]
    texts = ["assistant text", "tool text"]

    with patch.object(
        guardrail, "make_bedrock_api_request", new_callable=AsyncMock
    ) as mock_api:
        result = await guardrail.apply_guardrail(
            inputs={"texts": texts, "structured_messages": structured_messages},
            request_data={"messages": structured_messages},
            input_type="request",
        )

        assert not mock_api.called
        assert result.get("texts") == texts


@pytest.mark.asyncio
async def test_bedrock_apply_guardrail_masking_written_back_to_correct_position():
    """When only the latest user message is scanned, masked content must be
    written back to that message's position in the flat texts list — other
    entries stay untouched."""
    guardrail = BedrockGuardrail(
        guardrail_name="test-bedrock-guard",
        guardrailIdentifier="test-guard-id",
        guardrailVersion="DRAFT",
        experimental_use_latest_role_message_only=True,
    )

    structured_messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "my SSN is 123-45-6789"},
        {"role": "assistant", "content": "assistant text"},
        {"role": "tool", "tool_call_id": "call_1", "content": "tool text"},
    ]
    texts = ["rules", "my SSN is 123-45-6789", "assistant text", "tool text"]

    with patch.object(
        guardrail, "make_bedrock_api_request", new_callable=AsyncMock
    ) as mock_api:
        mock_api.return_value = {
            "action": "GUARDRAIL_INTERVENED",
            "output": [{"text": "my SSN is {SSN}"}],
        }

        guardrailed_inputs = await guardrail.apply_guardrail(
            inputs={"texts": texts, "structured_messages": structured_messages},
            request_data={"messages": structured_messages},
            input_type="request",
        )

        assert guardrailed_inputs.get("texts") == [
            "rules",
            "my SSN is {SSN}",
            "assistant text",
            "tool text",
        ]


@pytest.mark.asyncio
async def test_bedrock_apply_guardrail_skips_writeback_when_slice_unresolved_and_lengths_match():
    """Regression: a role-subset scan that can't be mapped back to flat-text
    positions (scanned_slice is None) must NOT write masked content back, even
    when the masked output happens to match len(texts) (e.g. both length 1).

    Here structured_messages carries two user texts but the flat `texts` list
    holds only one entry, so _locate_message_texts_slice returns None. The
    guardrail still scans the latest user message and returns a single masked
    text; a length-only guard would let that mask clobber the unrelated single
    `texts` entry. The slice-based guard keeps the original text instead.
    """
    guardrail = BedrockGuardrail(
        guardrail_name="test-bedrock-guard",
        guardrailIdentifier="test-guard-id",
        guardrailVersion="DRAFT",
        experimental_use_latest_role_message_only=True,
    )

    structured_messages = [
        {"role": "user", "content": "first user text"},
        {"role": "user", "content": "second user text"},
    ]
    # Flat texts is out of sync with structured_messages (one entry vs two
    # user texts) -> slice cannot be resolved.
    texts = ["unrelated flat text"]

    with patch.object(
        guardrail, "make_bedrock_api_request", new_callable=AsyncMock
    ) as mock_api:
        mock_api.return_value = {
            "action": "GUARDRAIL_INTERVENED",
            "output": [{"text": "[MASKED]"}],
        }

        guardrailed_inputs = await guardrail.apply_guardrail(
            inputs={"texts": texts, "structured_messages": structured_messages},
            request_data={"messages": structured_messages},
            input_type="request",
        )

    # The scan ran on the latest user message...
    assert mock_api.called
    # ...but because the slice was unresolved, the original flat text is kept
    # rather than clobbered by the single masked output.
    assert guardrailed_inputs.get("texts") == ["unrelated flat text"]


@pytest.mark.asyncio
async def test_bedrock_masking_writes_back_to_user_message_through_handler():
    """End-to-end through the unified translation handler (issue #23476).

    `apply_guardrail` only sees a flat `texts` list; the handler is what
    flattens messages into that list and re-inflates the guardrailed texts back
    onto messages — positionally, by the order they were extracted. This test
    exercises that whole contract with a real
    OpenAIChatCompletionsHandler.process_input_messages (only the network call
    is mocked), so a divergence between the handler's flattening and
    apply_guardrail's slice reconstruction would surface as masked content
    landing on the wrong message.

    With experimental_use_latest_role_message_only enabled and a conversation
    ending in a tool message, only the latest USER message is scanned; the
    returned mask must overwrite that user message and leave system / assistant
    / tool content untouched.
    """
    from litellm.llms.openai.chat.guardrail_translation.handler import (
        OpenAIChatCompletionsHandler,
    )

    guardrail = BedrockGuardrail(
        guardrail_name="test-bedrock-guard",
        guardrailIdentifier="test-guard-id",
        guardrailVersion="DRAFT",
        experimental_use_latest_role_message_only=True,
    )

    data = {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "SYSTEM rules"},
            {"role": "user", "content": "my SSN is 123-45-6789"},
            {"role": "assistant", "content": "ASSISTANT let me check"},
            {"role": "tool", "tool_call_id": "call_1", "content": "TOOL secret output"},
        ],
    }

    async def _fake_api(*args, **kwargs):
        # Mask every text actually scanned (the handler/fix should only send the
        # latest user message), one masked output per scanned message.
        scanned = kwargs.get("messages") or []
        return {
            "action": "GUARDRAIL_INTERVENED",
            "output": [{"text": "[MASKED]"} for _ in scanned],
        }

    with patch.object(
        guardrail, "make_bedrock_api_request", new_callable=AsyncMock
    ) as mock_api:
        mock_api.side_effect = _fake_api

        result = await OpenAIChatCompletionsHandler().process_input_messages(
            data=data, guardrail_to_apply=guardrail
        )

    # Only the latest user message was scanned...
    assert mock_api.called
    _, kwargs = mock_api.call_args
    assert kwargs["messages"] == [data["messages"][1]]

    # ...and the mask landed on the user message only — nothing else clobbered.
    msgs = result["messages"]
    assert msgs[0]["content"] == "SYSTEM rules"
    assert msgs[1]["content"] == "[MASKED]"
    assert msgs[2]["content"] == "ASSISTANT let me check"
    assert msgs[3]["content"] == "TOOL secret output"


@pytest.mark.asyncio
async def test_bedrock_handler_leaves_messages_untouched_when_no_user_message():
    """Through the real handler: no user-role message → INPUT scan skipped and
    every message is forwarded unchanged (no masking, no leak)."""
    from litellm.llms.openai.chat.guardrail_translation.handler import (
        OpenAIChatCompletionsHandler,
    )

    guardrail = BedrockGuardrail(
        guardrail_name="test-bedrock-guard",
        guardrailIdentifier="test-guard-id",
        guardrailVersion="DRAFT",
        experimental_use_latest_role_message_only=True,
    )

    data = {
        "model": "test-model",
        "messages": [
            {"role": "assistant", "content": "ASSISTANT text"},
            {"role": "tool", "tool_call_id": "call_1", "content": "TOOL output"},
        ],
    }

    with patch.object(
        guardrail, "make_bedrock_api_request", new_callable=AsyncMock
    ) as mock_api:
        result = await OpenAIChatCompletionsHandler().process_input_messages(
            data=data, guardrail_to_apply=guardrail
        )

    assert not mock_api.called
    assert result["messages"][0]["content"] == "ASSISTANT text"
    assert result["messages"][1]["content"] == "TOOL output"
