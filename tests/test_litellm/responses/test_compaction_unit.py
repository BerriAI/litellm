"""
Unit tests for the compaction implementation.
These tests mock LLM calls and run without API keys.
"""
import base64
import json
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)


# ---------------------------------------------------------------------------
# Token counter tests
# ---------------------------------------------------------------------------


def test_cheap_token_counter_string():
    result = LiteLLMCompletionResponsesConfig._cheap_token_counter("hello world")
    assert result == len("hello world") // 4


def test_cheap_token_counter_list():
    input_items = [
        {"type": "message", "role": "user", "content": "word " * 100},
        {"type": "message", "role": "assistant", "content": "reply " * 50},
    ]
    count = LiteLLMCompletionResponsesConfig._cheap_token_counter(input_items)
    expected = (len("word " * 100) + len("reply " * 50)) // 4
    assert count == expected


def test_cheap_token_counter_compaction_item():
    summary = "This is a summary."
    encrypted = base64.b64encode(summary.encode()).decode()
    input_items = [
        {"type": "compaction", "encrypted_content": encrypted},
        {"type": "message", "role": "user", "content": "hello"},
    ]
    count = LiteLLMCompletionResponsesConfig._cheap_token_counter(input_items)
    # Should count decoded length of encrypted content, not raw length
    assert count > 0


def test_cheap_token_counter_for_messages():
    messages = [
        {"role": "user", "content": "word " * 100},
        {"role": "assistant", "content": "answer " * 50},
    ]
    count = LiteLLMCompletionResponsesConfig._cheap_token_counter_for_messages(messages)
    expected = (len("word " * 100) + len("answer " * 50)) // 4
    assert count == expected


# ---------------------------------------------------------------------------
# should_execute_compaction tests
# ---------------------------------------------------------------------------


def test_should_execute_compaction_over_threshold():
    result = LiteLLMCompletionResponsesConfig.should_execute_compaction(
        input_token_size=5000,
        context_management=[{"type": "compaction", "compact_threshold": 1000}],
    )
    assert result is True


def test_should_execute_compaction_under_threshold():
    result = LiteLLMCompletionResponsesConfig.should_execute_compaction(
        input_token_size=500,
        context_management=[{"type": "compaction", "compact_threshold": 1000}],
    )
    assert result is False


def test_should_execute_compaction_no_context_management():
    result = LiteLLMCompletionResponsesConfig.should_execute_compaction(
        input_token_size=999999,
        context_management=None,
    )
    assert result is False


def test_should_execute_compaction_empty_list():
    result = LiteLLMCompletionResponsesConfig.should_execute_compaction(
        input_token_size=999999,
        context_management=[],
    )
    assert result is False


def test_should_execute_compaction_exact_threshold():
    # At exactly the threshold: should compact
    result = LiteLLMCompletionResponsesConfig.should_execute_compaction(
        input_token_size=1000,
        context_management=[{"type": "compaction", "compact_threshold": 1000}],
    )
    assert result is True


# ---------------------------------------------------------------------------
# Compaction item decrypt in input transformation
# ---------------------------------------------------------------------------


def test_compaction_item_in_input_decrypts_to_message():
    """When input contains type:compaction, it must be converted to a context message."""
    summary = "We were discussing the payment service incident."
    encrypted = base64.b64encode(summary.encode()).decode()
    compaction_item = {
        "type": "compaction",
        "id": "cmp_123",
        "encrypted_content": encrypted,
    }
    messages = LiteLLMCompletionResponsesConfig._transform_responses_api_input_item_to_chat_completion_message(
        input_item=compaction_item
    )
    assert len(messages) == 1
    msg = messages[0]
    content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
    assert summary in str(content), (
        f"Decoded summary not found in message content. Got: {content}"
    )


def test_compaction_item_empty_encrypted_content_returns_empty():
    """Compaction item with no encrypted_content → empty list, not an error."""
    compaction_item = {"type": "compaction", "id": "cmp_123", "encrypted_content": ""}
    messages = LiteLLMCompletionResponsesConfig._transform_responses_api_input_item_to_chat_completion_message(
        input_item=compaction_item
    )
    assert messages == []


# ---------------------------------------------------------------------------
# _apply_compaction_to_messages (with mocked LLM)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_compaction_returns_compaction_item_and_new_messages():
    """_apply_compaction_to_messages returns (new_msgs, compaction_item_dict)."""
    messages = [
        {"role": "user", "content": "Start of conversation."},
        {"role": "assistant", "content": "Sure, let's talk."},
        {"role": "user", "content": "What is the answer?"},  # last user msg kept
    ]

    fake_model_response = MagicMock()
    fake_model_response.choices = [MagicMock()]
    fake_model_response.choices[0].message.content = "The conversation was about debugging."

    from litellm.types.utils import ModelResponse

    fake_model_response.__class__ = ModelResponse

    with patch("litellm.acompletion", new=AsyncMock(return_value=fake_model_response)):
        new_messages, compaction_item = await LiteLLMCompletionResponsesConfig._apply_compaction_to_messages(
            model="claude-3-haiku-20240307",
            messages=messages,
        )

    # The compaction item must have required fields
    assert compaction_item.get("type") == "compaction"
    assert compaction_item.get("id", "").startswith("cmp_")
    assert compaction_item.get("encrypted_content"), "encrypted_content must be non-empty"

    # The encrypted content must be base64-decodable
    decoded = base64.b64decode(compaction_item["encrypted_content"] + "==").decode("utf-8")
    assert len(decoded) > 0

    # The new messages must contain the last user message
    assert new_messages[-1]["role"] == "user"
    assert new_messages[-1]["content"] == "What is the answer?"

    # The first new message should be the summary
    assert "[Previous conversation summary:" in new_messages[0]["content"]


@pytest.mark.asyncio
async def test_apply_compaction_keeps_last_user_message():
    """The last user message must be in the tail, not summarized away."""
    CURRENT_QUESTION = "What is the answer to the secret question?"
    messages = [
        {"role": "user", "content": "word " * 500},
        {"role": "assistant", "content": "OK."},
        {"role": "user", "content": CURRENT_QUESTION},
    ]

    fake_model_response = MagicMock()
    fake_model_response.choices = [MagicMock()]
    fake_model_response.choices[0].message.content = "Summary of prior conversation."

    from litellm.types.utils import ModelResponse

    fake_model_response.__class__ = ModelResponse

    with patch("litellm.acompletion", new=AsyncMock(return_value=fake_model_response)):
        new_messages, _ = await LiteLLMCompletionResponsesConfig._apply_compaction_to_messages(
            model="claude-3-haiku-20240307",
            messages=messages,
        )

    last_user = [m for m in new_messages if m.get("role") == "user"][-1]
    assert last_user["content"] == CURRENT_QUESTION, (
        "Current question must be preserved verbatim in new messages"
    )


# ---------------------------------------------------------------------------
# Case 3: no recompaction logic (threshold check)
# ---------------------------------------------------------------------------


def test_no_recompaction_when_summary_msg_plus_new_under_threshold():
    """After decryption, the summary msg + new user msg should be small → no recompact."""
    summary = "Short summary."  # ~3 tokens
    summary_msg = {"role": "user", "content": f"[Previous conversation summary: {summary}]"}
    new_user_msg = {"role": "user", "content": "What is 3+3?"}
    messages = [summary_msg, new_user_msg]

    token_count = LiteLLMCompletionResponsesConfig._cheap_token_counter_for_messages(messages)
    should_compact = LiteLLMCompletionResponsesConfig.should_execute_compaction(
        input_token_size=token_count,
        context_management=[{"type": "compaction", "compact_threshold": 200000}],
    )
    assert not should_compact, (
        f"Should NOT compact small messages ({token_count} tokens) under high threshold"
    )


# ---------------------------------------------------------------------------
# Case 2: full end-to-end with mocked LLM (async_response_api_handler)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handler_injects_compaction_item_in_response():
    """Full handler path: compaction triggers, item injected at output[0]."""
    from litellm.responses.litellm_completion_transformation.handler import (
        LiteLLMCompletionTransformationHandler,
    )
    from litellm.types.llms.openai import ResponsesAPIResponse

    fat_content = "word " * 5000
    request_input = [
        {"type": "message", "role": "user", "content": "Start."},
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": fat_content}],
        },
        {"type": "message", "role": "user", "content": "Continue."},
    ]
    responses_api_request = {
        "context_management": [{"type": "compaction", "compact_threshold": 100}],
        "store": False,
    }

    # Fake the summarization call
    fake_summary_response = MagicMock()
    fake_summary_response.choices = [MagicMock()]
    fake_summary_response.choices[0].message.content = "The user discussed a start and wanted to continue."

    from litellm.types.utils import ModelResponse as MR

    fake_summary_response.__class__ = MR

    # Fake the actual LLM call — set reasoning_content=None so Pydantic doesn't choke
    choice_mock = MagicMock()
    choice_mock.message.content = "OK, continuing."
    choice_mock.message.role = "assistant"
    choice_mock.message.reasoning_content = None  # must be falsy to skip reasoning path
    choice_mock.message.tool_calls = None
    choice_mock.finish_reason = "stop"
    choice_mock.index = 0

    fake_llm_response = MagicMock(spec=MR)
    fake_llm_response.choices = [choice_mock]
    fake_llm_response.model = "claude-3-haiku-20240307"
    fake_llm_response.id = "resp_test_123"
    fake_llm_response.created = 1234567890
    fake_llm_response.object = "chat.completion"
    fake_llm_response.usage = MagicMock()
    fake_llm_response.usage.prompt_tokens = 10
    fake_llm_response.usage.completion_tokens = 5
    fake_llm_response.usage.total_tokens = 15

    call_count = {"n": 0}

    async def fake_acompletion(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # First call is summarization
            return fake_summary_response
        # Second call is actual LLM
        return fake_llm_response

    from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams

    litellm_completion_request = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
        model="claude-3-haiku-20240307",
        input=request_input,
        responses_api_request=ResponsesAPIOptionalRequestParams(**responses_api_request),
    )

    handler = LiteLLMCompletionTransformationHandler()
    with patch("litellm.acompletion", new=AsyncMock(side_effect=fake_acompletion)):
        response = await handler.async_response_api_handler(
            litellm_completion_request=litellm_completion_request,
            request_input=request_input,
            responses_api_request=ResponsesAPIOptionalRequestParams(**responses_api_request),
        )

    assert isinstance(response, ResponsesAPIResponse)
    assert response.output is not None
    assert len(response.output) > 0

    first_item = response.output[0]
    first_type = first_item.get("type") if isinstance(first_item, dict) else getattr(first_item, "type", None)
    assert first_type == "compaction", (
        f"Expected compaction at output[0], got: {first_type}"
    )

    # The summarization call should have been made (2 total calls: summarize + actual)
    assert call_count["n"] == 2, f"Expected 2 LLM calls, got {call_count['n']}"
