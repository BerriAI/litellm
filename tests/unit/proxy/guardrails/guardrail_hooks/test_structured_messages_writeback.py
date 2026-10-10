"""
Tests that when apply_guardrail returns structured_messages, both OpenAI and Anthropic
translation handlers write them back to data["messages"] correctly.

For OpenAI: structured_messages (OpenAI format) written directly to data["messages"].
For Anthropic: structured_messages (OpenAI format) converted back to Anthropic format
              via anthropic_messages_pt before writing to data["messages"].
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from litellm.llms.openai.chat.guardrail_translation.handler import (
    OpenAIChatCompletionsHandler,
)
from litellm.types.utils import GenericGuardrailAPIInputs

ORIGINAL_MESSAGES = [
    {"role": "system", "content": "Be concise."},
    {"role": "user", "content": "A" * 5000},
]
COMPRESSED_MESSAGES = [
    {"role": "system", "content": "Be concise."},
    {"role": "user", "content": "A" * 200},
]
COMPRESSED_MESSAGES_NON_SYSTEM = [
    {"role": "user", "content": "A" * 200},
]


def _make_guardrail_returning_structured_messages(compressed: list) -> MagicMock:
    guardrail = MagicMock()
    guardrail.should_run_guardrail.return_value = True
    guardrail.skip_system_message_in_guardrail = None
    guardrail.skip_tool_message_in_guardrail = None
    guardrail.experimental_use_latest_role_message_only = False

    async def apply_guardrail(inputs, request_data, input_type, logging_obj=None):
        result = dict(inputs)
        result["structured_messages"] = compressed
        return result

    guardrail.apply_guardrail = apply_guardrail
    return guardrail


@pytest.mark.asyncio
async def test_openai_handler_writes_structured_messages_back():
    handler = OpenAIChatCompletionsHandler()
    guardrail = _make_guardrail_returning_structured_messages(COMPRESSED_MESSAGES)

    data = {
        "model": "gpt-4o",
        "messages": ORIGINAL_MESSAGES,
    }
    result = await handler.process_input_messages(
        data=data,
        guardrail_to_apply=guardrail,
    )

    assert result["messages"] == COMPRESSED_MESSAGES


@pytest.mark.asyncio
async def test_openai_handler_uses_text_patchback_when_no_structured_messages():
    handler = OpenAIChatCompletionsHandler()

    guardrail = MagicMock()
    guardrail.should_run_guardrail.return_value = True
    guardrail.skip_system_message_in_guardrail = None
    guardrail.skip_tool_message_in_guardrail = None
    guardrail.experimental_use_latest_role_message_only = False

    original_text = "hello world"
    modified_text = "HELLO WORLD"
    messages = [{"role": "user", "content": original_text}]

    async def apply_guardrail(inputs, request_data, input_type, logging_obj=None):
        return GenericGuardrailAPIInputs(texts=[modified_text])

    guardrail.apply_guardrail = apply_guardrail

    data = {"model": "gpt-4o", "messages": messages}
    result = await handler.process_input_messages(
        data=data,
        guardrail_to_apply=guardrail,
    )

    assert result["messages"][0]["content"] == modified_text


@pytest.mark.asyncio
async def test_anthropic_handler_converts_structured_messages_to_anthropic_format():
    from litellm.llms.anthropic.chat.guardrail_translation.handler import (
        AnthropicMessagesHandler,
    )

    handler = AnthropicMessagesHandler()
    guardrail = _make_guardrail_returning_structured_messages(COMPRESSED_MESSAGES)

    anthropic_messages = [
        {"role": "user", "content": [{"type": "text", "text": "A" * 5000}]}
    ]
    data = {
        "model": "claude-3-5-sonnet-20241022",
        "messages": anthropic_messages,
        "max_tokens": 1024,
    }

    converted_back = [
        {"role": "user", "content": [{"type": "text", "text": "A" * 200}]}
    ]

    with patch(
        "litellm.litellm_core_utils.prompt_templates.factory.anthropic_messages_pt",
        return_value=converted_back,
    ) as mock_pt:
        result = await handler.process_input_messages(
            data=data,
            guardrail_to_apply=guardrail,
        )

    mock_pt.assert_called_once_with(
        messages=COMPRESSED_MESSAGES_NON_SYSTEM,
        model="claude-3-5-sonnet-20241022",
        llm_provider="anthropic",
    )
    assert result["messages"] == converted_back


# ---------------------------------------------------------------------------
# LIT-5018: the write-back must not restructure the conversation.
#
# anthropic_messages_pt merges every run of consecutive user/tool rows into one
# message, so a tool_result-only turn and the live user turn that follows it
# came back fused: the current instruction stopped being its own turn purely
# because a compression guardrail was enabled.
# ---------------------------------------------------------------------------

AGENTIC_ANTHROPIC_MESSAGES = [
    {"role": "user", "content": [{"type": "text", "text": "first turn"}]},
    {"role": "assistant", "content": [{"type": "tool_use", "id": "tu_1", "name": "Read", "input": {}}]},
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "FILE BODY"}]},
    {"role": "user", "content": [{"type": "text", "text": "can we run /team to fix this"}]},
]


async def _write_back_identity(messages: list) -> list:
    """Run the request through a guardrail that changes nothing but returns a
    new list, which is what puts a compression guardrail on the write-back
    path, and return the resulting Anthropic messages."""
    from litellm.llms.anthropic.chat.guardrail_translation.handler import (
        AnthropicMessagesHandler,
    )

    guardrail = MagicMock()
    guardrail.should_run_guardrail.return_value = True
    guardrail.skip_system_message_in_guardrail = None
    guardrail.skip_tool_message_in_guardrail = None
    guardrail.experimental_use_latest_role_message_only = False

    async def apply_guardrail(inputs, request_data, input_type, logging_obj=None):
        return {**inputs, "structured_messages": list(inputs["structured_messages"])}

    guardrail.apply_guardrail = apply_guardrail

    data = {"model": "claude-sonnet-4-5-20250929", "messages": messages, "max_tokens": 1024}
    result = await AnthropicMessagesHandler().process_input_messages(data=data, guardrail_to_apply=guardrail)
    return result["messages"]


@pytest.mark.asyncio
async def test_write_back_keeps_the_live_user_turn_separate_from_the_tool_result_turn():
    written = await _write_back_identity([dict(m) for m in AGENTIC_ANTHROPIC_MESSAGES])

    assert [m["role"] for m in written] == ["user", "assistant", "user", "user"]
    assert written[2]["content"] == [{"type": "tool_result", "tool_use_id": "tu_1", "content": "FILE BODY"}]
    assert written[3]["content"] == [{"type": "text", "text": "can we run /team to fix this"}]


@pytest.mark.asyncio
async def test_write_back_keeps_real_tool_results_under_modify_params():
    """Converting one row at a time would keep the turns apart too, but an
    assistant row whose results are converted separately reads as an orphaned
    tool call: with modify_params on, the sanitizer answers it with a synthetic
    "tool execution skipped" result and drops the real one."""
    import litellm

    original = litellm.modify_params
    litellm.modify_params = True
    try:
        written = await _write_back_identity([dict(m) for m in AGENTIC_ANTHROPIC_MESSAGES])
    finally:
        litellm.modify_params = original

    serialized = json.dumps(written)
    assert "FILE BODY" in serialized
    assert "skipped" not in serialized
    assert "Please continue" not in serialized
    assert [m["role"] for m in written] == ["user", "assistant", "user", "user"]
