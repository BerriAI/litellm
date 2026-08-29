import base64
from typing import Any, cast

import pytest

import litellm



from litellm.litellm_core_utils.prompt_templates.common_utils import (
    TOOL_RESULT_IMAGE_PLACEHOLDER,
)
from litellm.litellm_core_utils.prompt_templates.factory import (
    THOUGHT_SIGNATURE_SEPARATOR,
    _bedrock_converse_messages_pt,
)
from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
    OPENAI_MAX_TOOL_NAME_LENGTH,
    AnthropicAdapter,
    LiteLLMAnthropicMessagesAdapter,
    create_tool_name_mapping,
    truncate_tool_name,
)
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.types.llms.anthropic import (
    AnthopicMessagesAssistantMessageParam,
    AnthropicMessagesUserMessageParam,
)
from litellm.types.llms.openai import ChatCompletionAssistantToolCall
from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    Choices,
    Delta,
    Function,
    Message,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
    Usage,
)


def test_translate_streaming_openai_chunk_to_anthropic_content_block():
    choices = [
        StreamingChoices(
            finish_reason=None,
            index=0,
            delta=Delta(
                provider_specific_fields=None,
                content=None,
                role="assistant",
                function_call=None,
                tool_calls=[
                    ChatCompletionDeltaToolCall(
                        id="call_d581d130-e234-4315-94e8-27e7ff7c4e55",
                        function=Function(arguments='{"location": "Boston"}', name="get_weather"),
                        type="function",
                        index=0,
                    )
                ],
                audio=None,
            ),
            logprobs=None,
        )
    ]

    (
        block_type,
        content_block_start,
    ) = LiteLLMAnthropicMessagesAdapter()._translate_streaming_openai_chunk_to_anthropic_content_block(choices=choices)

    print(content_block_start)

    assert block_type == "tool_use"
    assert content_block_start == {
        "type": "tool_use",
        "id": "call_d581d130-e234-4315-94e8-27e7ff7c4e55",
        "name": "get_weather",
        "input": {},
    }


def test_translate_streaming_openai_chunk_strips_gemini_thought_from_tool_call_id():
    """Gemini embeds thought signatures in OpenAI tool ids; Anthropic SSE should expose a clean id."""
    base = "call_3e9417b7925e49aca9a71dc1885e"
    sig = "CiIBDDnWx"
    combined = f"{base}{THOUGHT_SIGNATURE_SEPARATOR}{sig}"
    choices = [
        StreamingChoices(
            finish_reason=None,
            index=0,
            delta=Delta(
                provider_specific_fields=None,
                content=None,
                role="assistant",
                function_call=None,
                tool_calls=[
                    ChatCompletionDeltaToolCall(
                        id=combined,
                        function=Function(arguments='{"a": 17, "b": 25}', name="add_numbers"),
                        type="function",
                        index=0,
                    )
                ],
                audio=None,
            ),
            logprobs=None,
        )
    ]

    (
        block_type,
        content_block_start,
    ) = LiteLLMAnthropicMessagesAdapter()._translate_streaming_openai_chunk_to_anthropic_content_block(choices=choices)

    assert block_type == "tool_use"
    assert content_block_start["id"] == base
    assert content_block_start["name"] == "add_numbers"
    assert content_block_start["input"] == {}
    assert content_block_start["provider_specific_fields"]["signature"] == sig


def test_translate_streaming_openai_chunk_to_anthropic_thinking_content_block():
    choices = [
        StreamingChoices(
            finish_reason=None,
            index=0,
            delta=Delta(
                reasoning_content="I need to summar",
                thinking_blocks=[
                    {
                        "type": "thinking",
                        "thinking": "I need to summar",
                        "signature": None,
                    }
                ],
                provider_specific_fields={
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": "I need to summar",
                            "signature": None,
                        }
                    ]
                },
                content="",
                role="assistant",
                function_call=None,
                tool_calls=None,
                audio=None,
            ),
            logprobs=None,
        )
    ]

    (
        block_type,
        content_block_start,
    ) = LiteLLMAnthropicMessagesAdapter()._translate_streaming_openai_chunk_to_anthropic_content_block(choices=choices)

    assert block_type == "thinking"
    assert content_block_start == {
        "type": "thinking",
        "thinking": "I need to summar",
        "signature": "",
    }


def test_translate_streaming_openai_chunk_to_anthropic_reasoning_content_only_content_block():
    """OpenAI-compatible reasoning backends (vLLM/SGLang) emit ``reasoning_content``
    without ``thinking_blocks``. The content-block classifier must still open a
    ``thinking`` block so the matching ``thinking_delta`` stream is not emitted
    inside a text block (which silently drops chain-of-thought for /v1/messages
    streaming clients)."""
    choices = [
        StreamingChoices(
            finish_reason=None,
            index=0,
            delta=Delta(
                reasoning_content="Let me think",
                thinking_blocks=None,
                content=None,
                role="assistant",
                function_call=None,
                tool_calls=None,
                audio=None,
            ),
            logprobs=None,
        )
    ]

    (
        block_type,
        content_block_start,
    ) = LiteLLMAnthropicMessagesAdapter()._translate_streaming_openai_chunk_to_anthropic_content_block(choices=choices)

    assert block_type == "thinking"
    assert content_block_start == {
        "type": "thinking",
        "thinking": "",
        "signature": "",
    }


def test_translate_streaming_openai_chunk_to_anthropic_thinking_signature_block():
    choices = [
        StreamingChoices(
            finish_reason=None,
            index=0,
            delta=Delta(
                reasoning_content="",
                thinking_blocks=[
                    {
                        "type": "thinking",
                        "thinking": None,
                        "signature": "sigsig",
                    }
                ],
                provider_specific_fields={
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": None,
                            "signature": "sigsig",
                        }
                    ]
                },
                content="",
                role="assistant",
                function_call=None,
                tool_calls=None,
                audio=None,
            ),
            logprobs=None,
        )
    ]

    (
        block_type,
        content_block_start,
    ) = LiteLLMAnthropicMessagesAdapter()._translate_streaming_openai_chunk_to_anthropic_content_block(choices=choices)

    assert block_type == "thinking"
    assert content_block_start == {
        "type": "thinking",
        "thinking": "",
        "signature": "sigsig",
    }


def test_translate_streaming_openai_chunk_to_anthropic_content_block_thinking_and_signature():
    """The content-block classifier must treat a chunk carrying both ``thinking``
    and ``signature`` as a ``thinking`` block instead of raising.

    Such a chunk is the terminal signature event of an already-open thinking block,
    so classifying it as ``thinking`` keeps the stream on the same block rather than
    500'ing. Before the fix this raised ``ValueError``.
    """
    choices = [
        StreamingChoices(
            finish_reason=None,
            index=0,
            delta=Delta(
                reasoning_content="",
                thinking_blocks=[
                    {
                        "type": "thinking",
                        "thinking": "I need to summar",
                        "signature": "sigsig",
                    }
                ],
                provider_specific_fields={
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": "I need to summar",
                            "signature": "sigsig",
                        }
                    ]
                },
                content="",
                role="assistant",
                function_call=None,
                tool_calls=None,
                audio=None,
            ),
            logprobs=None,
        )
    ]

    (
        block_type,
        content_block_start,
    ) = LiteLLMAnthropicMessagesAdapter()._translate_streaming_openai_chunk_to_anthropic_content_block(choices=choices)

    assert block_type == "thinking"


def test_translate_anthropic_messages_to_openai_thinking_blocks():
    """Test that tool result messages are placed before user messages in the conversation order."""

    anthropic_messages = [
        AnthropicMessagesUserMessageParam(
            role="user",
            content=[{"type": "text", "text": "What's the weather in Boston?"}],
        ),
        AnthopicMessagesAssistantMessageParam(
            role="assistant",
            content=[
                {
                    "type": "thinking",
                    "thinking": "I will call the get_weather tool.",
                    "signature": "sigsig",
                },
                {
                    "type": "redacted_thinking",
                    "data": "REDACTED",
                },
                {
                    "type": "tool_use",
                    "id": "toolu_01234",
                    "name": "get_weather",
                    "input": {"location": "Boston"},
                },
            ],
        ),
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter.translate_anthropic_messages_to_openai(messages=anthropic_messages)

    assert len(result) == 2
    assert result[1]["role"] == "assistant"
    assert "thinking_blocks" in result[1]
    assert len(result[1]["thinking_blocks"]) == 2
    assert result[1]["thinking_blocks"][0]["type"] == "thinking"
    assert result[1]["thinking_blocks"][0]["thinking"] == "I will call the get_weather tool."
    assert result[1]["thinking_blocks"][0]["signature"] == "sigsig"
    assert result[1]["thinking_blocks"][1]["type"] == "redacted_thinking"
    assert result[1]["thinking_blocks"][1]["data"] == "REDACTED"
    assert "tool_calls" in result[1]
    assert len(result[1]["tool_calls"]) == 1
    assert result[1]["tool_calls"][0]["id"] == "toolu_01234"


def test_translate_anthropic_messages_to_openai_sets_reasoning_content():
    """Reasoning-aware chat providers read reasoning_content, so thinking text must land there.

    Without it Moonshot and DeepSeek fill in a single-space placeholder and the model gets
    a blank where its own prior reasoning belongs.
    """

    anthropic_messages = [
        AnthropicMessagesUserMessageParam(
            role="user",
            content=[{"type": "text", "text": "Which city is best for a picnic?"}],
        ),
        AnthopicMessagesAssistantMessageParam(
            role="assistant",
            content=[
                {"type": "thinking", "thinking": "Denver is dry in August.", "signature": "sig1"},
                {"type": "thinking", "thinking": "San Francisco is foggy.", "signature": "sig2"},
                {"type": "redacted_thinking", "data": "REDACTED"},
                {"type": "text", "text": "Denver."},
            ],
        ),
    ]

    result = LiteLLMAnthropicMessagesAdapter().translate_anthropic_messages_to_openai(messages=anthropic_messages)

    assert result[1]["reasoning_content"] == "Denver is dry in August.\nSan Francisco is foggy."
    assert result[1]["content"] == "Denver."


def test_translate_anthropic_messages_to_openai_sets_no_reasoning_content_without_thinking():
    anthropic_messages = [
        AnthopicMessagesAssistantMessageParam(
            role="assistant",
            content=[{"type": "text", "text": "Denver."}],
        ),
    ]

    result = LiteLLMAnthropicMessagesAdapter().translate_anthropic_messages_to_openai(messages=anthropic_messages)

    assert "reasoning_content" not in result[0]


def test_translate_anthropic_messages_to_openai_tool_message_placement():
    """Test that tool result messages are placed before user messages in the conversation order."""

    anthropic_messages = [
        AnthropicMessagesUserMessageParam(
            role="user",
            content=[{"type": "text", "text": "What's the weather in Boston?"}],
        ),
        AnthopicMessagesAssistantMessageParam(
            role="assistant",
            content=[
                {
                    "type": "tool_use",
                    "id": "toolu_01234",
                    "name": "get_weather",
                    "input": {"location": "Boston"},
                }
            ],
        ),
        AnthropicMessagesUserMessageParam(
            role="user",
            content=[
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01234",
                    "content": "Sunny, 75°F",
                },
                {"type": "text", "text": "What about tomorrow?"},
            ],
        ),
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter.translate_anthropic_messages_to_openai(messages=anthropic_messages)

    # find the indices of tool and user messages in the result
    tool_message_idx = None
    user_message_idx = None

    for i, msg in enumerate(result):
        if isinstance(msg, dict) and msg.get("role") == "tool":
            tool_message_idx = i
        elif (
            isinstance(msg, dict)
            and msg.get("role") == "user"
            and "What about tomorrow?" in str(msg.get("content", ""))
        ):
            user_message_idx = i
            break

    assert tool_message_idx is not None, "Tool message not found"
    assert user_message_idx is not None, "User message not found"
    assert tool_message_idx < user_message_idx, "Tool message should be placed before user message"


@pytest.mark.parametrize(
    ("system_content", "expected_content"),
    [
        ("Use the corrected result.", "Use the corrected result."),
        (
            [{"type": "text", "text": "Use the corrected result."}],
            [{"type": "text", "text": "Use the corrected result."}],
        ),
        (
            [
                {
                    "type": "image",
                    "source": {"type": "url", "url": "https://example.com/a.png"},
                },
                {"type": "text", "text": "Use the corrected result."},
            ],
            [{"type": "text", "text": "Use the corrected result."}],
        ),
        (
            [
                {"type": "text", "text": "First correction."},
                {"type": "text", "text": "Second correction."},
            ],
            [
                {"type": "text", "text": "First correction."},
                {"type": "text", "text": "Second correction."},
            ],
        ),
    ],
)
def test_translate_anthropic_messages_to_openai_preserves_midturn_system_correction(
    system_content: object,
    expected_content: object,
):
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_01234",
                    "name": "get_weather",
                    "input": {"location": "Boston"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01234",
                    "content": "Rainy, 55°F",
                }
            ],
        },
        {"role": "system", "content": system_content},
        {"role": "user", "content": "Continue."},
    ]

    result = LiteLLMAnthropicMessagesAdapter().translate_anthropic_messages_to_openai(
        messages=messages,
        model="claude-3-5-sonnet-20240620",
    )

    assert result == [
        {
            "role": "assistant",
            "content": None,
            "thinking_blocks": None,
            "tool_calls": [
                {
                    "id": "toolu_01234",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Boston"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "toolu_01234",
            "content": "Rainy, 55°F",
        },
        {"role": "system", "content": expected_content},
        {"role": "user", "content": "Continue."},
    ]


def test_translate_anthropic_messages_to_openai_preserves_midturn_system_cache_control():
    """
    `cache_control` on an in-sequence system text block survives, matching how the
    hoisted top-level `system` prompt and user text blocks are already handled.
    """
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "Use the corrected result.",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
    ]

    result = LiteLLMAnthropicMessagesAdapter().translate_anthropic_messages_to_openai(
        messages=messages,
        model="claude-3-5-sonnet-20240620",
    )

    assert result == [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "Use the corrected result.",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
    ]


def test_translate_anthropic_messages_to_openai_drops_midturn_system_cache_control_for_non_claude():
    """
    `cache_control` goes through the same `_add_cache_control_if_applicable` gate as the
    hoisted top-level prompt and user text blocks, so a non-Claude *requested model name*
    does not get it. That gate is a best-effort check of the requested name before routing
    (behind the proxy it is often a public alias), not a guarantee about the backend that
    ultimately serves the request.
    """
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "Use the corrected result.",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
    ]

    result = LiteLLMAnthropicMessagesAdapter().translate_anthropic_messages_to_openai(
        messages=messages,
        model="gpt-4o",
    )

    assert result == [
        {
            "role": "system",
            "content": [{"type": "text", "text": "Use the corrected result."}],
        }
    ]


@pytest.mark.parametrize(
    "system_content",
    [
        "",
        [{"type": "text", "text": ""}],
        [
            {
                "type": "image",
                "source": {"type": "url", "url": "https://example.com/a.png"},
            }
        ],
        None,
    ],
)
def test_translate_anthropic_messages_to_openai_drops_empty_midturn_system(
    system_content: object,
):
    messages = [{"role": "system", "content": system_content}]

    result = LiteLLMAnthropicMessagesAdapter().translate_anthropic_messages_to_openai(
        messages=messages,
        model="claude-3-5-sonnet-20240620",
    )

    assert result == []


def test_translate_anthropic_to_openai_orders_top_level_and_midturn_system():
    """
    Request level: the trusted top-level prompt is hoisted to index 0 exactly once and the
    in-sequence correction keeps its own position and `role: "system"` -- no duplication of
    either, and no reordering of the surrounding turns.
    """
    openai_request, _ = LiteLLMAnthropicMessagesAdapter().translate_anthropic_to_openai(
        anthropic_message_request={
            "model": "claude-3-5-sonnet-20240620",
            "max_tokens": 100,
            "system": "Trusted top-level prompt.",
            "messages": [
                {"role": "user", "content": "First question."},
                {"role": "assistant", "content": "First answer."},
                {"role": "system", "content": "Use the corrected result."},
                {"role": "user", "content": "Continue."},
            ],
        }
    )

    assert openai_request["messages"] == [
        {"role": "system", "content": "Trusted top-level prompt."},
        {"role": "user", "content": "First question."},
        {"role": "assistant", "content": "First answer.", "thinking_blocks": None},
        {"role": "system", "content": "Use the corrected result."},
        {"role": "user", "content": "Continue."},
    ]


def _translate_with_metadata(
    model: str, metadata: dict[str, str], custom_llm_provider: str | None
) -> dict[str, Any]:
    openai_request, _ = LiteLLMAnthropicMessagesAdapter().translate_anthropic_to_openai(
        anthropic_message_request={
            "model": model,
            "max_tokens": 100,
            "metadata": metadata,
            "messages": [{"role": "user", "content": "hi"}],
        },
        custom_llm_provider=custom_llm_provider,
    )
    return cast(dict[str, Any], openai_request)


def test_translate_anthropic_to_openai_maps_user_id_to_prompt_cache_key_for_openai():
    openai_request = _translate_with_metadata("openai/gpt-5.6-luna", {"user_id": "session-abc"}, "openai")
    assert openai_request["user"] == "session-abc"
    assert openai_request["prompt_cache_key"] == "session-abc"


def test_translate_anthropic_to_openai_truncates_prompt_cache_key_but_keeps_full_user():
    long_id = "".join(str(i % 10) for i in range(100))
    openai_request = _translate_with_metadata("openai/gpt-5.6-luna", {"user_id": long_id}, "openai")
    assert openai_request["user"] == long_id
    assert openai_request["prompt_cache_key"] == long_id[:64]
    assert len(openai_request["prompt_cache_key"]) == 64


@pytest.mark.parametrize("model", ["azure/my-gpt-5-deployment", "my-gpt-5-deployment"])
def test_translate_anthropic_to_openai_sets_prompt_cache_key_for_azure(model: str):
    openai_request = _translate_with_metadata(model, {"user_id": "session-abc"}, "azure")
    assert openai_request["prompt_cache_key"] == "session-abc"


@pytest.mark.parametrize(
    "model, custom_llm_provider",
    [
        ("gemini/gemini-2.5-pro", "gemini"),
        ("vertex_ai/gemini-2.5-pro", "vertex_ai"),
        ("anthropic/claude-sonnet-4-5", "anthropic"),
        ("bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0", "bedrock"),
        ("no-such-model-lit5875", "no-such-provider-lit5875"),
    ],
)
def test_translate_anthropic_to_openai_skips_prompt_cache_key_when_provider_lacks_it(
    model: str, custom_llm_provider: str
):
    openai_request = _translate_with_metadata(model, {"user_id": "session-abc"}, custom_llm_provider)
    assert openai_request["user"] == "session-abc"
    assert "prompt_cache_key" not in openai_request


def test_translate_anthropic_to_openai_skips_prompt_cache_key_for_chained_litellm_proxy():
    assert "prompt_cache_key" in litellm.get_supported_openai_params(
        model="xai", custom_llm_provider="litellm_proxy"
    )
    openai_request = _translate_with_metadata("litellm_proxy/xai", {"user_id": "session-abc"}, "litellm_proxy")
    assert openai_request["user"] == "session-abc"
    assert "prompt_cache_key" not in openai_request


def test_translate_anthropic_to_openai_skips_prompt_cache_key_without_provider():
    openai_request = _translate_with_metadata("openai/gpt-5.6-luna", {"user_id": "session-abc"}, None)
    assert openai_request["user"] == "session-abc"
    assert "prompt_cache_key" not in openai_request


@pytest.mark.parametrize("user_id", ["", None])
def test_translate_anthropic_to_openai_skips_prompt_cache_key_for_empty_or_null_user_id(user_id: str | None):
    openai_request = _translate_with_metadata("openai/gpt-5.6-luna", {"user_id": user_id}, "openai")
    assert openai_request["user"] == user_id
    assert "prompt_cache_key" not in openai_request


def test_translate_anthropic_to_openai_without_metadata_sets_neither_user_nor_prompt_cache_key():
    openai_request, _ = LiteLLMAnthropicMessagesAdapter().translate_anthropic_to_openai(
        anthropic_message_request={
            "model": "openai/gpt-5.6-luna",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
        },
        custom_llm_provider="openai",
    )
    assert "user" not in openai_request
    assert "prompt_cache_key" not in openai_request


@pytest.mark.parametrize("env_value", ["true", "demote"])
def test_translate_anthropic_to_openai_demotes_midturn_system_when_enabled(
    monkeypatch,
    env_value: str,
):
    """
    With LITELLM_DEMOTE_MIDTURN_SYSTEM=true (alias "demote"), in-sequence system rows are
    rewritten as user rows so chat templates that reject non-leading system messages
    (e.g. Qwen3 on vLLM) accept the request. The hoisted top-level prompt at index 0
    keeps `role: "system"`.
    """
    monkeypatch.setenv("LITELLM_DEMOTE_MIDTURN_SYSTEM", env_value)

    openai_request, _ = LiteLLMAnthropicMessagesAdapter().translate_anthropic_to_openai(
        anthropic_message_request={
            "model": "claude-3-5-sonnet-20240620",
            "max_tokens": 100,
            "system": "Trusted top-level prompt.",
            "messages": [
                {"role": "user", "content": "First question."},
                {"role": "assistant", "content": "First answer."},
                {"role": "system", "content": "Use the corrected result."},
                {"role": "user", "content": "Continue."},
            ],
        }
    )

    assert openai_request["messages"] == [
        {"role": "system", "content": "Trusted top-level prompt."},
        {"role": "user", "content": "First question."},
        {"role": "assistant", "content": "First answer.", "thinking_blocks": None},
        {"role": "user", "content": "Use the corrected result."},
        {"role": "user", "content": "Continue."},
    ]


def test_translate_anthropic_to_openai_demotes_midturn_system_block_content(monkeypatch):
    """Demoted rows keep their translated content-block list, including cache_control."""
    monkeypatch.setenv("LITELLM_DEMOTE_MIDTURN_SYSTEM", "true")

    openai_request, _ = LiteLLMAnthropicMessagesAdapter().translate_anthropic_to_openai(
        anthropic_message_request={
            "model": "claude-3-5-sonnet-20240620",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "First question."},
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "Use the corrected result.",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                },
            ],
        }
    )

    assert openai_request["messages"] == [
        {"role": "user", "content": "First question."},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Use the corrected result.",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
    ]


def test_translate_anthropic_to_openai_drops_midturn_system_when_requested(monkeypatch):
    """
    With LITELLM_DEMOTE_MIDTURN_SYSTEM=drop, in-sequence system rows are removed entirely;
    the hoisted top-level prompt at index 0 keeps `role: "system"` and every other turn
    is untouched.
    """
    monkeypatch.setenv("LITELLM_DEMOTE_MIDTURN_SYSTEM", "drop")

    openai_request, _ = LiteLLMAnthropicMessagesAdapter().translate_anthropic_to_openai(
        anthropic_message_request={
            "model": "claude-3-5-sonnet-20240620",
            "max_tokens": 100,
            "system": "Trusted top-level prompt.",
            "messages": [
                {"role": "user", "content": "First question."},
                {"role": "assistant", "content": "First answer."},
                {"role": "system", "content": "Use the corrected result."},
                {"role": "user", "content": "Continue."},
            ],
        }
    )

    assert openai_request["messages"] == [
        {"role": "system", "content": "Trusted top-level prompt."},
        {"role": "user", "content": "First question."},
        {"role": "assistant", "content": "First answer.", "thinking_blocks": None},
        {"role": "user", "content": "Continue."},
    ]


def test_translate_anthropic_to_openai_drop_keeps_leading_system_row(monkeypatch):
    """Without a top-level system param, a system row already at index 0 survives drop mode."""
    monkeypatch.setenv("LITELLM_DEMOTE_MIDTURN_SYSTEM", "drop")

    openai_request, _ = LiteLLMAnthropicMessagesAdapter().translate_anthropic_to_openai(
        anthropic_message_request={
            "model": "claude-3-5-sonnet-20240620",
            "max_tokens": 100,
            "messages": [
                {"role": "system", "content": "Leading system row."},
                {"role": "user", "content": "First question."},
                {"role": "system", "content": "Use the corrected result."},
            ],
        }
    )

    assert openai_request["messages"] == [
        {"role": "system", "content": "Leading system row."},
        {"role": "user", "content": "First question."},
    ]


@pytest.mark.parametrize("env_value", ["", "false", "1", "TRUE_", "drop_"])
def test_translate_anthropic_to_openai_midturn_system_preserved_unless_opted_in(
    monkeypatch,
    env_value: str,
):
    """Anything other than "true" keeps the default in-place behavior."""
    monkeypatch.setenv("LITELLM_DEMOTE_MIDTURN_SYSTEM", env_value)

    openai_request, _ = LiteLLMAnthropicMessagesAdapter().translate_anthropic_to_openai(
        anthropic_message_request={
            "model": "claude-3-5-sonnet-20240620",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "First question."},
                {"role": "system", "content": "Use the corrected result."},
            ],
        }
    )

    assert openai_request["messages"] == [
        {"role": "user", "content": "First question."},
        {"role": "system", "content": "Use the corrected result."},
    ]


def test_translate_openai_content_to_anthropic_empty_function_arguments():
    """Test that empty function arguments are handled safely and don't cause JSON parsing errors."""

    openai_choices = [
        Choices(
            message=Message(
                role="assistant",
                content=None,
                tool_calls=[
                    ChatCompletionAssistantToolCall(
                        id="call_empty_args",
                        type="function",
                        function=Function(
                            name="test_function",
                            arguments="",  # empty arguments string
                        ),
                    )
                ],
            )
        )
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter._translate_openai_content_to_anthropic(choices=openai_choices)

    assert len(result) == 1
    assert result[0]["type"] == "tool_use"
    assert result[0]["id"] == "call_empty_args"
    assert result[0]["name"] == "test_function"
    assert result[0]["input"] == {}, "Empty function arguments should result in empty dict"


def test_translate_openai_content_to_anthropic_text_and_tool_calls():
    """Ensure content blocks contain both the assistant text + tool call data."""
    openai_choices = [
        Choices(
            message=Message(
                role="assistant",
                content="Calling get_weather now.",
                tool_calls=[
                    ChatCompletionAssistantToolCall(
                        id="call_weather",
                        type="function",
                        function=Function(
                            name="get_weather",
                            arguments='{"location": "Boston"}',
                        ),
                    )
                ],
            )
        )
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter._translate_openai_content_to_anthropic(choices=openai_choices)

    assert len(result) == 2
    assert result[0]["type"] == "text"
    assert result[0]["text"] == "Calling get_weather now."
    assert result[1]["type"] == "tool_use"
    assert result[1]["id"] == "call_weather"
    assert result[1]["name"] == "get_weather"
    assert result[1]["input"] == {"location": "Boston"}


def test_translate_openai_content_to_anthropic_strips_gemini_thought_from_tool_call_id():
    """
    Non-streaming path must strip the Gemini thought-signature suffix from
    tool_call.id, same as the streaming path. The base64 signature contains
    `+ / =` which violate Anthropic's `^[a-zA-Z0-9_-]+$` tool_use.id pattern
    and 400 when the history is replayed to an Anthropic-native provider.
    """
    base = "call_3e9417b7925e49aca9a71dc1885e"
    sig = "CiIBDDnWx+/a=="
    combined = f"{base}{THOUGHT_SIGNATURE_SEPARATOR}{sig}"
    openai_choices = [
        Choices(
            message=Message(
                role="assistant",
                content=None,
                tool_calls=[
                    ChatCompletionAssistantToolCall(
                        id=combined,
                        type="function",
                        function=Function(
                            name="get_weather",
                            arguments='{"location": "Boston"}',
                        ),
                    )
                ],
            )
        )
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter._translate_openai_content_to_anthropic(choices=openai_choices)

    assert len(result) == 1
    assert result[0]["type"] == "tool_use"
    assert result[0]["id"] == base
    assert THOUGHT_SIGNATURE_SEPARATOR not in result[0]["id"]
    assert result[0]["name"] == "get_weather"
    assert result[0]["input"] == {"location": "Boston"}


def test_translate_openai_content_to_anthropic_sanitizes_colon_dot_tool_call_ids():
    """Cross-provider ids like ``functions.Bash:0`` must be normalized for Anthropic replay."""
    openai_choices = [
        Choices(
            message=Message(
                role="assistant",
                content=None,
                tool_calls=[
                    ChatCompletionAssistantToolCall(
                        id="functions.Bash:0",
                        type="function",
                        function=Function(
                            name="Bash",
                            arguments='{"command": "ls"}',
                        ),
                    )
                ],
            )
        )
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter._translate_openai_content_to_anthropic(choices=openai_choices)

    assert len(result) == 1
    assert result[0]["type"] == "tool_use"
    assert result[0]["id"] == "functions_Bash_0"


def test_translate_openai_response_to_anthropic_text_and_tool_calls():
    """`translate_openai_response_to_anthropic` should surface assistant text even when tools fire."""
    openai_response = ModelResponse(
        id="resp_text_tool",
        model="gpt-4o-mini",
        choices=[
            Choices(
                finish_reason="tool_calls",
                message=Message(
                    role="assistant",
                    content="Let me grab the current weather.",
                    tool_calls=[
                        ChatCompletionAssistantToolCall(
                            id="call_tool_combo",
                            type="function",
                            function=Function(name="get_weather", arguments='{"location": "Paris"}'),
                        )
                    ],
                ),
            )
        ],
        usage=Usage(prompt_tokens=5, completion_tokens=2),
    )

    adapter = LiteLLMAnthropicMessagesAdapter()
    anthropic_response = adapter.translate_openai_response_to_anthropic(response=openai_response)

    anthropic_content = anthropic_response.get("content")
    assert anthropic_content is not None
    assert len(anthropic_content) == 2
    assert anthropic_content[0]["type"] == "text"
    assert anthropic_content[0]["text"] == "Let me grab the current weather."
    assert anthropic_content[1]["type"] == "tool_use"
    assert anthropic_content[1]["id"] == "call_tool_combo"
    assert anthropic_content[1]["input"] == {"location": "Paris"}
    assert anthropic_response.get("stop_reason") == "tool_use"


def test_translate_streaming_openai_chunk_to_anthropic_with_partial_json():
    """Test that partial tool arguments are correctly handled as input_json_delta."""
    choices = [
        StreamingChoices(
            finish_reason=None,
            index=1,
            delta=Delta(
                provider_specific_fields=None,
                content="",
                role="assistant",
                function_call=None,
                tool_calls=[
                    ChatCompletionDeltaToolCall(
                        id=None,
                        function=Function(arguments=': "San ', name=None),
                        type="function",
                        index=0,
                    )
                ],
                audio=None,
            ),
            logprobs=None,
        )
    ]

    (
        type_of_content,
        content_block_delta,
    ) = LiteLLMAnthropicMessagesAdapter()._translate_streaming_openai_chunk_to_anthropic(choices=choices)

    print("Type of content:", type_of_content)
    print("Content block delta:", content_block_delta)

    assert type_of_content == "input_json_delta"
    assert content_block_delta["type"] == "input_json_delta"
    assert content_block_delta["partial_json"] == ': "San '


def test_translate_openai_content_to_anthropic_thinking_and_redacted_thinking():
    openai_choices = [
        Choices(
            message=Message(
                role="assistant",
                content=None,
                thinking_blocks=[
                    {
                        "type": "thinking",
                        "thinking": "I need to summar",
                        "signature": "sigsig",
                    },
                    {"type": "redacted_thinking", "data": "REDACTED"},
                ],
            )
        )
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter._translate_openai_content_to_anthropic(choices=openai_choices)

    assert len(result) == 2
    assert result[0]["type"] == "thinking"
    assert result[0]["thinking"] == "I need to summar"
    assert result[0]["signature"] == "sigsig"
    assert result[1]["type"] == "redacted_thinking"
    assert result[1]["data"] == "REDACTED"


def test_translate_openai_content_to_anthropic_drops_empty_unsigned_thinking_blocks():
    """LIT-6357 non-streaming producer half, narrowed to unsigned blocks: a
    bridged reasoning model whose thinking_blocks entry has empty or
    whitespace-only text and no signature must not surface as
    {"type": "thinking", "thinking": ""}. A signature-only block (Bedrock
    Converse adaptive thinking) must be emitted so the client keeps the
    signature for tool-use replay; the inbound strip self-heals it if the
    client loops it back. Non-empty thinking and redacted_thinking pass
    through."""
    openai_choices = [
        Choices(
            message=Message(
                role="assistant",
                content="the answer",
                thinking_blocks=[
                    {"type": "thinking", "thinking": "", "signature": "sig_abc"},
                    {"type": "thinking", "thinking": " \n "},
                    {"type": "thinking", "thinking": "real plan", "signature": "sigsig"},
                    {"type": "redacted_thinking", "data": "REDACTED"},
                ],
            )
        )
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter._translate_openai_content_to_anthropic(choices=openai_choices)

    assert [b["type"] for b in result] == ["thinking", "thinking", "redacted_thinking", "text"]
    assert result[0]["thinking"] == ""
    assert result[0]["signature"] == "sig_abc"
    assert result[1]["thinking"] == "real plan"
    assert result[2]["data"] == "REDACTED"


def test_translate_streaming_openai_chunk_to_anthropic_thinking_delta():
    choices = [
        StreamingChoices(
            finish_reason=None,
            index=0,
            delta=Delta(
                reasoning_content="I need to summar",
                thinking_blocks=[
                    {
                        "type": "thinking",
                        "thinking": "I need to summar",
                        "signature": None,
                    }
                ],
                provider_specific_fields={
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": "I need to summar",
                            "signature": None,
                        }
                    ]
                },
                content="",
                role="assistant",
                function_call=None,
                tool_calls=None,
                audio=None,
            ),
            logprobs=None,
        )
    ]

    (
        type_of_content,
        content_block_delta,
    ) = LiteLLMAnthropicMessagesAdapter()._translate_streaming_openai_chunk_to_anthropic(choices=choices)

    assert type_of_content == "thinking_delta"
    assert content_block_delta["type"] == "thinking_delta"
    assert content_block_delta["thinking"] == "I need to summar"


def test_translate_streaming_openai_chunk_to_anthropic_with_thinking():
    choices = [
        StreamingChoices(
            finish_reason=None,
            index=0,
            delta=Delta(
                reasoning_content="",
                thinking_blocks=[
                    {
                        "type": "thinking",
                        "thinking": None,
                        "signature": "sigsig",
                    }
                ],
                provider_specific_fields={
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": None,
                            "signature": "sigsig",
                        }
                    ]
                },
                content="",
                role="assistant",
                function_call=None,
                tool_calls=None,
                audio=None,
            ),
            logprobs=None,
        )
    ]

    (
        type_of_content,
        content_block_delta,
    ) = LiteLLMAnthropicMessagesAdapter()._translate_streaming_openai_chunk_to_anthropic(choices=choices)

    assert type_of_content == "signature_delta"
    assert content_block_delta["type"] == "signature_delta"
    assert content_block_delta["signature"] == "sigsig"


def test_translate_streaming_openai_chunk_to_anthropic_emits_signature_when_thinking_and_signature():
    """A single streaming chunk carrying both ``thinking`` and ``signature`` must
    translate to a ``signature_delta``, not crash.

    litellm's Anthropic streaming handler emits the ``signature_delta`` event as an
    OpenAI chunk whose ``thinking_blocks`` entry re-states the full accumulated
    thinking text alongside the signature (see anthropic/chat/handler.py). That text
    was already streamed as ``thinking_delta`` chunks, so the signature must win and
    the duplicate thinking must not be re-emitted. Before the fix this raised
    ``ValueError`` and 500'd the whole stream, breaking Claude Code through the proxy.
    """
    choices = [
        StreamingChoices(
            finish_reason=None,
            index=0,
            delta=Delta(
                reasoning_content="",
                thinking_blocks=[
                    {
                        "type": "thinking",
                        "thinking": "I need to summar",
                        "signature": "sigsig",
                    }
                ],
                provider_specific_fields={
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": "I need to summar",
                            "signature": "sigsig",
                        }
                    ]
                },
                content="",
                role="assistant",
                function_call=None,
                tool_calls=None,
                audio=None,
            ),
            logprobs=None,
        )
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()

    (
        type_of_content,
        content_block_delta,
    ) = adapter._translate_streaming_openai_chunk_to_anthropic(choices=choices)

    assert type_of_content == "signature_delta"
    assert content_block_delta["type"] == "signature_delta"
    assert content_block_delta["signature"] == "sigsig"

    (
        block_type,
        content_block_start,
    ) = adapter._translate_streaming_openai_chunk_to_anthropic_content_block(choices=choices)

    assert block_type == "thinking"


def test_translate_anthropic_messages_to_openai_user_message_with_base64_image():
    """Test that base64 images in user messages are correctly translated to OpenAI format."""

    anthropic_messages = [
        AnthropicMessagesUserMessageParam(
            role="user",
            content=[
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                    },
                },
            ],
        )
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter.translate_anthropic_messages_to_openai(messages=anthropic_messages)

    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert isinstance(result[0]["content"], list)
    assert len(result[0]["content"]) == 2

    # Check text content
    assert result[0]["content"][0]["type"] == "text"
    assert result[0]["content"][0]["text"] == "What's in this image?"

    # Check image content
    assert result[0]["content"][1]["type"] == "image_url"
    assert "image_url" in result[0]["content"][1]
    assert result[0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        in result[0]["content"][1]["image_url"]["url"]
    )


def test_translate_anthropic_messages_to_openai_user_message_with_url_image():
    """Test that URL-based images in user messages are correctly translated to OpenAI format."""

    anthropic_messages = [
        AnthropicMessagesUserMessageParam(
            role="user",
            content=[
                {"type": "text", "text": "Describe this forest path"},
                {
                    "type": "image",
                    "source": {"type": "url", "url": "https://example.com/forest.jpg"},
                },
            ],
        )
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter.translate_anthropic_messages_to_openai(messages=anthropic_messages)

    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert isinstance(result[0]["content"], list)
    assert len(result[0]["content"]) == 2

    # Check text content
    assert result[0]["content"][0]["type"] == "text"
    assert result[0]["content"][0]["text"] == "Describe this forest path"

    # Check image content
    assert result[0]["content"][1]["type"] == "image_url"
    assert "image_url" in result[0]["content"][1]
    assert result[0]["content"][1]["image_url"]["url"] == "https://example.com/forest.jpg"


def test_translate_anthropic_messages_to_openai_tool_result_with_base64_image():
    """Test that base64 images in tool results are correctly translated to OpenAI format."""

    anthropic_messages = [
        AnthropicMessagesUserMessageParam(role="user", content=[{"type": "text", "text": "Take a screenshot"}]),
        AnthopicMessagesAssistantMessageParam(
            role="assistant",
            content=[
                {
                    "type": "tool_use",
                    "id": "toolu_01A09q90qw90lq917835lq9",
                    "name": "get_screenshot",
                    "input": {"area": "desktop"},
                }
            ],
        ),
        AnthropicMessagesUserMessageParam(
            role="user",
            content=[
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01A09q90qw90lq917835lq9",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/2wBDAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwA/wA/==",
                            },
                        }
                    ],
                }
            ],
        ),
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter.translate_anthropic_messages_to_openai(messages=anthropic_messages)

    # Find the tool message in the result
    tool_message = None
    for msg in result:
        if isinstance(msg, dict) and msg.get("role") == "tool":
            tool_message = msg
            break

    assert tool_message is not None, "Tool message not found in result"
    assert isinstance(tool_message["content"], list)
    assert len(tool_message["content"]) == 1
    image_part = tool_message["content"][0]
    assert image_part["type"] == "image_url"
    assert image_part["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert "/9j/4AAQSkZJRgABAQAAAQABAAD" in image_part["image_url"]["url"]


def test_translate_anthropic_messages_to_openai_tool_result_with_url_image():
    """Test that URL-based images in tool results are correctly translated to OpenAI format."""

    anthropic_messages = [
        AnthropicMessagesUserMessageParam(
            role="user",
            content=[{"type": "text", "text": "Take a screenshot of the forest"}],
        ),
        AnthopicMessagesAssistantMessageParam(
            role="assistant",
            content=[
                {
                    "type": "tool_use",
                    "id": "toolu_01A09q90qw90lq917835lq9",
                    "name": "get_screenshot",
                    "input": {"area": "forest_path"},
                }
            ],
        ),
        AnthropicMessagesUserMessageParam(
            role="user",
            content=[
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01A09q90qw90lq917835lq9",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": "https://i0.wp.com/picjumbo.com/wp-content/uploads/amazing-stone-path-in-forest-free-image.jpg",
                            },
                        }
                    ],
                }
            ],
        ),
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter.translate_anthropic_messages_to_openai(messages=anthropic_messages)

    # Find the tool message in the result
    tool_message = None
    for msg in result:
        if isinstance(msg, dict) and msg.get("role") == "tool":
            tool_message = msg
            break

    assert tool_message is not None, "Tool message not found in result"
    assert isinstance(tool_message["content"], list)
    assert len(tool_message["content"]) == 1
    image_part = tool_message["content"][0]
    assert image_part["type"] == "image_url"
    assert (
        image_part["image_url"]["url"]
        == "https://i0.wp.com/picjumbo.com/wp-content/uploads/amazing-stone-path-in-forest-free-image.jpg"
    )


def test_translate_anthropic_messages_to_openai_mixed_content_with_image():
    """Test that messages with mixed text and image content are correctly translated."""

    anthropic_messages = [
        AnthropicMessagesUserMessageParam(
            role="user",
            content=[
                {"type": "text", "text": "Here are two images:"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                    },
                },
                {"type": "text", "text": "and this one:"},
                {
                    "type": "image",
                    "source": {"type": "url", "url": "https://example.com/image2.jpg"},
                },
                {"type": "text", "text": "What's the difference?"},
            ],
        )
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter.translate_anthropic_messages_to_openai(messages=anthropic_messages)

    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert isinstance(result[0]["content"], list)
    assert len(result[0]["content"]) == 5

    # Check text content
    assert result[0]["content"][0]["type"] == "text"
    assert result[0]["content"][0]["text"] == "Here are two images:"

    # Check first image (base64)
    assert result[0]["content"][1]["type"] == "image_url"
    assert result[0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")

    # Check middle text
    assert result[0]["content"][2]["type"] == "text"
    assert result[0]["content"][2]["text"] == "and this one:"

    # Check second image (URL)
    assert result[0]["content"][3]["type"] == "image_url"
    assert result[0]["content"][3]["image_url"]["url"] == "https://example.com/image2.jpg"

    # Check final text
    assert result[0]["content"][4]["type"] == "text"
    assert result[0]["content"][4]["text"] == "What's the difference?"


def test_translate_anthropic_messages_to_openai_tool_use_with_signature():
    """Test that thought signatures from tool_use blocks are correctly extracted and placed in provider_specific_fields."""

    test_signature = "EpYECpMEAdHtim9iBECdK1l5uVIIXoZZmq+PUBH9nz3Q6EMeIdEqWwVb5GlxSNtxuSkFoseFco5U4zxN/lacJxD2WUjFvEyL2GOkbPgXFeCcgNBMEYVRg7UAr45KGeWJJmJMoheLHezKawI1L94vi2PsB9TDpWv4vyAx1vKG2PByiVmWWtd0rondsdbENNp2Rrz3ol1zha+XhOtyhTCdSWce8GVD/zElklL3C0h9HrsTQrnNyouaZa9KlXZJ72XDCIkIlV0m6EtxbzdMwbH4sLFOpifRlRn+AmzXjxvLovRtn2bXh/X3bUgPxqypaST57Dlpddlk1Mt0oJmGFtwB/FH1JmK21cIC06uXtlUc8lm/9cTQLd5hcEUX+XRrmTdzqxDgRttN8CRfVUAGE7Er+prN4yCIdNtEQdZm8zymEpHTkYplJ/hK7SMf9Iu1k+eCDFYCzvQuzLcJtNpRaGS1BbVA3va5JKrEu96G7a3Wl3DyzmrH8N3+RA+UIHvP6P5v93tI/eTyfMY54rKpLGkfFeeSMAr5aSoUZVYkvFI8xGEcIrqLWPDF91MclLZa7USSVql0wYu1G9KD10IkopeKkTIAl81WfoY5+Kw1o4CHo7bEQ6tfTuTB4IEywf1XKMBYHmsfAe5B9ferkLYtnAzzt1hoiK1m/2CjX8yQAknRLsnAuyeXfJZRZidVKYOKaSDftddbXJpIlJApC"

    anthropic_messages = [
        AnthropicMessagesUserMessageParam(
            role="user",
            content=[{"type": "text", "text": "What's the weather like in London?"}],
        ),
        AnthopicMessagesAssistantMessageParam(
            role="assistant",
            content=[
                {
                    "type": "tool_use",
                    "id": "call_386f67af31f9415781bc35071405",
                    "name": "get_weather",
                    "input": {"location": "London"},
                    "provider_specific_fields": {
                        "signature": test_signature,
                    },
                }
            ],
        ),
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter.translate_anthropic_messages_to_openai(messages=anthropic_messages)

    assert len(result) == 2
    assert result[1]["role"] == "assistant"
    assert "tool_calls" in result[1]
    assert len(result[1]["tool_calls"]) == 1

    # Verify thought signature is extracted and placed in provider_specific_fields
    tool_call = result[1]["tool_calls"][0]
    assert tool_call["id"] == "call_386f67af31f9415781bc35071405"
    assert "function" in tool_call
    assert "provider_specific_fields" in tool_call["function"]
    assert tool_call["function"]["provider_specific_fields"]["thought_signature"] == test_signature


def test_translate_anthropic_messages_to_openai_tool_result_with_multiple_content_items():
    """
    Test that tool_result with multiple content items creates a single tool message
    (not multiple messages with the same tool_call_id).

    This is a regression test for the bug:
    "each tool_use must have a single result. Found multiple `tool_result` blocks with id"

    When a tool_result has a list of content items (e.g., text + image), we should create
    ONE tool message with combined content, not multiple tool messages with the same ID.
    """

    anthropic_messages = [
        AnthropicMessagesUserMessageParam(
            role="user",
            content=[{"type": "text", "text": "Take a screenshot and describe it"}],
        ),
        AnthopicMessagesAssistantMessageParam(
            role="assistant",
            content=[
                {
                    "type": "tool_use",
                    "id": "toolu_016hYHBkTf4JDF3p22UoYk5C",
                    "name": "screenshot_tool",
                    "input": {},
                }
            ],
        ),
        AnthropicMessagesUserMessageParam(
            role="user",
            content=[
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_016hYHBkTf4JDF3p22UoYk5C",
                    "content": [
                        {"type": "text", "text": "Here is the screenshot:"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                            },
                        },
                        {"type": "text", "text": "Screenshot captured successfully."},
                    ],
                }
            ],
        ),
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter.translate_anthropic_messages_to_openai(messages=anthropic_messages)

    # Count how many tool messages have the same tool_call_id
    tool_messages = [msg for msg in result if isinstance(msg, dict) and msg.get("role") == "tool"]
    tool_call_ids = [msg.get("tool_call_id") for msg in tool_messages]

    # The critical assertion: each tool_call_id should appear only ONCE
    assert len(tool_call_ids) == len(set(tool_call_ids)), (
        f"Bug: Found duplicate tool_call_ids! "
        f"Each tool_use must have exactly one tool_result. "
        f"tool_call_ids: {tool_call_ids}"
    )

    # There should be exactly one tool message
    assert len(tool_messages) == 1, f"Expected 1 tool message, got {len(tool_messages)}"

    # The content should be a list with all items combined
    tool_message = tool_messages[0]
    assert tool_message["tool_call_id"] == "toolu_016hYHBkTf4JDF3p22UoYk5C"
    assert isinstance(tool_message["content"], list), "Multiple content items should be combined into a list"
    assert len(tool_message["content"]) == 3, f"Expected 3 content items, got {len(tool_message['content'])}"

    # Verify content types
    assert tool_message["content"][0]["type"] == "text"
    assert tool_message["content"][0]["text"] == "Here is the screenshot:"
    assert tool_message["content"][1]["type"] == "image_url"
    assert tool_message["content"][2]["type"] == "text"
    assert tool_message["content"][2]["text"] == "Screenshot captured successfully."


def test_translate_anthropic_messages_to_openai_tool_result_single_item_backward_compat():
    """
    Test that tool_result with a single content item maintains backward compatibility
    by returning a string content (not a list).
    """

    anthropic_messages = [
        AnthropicMessagesUserMessageParam(
            role="user",
            content=[{"type": "text", "text": "Get the weather"}],
        ),
        AnthopicMessagesAssistantMessageParam(
            role="assistant",
            content=[
                {
                    "type": "tool_use",
                    "id": "toolu_single_item",
                    "name": "get_weather",
                    "input": {"location": "Boston"},
                }
            ],
        ),
        AnthropicMessagesUserMessageParam(
            role="user",
            content=[
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_single_item",
                    "content": [
                        {"type": "text", "text": "72°F and sunny"},
                    ],
                }
            ],
        ),
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter.translate_anthropic_messages_to_openai(messages=anthropic_messages)

    tool_messages = [msg for msg in result if isinstance(msg, dict) and msg.get("role") == "tool"]

    assert len(tool_messages) == 1
    tool_message = tool_messages[0]

    # Single item should be a string for backward compatibility
    assert isinstance(tool_message["content"], str), (
        f"Single content item should be a string for backward compatibility, got {type(tool_message['content'])}"
    )
    assert tool_message["content"] == "72°F and sunny"


def test_streaming_chunk_with_both_text_and_tool_calls_issue_18238():
    """
    When a streaming choice contains both text content and tool_calls,
    both should be processed (tool_calls should not be ignored).
    """
    # streaming choice with both text and tool_calls
    choices = [
        StreamingChoices(
            finish_reason=None,
            index=0,
            delta=Delta(
                provider_specific_fields=None,
                content="Here is some text for litellm",
                role=None,
                function_call=None,
                tool_calls=[
                    ChatCompletionDeltaToolCall(
                        id="toolu_bdrk_013xRVejhv3ybmLEGCoZib2b",
                        function=Function(arguments='{"cmd": "init"}', name="Bash"),
                        type="function",
                        index=0,
                    )
                ],
                audio=None,
            ),
            logprobs=None,
        )
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()

    # When both text and tool_calls exist, tool_calls (input_json_delta) takes priority
    (
        type_of_content,
        content_block_delta,
    ) = adapter._translate_streaming_openai_chunk_to_anthropic(choices=choices)

    assert type_of_content == "input_json_delta"
    assert content_block_delta["partial_json"] == '{"cmd": "init"}'

    # When both text and tool_calls exist, tool_use should be detected and tool name captured
    (
        block_type,
        content_block_start,
    ) = adapter._translate_streaming_openai_chunk_to_anthropic_content_block(choices=choices)

    assert block_type == "tool_use"
    assert content_block_start["name"] == "Bash"
    assert content_block_start["id"] == "toolu_bdrk_013xRVejhv3ybmLEGCoZib2b"


def test_streaming_chunk_with_text_and_empty_tool_calls_returns_text_delta():
    """
    Some OpenAI-compatible providers emit `tool_calls: []` on regular text chunks.

    Empty tool_calls should be treated as no tool call so the Anthropic adapter
    does not shadow text with an empty input_json_delta.
    """
    choices = [
        StreamingChoices(
            finish_reason=None,
            index=0,
            delta=Delta(
                provider_specific_fields=None,
                content="Hello from vLLM",
                role="assistant",
                function_call=None,
                tool_calls=[],
                audio=None,
            ),
            logprobs=None,
        )
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()

    (
        type_of_content,
        content_block_delta,
    ) = adapter._translate_streaming_openai_chunk_to_anthropic(choices=choices)

    assert type_of_content == "text_delta"
    assert content_block_delta["type"] == "text_delta"
    assert content_block_delta["text"] == "Hello from vLLM"

    (
        block_type,
        content_block_start,
    ) = adapter._translate_streaming_openai_chunk_to_anthropic_content_block(choices=choices)

    assert block_type == "text"
    assert content_block_start == {"type": "text", "text": ""}


# ============================================================================
# Cache Control Transformation Tests
# ============================================================================

# Model constant for cache control tests
CACHE_CONTROL_BEDROCK_CONVERSE_MODEL = "bedrock/converse/global.anthropic.claude-opus-4-5-20251101-v1:0"
CACHE_CONTROL_NON_ANTHROPIC_MODEL = "gpt-4"
# Bedrock Application Inference Profile ARN: the string contains neither
# "anthropic" nor "claude", so the model can only be recognized via its ARN shape
CACHE_CONTROL_BEDROCK_ARN_MODEL = (
    "bedrock/converse/arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abcdef123456"
)


def test_should_add_cache_control_for_anthropic_model():
    """Should add cache_control to target for Anthropic Claude models."""
    adapter = LiteLLMAnthropicMessagesAdapter()
    cache_control = {"type": "ephemeral"}

    for model in [
        CACHE_CONTROL_BEDROCK_CONVERSE_MODEL,
        "anthropic/claude-sonnet-4-5",
        "claude-opus-4-5-20251101",
        "vertex_ai/claude-3-sonnet@20240229",
    ]:
        target = {}
        adapter._add_cache_control_if_applicable({"cache_control": cache_control}, target, model)
        assert "cache_control" in target
        assert target["cache_control"] == cache_control


def test_should_not_add_cache_control_for_non_anthropic_model():
    """Should not add cache_control for non-Anthropic models."""
    adapter = LiteLLMAnthropicMessagesAdapter()
    cache_control = {"type": "ephemeral"}

    for model in [
        CACHE_CONTROL_NON_ANTHROPIC_MODEL,
        "openai/gpt-4-turbo",
        "gemini-pro",
    ]:
        target = {}
        adapter._add_cache_control_if_applicable({"cache_control": cache_control}, target, model)
        assert "cache_control" not in target


def test_should_not_add_cache_control_when_none():
    """Should not add cache_control when source has None or empty cache_control."""
    adapter = LiteLLMAnthropicMessagesAdapter()

    for source in [
        {"cache_control": None},
        {"cache_control": {}},
        {"cache_control": ""},
        {},
    ]:
        target = {}
        adapter._add_cache_control_if_applicable(source, target, CACHE_CONTROL_BEDROCK_CONVERSE_MODEL)
        assert "cache_control" not in target


def test_should_not_add_cache_control_when_model_none():
    """Should not add cache_control when model is None or empty."""
    adapter = LiteLLMAnthropicMessagesAdapter()
    cache_control = {"type": "ephemeral"}

    for model in [None, ""]:
        target = {}
        adapter._add_cache_control_if_applicable({"cache_control": cache_control}, target, model)
        assert "cache_control" not in target


def test_cache_control_preserved_in_text_content_for_claude():
    """Cache control should be preserved in text content for Claude models."""
    anthropic_messages = [
        AnthropicMessagesUserMessageParam(
            role="user",
            content=[
                {
                    "type": "text",
                    "text": "This is cached content",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        )
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter.translate_anthropic_messages_to_openai(
        messages=anthropic_messages, model=CACHE_CONTROL_BEDROCK_CONVERSE_MODEL
    )

    assert len(result) == 1
    assert result[0]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_cache_control_not_preserved_for_non_claude_model():
    """Cache control should NOT be preserved for non-Claude models."""
    anthropic_messages = [
        AnthropicMessagesUserMessageParam(
            role="user",
            content=[
                {
                    "type": "text",
                    "text": "This is cached content",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        )
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter.translate_anthropic_messages_to_openai(
        messages=anthropic_messages, model=CACHE_CONTROL_NON_ANTHROPIC_MODEL
    )

    assert len(result) == 1
    assert "cache_control" not in result[0]["content"][0]


@pytest.mark.parametrize(
    "model, expected",
    [
        (CACHE_CONTROL_BEDROCK_ARN_MODEL, True),
        (
            "arn:aws-us-gov:bedrock:us-gov-west-1:123:application-inference-profile/x",
            True,
        ),
        ("bedrock/amazon.titan-text-express-v1", False),
        ("arn:aws:sagemaker:us-east-1:123:endpoint/my-endpoint", False),
        ("arn:aws:sagemaker:us-east-1:123:endpoint/my-bedrock-transcriber", False),
        (CACHE_CONTROL_NON_ANTHROPIC_MODEL, False),
    ],
)
def test_is_bedrock_arn_model(model, expected):
    """is_bedrock_arn_model requires an ARN with bedrock in the service field, not just anywhere."""
    assert LiteLLMAnthropicMessagesAdapter.is_bedrock_arn_model(model) is expected


def test_cache_control_preserved_for_bedrock_arn_inference_profile():
    """
    Regression for https://github.com/BerriAI/litellm/issues/26625

    Bedrock Application Inference Profile ARNs hide the underlying Claude model
    name, so cache_control must still be preserved through the /v1/messages adapter.
    """
    anthropic_messages = [
        AnthropicMessagesUserMessageParam(
            role="user",
            content=[
                {
                    "type": "text",
                    "text": "This is cached content",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        )
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter.translate_anthropic_messages_to_openai(
        messages=anthropic_messages, model=CACHE_CONTROL_BEDROCK_ARN_MODEL
    )

    assert len(result) == 1
    assert result[0]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_cache_control_fix_does_not_broaden_claude_detection():
    """
    The cache_control fix is scoped to _add_cache_control_if_applicable; it must not
    make is_anthropic_claude_model treat ARN profiles as Claude, which would route
    thinking params through unmodified and break non-Claude Bedrock profiles.
    """
    assert LiteLLMAnthropicMessagesAdapter.is_anthropic_claude_model(CACHE_CONTROL_BEDROCK_ARN_MODEL) is False


def test_thinking_preserved_for_bedrock_arn_inference_profile():
    """
    Regression: opaque Bedrock Application Inference Profile ARNs hide the underlying
    Claude model name, so on /v1/messages a `thinking` param must be preserved as
    `thinking` (not rewritten to `reasoning_effort`). Otherwise `additional_drop_params:
    ["thinking"]` runs after the rewrite and has nothing left to drop, and the Bedrock
    Converse body re-expands reasoning_effort back into additionalModelRequestFields.thinking.
    """
    adapter = LiteLLMAnthropicMessagesAdapter()
    thinking = {"type": "enabled", "budget_tokens": 1024}

    new_kwargs = {"model": CACHE_CONTROL_BEDROCK_ARN_MODEL}
    adapter._translate_thinking_to_openai(cast(Any, {"thinking": thinking}), cast(Any, new_kwargs))

    assert new_kwargs["thinking"] == thinking
    assert "reasoning_effort" not in new_kwargs

    assert LiteLLMAnthropicMessagesAdapter.translate_thinking_for_model(thinking, CACHE_CONTROL_BEDROCK_ARN_MODEL) == {
        "thinking": thinking
    }


def test_thinking_still_translated_to_reasoning_effort_for_non_claude_model():
    """
    The bedrock-ARN gate must not broaden to every model: a genuine non-Claude model
    still has `thinking` converted to `reasoning_effort` so it does not hit an
    UnsupportedParamsError downstream.
    """
    adapter = LiteLLMAnthropicMessagesAdapter()
    thinking = {"type": "enabled", "budget_tokens": 1024}

    new_kwargs = {"model": CACHE_CONTROL_NON_ANTHROPIC_MODEL}
    adapter._translate_thinking_to_openai(cast(Any, {"thinking": thinking}), cast(Any, new_kwargs))

    assert "thinking" not in new_kwargs
    assert new_kwargs["reasoning_effort"] == "low"


def test_thinking_disabled_translated_to_reasoning_effort_none_for_non_claude_model():
    adapter = LiteLLMAnthropicMessagesAdapter()
    thinking = {"type": "disabled"}

    new_kwargs = {"model": CACHE_CONTROL_NON_ANTHROPIC_MODEL}
    adapter._translate_thinking_to_openai(cast(Any, {"thinking": thinking}), cast(Any, new_kwargs))

    assert "thinking" not in new_kwargs
    assert new_kwargs["reasoning_effort"] == "none"


def test_thinking_disabled_stays_plain_string_when_auto_summary_enabled():
    import litellm

    adapter = LiteLLMAnthropicMessagesAdapter()
    thinking = {"type": "disabled"}

    original = litellm.reasoning_auto_summary
    try:
        litellm.reasoning_auto_summary = True
        new_kwargs = {"model": CACHE_CONTROL_NON_ANTHROPIC_MODEL}
        adapter._translate_thinking_to_openai(cast(Any, {"thinking": thinking}), cast(Any, new_kwargs))
    finally:
        litellm.reasoning_auto_summary = original

    assert new_kwargs["reasoning_effort"] == "none"


@pytest.mark.parametrize(
    "model",
    [
        # SDK-style model with the provider prefix intact
        "bedrock/converse/us.anthropic.claude-opus-4-7",
        # what the bridge actually sees in the proxy: get_llm_provider has
        # already stripped the `bedrock/` prefix by the time it translates
        "converse/us.anthropic.claude-opus-4-7",
    ],
)
def test_adaptive_thinking_output_config_effort_preserved_for_claude_model(model):
    """
    Regression: Claude Code drives adaptive thinking as `thinking: {"type": "adaptive"}`
    plus `output_config: {"effort": "max"}`. The Claude branch of the thinking translator
    forwarded `thinking` verbatim but returned early without reading `output_config`, and
    the handler strips the raw key from extra_kwargs, so the effort tier never reached the
    backend. On Bedrock Converse, adaptive thinking without effort streams zero reasoning
    blocks. The `format` subkey must still be excluded (it is translated to
    `response_format` separately).

    Bedrock keeps taking the tier as `output_config`, which attaches it without disturbing
    `thinking`. Driving the translated request through the provider's own param mapping is what
    makes the second half a claim about the wire rather than about an intermediate key.
    """
    from litellm.types.llms.anthropic import AnthropicMessagesRequest
    from litellm.utils import get_optional_params

    anthropic_request = AnthropicMessagesRequest(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": "hi"}],
        thinking={"type": "adaptive"},
        output_config={
            "effort": "max",
            "format": {"type": "json_schema", "schema": {"type": "object", "properties": {}}},
        },
    )

    adapter = LiteLLMAnthropicMessagesAdapter()
    openai_request, _ = adapter.translate_anthropic_to_openai(anthropic_message_request=anthropic_request)

    assert openai_request["thinking"] == {"type": "adaptive"}
    assert openai_request["output_config"] == {"effort": "max"}
    assert "reasoning_effort" not in openai_request
    assert "response_format" in openai_request

    on_the_wire = get_optional_params(
        model=model,
        custom_llm_provider="bedrock",
        thinking=openai_request["thinking"],
        output_config=openai_request["output_config"],
    )

    assert on_the_wire["output_config"] == {"effort": "max"}


def test_adaptive_thinking_format_only_output_config_not_forwarded_for_claude_model():
    """When `output_config` carries only `format`, nothing effort-bearing remains, so the
    translator must not forward an empty `output_config` dict."""
    from litellm.types.llms.anthropic import AnthropicMessagesRequest

    anthropic_request = AnthropicMessagesRequest(
        model="bedrock/converse/us.anthropic.claude-opus-4-7",
        max_tokens=1024,
        messages=[{"role": "user", "content": "hi"}],
        thinking={"type": "adaptive"},
        output_config={"format": {"type": "json_schema", "schema": {"type": "object", "properties": {}}}},
    )

    adapter = LiteLLMAnthropicMessagesAdapter()
    openai_request, _ = adapter.translate_anthropic_to_openai(anthropic_message_request=anthropic_request)

    assert openai_request["thinking"] == {"type": "adaptive"}
    assert "output_config" not in openai_request


def test_adaptive_thinking_output_config_not_forwarded_for_non_bedrock_claude_model():
    """`output_config` is never forwarded raw to a bridged provider: openrouter and friends accept
    `thinking` but reject that param with UnsupportedParamsError when drop_params is off.

    Regression: the tier used to be dropped along with it, so an openrouter Claude deployment got a
    bare adaptive `thinking` block and the caller's effort did nothing, byte-identical for `max` and
    `minimal`. It now travels as `reasoning_effort`, which that provider does accept."""
    from litellm.types.llms.anthropic import AnthropicMessagesRequest

    anthropic_request = AnthropicMessagesRequest(
        model="openrouter/anthropic/claude-opus-4.7",
        max_tokens=1024,
        messages=[{"role": "user", "content": "hi"}],
        thinking={"type": "adaptive"},
        output_config={"effort": "max"},
    )

    adapter = LiteLLMAnthropicMessagesAdapter()
    openai_request, _ = adapter.translate_anthropic_to_openai(
        anthropic_message_request=anthropic_request, custom_llm_provider="openrouter"
    )

    assert openai_request["thinking"] == {"type": "adaptive"}
    assert "output_config" not in openai_request
    assert openai_request["reasoning_effort"] == "max"


@pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high", "xhigh", "max"])
def test_every_adaptive_effort_tier_reaches_a_bridged_claude_target(effort):
    """The tier the caller asked for is the tier the bridge carries, for every level. The bug was
    invisible per-request because each call returned 200; only comparing two tiers showed the
    upstream body was the same either way."""
    from litellm.types.llms.anthropic import AnthropicMessagesRequest

    adapter = LiteLLMAnthropicMessagesAdapter()
    openai_request, _ = adapter.translate_anthropic_to_openai(
        anthropic_message_request=AnthropicMessagesRequest(
            model="openrouter/anthropic/claude-opus-4.7",
            max_tokens=1024,
            messages=[{"role": "user", "content": "hi"}],
            thinking={"type": "adaptive"},
            output_config={"effort": effort},
        ),
        custom_llm_provider="openrouter",
    )

    assert openai_request["reasoning_effort"] == effort


def test_adaptive_thinking_without_a_tier_leaves_a_claude_target_on_its_own_default():
    """Adaptive with no `output_config.effort` must stay bare, so the provider's own adaptive
    default still decides. Inventing a tier here would silently override it."""
    from litellm.types.llms.anthropic import AnthropicMessagesRequest

    adapter = LiteLLMAnthropicMessagesAdapter()
    openai_request, _ = adapter.translate_anthropic_to_openai(
        anthropic_message_request=AnthropicMessagesRequest(
            model="openrouter/anthropic/claude-opus-4-7",
            max_tokens=1024,
            messages=[{"role": "user", "content": "hi"}],
            thinking={"type": "adaptive"},
        )
    )

    assert openai_request["thinking"] == {"type": "adaptive"}
    assert "reasoning_effort" not in openai_request
    assert "output_config" not in openai_request


def test_budgeted_thinking_on_a_claude_target_keeps_its_budget_and_gains_no_tier():
    """`enabled` + `budget_tokens` is more precise than any tier, so the bridge must forward it
    untouched rather than coarsening it into a `reasoning_effort` bucket."""
    from litellm.types.llms.anthropic import AnthropicMessagesRequest

    adapter = LiteLLMAnthropicMessagesAdapter()
    openai_request, _ = adapter.translate_anthropic_to_openai(
        anthropic_message_request=AnthropicMessagesRequest(
            model="openrouter/anthropic/claude-opus-4-7",
            max_tokens=1024,
            messages=[{"role": "user", "content": "hi"}],
            thinking={"type": "enabled", "budget_tokens": 8000},
            output_config={"effort": "max"},
        )
    )

    assert openai_request["thinking"] == {"type": "enabled", "budget_tokens": 8000}
    assert "reasoning_effort" not in openai_request


def test_stop_sequences_translated_to_stop_for_non_claude_model():
    from litellm.types.llms.anthropic import AnthropicMessagesRequest

    anthropic_request = AnthropicMessagesRequest(
        model=CACHE_CONTROL_NON_ANTHROPIC_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": "hi"}],
        stop_sequences=["</block>"],
    )

    adapter = LiteLLMAnthropicMessagesAdapter()
    openai_request, _ = adapter.translate_anthropic_to_openai(anthropic_message_request=anthropic_request)

    assert openai_request["stop"] == ["</block>"]
    assert "stop_sequences" not in openai_request


def test_empty_stop_sequences_does_not_set_stop():
    from litellm.types.llms.anthropic import AnthropicMessagesRequest

    anthropic_request = AnthropicMessagesRequest(
        model=CACHE_CONTROL_NON_ANTHROPIC_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": "hi"}],
        stop_sequences=[],
    )

    adapter = LiteLLMAnthropicMessagesAdapter()
    openai_request, _ = adapter.translate_anthropic_to_openai(anthropic_message_request=anthropic_request)

    assert "stop" not in openai_request


def test_cache_control_preserved_in_image_content_for_claude():
    """Cache control should be preserved in image content for Claude models."""
    anthropic_messages = [
        AnthropicMessagesUserMessageParam(
            role="user",
            content=[
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                    },
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        )
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter.translate_anthropic_messages_to_openai(
        messages=anthropic_messages, model=CACHE_CONTROL_BEDROCK_CONVERSE_MODEL
    )

    assert len(result) == 1
    assert result[0]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_cache_control_preserved_in_document_content_for_claude():
    """Cache control should be preserved in document content for Claude models."""
    anthropic_messages = [
        AnthropicMessagesUserMessageParam(
            role="user",
            content=[
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": "JVBERi0xLjQKJeLjz9MK",
                    },
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        )
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter.translate_anthropic_messages_to_openai(
        messages=anthropic_messages, model=CACHE_CONTROL_BEDROCK_CONVERSE_MODEL
    )

    assert len(result) == 1
    assert result[0]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_cache_control_preserved_in_tool_result_for_claude():
    """Cache control should be preserved in tool_result for Claude models."""
    anthropic_messages = [
        AnthropicMessagesUserMessageParam(
            role="user",
            content=[
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01234",
                    "content": "Tool result content",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        )
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter.translate_anthropic_messages_to_openai(
        messages=anthropic_messages, model=CACHE_CONTROL_BEDROCK_CONVERSE_MODEL
    )

    tool_message = next(msg for msg in result if msg.get("role") == "tool")
    assert tool_message["cache_control"] == {"type": "ephemeral"}


def test_cache_control_not_preserved_in_tool_result_for_non_claude():
    """Cache control should NOT be preserved in tool_result for non-Claude models."""
    anthropic_messages = [
        AnthropicMessagesUserMessageParam(
            role="user",
            content=[
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01234",
                    "content": "Tool result content",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        )
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter.translate_anthropic_messages_to_openai(
        messages=anthropic_messages, model=CACHE_CONTROL_NON_ANTHROPIC_MODEL
    )

    tool_message = next(msg for msg in result if msg.get("role") == "tool")
    assert "cache_control" not in tool_message


def test_cache_control_preserved_in_assistant_text_for_claude():
    """Cache control should be preserved in assistant text blocks for Claude models."""
    anthropic_messages = [
        AnthopicMessagesAssistantMessageParam(
            role="assistant",
            content=[
                {
                    "type": "text",
                    "text": "Assistant response",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        )
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter.translate_anthropic_messages_to_openai(
        messages=anthropic_messages, model=CACHE_CONTROL_BEDROCK_CONVERSE_MODEL
    )

    assert len(result) == 1
    assert result[0]["role"] == "assistant"
    # When cache_control is present, content should be a list
    assert isinstance(result[0]["content"], list)
    assert result[0]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_cache_control_preserved_in_tool_use_for_claude():
    """Cache control should be preserved in tool_use blocks for Claude models."""
    anthropic_messages = [
        AnthopicMessagesAssistantMessageParam(
            role="assistant",
            content=[
                {
                    "type": "tool_use",
                    "id": "toolu_01234",
                    "name": "get_weather",
                    "input": {"location": "Boston"},
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        )
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter.translate_anthropic_messages_to_openai(
        messages=anthropic_messages, model=CACHE_CONTROL_BEDROCK_CONVERSE_MODEL
    )

    assert len(result) == 1
    assert "tool_calls" in result[0]
    assert result[0]["tool_calls"][0]["cache_control"] == {"type": "ephemeral"}


def test_cache_control_preserved_in_tools_for_claude():
    """Cache control should be preserved in tools for Claude models."""
    tools = [
        {
            "name": "get_weather",
            "description": "Get weather for a location",
            "input_schema": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
            },
            "cache_control": {"type": "ephemeral"},
        }
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result, tool_name_mapping = adapter.translate_anthropic_tools_to_openai(
        tools=tools, model=CACHE_CONTROL_BEDROCK_CONVERSE_MODEL
    )

    assert len(result) == 1
    assert result[0]["cache_control"] == {"type": "ephemeral"}
    assert tool_name_mapping == {}  # No truncation needed for short names


def test_cache_control_not_preserved_in_tools_for_non_claude():
    """Cache control should NOT be preserved in tools for non-Claude models."""
    tools = [
        {
            "name": "get_weather",
            "description": "Get weather for a location",
            "input_schema": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
            },
            "cache_control": {"type": "ephemeral"},
        }
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result, tool_name_mapping = adapter.translate_anthropic_tools_to_openai(
        tools=tools, model=CACHE_CONTROL_NON_ANTHROPIC_MODEL
    )

    assert len(result) == 1
    assert "cache_control" not in result[0]


def test_translate_anthropic_tools_to_openai_fills_missing_tool_name():
    """Schema-only tools (no ``name``) must not crash the Converse adapter path."""
    tools = [
        {
            "input_schema": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        },
        {"name": "", "input_schema": {"type": "object", "properties": {}}},
    ]
    adapter = LiteLLMAnthropicMessagesAdapter()
    result, _ = adapter.translate_anthropic_tools_to_openai(tools=tools, model=None)
    assert result[0]["function"]["name"] == "litellm_unnamed_tool_0"
    assert result[1]["function"]["name"] == "litellm_unnamed_tool_1"


def test_translate_anthropic_tools_to_openai_passes_provider_native_tool_dicts_through():
    """Deployment-level provider-native tools (e.g. Gemini googleMaps) must reach the provider transformation verbatim (LIT-6286)."""
    tools = [
        {"googleMaps": {}},
        {"googleSearch": {}},
        {
            "name": "get_weather",
            "input_schema": {"type": "object", "properties": {"location": {"type": "string"}}},
        },
    ]
    adapter = LiteLLMAnthropicMessagesAdapter()
    result, tool_name_mapping = adapter.translate_anthropic_tools_to_openai(tools=tools, model=None)
    assert result[0] == {"googleMaps": {}}
    assert result[1] == {"googleSearch": {}}
    assert result[2]["function"]["name"] == "get_weather"
    assert tool_name_mapping == {}


def test_translate_anthropic_tools_to_openai_passes_openai_function_tools_through():
    """A tool already in OpenAI function format must pass through unchanged instead of becoming litellm_unnamed_tool_N."""
    openai_tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
        },
    }
    adapter = LiteLLMAnthropicMessagesAdapter()
    result, _ = adapter.translate_anthropic_tools_to_openai(tools=[openai_tool], model=None)
    assert result == [openai_tool]


def test_translate_completion_input_params_keeps_provider_native_tools():
    """/v1/messages request translation must keep router-merged provider-native tools in kwargs['tools'] (LIT-6286)."""
    adapter = AnthropicAdapter()
    translated = adapter.translate_completion_input_params(
        {
            "model": "gemini/gemini-2.5-flash",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "coffee shops near Union Square"}],
            "tools": [{"googleMaps": {}}],
        }
    )
    assert translated is not None
    assert translated["tools"] == [{"googleMaps": {}}]


def test_translate_openai_content_to_anthropic_reasoning_content_without_thinking_blocks():
    """
    Test that reasoning_content is converted to thinking block when thinking_blocks is not present.
    This handles providers like OpenRouter that return reasoning_content instead of thinking_blocks.

    Regression test for: OpenRouter models returning reasoning_content in /v1/messages endpoint
    should be converted to Anthropic's thinking block format.
    """
    openai_choices = [
        Choices(
            message=Message(
                role="assistant",
                content='There are **3** "r"s in the word strawberry.',
                reasoning_content="**Considering Letter Frequency**\n\nI've homed in on the specifics: The task focuses on counting the letter 'r'. I've identified the target word, \"strawberry,\" and confirmed my understanding of the letter's location. The first 'r' follows 't', the second after 'e', and the third… well, I'm almost there.\n\n\n**Calculating the Count**\n\nMy analysis is complete! I've confirmed that the letter \"r\" appears three times in \"strawberry.\" The first follows \"t,\" the second \"e,\" and the third immediately follows the second. The count is definitively three.",
            )
        )
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter._translate_openai_content_to_anthropic(choices=openai_choices)

    assert len(result) == 2
    # First block should be thinking block with reasoning_content
    assert result[0]["type"] == "thinking"
    assert "Considering Letter Frequency" in result[0]["thinking"]
    assert "Calculating the Count" in result[0]["thinking"]
    assert result[0]["signature"] is None
    # Second block should be text block with content
    assert result[1]["type"] == "text"
    assert result[1]["text"] == 'There are **3** "r"s in the word strawberry.'


def test_translate_streaming_openai_chunk_to_anthropic_reasoning_content_without_thinking_blocks():
    """
    Test that reasoning_content in streaming chunks is converted to thinking_delta
    when thinking_blocks is not present.

    This handles providers like OpenRouter that return reasoning_content in streaming
    responses without thinking_blocks.
    """
    choices = [
        StreamingChoices(
            finish_reason=None,
            index=0,
            delta=Delta(
                reasoning_content="I need to analyze this carefully...",
                content="",
                role="assistant",
                function_call=None,
                tool_calls=None,
                audio=None,
            ),
            logprobs=None,
        )
    ]

    (
        type_of_content,
        content_block_delta,
    ) = LiteLLMAnthropicMessagesAdapter()._translate_streaming_openai_chunk_to_anthropic(choices=choices)

    assert type_of_content == "thinking_delta"
    assert content_block_delta["type"] == "thinking_delta"
    assert content_block_delta["thinking"] == "I need to analyze this carefully..."


def test_translate_openai_response_to_anthropic_with_reasoning_content_only():
    """
    Test the full response translation when only reasoning_content is present
    (no thinking_blocks).

    This simulates OpenRouter's response format being translated to Anthropic format
    through /v1/messages endpoint.
    """
    openai_response = ModelResponse(
        id="gen-1770027855-HyrqYvLcX8oTLNgfyDob",
        model="gemini-3-flash",
        choices=[
            Choices(
                finish_reason="stop",
                message=Message(
                    role="assistant",
                    content='There are **3** "r"s in the word strawberry.',
                    reasoning_content="**Considering Letter Frequency**\n\nI've homed in on the specifics: The task focuses on counting the letter 'r'.",
                ),
            )
        ],
        usage=Usage(prompt_tokens=13, completion_tokens=138),
    )

    adapter = LiteLLMAnthropicMessagesAdapter()
    anthropic_response = adapter.translate_openai_response_to_anthropic(response=openai_response)

    anthropic_content = anthropic_response.get("content")
    assert anthropic_content is not None
    assert len(anthropic_content) == 2

    # First block should be thinking
    assert anthropic_content[0]["type"] == "thinking"
    assert "Considering Letter Frequency" in anthropic_content[0]["thinking"]
    assert anthropic_content[0].get("signature") is None

    # Second block should be text
    assert anthropic_content[1]["type"] == "text"
    assert anthropic_content[1]["text"] == 'There are **3** "r"s in the word strawberry.'

    assert anthropic_response.get("stop_reason") == "end_turn"


# =====================================================================
# Tool Name Truncation Tests (Issue #17904)
# OpenAI has a 64-character limit for function/tool names
# =====================================================================


def test_truncate_tool_name_short_name():
    """Short tool names should not be truncated."""
    short_name = "get_weather"
    result = truncate_tool_name(short_name)
    assert result == short_name
    assert len(result) <= OPENAI_MAX_TOOL_NAME_LENGTH


def test_truncate_tool_name_exactly_64_chars():
    """Tool names exactly 64 chars should not be truncated."""
    name_64_chars = "a" * 64
    result = truncate_tool_name(name_64_chars)
    assert result == name_64_chars
    assert len(result) == 64


def test_truncate_tool_name_long_name():
    """Long tool names should be truncated with hash suffix."""
    long_name = "computer_tool_with_very_long_name_that_exceeds_openai_64_character_limit_and_keeps_going"
    result = truncate_tool_name(long_name)

    assert len(result) == OPENAI_MAX_TOOL_NAME_LENGTH
    assert result != long_name
    # Should have format: {55-char-prefix}_{8-char-hash}
    assert "_" in result
    parts = result.rsplit("_", 1)
    assert len(parts[0]) == 55
    assert len(parts[1]) == 8


def test_truncate_tool_name_deterministic():
    """Truncation should be deterministic (same input = same output)."""
    long_name = "a_very_long_tool_name_that_needs_to_be_truncated_for_openai_compatibility_reasons"
    result1 = truncate_tool_name(long_name)
    result2 = truncate_tool_name(long_name)
    assert result1 == result2


def test_truncate_tool_name_avoids_collisions():
    """Similar long names should produce different truncated names."""
    name1 = "process_user_data_with_validation_and_error_handling_for_production_environment"
    name2 = "process_user_data_with_validation_and_error_handling_for_staging_environment"

    result1 = truncate_tool_name(name1)
    result2 = truncate_tool_name(name2)

    assert result1 != result2  # Different hashes prevent collision


def test_create_tool_name_mapping_no_long_names():
    """Mapping should be empty when no names need truncation."""
    tools = [
        {"name": "get_weather"},
        {"name": "search_web"},
    ]
    mapping = create_tool_name_mapping(tools)
    assert mapping == {}


def test_create_tool_name_mapping_with_long_names():
    """Mapping should contain entries for truncated names."""
    long_name = "a_very_long_tool_name_that_exceeds_the_64_character_limit_imposed_by_openai"
    tools = [
        {"name": "short_name"},
        {"name": long_name},
    ]
    mapping = create_tool_name_mapping(tools)

    assert len(mapping) == 1
    truncated = truncate_tool_name(long_name)
    assert truncated in mapping
    assert mapping[truncated] == long_name


def test_translate_anthropic_tools_with_long_names():
    """Tools with long names should be truncated and mapped."""
    long_name = "computer_tool_with_very_long_descriptive_name_that_exceeds_openai_limit_completely"
    tools = [
        {
            "name": long_name,
            "description": "A tool with a very long name",
            "input_schema": {"type": "object", "properties": {}},
        }
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result, tool_name_mapping = adapter.translate_anthropic_tools_to_openai(tools=tools, model="gpt-4")

    assert len(result) == 1
    # The tool name should be truncated
    truncated_name = result[0]["function"]["name"]
    assert len(truncated_name) <= 64
    assert truncated_name != long_name
    # Mapping should have the reverse lookup
    assert truncated_name in tool_name_mapping
    assert tool_name_mapping[truncated_name] == long_name


def test_translate_anthropic_tools_mixed_names():
    """Mix of short and long names should work correctly."""
    short_name = "get_weather"
    long_name = "process_complex_data_transformation_with_validation_and_error_handling_pipeline"
    tools = [
        {"name": short_name, "input_schema": {"type": "object"}},
        {"name": long_name, "input_schema": {"type": "object"}},
    ]

    adapter = LiteLLMAnthropicMessagesAdapter()
    result, tool_name_mapping = adapter.translate_anthropic_tools_to_openai(tools=tools, model="gpt-4")

    assert len(result) == 2
    # Short name unchanged
    assert result[0]["function"]["name"] == short_name
    # Long name truncated
    assert result[1]["function"]["name"] != long_name
    assert len(result[1]["function"]["name"]) <= 64
    # Only long name in mapping
    assert len(tool_name_mapping) == 1


def test_translate_openai_response_restores_tool_names():
    """Tool names in responses should be restored to original."""
    original_name = "a_very_long_tool_name_that_needs_truncation_for_openai_api_compatibility"
    truncated_name = truncate_tool_name(original_name)
    tool_name_mapping = {truncated_name: original_name}

    # Create a mock OpenAI response with the truncated name
    response = ModelResponse(
        id="test-id",
        choices=[
            Choices(
                index=0,
                finish_reason="tool_calls",
                message=Message(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ChatCompletionAssistantToolCall(
                            id="call_123",
                            type="function",
                            function=Function(
                                name=truncated_name,
                                arguments='{"arg": "value"}',
                            ),
                        )
                    ],
                ),
            )
        ],
        model="gpt-4",
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter.translate_openai_response_to_anthropic(response=response, tool_name_mapping=tool_name_mapping)

    # Find the tool_use block in the response
    tool_use_blocks = [c for c in result["content"] if c.get("type") == "tool_use"]
    assert len(tool_use_blocks) == 1
    # Name should be restored to original
    assert tool_use_blocks[0]["name"] == original_name


def test_translate_openai_response_to_anthropic_input_tokens_excludes_cached_tokens():
    """
    Regression test: input_tokens in Anthropic format should NOT include cached tokens.

    Issue: v1/messages API was returning incorrect input_token count when using prompt caching.
    The OpenAI format includes cached tokens in prompt_tokens, but Anthropic format should not.

    According to Anthropic's spec:
    - input_tokens = uncached input tokens only
    - cache_read_input_tokens = tokens read from cache

    In OpenAI format:
    - prompt_tokens = all input tokens (including cached)
    - prompt_tokens_details.cached_tokens = cached tokens

    Expected: anthropic.input_tokens = openai.prompt_tokens - openai.prompt_tokens_details.cached_tokens
    """
    from litellm.types.utils import PromptTokensDetailsWrapper

    # Create OpenAI format response with cached tokens
    # Scenario: 100 total prompt tokens, 30 of which are cached
    usage = Usage(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=30),
        cache_read_input_tokens=30,  # Anthropic format cache info
    )

    response = ModelResponse(
        id="test-id",
        choices=[
            Choices(
                index=0,
                finish_reason="stop",
                message=Message(
                    role="assistant",
                    content="Test response",
                ),
            )
        ],
        model="claude-3-sonnet-20240229",
        usage=usage,
    )

    # Convert to Anthropic format
    adapter = LiteLLMAnthropicMessagesAdapter()
    anthropic_response = adapter.translate_openai_response_to_anthropic(
        response=response,
        tool_name_mapping=None,
    )

    # Validate: input_tokens should be 70 (100 - 30 cached), not 100
    assert anthropic_response["usage"]["input_tokens"] == 70, (
        f"Expected input_tokens=70 (100 total - 30 cached), "
        f"but got {anthropic_response['usage']['input_tokens']}. "
        f"input_tokens should NOT include cached tokens per Anthropic spec."
    )
    assert anthropic_response["usage"]["output_tokens"] == 50
    assert anthropic_response["usage"]["cache_read_input_tokens"] == 30


def test_translate_openai_response_to_anthropic_input_tokens_no_cache():
    """
    Regression test: input_tokens should equal prompt_tokens when there are no cached tokens.
    """
    from litellm.types.utils import PromptTokensDetailsWrapper

    # Create OpenAI format response without cached tokens
    usage = Usage(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
    )

    response = ModelResponse(
        id="test-id",
        choices=[
            Choices(
                index=0,
                finish_reason="stop",
                message=Message(
                    role="assistant",
                    content="Test response",
                ),
            )
        ],
        model="claude-3-sonnet-20240229",
        usage=usage,
    )

    # Convert to Anthropic format
    adapter = LiteLLMAnthropicMessagesAdapter()
    anthropic_response = adapter.translate_openai_response_to_anthropic(
        response=response,
        tool_name_mapping=None,
    )

    # Validate: input_tokens should equal prompt_tokens when no caching
    assert anthropic_response["usage"]["input_tokens"] == 100
    assert anthropic_response["usage"]["output_tokens"] == 50


def test_translate_openai_response_to_anthropic_cache_tokens_from_prompt_tokens_details():
    """
    OpenAI/Azure providers set prompt_tokens_details.cached_tokens but not
    _cache_read_input_tokens.  The adapter should populate cache_read_input_tokens
    from prompt_tokens_details.cached_tokens directly.
    """
    from litellm.types.utils import PromptTokensDetailsWrapper

    # OpenAI-style usage: only prompt_tokens_details, no cache_read_input_tokens kwarg
    usage = Usage(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=30),
    )

    response = ModelResponse(
        id="test-id",
        choices=[
            Choices(
                index=0,
                finish_reason="stop",
                message=Message(
                    role="assistant",
                    content="Test response",
                ),
            )
        ],
        model="gpt-4o-2024-08-06",
        usage=usage,
    )

    adapter = LiteLLMAnthropicMessagesAdapter()
    anthropic_response = adapter.translate_openai_response_to_anthropic(
        response=response,
        tool_name_mapping=None,
    )

    assert anthropic_response["usage"]["input_tokens"] == 70
    assert anthropic_response["usage"]["output_tokens"] == 50
    assert anthropic_response["usage"]["cache_read_input_tokens"] == 30


def test_translate_openai_usage_to_anthropic_cache_tokens_from_dict_details_with_integral_floats():
    usage = Usage(
        prompt_tokens=120,
        completion_tokens=50,
        total_tokens=170,
    )
    usage.prompt_tokens_details = {
        "cached_tokens": 30.0,
        "cache_write_tokens": 20.0,
    }

    anthropic_usage = LiteLLMAnthropicMessagesAdapter._translate_openai_usage_to_anthropic_usage_delta(usage)

    assert anthropic_usage["input_tokens"] == 70
    assert anthropic_usage["output_tokens"] == 50
    assert anthropic_usage["cache_read_input_tokens"] == 30
    assert anthropic_usage["cache_creation_input_tokens"] == 20


def test_translate_openai_usage_to_anthropic_ignores_fractional_cache_tokens():
    usage = Usage(
        prompt_tokens=120,
        completion_tokens=50,
        total_tokens=170,
    )
    usage.prompt_tokens_details = {
        "cached_tokens": 30.5,
        "cache_creation_tokens": 20.25,
    }

    anthropic_usage = LiteLLMAnthropicMessagesAdapter._translate_openai_usage_to_anthropic_usage_delta(usage)

    assert anthropic_usage["input_tokens"] == 120
    assert anthropic_usage["output_tokens"] == 50
    assert "cache_read_input_tokens" not in anthropic_usage
    assert "cache_creation_input_tokens" not in anthropic_usage


def test_translate_openai_usage_to_anthropic_ignores_bool_cache_tokens():
    usage = Usage(
        prompt_tokens=120,
        completion_tokens=50,
        total_tokens=170,
    )
    usage.cache_read_input_tokens = True
    usage.cache_creation_input_tokens = True

    anthropic_usage = LiteLLMAnthropicMessagesAdapter._translate_openai_usage_to_anthropic_usage_delta(usage)

    assert anthropic_usage["input_tokens"] == 120
    assert anthropic_usage["output_tokens"] == 50
    assert "cache_read_input_tokens" not in anthropic_usage
    assert "cache_creation_input_tokens" not in anthropic_usage


def test_translate_openai_response_to_anthropic_cache_creation_from_prompt_tokens_details():
    from litellm.types.utils import PromptTokensDetailsWrapper

    usage = Usage(
        prompt_tokens=120,
        completion_tokens=50,
        total_tokens=170,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=30,
            cache_creation_tokens=20,
        ),
    )

    response = ModelResponse(
        id="test-id",
        choices=[
            Choices(
                index=0,
                finish_reason="stop",
                message=Message(
                    role="assistant",
                    content="Test response",
                ),
            )
        ],
        model="gpt-4o-2024-08-06",
        usage=usage,
    )

    adapter = LiteLLMAnthropicMessagesAdapter()
    anthropic_response = adapter.translate_openai_response_to_anthropic(
        response=response,
        tool_name_mapping=None,
    )

    assert anthropic_response["usage"]["input_tokens"] == 70
    assert anthropic_response["usage"]["output_tokens"] == 50
    assert anthropic_response["usage"]["cache_read_input_tokens"] == 30
    assert anthropic_response["usage"]["cache_creation_input_tokens"] == 20


def test_translate_openai_response_to_anthropic_cache_tokens_from_usage_fields():
    usage = Usage(prompt_tokens=120, completion_tokens=50, total_tokens=170)
    usage.cache_read_input_tokens = 30
    usage.cache_creation_input_tokens = 20

    response = ModelResponse(
        id="test-id",
        choices=[
            Choices(
                index=0,
                finish_reason="stop",
                message=Message(
                    role="assistant",
                    content="Test response",
                ),
            )
        ],
        model="claude-3-sonnet-20240229",
        usage=usage,
    )

    adapter = LiteLLMAnthropicMessagesAdapter()
    anthropic_response = adapter.translate_openai_response_to_anthropic(
        response=response,
        tool_name_mapping=None,
    )

    assert anthropic_response["usage"]["input_tokens"] == 70
    assert anthropic_response["usage"]["output_tokens"] == 50
    assert anthropic_response["usage"]["cache_read_input_tokens"] == 30
    assert anthropic_response["usage"]["cache_creation_input_tokens"] == 20


def test_translate_openai_response_to_anthropic_cache_tokens_from_private_usage_fields():
    usage = Usage(prompt_tokens=120, completion_tokens=50, total_tokens=170)

    response = ModelResponse(
        id="test-id",
        choices=[
            Choices(
                index=0,
                finish_reason="stop",
                message=Message(
                    role="assistant",
                    content="Test response",
                ),
            )
        ],
        model="claude-3-sonnet-20240229",
        usage=usage,
    )
    response.usage._cache_read_input_tokens = 30
    response.usage._cache_creation_input_tokens = 20

    adapter = LiteLLMAnthropicMessagesAdapter()
    anthropic_response = adapter.translate_openai_response_to_anthropic(
        response=response,
        tool_name_mapping=None,
    )

    assert anthropic_response["usage"]["input_tokens"] == 70
    assert anthropic_response["usage"]["output_tokens"] == 50
    assert anthropic_response["usage"]["cache_read_input_tokens"] == 30
    assert anthropic_response["usage"]["cache_creation_input_tokens"] == 20


def test_translate_streaming_openai_response_to_anthropic_cache_tokens_from_prompt_tokens_details():
    from litellm.types.utils import PromptTokensDetailsWrapper

    usage = Usage(
        prompt_tokens=120,
        completion_tokens=50,
        total_tokens=170,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=30,
            cache_creation_tokens=20,
        ),
    )
    response = ModelResponseStream(
        choices=[
            StreamingChoices(
                index=0,
                delta=Delta(),
                finish_reason="stop",
            )
        ],
        usage=usage,
    )

    adapter = LiteLLMAnthropicMessagesAdapter()
    message_delta = adapter.translate_streaming_openai_response_to_anthropic(
        response=response,
        current_content_block_index=0,
    )

    assert message_delta["usage"]["input_tokens"] == 70
    assert message_delta["usage"]["output_tokens"] == 50
    assert message_delta["usage"]["cache_read_input_tokens"] == 30
    assert message_delta["usage"]["cache_creation_input_tokens"] == 20


def test_translate_streaming_openai_response_to_anthropic_cache_tokens_from_hidden_params_usage():
    from litellm.types.utils import PromptTokensDetailsWrapper

    usage = Usage(
        prompt_tokens=120,
        completion_tokens=50,
        total_tokens=170,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=30,
            cache_creation_tokens=20,
        ),
    )
    response = ModelResponseStream(
        choices=[
            StreamingChoices(
                index=0,
                delta=Delta(),
                finish_reason="stop",
            )
        ],
    )
    response._hidden_params = {"usage": usage}

    adapter = LiteLLMAnthropicMessagesAdapter()
    message_delta = adapter.translate_streaming_openai_response_to_anthropic(
        response=response,
        current_content_block_index=0,
    )

    assert message_delta["usage"]["input_tokens"] == 70
    assert message_delta["usage"]["output_tokens"] == 50
    assert message_delta["usage"]["cache_read_input_tokens"] == 30
    assert message_delta["usage"]["cache_creation_input_tokens"] == 20


def test_translate_streaming_openai_response_to_anthropic_cache_tokens_with_applied_edits():
    from litellm.types.utils import PromptTokensDetailsWrapper

    usage = Usage(
        prompt_tokens=120,
        completion_tokens=50,
        total_tokens=170,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=30,
            cache_creation_tokens=20,
        ),
    )
    response = ModelResponseStream(
        choices=[
            StreamingChoices(
                index=0,
                delta=Delta(),
                finish_reason="stop",
            )
        ],
        usage=usage,
    )

    adapter = LiteLLMAnthropicMessagesAdapter()
    message_delta = adapter.translate_streaming_openai_response_to_anthropic(
        response=response,
        current_content_block_index=0,
        applied_edits=[{"type": "compact_20260112"}],
    )

    assert message_delta["usage"]["input_tokens"] == 70
    assert message_delta["usage"]["output_tokens"] == 50
    assert message_delta["usage"]["cache_read_input_tokens"] == 30
    assert message_delta["usage"]["cache_creation_input_tokens"] == 20
    assert message_delta["context_management"]["applied_edits"][0]["type"] == ("compact_20260112")


# =====================================================================
# Web Search Tool Transformation Tests
# =====================================================================


def test_is_web_search_tool():
    """Test detection of Anthropic web search tools."""
    adapter = LiteLLMAnthropicMessagesAdapter()

    # Tool with type starting with "web_search" should be detected
    web_search_tool_with_type = {
        "type": "web_search_20260209",
        "name": "web_search",
    }
    assert adapter._is_web_search_tool(web_search_tool_with_type) is True

    # Tool with name "web_search" should be detected
    web_search_tool_with_name = {
        "name": "web_search",
    }
    assert adapter._is_web_search_tool(web_search_tool_with_name) is True

    # Regular function tool should not be detected
    regular_tool = {
        "name": "get_weather",
        "description": "Get weather info",
        "input_schema": {"type": "object"},
    }
    assert adapter._is_web_search_tool(regular_tool) is False


def test_translate_anthropic_to_openai_with_web_search_tool():
    """
    Test that Anthropic web search tools are converted to web_search_options parameter.

    When a user sends an Anthropic /v1/messages request with {"type": "web_search_20260209"}
    tool, it should be transformed to OpenAI format with web_search_options: {} parameter.
    """
    from litellm.types.llms.anthropic import AnthropicMessagesRequest

    anthropic_request = AnthropicMessagesRequest(
        model="gemini-2.5-flash-lite",
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": "Search for the current prices of AAPL and GOOGL",
            }
        ],
        tools=[
            {
                "type": "web_search_20260209",
                "name": "web_search",
            }
        ],
    )

    adapter = LiteLLMAnthropicMessagesAdapter()
    openai_request, tool_name_mapping = adapter.translate_anthropic_to_openai(
        anthropic_message_request=anthropic_request
    )

    # web_search_options should be added
    assert "web_search_options" in openai_request
    assert openai_request["web_search_options"] == {}

    # web search tool should NOT be in the tools array
    assert "tools" not in openai_request or openai_request.get("tools") == []

    # tool_name_mapping should be empty since no regular tools were present
    assert tool_name_mapping == {}


def test_translate_anthropic_to_openai_with_mixed_tools():
    """
    Test that web search tools are separated from regular tools.

    When a request has both web search tools and regular function tools,
    only the regular tools should be in the tools array, and web_search_options
    should be added.
    """
    from litellm.types.llms.anthropic import AnthropicMessagesRequest

    anthropic_request = AnthropicMessagesRequest(
        model="gemini-2.5-flash-lite",
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": "Get weather and search the web",
            }
        ],
        tools=[
            {
                "type": "web_search_20260209",
                "name": "web_search",
            },
            {
                "name": "get_weather",
                "description": "Get weather information",
                "input_schema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            },
        ],
    )

    adapter = LiteLLMAnthropicMessagesAdapter()
    openai_request, tool_name_mapping = adapter.translate_anthropic_to_openai(
        anthropic_message_request=anthropic_request
    )

    # web_search_options should be added
    assert "web_search_options" in openai_request
    assert openai_request["web_search_options"] == {}

    # Only get_weather tool should be in the tools array
    assert "tools" in openai_request
    assert len(openai_request["tools"]) == 1
    assert openai_request["tools"][0]["function"]["name"] == "get_weather"

    # tool_name_mapping should be empty for short tool names
    assert tool_name_mapping == {}


class TestTranslateAnthropicOutputFormatToOpenAI:
    """Tests for translate_anthropic_output_format_to_openai adding additionalProperties: false."""

    def setup_method(self):
        self.adapter = LiteLLMAnthropicMessagesAdapter()

    def test_simple_object_adds_additional_properties_false(self):
        output_format = {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
            },
        }
        result = self.adapter.translate_anthropic_output_format_to_openai(output_format)
        assert result is not None
        schema = result["json_schema"]["schema"]
        assert schema["additionalProperties"] is False
        assert schema["required"] == ["name"]

    def test_nested_objects_adds_additional_properties_false(self):
        output_format = {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "user": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "address": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                            },
                        },
                    }
                },
            },
        }
        result = self.adapter.translate_anthropic_output_format_to_openai(output_format)
        assert result is not None
        schema = result["json_schema"]["schema"]
        assert schema["additionalProperties"] is False
        assert schema["required"] == ["user"]
        assert schema["properties"]["user"]["additionalProperties"] is False
        assert schema["properties"]["user"]["required"] == ["name", "address"]
        assert schema["properties"]["user"]["properties"]["address"]["additionalProperties"] is False
        assert schema["properties"]["user"]["properties"]["address"]["required"] == ["city"]

    def test_array_items_object_adds_additional_properties_false(self):
        output_format = {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"id": {"type": "integer"}},
                        },
                    }
                },
            },
        }
        result = self.adapter.translate_anthropic_output_format_to_openai(output_format)
        assert result is not None
        schema = result["json_schema"]["schema"]
        assert schema["additionalProperties"] is False
        assert schema["properties"]["items"]["items"]["additionalProperties"] is False

    def test_does_not_mutate_original_schema(self):
        original_schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        output_format = {"type": "json_schema", "schema": original_schema}
        self.adapter.translate_anthropic_output_format_to_openai(output_format)
        assert "additionalProperties" not in original_schema
        assert "required" not in original_schema

    def test_defs_adds_additional_properties_false(self):
        output_format = {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {"ref": {"$ref": "#/$defs/Item"}},
                "$defs": {
                    "Item": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                    }
                },
            },
        }
        result = self.adapter.translate_anthropic_output_format_to_openai(output_format)
        assert result is not None
        schema = result["json_schema"]["schema"]
        assert schema["$defs"]["Item"]["additionalProperties"] is False
        assert schema["$defs"]["Item"]["required"] == ["value"]

    def test_incomplete_required_gets_completed(self):
        """OpenAI strict mode requires ALL properties in required."""
        output_format = {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "email": {"type": "string"},
                },
                "required": ["name"],  # only 1 of 3
            },
        }
        result = self.adapter.translate_anthropic_output_format_to_openai(output_format)
        assert result is not None
        schema = result["json_schema"]["schema"]
        assert schema["additionalProperties"] is False
        assert sorted(schema["required"]) == ["age", "email", "name"]

    def test_invalid_output_format_returns_none(self):
        assert self.adapter.translate_anthropic_output_format_to_openai("invalid") is None
        assert self.adapter.translate_anthropic_output_format_to_openai({"type": "text"}) is None
        assert self.adapter.translate_anthropic_output_format_to_openai({"type": "json_schema"}) is None


class TestAnthropicStreamWrapperToolArgs:
    """
    Regression test for https://github.com/BerriAI/litellm/issues/24134

    When Gemini sends tool call args in the same streaming chunk as a content
    block transition, the Anthropic adapter was discarding the processed_chunk
    containing input_json_delta. This verifies the args are preserved.
    """

    def _build_chunks(self):
        """Build mock OpenAI-format chunks simulating Gemini tool call response."""
        # Chunk 1: text content
        text_chunk = ModelResponseStream(
            id="chatcmpl-123",
            created=1700000000,
            model="gemini-2.0-flash",
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(content="Let me check", role="assistant"),
                    finish_reason=None,
                )
            ],
        )

        # Chunk 2: tool call (triggers new content block + carries args)
        tool_chunk = ModelResponseStream(
            id="chatcmpl-123",
            created=1700000000,
            model="gemini-2.0-flash",
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(
                        tool_calls=[
                            ChatCompletionDeltaToolCall(
                                id="call_123",
                                type="function",
                                function=Function(
                                    name="get_weather",
                                    arguments='{"city": "Tokyo"}',
                                ),
                                index=0,
                            )
                        ]
                    ),
                    finish_reason=None,
                )
            ],
        )

        # Chunk 3: finish
        finish_chunk = ModelResponseStream(
            id="chatcmpl-123",
            created=1700000000,
            model="gemini-2.0-flash",
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

        return [text_chunk, tool_chunk, finish_chunk]

    def _make_stream_wrapper(self, chunks):
        from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
            AnthropicStreamWrapper,
        )

        class SimpleIterator:
            def __init__(self, items):
                self._items = iter(items)

            def __iter__(self):
                return self

            def __next__(self):
                return next(self._items)

            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    return next(self._items)
                except StopIteration:
                    raise StopAsyncIteration

        return AnthropicStreamWrapper(
            completion_stream=SimpleIterator(chunks),
            model="gemini/gemini-2.0-flash",
        )

    def _find_tool_deltas(self, events):
        return [
            e
            for e in events
            if isinstance(e, dict)
            and e.get("type") == "content_block_delta"
            and isinstance(e.get("delta"), dict)
            and e["delta"].get("type") == "input_json_delta"
        ]

    def test_sync_tool_args_not_dropped(self):
        import json

        chunks = self._build_chunks()
        wrapper = self._make_stream_wrapper(chunks)

        events = list(wrapper)
        tool_deltas = self._find_tool_deltas(events)

        assert len(tool_deltas) > 0, (
            f"No input_json_delta events found (issue #24134). "
            f"Event types: {[e.get('type') for e in events if isinstance(e, dict)]}"
        )

        combined = "".join(d["delta"]["partial_json"] for d in tool_deltas)
        parsed = json.loads(combined)
        assert parsed == {"city": "Tokyo"}

    @pytest.mark.asyncio
    async def test_async_tool_args_not_dropped(self):
        import json

        chunks = self._build_chunks()
        wrapper = self._make_stream_wrapper(chunks)

        events = []
        async for event in wrapper:
            events.append(event)

        tool_deltas = self._find_tool_deltas(events)

        assert len(tool_deltas) > 0, (
            f"No input_json_delta events found (issue #24134). "
            f"Event types: {[e.get('type') for e in events if isinstance(e, dict)]}"
        )

        combined = "".join(d["delta"]["partial_json"] for d in tool_deltas)
        parsed = json.loads(combined)
        assert parsed == {"city": "Tokyo"}


def test_translate_anthropic_tool_choice_none():
    """
    Regression test for issue #24443.

    tool_choice={"type": "none"} should be translated to "none" for OpenAI format,
    not raise a ValueError.
    """
    adapter = LiteLLMAnthropicMessagesAdapter()

    result = adapter.translate_anthropic_tool_choice_to_openai({"type": "none"})
    assert result == "none"


# ---------------------------------------------------------------------------
# PolyfillResult integration tests
# ---------------------------------------------------------------------------


def _make_simple_openai_response(
    text: str = "Hello", prompt_tokens: int = 10, completion_tokens: int = 5
) -> ModelResponse:
    return ModelResponse(
        id="resp_polyfill_test",
        model="gpt-4o",
        choices=[
            Choices(
                finish_reason="stop",
                message=Message(role="assistant", content=text),
            )
        ],
        usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


def test_translate_openai_response_to_anthropic_with_polyfill_compaction_block():
    """compaction_block from PolyfillResult must be prepended to content at index 0."""
    from litellm.llms.anthropic.experimental_pass_through.context_management.result import (
        PolyfillResult,
    )

    compaction_block = {"type": "compaction", "content": "Summary of prior turns."}
    polyfill = PolyfillResult(
        messages=[],
        system=None,
        applied_edits=[{"type": "compact_20260112"}],
        compaction_block=compaction_block,
        iterations_usage=None,
    )
    response = _make_simple_openai_response(text="Hello after compaction.")
    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter.translate_openai_response_to_anthropic(response=response, polyfill_result=polyfill)

    content = result.get("content")
    assert content is not None
    assert content[0]["type"] == "compaction"
    assert content[0]["content"] == "Summary of prior turns."
    assert content[1]["type"] == "text"
    assert content[1]["text"] == "Hello after compaction."

    # applied_edits must surface on context_management
    cm = result.get("context_management")
    assert cm is not None
    assert cm["applied_edits"][0]["type"] == "compact_20260112"


def test_translate_openai_response_to_anthropic_with_polyfill_iterations_usage():
    """iterations_usage from PolyfillResult must produce usage['iterations'] with a message entry."""
    from litellm.llms.anthropic.experimental_pass_through.context_management.result import (
        PolyfillResult,
    )

    polyfill = PolyfillResult(
        messages=[],
        system=None,
        applied_edits=[{"type": "compact_20260112"}],
        compaction_block=None,
        iterations_usage=[
            {"type": "compaction", "input_tokens": 200, "output_tokens": 50},
        ],
    )
    response = _make_simple_openai_response(prompt_tokens=100, completion_tokens=30)
    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter.translate_openai_response_to_anthropic(response=response, polyfill_result=polyfill)

    usage = result.get("usage")
    assert usage is not None
    iterations = usage.get("iterations")
    assert iterations is not None
    assert len(iterations) == 2
    assert iterations[0] == {
        "type": "compaction",
        "input_tokens": 200,
        "output_tokens": 50,
    }
    assert iterations[1]["type"] == "message"
    assert iterations[1]["input_tokens"] == 100
    assert iterations[1]["output_tokens"] == 30

    # Top-level tokens must still reflect the message iteration
    assert usage["input_tokens"] == 100
    assert usage["output_tokens"] == 30


def test_translate_openai_response_to_anthropic_no_polyfill_no_change():
    """Without a PolyfillResult the response must be unchanged (no compaction, no iterations)."""
    response = _make_simple_openai_response()
    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter.translate_openai_response_to_anthropic(response=response)

    content = result.get("content")
    assert content is not None
    assert content[0]["type"] == "text"

    usage = result.get("usage")
    assert usage is not None
    assert "iterations" not in usage


def test_translate_openai_response_to_anthropic_with_polyfill_both_compaction_and_iterations():
    """Full summary path: compaction_block and iterations_usage both present simultaneously."""
    from litellm.llms.anthropic.experimental_pass_through.context_management.result import (
        PolyfillResult,
    )

    compaction_block = {
        "type": "compaction",
        "content": "Summary of a long conversation.",
    }
    polyfill = PolyfillResult(
        messages=[],
        system=None,
        applied_edits=[{"type": "compact_20260112"}],
        compaction_block=compaction_block,
        iterations_usage=[
            {"type": "compaction", "input_tokens": 300, "output_tokens": 75},
        ],
    )
    response = _make_simple_openai_response(text="After compaction.", prompt_tokens=120, completion_tokens=40)
    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter.translate_openai_response_to_anthropic(response=response, polyfill_result=polyfill)

    # compaction block must come first
    content = result.get("content")
    assert content is not None
    assert content[0]["type"] == "compaction"
    assert content[0]["content"] == "Summary of a long conversation."
    assert content[1]["type"] == "text"
    assert content[1]["text"] == "After compaction."

    # iterations: compaction entry + message entry
    usage = result.get("usage")
    assert usage is not None
    iterations = usage.get("iterations")
    assert iterations is not None
    assert len(iterations) == 2
    assert iterations[0] == {
        "type": "compaction",
        "input_tokens": 300,
        "output_tokens": 75,
    }
    assert iterations[1]["type"] == "message"
    assert iterations[1]["input_tokens"] == 120
    assert iterations[1]["output_tokens"] == 40

    # top-level tokens match the message iteration
    assert usage["input_tokens"] == 120
    assert usage["output_tokens"] == 40

    # context_management applied_edits must surface
    cm = result.get("context_management")
    assert cm is not None
    assert cm["applied_edits"][0]["type"] == "compact_20260112"


def test_translate_anthropic_tools_to_openai_preserves_parameters_type():
    """Regression for #30557: the Anthropic tool `type` ("custom") must not be
    merged into the OpenAI function `parameters`, overwriting parameters.type."""
    adapter = LiteLLMAnthropicMessagesAdapter()
    tools = [
        {
            "type": "custom",
            "name": "get_weather",
            "description": "Get weather",
            "input_schema": {"type": "object", "properties": {}},
        }
    ]

    new_tools, _ = adapter.translate_anthropic_tools_to_openai(tools=tools)

    params = new_tools[0]["function"]["parameters"]
    assert params["type"] == "object"
    assert new_tools[0]["type"] == "function"


def test_translate_anthropic_tools_to_openai_maps_strict_onto_function_not_parameters():
    """A tool-level `strict` lands on the OpenAI function, leaving the caller's `input_schema` untouched."""
    adapter = LiteLLMAnthropicMessagesAdapter()
    input_schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": False,
    }
    tools = [{"type": "custom", "name": "get_weather", "strict": True, "input_schema": input_schema}]

    new_tools, _ = adapter.translate_anthropic_tools_to_openai(tools=tools)

    function = new_tools[0]["function"]
    assert function["strict"] is True
    assert "strict" not in function["parameters"]
    assert input_schema == {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": False,
    }


def test_translate_anthropic_tools_to_openai_omits_unset_strict():
    """Chat Completions already defaults to non-strict, so an unset `strict` stays unset."""
    adapter = LiteLLMAnthropicMessagesAdapter()
    tools = [
        {
            "type": "custom",
            "name": "search",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "cursor": {"type": "string"}},
                "required": ["query"],
            },
        }
    ]

    new_tools, _ = adapter.translate_anthropic_tools_to_openai(tools=tools)

    function = new_tools[0]["function"]
    assert "strict" not in function
    assert "strict" not in function["parameters"]
    assert function["parameters"]["required"] == ["query"]


TOOL_RESULT_IMAGE_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)
TOOL_RESULT_IMAGE_URL = "https://example.com/screenshot.png"


def _anthropic_tool_use_turn(*tool_use_ids):
    return AnthopicMessagesAssistantMessageParam(
        role="assistant",
        content=[
            {"type": "tool_use", "id": tid, "name": "read_file", "input": {"path": "img.png"}} for tid in tool_use_ids
        ],
    )


def _anthropic_tool_result_turn(blocks_by_tool_use_id):
    return AnthropicMessagesUserMessageParam(
        role="user",
        content=[
            {"type": "tool_result", "tool_use_id": tid, "content": blocks}
            for tid, blocks in blocks_by_tool_use_id.items()
        ],
    )


def _base64_image_block():
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": TOOL_RESULT_IMAGE_B64},
    }


def _url_image_block():
    return {"type": "image", "source": {"type": "url", "url": TOOL_RESULT_IMAGE_URL}}


def _run_chat_completions_pipeline(anthropic_messages):
    """Anthropic /v1/messages input -> chat adapter -> the OpenAI-compatible
    request transformation every OpenAIGPTConfig-based provider runs."""
    adapter = LiteLLMAnthropicMessagesAdapter()
    translated = adapter.translate_anthropic_messages_to_openai(messages=anthropic_messages)
    request = OpenAIGPTConfig().transform_request(
        model="gpt-5.4-mini", messages=translated, optional_params={}, litellm_params={}, headers={}
    )
    return request["messages"]


def _images_in_tool_messages(messages):
    found = []
    for message in messages:
        if message.get("role") != "tool":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.startswith("data:image"):
            found.append(content)
        elif isinstance(content, list):
            found.extend(p for p in content if isinstance(p, dict) and p.get("type") == "image_url")
    return found


def _image_urls_in_user_messages(messages):
    return [
        part["image_url"]["url"]
        for message in messages
        if message.get("role") == "user" and isinstance(message.get("content"), list)
        for part in message["content"]
        if isinstance(part, dict) and part.get("type") == "image_url"
    ]


@pytest.mark.parametrize(
    "image_block,expected_url_prefix",
    [
        (_base64_image_block(), "data:image/png;base64,"),
        (_url_image_block(), TOOL_RESULT_IMAGE_URL),
    ],
    ids=["base64_source", "url_source"],
)
def test_tool_result_single_image_visible_after_openai_transform(image_block, expected_url_prefix):
    result = _run_chat_completions_pipeline(
        [
            _anthropic_tool_use_turn("toolu_01"),
            _anthropic_tool_result_turn({"toolu_01": [image_block]}),
        ]
    )

    assert _images_in_tool_messages(result) == []
    user_image_urls = _image_urls_in_user_messages(result)
    assert len(user_image_urls) == 1
    assert user_image_urls[0].startswith(expected_url_prefix)

    tool_messages = [m for m in result if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "toolu_01"
    assert tool_messages[0]["content"] == TOOL_RESULT_IMAGE_PLACEHOLDER


def test_tool_result_text_and_image_visible_after_openai_transform():
    result = _run_chat_completions_pipeline(
        [
            _anthropic_tool_use_turn("toolu_01"),
            _anthropic_tool_result_turn(
                {"toolu_01": [{"type": "text", "text": "screenshot saved"}, _base64_image_block()]}
            ),
        ]
    )

    assert _images_in_tool_messages(result) == []
    assert len(_image_urls_in_user_messages(result)) == 1

    tool_messages = [m for m in result if m.get("role") == "tool"]
    assert tool_messages[0]["content"] == [{"type": "text", "text": "screenshot saved"}]


def test_tool_result_two_images_visible_after_openai_transform():
    result = _run_chat_completions_pipeline(
        [
            _anthropic_tool_use_turn("toolu_01"),
            _anthropic_tool_result_turn({"toolu_01": [_base64_image_block(), _base64_image_block()]}),
        ]
    )

    assert _images_in_tool_messages(result) == []
    assert len(_image_urls_in_user_messages(result)) == 2


def test_tool_result_parallel_tool_calls_keep_tool_message_adjacency():
    result = _run_chat_completions_pipeline(
        [
            _anthropic_tool_use_turn("toolu_01", "toolu_02"),
            _anthropic_tool_result_turn({"toolu_01": [_base64_image_block()], "toolu_02": [_url_image_block()]}),
        ]
    )

    roles = [m.get("role") for m in result]
    assert roles == ["assistant", "tool", "tool", "user"]
    assert _images_in_tool_messages(result) == []
    assert len(_image_urls_in_user_messages(result)) == 2


@pytest.mark.parametrize(
    "image_block",
    [
        {"type": "image", "source": {"type": "unsupported"}},
        {"type": "image"},
        {"type": "image", "source": "https://example.com/screenshot.png"},
    ],
    ids=["untranslatable_source", "missing_source", "non_dict_source"],
)
def test_tool_result_malformed_image_source_keeps_empty_tool_content(image_block):
    adapter = LiteLLMAnthropicMessagesAdapter()
    translated = adapter.translate_anthropic_messages_to_openai(
        messages=[
            _anthropic_tool_use_turn("toolu_01"),
            _anthropic_tool_result_turn({"toolu_01": [image_block]}),
        ]
    )

    tool_messages = [m for m in translated if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["content"] == ""


def test_tool_result_plain_text_unchanged_by_openai_transform():
    result = _run_chat_completions_pipeline(
        [
            _anthropic_tool_use_turn("toolu_01"),
            _anthropic_tool_result_turn({"toolu_01": [{"type": "text", "text": "42 files found"}]}),
        ]
    )

    tool_messages = [m for m in result if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["content"] == "42 files found"
    assert _image_urls_in_user_messages(result) == []


TOOL_RESULT_PDF_B64 = base64.b64encode(b"%PDF-1.4 minimal regression fixture").decode()


def _base64_pdf_block():
    return {
        "type": "document",
        "source": {"type": "base64", "media_type": "application/pdf", "data": TOOL_RESULT_PDF_B64},
    }


def test_tool_result_single_document_kept_as_pdf_data_url():
    adapter = LiteLLMAnthropicMessagesAdapter()
    translated = adapter.translate_anthropic_messages_to_openai(
        messages=[
            _anthropic_tool_use_turn("toolu_01"),
            _anthropic_tool_result_turn({"toolu_01": [_base64_pdf_block()]}),
        ]
    )

    tool_messages = [m for m in translated if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["content"] == [
        {
            "type": "image_url",
            "image_url": {"url": f"data:application/pdf;base64,{TOOL_RESULT_PDF_B64}"},
        }
    ]


def test_tool_result_text_and_document_reach_bedrock_converse_tool_result():
    """Claude Code >= 2.1.245 sends Read-tool PDF output as a document block inside
    tool_result; dropping it left bedrock converse models blind to the PDF content."""
    adapter = LiteLLMAnthropicMessagesAdapter()
    translated = adapter.translate_anthropic_messages_to_openai(
        messages=[
            AnthropicMessagesUserMessageParam(role="user", content="Read pong.pdf"),
            _anthropic_tool_use_turn("toolu_01"),
            _anthropic_tool_result_turn(
                {
                    "toolu_01": [
                        {"type": "text", "text": "PDF file read: pong.pdf (579 bytes)"},
                        _base64_pdf_block(),
                    ]
                }
            ),
        ]
    )

    converse_messages = _bedrock_converse_messages_pt(
        messages=translated,
        model="anthropic.claude-haiku-4-5-20251001-v1:0",
        llm_provider="bedrock_converse",
    )

    tool_results = [
        block["toolResult"]
        for message in converse_messages
        for block in message["content"]
        if "toolResult" in block
    ]
    assert len(tool_results) == 1
    documents = [part["document"] for part in tool_results[0]["content"] if "document" in part]
    assert len(documents) == 1
    assert documents[0]["format"] == "pdf"
    assert documents[0]["source"]["bytes"] == TOOL_RESULT_PDF_B64
    texts = [part["text"] for part in tool_results[0]["content"] if "text" in part]
    assert texts == ["PDF file read: pong.pdf (579 bytes)"]


def test_translate_anthropic_to_openai_carries_prompt_cache_breakpoint_on_system_and_user_blocks():
    explicit = {"mode": "explicit"}
    openai_request, _ = LiteLLMAnthropicMessagesAdapter().translate_anthropic_to_openai(
        anthropic_message_request={
            "model": "gpt-5.6",
            "max_tokens": 64,
            "system": [{"type": "text", "text": "sys", "prompt_cache_breakpoint": explicit}],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hi", "prompt_cache_breakpoint": explicit},
                        {
                            "type": "image",
                            "source": {"type": "url", "url": "https://example.com/a.png"},
                            "prompt_cache_breakpoint": explicit,
                        },
                    ],
                }
            ],
        }
    )
    assert openai_request["messages"][0] == {
        "role": "system",
        "content": [{"type": "text", "text": "sys", "prompt_cache_breakpoint": explicit}],
    }
    user_content = openai_request["messages"][1]["content"]
    assert user_content[0] == {"type": "text", "text": "hi", "prompt_cache_breakpoint": explicit}
    assert user_content[1]["type"] == "image_url"
    assert user_content[1]["prompt_cache_breakpoint"] == explicit


def test_translate_anthropic_to_openai_without_prompt_cache_breakpoint_adds_nothing():
    openai_request, _ = LiteLLMAnthropicMessagesAdapter().translate_anthropic_to_openai(
        anthropic_message_request={
            "model": "gpt-5.6",
            "max_tokens": 64,
            "system": [{"type": "text", "text": "sys"}],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        }
    )
    assert openai_request["messages"][0] == {"role": "system", "content": [{"type": "text", "text": "sys"}]}
    assert openai_request["messages"][1]["content"] == [{"type": "text", "text": "hi"}]


def test_translate_anthropic_messages_to_openai_carries_midturn_system_prompt_cache_breakpoint():
    explicit = {"mode": "explicit"}
    result = LiteLLMAnthropicMessagesAdapter().translate_anthropic_messages_to_openai(
        messages=[{"role": "system", "content": [{"type": "text", "text": "fix", "prompt_cache_breakpoint": explicit}]}],
        model="gpt-5.6",
    )
    assert result == [
        {"role": "system", "content": [{"type": "text", "text": "fix", "prompt_cache_breakpoint": explicit}]}
    ]


def _tool_reference_block(tool_name="WebFetch"):
    return {"type": "tool_reference", "tool_name": tool_name}


def test_tool_result_tool_reference_is_carried_through_untouched():
    adapter = LiteLLMAnthropicMessagesAdapter()

    result = adapter.translate_anthropic_messages_to_openai(
        messages=[
            _anthropic_tool_use_turn("toolu_01"),
            _anthropic_tool_result_turn({"toolu_01": [_tool_reference_block()]}),
        ]
    )

    assert [m["role"] for m in result] == ["assistant", "tool"]
    assert result[1]["tool_call_id"] == "toolu_01"
    assert result[1]["content"] == [{"type": "tool_reference", "tool_name": "WebFetch"}]


def test_tool_result_text_beside_tool_reference_keeps_both_parts_in_order():
    adapter = LiteLLMAnthropicMessagesAdapter()

    result = adapter.translate_anthropic_messages_to_openai(
        messages=[
            _anthropic_tool_use_turn("toolu_01"),
            _anthropic_tool_result_turn(
                {"toolu_01": [{"type": "text", "text": "loaded"}, _tool_reference_block("Grep")]}
            ),
        ]
    )

    assert result[1]["content"] == [
        {"type": "text", "text": "loaded"},
        {"type": "tool_reference", "tool_name": "Grep"},
    ]


@pytest.mark.parametrize(
    "tool_result_content",
    [
        [],
        None,
        "",
        {"not": "a list"},
        [{"type": "future_block", "payload": 1}],
        [{"type": "search_result", "source": "https://example.com", "title": "t", "content": []}],
    ],
    ids=["empty_list", "null", "empty_string", "non_list", "unknown_block", "search_result_only"],
)
def test_tool_result_without_translatable_content_still_answers_its_tool_use(tool_result_content):
    adapter = LiteLLMAnthropicMessagesAdapter()

    result = adapter.translate_anthropic_messages_to_openai(
        messages=[
            _anthropic_tool_use_turn("toolu_01"),
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_01", "content": tool_result_content}],
            },
        ]
    )

    assert result == [
        result[0],
        {"role": "tool", "tool_call_id": "toolu_01", "content": ""},
    ]
    assert result[0]["role"] == "assistant"


def _openai_response_with_usage(usage: Usage) -> ModelResponse:
    return ModelResponse(
        id="resp_web_search",
        model="gemini-3-flash-preview",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(role="assistant", content="searched"),
            )
        ],
        usage=usage,
    )


def test_translate_openai_response_to_anthropic_maps_gemini_web_search_usage():
    from litellm.types.utils import PromptTokensDetailsWrapper

    usage = Usage(
        prompt_tokens=385,
        completion_tokens=566,
        total_tokens=951,
        prompt_tokens_details=PromptTokensDetailsWrapper(text_tokens=385, web_search_requests=2),
    )

    anthropic_response = LiteLLMAnthropicMessagesAdapter().translate_openai_response_to_anthropic(
        response=_openai_response_with_usage(usage)
    )

    assert anthropic_response["usage"]["server_tool_use"] == {"web_search_requests": 2}


def test_translate_openai_response_to_anthropic_maps_server_tool_use_web_search_usage():
    from litellm.types.utils import ServerToolUse

    usage = Usage(
        prompt_tokens=100,
        completion_tokens=40,
        total_tokens=140,
        server_tool_use=ServerToolUse(web_search_requests=3),
    )

    anthropic_response = LiteLLMAnthropicMessagesAdapter().translate_openai_response_to_anthropic(
        response=_openai_response_with_usage(usage)
    )

    assert anthropic_response["usage"]["server_tool_use"] == {"web_search_requests": 3}


def test_translate_openai_response_to_anthropic_omits_server_tool_use_without_web_search():
    usage = Usage(prompt_tokens=100, completion_tokens=40, total_tokens=140)

    anthropic_response = LiteLLMAnthropicMessagesAdapter().translate_openai_response_to_anthropic(
        response=_openai_response_with_usage(usage)
    )

    assert "server_tool_use" not in anthropic_response["usage"]


def test_completion_cost_on_translated_anthropic_response_includes_web_search():
    from litellm.types.utils import PromptTokensDetailsWrapper

    adapter = LiteLLMAnthropicMessagesAdapter()
    with_search = adapter.translate_openai_response_to_anthropic(
        response=_openai_response_with_usage(
            Usage(
                prompt_tokens=385,
                completion_tokens=566,
                total_tokens=951,
                prompt_tokens_details=PromptTokensDetailsWrapper(text_tokens=385, web_search_requests=2),
            )
        )
    )
    without_search = adapter.translate_openai_response_to_anthropic(
        response=_openai_response_with_usage(Usage(prompt_tokens=385, completion_tokens=566, total_tokens=951))
    )

    cost_with_search = litellm.completion_cost(
        completion_response=with_search,
        model="gemini/gemini-3-flash-preview",
        call_type="anthropic_messages",
    )
    cost_without_search = litellm.completion_cost(
        completion_response=without_search,
        model="gemini/gemini-3-flash-preview",
        call_type="anthropic_messages",
    )

    per_query_cost = litellm.model_cost["gemini/gemini-3-flash-preview"]["search_context_cost_per_query"][
        "search_context_size_medium"
    ]
    assert per_query_cost > 0
    assert cost_with_search - cost_without_search == pytest.approx(2 * per_query_cost)


@pytest.mark.parametrize(
    "model, provider, carried",
    [
        ("databricks/databricks-claude-opus-4-7", "databricks", "max"),
        ("openrouter/anthropic/claude-opus-4.7", "openrouter", "xhigh"),
    ],
)
def test_a_summary_bearing_adaptive_request_still_delivers_its_tier(model, provider, carried):
    """The summary rides inside the forwarded `thinking` block for a Claude target, so the tier must
    stay a plain string. Wrapping it into `{"effort": ..., "summary": ...}` made databricks raise
    `Invalid reasoning_effort` and made bedrock drop `output_config` altogether, losing the tier on
    exactly the path this translator exists to serve.

    Each case names the exact tier that provider ends up sending, not merely that something arrived:
    bedrock and databricks rebuild `output_config`, and openrouter applies its own max to xhigh
    remap, so asserting presence alone would pass on a mapping that silently changed the tier."""
    from litellm.types.llms.anthropic import AnthropicMessagesRequest
    from litellm.utils import get_optional_params

    adapter = LiteLLMAnthropicMessagesAdapter()
    openai_request, _ = adapter.translate_anthropic_to_openai(
        anthropic_message_request=AnthropicMessagesRequest(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": "hi"}],
            thinking={"type": "adaptive", "summary": "detailed"},
            output_config={"effort": "max"},
        ),
        custom_llm_provider=provider,
    )

    assert openai_request["reasoning_effort"] == "max"

    on_the_wire = get_optional_params(
        model=model,
        custom_llm_provider=provider,
        thinking=openai_request["thinking"],
        reasoning_effort=openai_request["reasoning_effort"],
    )
    on_the_wire_tier = on_the_wire.get("output_config", {}).get("effort") or on_the_wire.get("reasoning_effort")

    assert on_the_wire_tier == carried


ARN_MODEL = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123"


def test_an_inference_profile_arn_keeps_taking_its_tier_as_output_config():
    """Regression: an ARN contains neither `anthropic` nor `claude`, so it reaches this branch only
    through `is_bedrock_arn_model`. Bedrock resolves no chat config for one, so `reasoning_effort`
    is dropped there and the tier vanishes; `output_config` is what survives."""
    from litellm.types.llms.anthropic import AnthropicMessagesRequest
    from litellm.utils import get_optional_params

    adapter = LiteLLMAnthropicMessagesAdapter()
    openai_request, _ = adapter.translate_anthropic_to_openai(
        anthropic_message_request=AnthropicMessagesRequest(
            model=ARN_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": "hi"}],
            thinking={"type": "adaptive"},
            output_config={"effort": "max"},
        )
    )

    assert openai_request["output_config"] == {"effort": "max"}
    assert "reasoning_effort" not in openai_request

    on_the_wire = get_optional_params(
        model=ARN_MODEL,
        custom_llm_provider="bedrock",
        thinking=openai_request["thinking"],
        output_config=openai_request["output_config"],
    )

    assert on_the_wire["output_config"] == {"effort": "max"}


def test_a_bedrock_target_keeps_a_caller_set_thinking_display():
    """`output_config` attaches the tier without touching `thinking`, so a caller who asked for
    `display: omitted` still gets it. Carrying the tier as `reasoning_effort` instead lets the
    provider mapping rewrite that block."""
    from litellm.types.llms.anthropic import AnthropicMessagesRequest
    from litellm.utils import get_optional_params

    thinking = {"type": "adaptive", "display": "omitted"}
    adapter = LiteLLMAnthropicMessagesAdapter()
    openai_request, _ = adapter.translate_anthropic_to_openai(
        anthropic_message_request=AnthropicMessagesRequest(
            model="bedrock/converse/us.anthropic.claude-opus-4-7",
            max_tokens=1024,
            messages=[{"role": "user", "content": "hi"}],
            thinking=thinking,
            output_config={"effort": "max"},
        )
    )

    on_the_wire = get_optional_params(
        model="converse/us.anthropic.claude-opus-4-7",
        custom_llm_provider="bedrock",
        thinking=openai_request["thinking"],
        output_config=openai_request["output_config"],
    )

    assert on_the_wire["thinking"] == thinking
    assert on_the_wire["output_config"] == {"effort": "max"}


def test_a_non_claude_target_keeps_its_summary_wrapping():
    """The negative class: a target that gets no `thinking` block has nowhere else to put the
    summary, so the wrapped dict is still the right shape there."""
    from litellm.types.llms.anthropic import AnthropicMessagesRequest

    adapter = LiteLLMAnthropicMessagesAdapter()
    openai_request, _ = adapter.translate_anthropic_to_openai(
        anthropic_message_request=AnthropicMessagesRequest(
            model="gpt-5-mini",
            max_tokens=1024,
            messages=[{"role": "user", "content": "hi"}],
            thinking={"type": "adaptive", "summary": "detailed"},
            output_config={"effort": "max"},
        )
    )

    assert openai_request["reasoning_effort"] == {"effort": "max", "summary": "detailed"}
    assert "thinking" not in openai_request


def test_a_databricks_target_trades_its_thinking_display_for_the_tier():
    """The one accepted cost of carrying the tier as `reasoning_effort`: databricks rebuilds the
    thinking block while mapping it, so a caller-set `display` is replaced. Pinned rather than left
    silent. It only takes `output_config` when litellm sends one, which this bridge cannot do for a
    provider whose own supported-params list omits it, so the tier is the thing worth keeping here.
    Bedrock avoids this entirely by taking `output_config` directly."""
    from litellm.types.llms.anthropic import AnthropicMessagesRequest
    from litellm.utils import get_optional_params

    adapter = LiteLLMAnthropicMessagesAdapter()
    openai_request, _ = adapter.translate_anthropic_to_openai(
        anthropic_message_request=AnthropicMessagesRequest(
            model="databricks/databricks-claude-opus-4-7",
            max_tokens=1024,
            messages=[{"role": "user", "content": "hi"}],
            thinking={"type": "adaptive", "display": "omitted"},
            output_config={"effort": "max"},
        ),
        custom_llm_provider="databricks",
    )

    on_the_wire = get_optional_params(
        model="databricks-claude-opus-4-7",
        custom_llm_provider="databricks",
        thinking=openai_request["thinking"],
        reasoning_effort=openai_request["reasoning_effort"],
    )

    assert on_the_wire["output_config"] == {"effort": "max"}
    assert on_the_wire["thinking"]["display"] == "summarized"


@pytest.mark.parametrize(
    "thinking, output_config",
    [
        ({"type": "adaptive"}, {"effort": "max"}),
        ({"type": "adaptive"}, {"effort": "minimal"}),
        ({"type": "adaptive", "summary": "detailed"}, {"effort": "high"}),
        ({"type": "adaptive", "display": "omitted"}, {"effort": "high"}),
    ],
)
def test_a_target_declaring_no_reasoning_effort_is_sent_none(thinking, output_config):
    """Regression: snowflake serves Claude over the Anthropic dialect and declares `thinking`
    alone, so storing the tier raised `UnsupportedParamsError` in `get_optional_params` before the
    request reached the wire. Every adaptive shape carrying a tier turned a 200 into a 400.

    Being Claude-family is a fact about the model, not about the params the provider in front of
    it accepts. The tier stays behind and the caller's `thinking` block travels untouched."""
    from litellm.types.llms.anthropic import AnthropicMessagesRequest
    from litellm.utils import get_optional_params

    adapter = LiteLLMAnthropicMessagesAdapter()
    openai_request, _ = adapter.translate_anthropic_to_openai(
        anthropic_message_request=AnthropicMessagesRequest(
            model="snowflake/claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": "hi"}],
            thinking=thinking,
            output_config=output_config,
        ),
        custom_llm_provider="snowflake",
    )

    assert "reasoning_effort" not in openai_request
    assert "output_config" not in openai_request
    assert openai_request["thinking"] == thinking

    on_the_wire = get_optional_params(
        model="snowflake/claude-sonnet-4-6",
        custom_llm_provider="snowflake",
        thinking=openai_request["thinking"],
    )

    assert on_the_wire["thinking"] == thinking


def test_a_target_declaring_reasoning_effort_still_gets_its_tier():
    """The negative class for the gate. Same request shape, a provider that does declare the
    param, so the tier must still travel: the gate must drop it for snowflake alone, not for
    every Claude target, or it would undo the fix it is protecting."""
    from litellm.types.llms.anthropic import AnthropicMessagesRequest

    adapter = LiteLLMAnthropicMessagesAdapter()
    openai_request, _ = adapter.translate_anthropic_to_openai(
        anthropic_message_request=AnthropicMessagesRequest(
            model="databricks/databricks-claude-opus-4-7",
            max_tokens=1024,
            messages=[{"role": "user", "content": "hi"}],
            thinking={"type": "adaptive"},
            output_config={"effort": "max"},
        ),
        custom_llm_provider="databricks",
    )

    assert openai_request["reasoning_effort"] == "max"


@pytest.mark.parametrize(
    "model",
    ["snowflake/claude-sonnet-4-6", "databricks/databricks-claude-opus-4-7", "github_copilot/claude-sonnet-4"],
)
def test_a_caller_that_names_no_provider_carries_no_tier(model):
    """`translate_anthropic_to_openai` is also called without a provider, by `adapter_completion`
    and by the shadow-eval logger. There is no declaration to read there, so the tier stays behind
    rather than being offered to a target that may reject it, which is what this bridge sent
    before it carried a tier at all.

    The databricks arm is the cost of that, stated rather than hidden: a provider that does take
    the tier does not get one from these two callers. The copilot arm is why the cost is worth
    paying, and why this must not be "fixed" by resolving the provider from the model prefix.
    That resolution runs an OAuth device flow for copilot and chatgpt, which would block this
    call for minutes, and one of the two callers is a logging callback. A test asserting the
    absence here is also a test that this stays fast."""
    from litellm.types.llms.anthropic import AnthropicMessagesRequest

    adapter = LiteLLMAnthropicMessagesAdapter()
    openai_request, _ = adapter.translate_anthropic_to_openai(
        anthropic_message_request=AnthropicMessagesRequest(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": "hi"}],
            thinking={"type": "adaptive"},
            output_config={"effort": "max"},
        )
    )

    assert openai_request["thinking"] == {"type": "adaptive"}
    assert "reasoning_effort" not in openai_request


def test_a_chained_litellm_proxy_target_still_takes_the_tier():
    """The one place this deliberately parts company with `_supports_prompt_cache_key`, which
    excludes a provider that proxies an unknown backend. That exclusion is right for a derived
    cache key and wrong here: the downstream proxy declares this param and resolves the real
    target itself, so excluding it would drop a tier that arrives perfectly well."""
    from litellm.types.llms.anthropic import AnthropicMessagesRequest

    adapter = LiteLLMAnthropicMessagesAdapter()
    openai_request, _ = adapter.translate_anthropic_to_openai(
        anthropic_message_request=AnthropicMessagesRequest(
            model="litellm_proxy/claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": "hi"}],
            thinking={"type": "adaptive"},
            output_config={"effort": "max"},
        ),
        custom_llm_provider="litellm_proxy",
    )

    assert openai_request["reasoning_effort"] == "max"
    assert openai_request["thinking"] == {"type": "adaptive"}


def test_a_bedrock_target_still_takes_output_config_not_the_declared_gate():
    """Bedrock declares both carriers, so the gate must not change which one it gets: the tier
    rides in `output_config`, which leaves `thinking` alone, and `reasoning_effort` is never
    stored alongside it."""
    from litellm.types.llms.anthropic import AnthropicMessagesRequest

    adapter = LiteLLMAnthropicMessagesAdapter()
    openai_request, _ = adapter.translate_anthropic_to_openai(
        anthropic_message_request=AnthropicMessagesRequest(
            model="bedrock/converse/us.anthropic.claude-opus-4-7",
            max_tokens=1024,
            messages=[{"role": "user", "content": "hi"}],
            thinking={"type": "adaptive", "display": "omitted"},
            output_config={"effort": "max"},
        ),
        custom_llm_provider="bedrock",
    )

    assert openai_request["output_config"] == {"effort": "max"}
    assert "reasoning_effort" not in openai_request
    assert openai_request["thinking"] == {"type": "adaptive", "display": "omitted"}
