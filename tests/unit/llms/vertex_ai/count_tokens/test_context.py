from typing import Final, Literal

import pytest
from pydantic import JsonValue

from litellm.llms.anthropic.pass_through.adapters.transformation import (
    LiteLLMAnthropicMessagesAdapter,
    truncate_tool_name,
)
from litellm.llms.vertex_ai.count_tokens.context import (
    VertexGeminiCountContext,
    VertexGeminiCountContextError,
    gemini_count_contents,
    vertex_gemini_count_context,
)
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig

_SCHEMA: Final[JsonValue] = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}
_FUNCTION_DECLARATION: Final = {
    "functionDeclarations": [
        {
            "name": "lookup",
            "description": "Look up a value",
            "parameters": {"type": "object"},
        }
    ]
}


@pytest.mark.parametrize(
    "tool",
    (
        {
            "name": "lookup",
            "description": "Look up a value",
            "input_schema": _SCHEMA,
        },
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up a value",
                "parameters": _SCHEMA,
            },
        },
        {
            "type": "function",
            "name": "lookup",
            "description": "Look up a value",
            "parameters": _SCHEMA,
        },
    ),
    ids=("anthropic", "openai-chat", "responses"),
)
def test_function_tool_shapes_map_to_vertex_declarations(tool: JsonValue) -> None:
    context: Final = vertex_gemini_count_context(system=None, tools=[tool])

    assert isinstance(context, VertexGeminiCountContext)
    assert context.tools == (_FUNCTION_DECLARATION,)
    assert _SCHEMA["additionalProperties"] is False


def test_long_anthropic_tool_name_is_truncated_for_vertex() -> None:
    name: Final = "mcp__docs__" + "a" * 70
    context: Final = vertex_gemini_count_context(
        system=None,
        tools=[{"name": name, "description": "Look up a value", "input_schema": _SCHEMA}],
    )

    assert isinstance(context, VertexGeminiCountContext)
    assert context.tools == (
        {
            "functionDeclarations": [
                {
                    "name": truncate_tool_name(name),
                    "description": "Look up a value",
                    "parameters": {"type": "object"},
                }
            ]
        },
    )
    assert len(truncate_tool_name(name)) == 64


def test_64_character_anthropic_tool_name_is_unchanged_for_vertex() -> None:
    name: Final = "mcp__docs__" + "a" * 53
    context: Final = vertex_gemini_count_context(
        system=None,
        tools=[{"name": name, "input_schema": _SCHEMA}],
    )

    assert isinstance(context, VertexGeminiCountContext)
    assert context.tools == ({"functionDeclarations": [{"name": name, "parameters": {"type": "object"}}]},)


def test_long_openai_chat_tool_name_is_unchanged_for_vertex() -> None:
    name: Final = "mcp__docs__" + "a" * 70
    context: Final = vertex_gemini_count_context(
        system=None,
        tools=[{"type": "function", "function": {"name": name, "parameters": _SCHEMA}}],
    )

    assert isinstance(context, VertexGeminiCountContext)
    assert context.tools == ({"functionDeclarations": [{"name": name, "parameters": {"type": "object"}}]},)


def test_unknown_tools_are_omitted() -> None:
    context: Final = vertex_gemini_count_context(
        system=None,
        tools=[{"type": "web_search_preview"}, {"type": "computer_use"}],
    )

    assert isinstance(context, VertexGeminiCountContext)
    assert context.tools is None


def test_text_block_system_instruction_keeps_one_part_per_block() -> None:
    context: Final = vertex_gemini_count_context(
        system=[{"type": "text", "text": "First instruction."}, {"type": "text", "text": "Second instruction."}],
        tools=None,
    )

    assert isinstance(context, VertexGeminiCountContext)
    assert context.system_instruction == {"parts": ({"text": "First instruction."}, {"text": "Second instruction."})}


def test_string_system_instruction_maps_to_one_part() -> None:
    context: Final = vertex_gemini_count_context(system="Follow this instruction.", tools=None)

    assert isinstance(context, VertexGeminiCountContext)
    assert context.system_instruction == {"parts": ({"text": "Follow this instruction."},)}


def test_non_text_system_block_returns_an_error_value() -> None:
    context: Final = vertex_gemini_count_context(
        system=[{"type": "image", "source": {"type": "base64", "data": "AAAA"}}],
        tools=None,
    )

    assert isinstance(context, VertexGeminiCountContextError)
    assert "text blocks" in context.error_message


@pytest.mark.parametrize(("system", "tools"), ((None, None), ("", None), ([], [])))
def test_empty_context_is_omitted(system: JsonValue | None, tools: JsonValue | None) -> None:
    context: Final = vertex_gemini_count_context(system=system, tools=tools)

    assert isinstance(context, VertexGeminiCountContext)
    assert context.system_instruction is None
    assert context.tools is None


@pytest.mark.parametrize(
    "tool",
    (
        {"description": "Look up a value", "input_schema": _SCHEMA},
        {"type": "bash_20250124", "name": "bash"},
    ),
    ids=("unnamed", "anthropic-hosted"),
)
def test_anthropic_tools_count_the_same_declarations_inference_sends(tool: dict[str, JsonValue]) -> None:
    translated_tools, _ = LiteLLMAnthropicMessagesAdapter().translate_anthropic_tools_to_openai([tool])  # pyright: ignore[reportArgumentType]  # raw Anthropic tool dict
    inference_tools: Final = VertexGeminiConfig().map_function(
        value=[dict(translated) for translated in translated_tools],
        optional_params={},
    )

    context: Final = vertex_gemini_count_context(system=None, tools=[tool])

    assert isinstance(context, VertexGeminiCountContext)
    assert context.tools is not None
    assert [tool_value["functionDeclarations"] for tool_value in context.tools] == [
        inference_tool["function_declarations"] for inference_tool in inference_tools
    ]


@pytest.mark.parametrize(
    "tools",
    (
        "lookup",
        [1],
        [{"name": "lookup", "input_schema": "not-a-schema"}],
        [{"type": "function", "function": {"name": "lookup", "parameters": [1]}}],
        [{"name": "lookup", "input_schema": {"type": "object", "properties": {"query": "not-a-schema"}}}],
    ),
    ids=("not-a-list", "non-object-tool", "string-schema", "list-parameters", "malformed-property"),
)
def test_invalid_tools_return_an_error_value(tools: JsonValue) -> None:
    context: Final = vertex_gemini_count_context(system=None, tools=tools)

    assert isinstance(context, VertexGeminiCountContextError)
    assert "tools" in context.error_message


_ANTHROPIC_CONVERSATION: Final = (
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Summarize both"},
            {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBERi0="}},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aW1hZ2U="}},
        ],
    },
    {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": "toolu_01", "name": "lookup", "input": {"query": "x"}}],
    },
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_01", "content": "42"}]},
)


@pytest.mark.parametrize("custom_llm_provider", ("vertex_ai", "gemini"))
def test_anthropic_documents_images_and_tool_turns_become_gemini_parts(
    custom_llm_provider: Literal["vertex_ai", "gemini"],
) -> None:
    contents: Final = gemini_count_contents(
        _ANTHROPIC_CONVERSATION, "anthropic", "gemini-2.5-flash", custom_llm_provider
    )

    assert contents == [
        {
            "role": "user",
            "parts": [
                {"text": "Summarize both"},
                {"inline_data": {"data": "JVBERi0=", "mime_type": "application/pdf"}},
                {"inline_data": {"data": "aW1hZ2U=", "mime_type": "image/png"}},
            ],
        },
        {"role": "model", "parts": [{"function_call": {"name": "lookup", "args": {"query": "x"}}}]},
        {"role": "user", "parts": [{"function_response": {"name": "lookup", "response": {"content": "42"}}}]},
    ]


@pytest.mark.parametrize("custom_llm_provider", ("vertex_ai", "gemini"))
def test_openai_messages_keep_their_chat_conversion(custom_llm_provider: Literal["vertex_ai", "gemini"]) -> None:
    contents: Final = gemini_count_contents(
        ({"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}),
        "openai",
        "gemini-2.5-flash",
        custom_llm_provider,
    )

    assert contents == [
        {"role": "user", "parts": [{"text": "hello"}]},
        {"role": "model", "parts": [{"text": "hi"}]},
    ]
