"""
Tests for MCP sampling handler response/tool conversion.

Covers the translation of a LiteLLM completion response back into MCP
`CreateMessageResult` / `CreateMessageResultWithTools`, plus the helpers that
convert MCP tool definitions, tool-choice modes, and image/audio content into
OpenAI request format.
"""

import json
from types import SimpleNamespace

from mcp.types import (
    CreateMessageResult,
    CreateMessageResultWithTools,
    ErrorData,
    TextContent,
    ToolUseContent,
)

from litellm.proxy._experimental.mcp_server.sampling_handler import (
    _convert_mcp_content_to_openai,
    _convert_mcp_tool_choice_to_openai,
    _convert_mcp_tools_to_openai,
    _convert_openai_response_to_mcp_result,
    _convert_single_content,
)


def _tool_call(*, call_id: str, name: str, arguments):
    return SimpleNamespace(
        id=call_id, function=SimpleNamespace(name=name, arguments=arguments)
    )


def _response(*, content=None, tool_calls=None, finish_reason="stop", model="gpt-4o"):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model=model)


class TestConvertOpenAIResponseToMcpResult:
    def test_should_return_error_data_when_no_choices(self):
        response = SimpleNamespace(choices=[], model="gpt-4o")
        result = _convert_openai_response_to_mcp_result(response, "gpt-4o")
        assert isinstance(result, ErrorData)
        assert "no choices" in result.message.lower()

    def test_should_convert_plain_text_response(self):
        result = _convert_openai_response_to_mcp_result(
            _response(content="hello world"), "gpt-4o"
        )
        assert isinstance(result, CreateMessageResult)
        assert isinstance(result.content, TextContent)
        assert result.content.text == "hello world"
        assert result.role == "assistant"
        assert result.stopReason == "endTurn"

    def test_should_map_length_finish_reason_to_max_tokens(self):
        result = _convert_openai_response_to_mcp_result(
            _response(content="truncated", finish_reason="length"), "gpt-4o"
        )
        assert result.stopReason == "maxTokens"

    def test_should_prefer_actual_model_from_response(self):
        result = _convert_openai_response_to_mcp_result(
            _response(content="hi", model="gpt-4o-2024-08-06"), "gpt-4o"
        )
        assert result.model == "gpt-4o-2024-08-06"

    def test_should_convert_tool_calls_response(self):
        tc = _tool_call(
            call_id="call_1",
            name="get_weather",
            arguments=json.dumps({"city": "NYC"}),
        )
        result = _convert_openai_response_to_mcp_result(
            _response(content=None, tool_calls=[tc], finish_reason="tool_calls"),
            "gpt-4o",
        )
        assert isinstance(result, CreateMessageResultWithTools)
        assert result.stopReason == "toolUse"
        tool_uses = [c for c in result.content if isinstance(c, ToolUseContent)]
        assert len(tool_uses) == 1
        assert tool_uses[0].name == "get_weather"
        assert tool_uses[0].id == "call_1"
        assert tool_uses[0].input == {"city": "NYC"}

    def test_should_keep_text_alongside_tool_calls(self):
        tc = _tool_call(call_id="call_1", name="search", arguments="{}")
        result = _convert_openai_response_to_mcp_result(
            _response(
                content="let me check", tool_calls=[tc], finish_reason="tool_calls"
            ),
            "gpt-4o",
        )
        texts = [c for c in result.content if isinstance(c, TextContent)]
        assert texts and texts[0].text == "let me check"

    def test_should_wrap_unparsable_tool_arguments_as_raw(self):
        tc = _tool_call(call_id="call_1", name="bad", arguments="not-json{")
        result = _convert_openai_response_to_mcp_result(
            _response(tool_calls=[tc], finish_reason="tool_calls"), "gpt-4o"
        )
        tool_uses = [c for c in result.content if isinstance(c, ToolUseContent)]
        assert tool_uses[0].input == {"raw": "not-json{"}


class TestConvertMcpToolsToOpenAI:
    def test_should_return_none_when_no_tools(self):
        assert _convert_mcp_tools_to_openai(None) is None

    def test_should_convert_tool_with_schema(self):
        schema = {"type": "object", "properties": {"q": {"type": "string"}}}
        tool = SimpleNamespace(
            name="search", description="search the web", inputSchema=schema
        )
        result = _convert_mcp_tools_to_openai([tool])
        assert result == [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "search the web",
                    "parameters": schema,
                },
            }
        ]

    def test_should_default_description_and_parameters(self):
        tool = SimpleNamespace(name="noop", description=None, inputSchema=None)
        result = _convert_mcp_tools_to_openai([tool])
        fn = result[0]["function"]
        assert fn["description"] == ""
        assert fn["parameters"] == {"type": "object", "properties": {}}


class TestConvertMcpToolChoiceToOpenAI:
    def test_should_return_none_when_no_choice(self):
        assert _convert_mcp_tool_choice_to_openai(None) is None

    def test_should_map_known_modes(self):
        for mode in ("auto", "required", "none"):
            choice = SimpleNamespace(mode=mode)
            assert _convert_mcp_tool_choice_to_openai(choice) == mode

    def test_should_default_unknown_mode_to_auto(self):
        choice = SimpleNamespace(mode="banana")
        assert _convert_mcp_tool_choice_to_openai(choice) == "auto"


class TestConvertImageAndAudioContent:
    def test_should_convert_image_to_data_uri(self):
        content = SimpleNamespace(type="image", data="aGVsbG8=", mimeType="image/jpeg")
        result = _convert_single_content(content)
        assert result == {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,aGVsbG8="},
        }

    def test_should_map_audio_mime_to_format(self):
        content = SimpleNamespace(type="audio", data="Zm9v", mimeType="audio/mp3")
        result = _convert_single_content(content)
        assert result["type"] == "input_audio"
        assert result["input_audio"] == {"data": "Zm9v", "format": "mp3"}

    def test_should_default_unknown_audio_mime_to_wav(self):
        content = SimpleNamespace(type="audio", data="Zm9v", mimeType="audio/weird")
        result = _convert_single_content(content)
        assert result["input_audio"]["format"] == "wav"

    def test_should_flatten_list_content(self):
        items = [
            SimpleNamespace(type="text", text="a"),
            SimpleNamespace(type="image", data="x", mimeType="image/png"),
        ]
        result = _convert_mcp_content_to_openai(items)
        assert isinstance(result, list)
        assert result[0] == {"type": "text", "text": "a"}
        assert result[1]["type"] == "image_url"
