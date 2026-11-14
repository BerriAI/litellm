import base64
import json
from typing import Final

import pytest

from litellm.llms.bedrock.count_tokens.transformation import (
    DEFAULT_ANTHROPIC_INVOKE_MODEL_MAX_TOKENS,
    BedrockCountTokensConfig,
)
from litellm.llms.bedrock.messages.invoke_transformations.anthropic_claude_transformation import (
    AmazonAnthropicClaudeMessagesConfig,
)
from litellm.types.router import GenericLiteLLMParams


def test_detect_input_type():
    """Test input type detection (converse vs invokeModel)"""
    config = BedrockCountTokensConfig()

    # Test messages format -> converse
    request_with_messages = {"messages": [{"role": "user", "content": "hi"}]}
    assert config._detect_input_type(request_with_messages) == "converse"

    # Test text format -> invokeModel
    request_with_text = {"inputText": "hello"}
    assert config._detect_input_type(request_with_text) == "invokeModel"


def test_detect_input_type_anthropic_blocks_route_to_invoke_model():
    """Anthropic-shape content blocks must not go through the Converse path,
    which Bedrock rejects with a 400 (and the caller then silently falls back
    to the local tokenizer)."""
    config = BedrockCountTokensConfig()

    request = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Reading the file."},
                    {
                        "type": "tool_use",
                        "id": "toolu_01",
                        "name": "read_file",
                        "input": {"path": "main.py"},
                    },
                ],
            },
        ],
    }
    assert config._detect_input_type(request) == "invokeModel"


def test_detect_input_type_converse_blocks_route_to_converse():
    """Converse-shape blocks (no "type" key) keep using the converse input."""
    config = BedrockCountTokensConfig()

    request = {"messages": [{"role": "user", "content": [{"text": "hi"}]}]}
    assert config._detect_input_type(request) == "converse"


@pytest.mark.parametrize("content", [[{"text": "hi"}], "hi"])
@pytest.mark.parametrize(
    "context_management", [None, {"edits": [{"type": "clear_tool_uses_20250919"}]}]
)
def test_a_pass_through_body_with_context_management_keeps_its_converse_input(content, context_management):
    request = {
        "model": "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "messages": [{"role": "user", "content": content}],
        "context_management": context_management,
    }

    transformed = BedrockCountTokensConfig().transform_anthropic_to_bedrock_count_tokens(request, request["model"])

    assert transformed["input"]["converse"]["messages"] == [{"role": "user", "content": [{"text": "hi"}]}]


def test_transform_to_invoke_model_format_base64_encodes_body():
    """The CountTokens API expects invokeModel.body as a base64-encoded blob;
    Anthropic Messages bodies additionally need anthropic_version/max_tokens
    to pass Bedrock's InvokeModel schema validation."""
    config = BedrockCountTokensConfig()

    request = {
        "model": "anthropic.claude-3-sonnet-20240229-v1:0",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
    }

    result = config.transform_anthropic_to_bedrock_count_tokens(request, request["model"])

    body = json.loads(base64.b64decode(result["input"]["invokeModel"]["body"]))
    assert body["messages"] == request["messages"]
    assert "model" not in body
    assert body["anthropic_version"] == "bedrock-2023-05-31"
    assert body["max_tokens"] == DEFAULT_ANTHROPIC_INVOKE_MODEL_MAX_TOKENS


def test_transform_to_invoke_model_format_counts_tools_the_way_bedrock_messages_sends_them():
    schema = {"type": "object", "properties": {}}

    result = BedrockCountTokensConfig().transform_anthropic_to_bedrock_count_tokens(
        {
            "model": "anthropic.claude-haiku-5-5",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
            "tools": [
                {"name": "read", "input_schema": schema, "custom": {"defer_loading": True}},
                {"name": "write", "input_schema": schema, "custom": {"eager_input_streaming": True}},
            ],
        },
        "anthropic.claude-haiku-5-5",
    )

    body = json.loads(base64.b64decode(result["input"]["invokeModel"]["body"]))
    assert body["tools"] == [
        {"name": "read", "input_schema": schema, "defer_loading": True},
        {"name": "write", "input_schema": schema},
    ]


def test_transform_to_invoke_model_format_raw_body_unchanged():
    """Non-messages bodies (e.g. Titan inputText) must not get Anthropic fields."""
    config = BedrockCountTokensConfig()

    result = config.transform_anthropic_to_bedrock_count_tokens(
        {"model": "amazon.titan-text-express-v1", "inputText": "hello"}, "amazon.titan-text-express-v1"
    )

    body = json.loads(base64.b64decode(result["input"]["invokeModel"]["body"]))
    assert body == {"inputText": "hello"}


def test_transform_anthropic_to_bedrock_request():
    """Test basic request transformation"""
    config = BedrockCountTokensConfig()

    anthropic_request = {
        "model": "anthropic.claude-3-sonnet-20240229-v1:0",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    result = config.transform_anthropic_to_bedrock_count_tokens(anthropic_request, anthropic_request["model"])

    assert "input" in result
    assert "converse" in result["input"]
    assert "messages" in result["input"]["converse"]


def test_transform_includes_system_prompt():
    """Test that system prompt is included in Bedrock converse format."""
    config = BedrockCountTokensConfig()

    request = {
        "model": "anthropic.claude-3-sonnet-20240229-v1:0",
        "messages": [{"role": "user", "content": "Hello"}],
        "system": "You are a helpful assistant.",
    }

    result = config.transform_anthropic_to_bedrock_count_tokens(request, request["model"])

    converse = result["input"]["converse"]
    assert "system" in converse
    assert converse["system"] == [{"text": "You are a helpful assistant."}]


def test_transform_includes_system_prompt_as_list():
    """Test that system prompt as list of blocks is handled."""
    config = BedrockCountTokensConfig()

    request = {
        "model": "anthropic.claude-3-sonnet-20240229-v1:0",
        "messages": [{"role": "user", "content": "Hello"}],
        "system": [
            {"type": "text", "text": "Block 1"},
            {"type": "text", "text": "Block 2"},
        ],
    }

    result = config.transform_anthropic_to_bedrock_count_tokens(request, request["model"])

    converse = result["input"]["converse"]
    assert converse["system"] == [{"text": "Block 1"}, {"text": "Block 2"}]


def test_transform_includes_tools():
    """Test that tools are transformed to Bedrock toolConfig format."""
    config = BedrockCountTokensConfig()

    request = {
        "model": "anthropic.claude-3-sonnet-20240229-v1:0",
        "messages": [{"role": "user", "content": "Hello"}],
        "tools": [
            {
                "name": "read_file",
                "description": "Read a file",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            }
        ],
    }

    result = config.transform_anthropic_to_bedrock_count_tokens(request, request["model"])

    converse = result["input"]["converse"]
    assert "toolConfig" in converse
    tools = converse["toolConfig"]["tools"]
    assert len(tools) == 1
    assert tools[0]["toolSpec"]["name"] == "read_file"
    assert tools[0]["toolSpec"]["description"] == "Read a file"
    assert tools[0]["toolSpec"]["inputSchema"]["json"]["type"] == "object"


def test_transform_maps_chat_function_tools():
    parameters = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    request = {
        "model": "amazon.nova-pro-v1:0",
        "messages": [{"role": "user", "content": "Weather in Paris?"}],
        "tools": [
            {
                "type": "function",
                "function": {"name": "get_weather", "description": "Look up the weather", "parameters": parameters},
            }
        ],
    }

    result = BedrockCountTokensConfig().transform_anthropic_to_bedrock_count_tokens(request, request["model"])

    assert result["input"]["converse"]["toolConfig"]["tools"] == [
        {"toolSpec": {"name": "get_weather", "description": "Look up the weather", "inputSchema": {"json": parameters}}}
    ]

def test_transform_includes_system_and_tools_together():
    """Test that both system and tools are included together."""
    config = BedrockCountTokensConfig()

    request = {
        "model": "anthropic.claude-3-sonnet-20240229-v1:0",
        "messages": [{"role": "user", "content": "Hello"}],
        "system": "Be helpful",
        "tools": [
            {
                "name": "my_tool",
                "description": "A tool",
                "input_schema": {"type": "object", "properties": {}},
            },
        ],
    }

    result = config.transform_anthropic_to_bedrock_count_tokens(request, request["model"])

    converse = result["input"]["converse"]
    assert "system" in converse
    assert "toolConfig" in converse
    assert "messages" in converse


def test_transform_no_system_no_tools():
    """Test that missing system and tools don't add extra keys."""
    config = BedrockCountTokensConfig()

    request = {
        "model": "anthropic.claude-3-sonnet-20240229-v1:0",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    result = config.transform_anthropic_to_bedrock_count_tokens(request, request["model"])

    converse = result["input"]["converse"]
    assert "system" not in converse
    assert "toolConfig" not in converse


def test_tool_name_sanitization():
    """Test that tool names are sanitized for Bedrock requirements."""
    config = BedrockCountTokensConfig()

    request = {
        "model": "anthropic.claude-3-sonnet-20240229-v1:0",
        "messages": [{"role": "user", "content": "Hello"}],
        "tools": [
            {
                "name": "my-tool!",
                "description": "A tool",
                "input_schema": {"type": "object", "properties": {}},
            },
        ],
    }

    result = config.transform_anthropic_to_bedrock_count_tokens(request, request["model"])

    tool_name = result["input"]["converse"]["toolConfig"]["tools"][0]["toolSpec"][
        "name"
    ]
    # Should be sanitized: only [a-zA-Z0-9_]
    assert tool_name == "my_tool_"


def test_count_tokens_endpoint_encodes_model_id(monkeypatch):
    """Test model IDs are treated as a single Bedrock path segment."""
    config = BedrockCountTokensConfig()

    monkeypatch.setattr(
        config,
        "get_runtime_endpoint",
        lambda **kwargs: ("https://bedrock-runtime.us-east-1.amazonaws.com", None),
    )

    endpoint = config.get_bedrock_count_tokens_endpoint(
        model="bedrock/../../model/other?x=1#frag",
        aws_region_name="us-east-1",
    )

    assert (
        endpoint
        == "https://bedrock-runtime.us-east-1.amazonaws.com/model/..%2F..%2Fmodel%2Fother%3Fx%3D1%23frag/count-tokens"
    )


@pytest.mark.parametrize(
    "request_data",
    [
        {"model": "us.anthropic.claude-opus-4-8", "messages": [{"role": "user", "content": "hi"}]},
        {"model": "us.anthropic.claude-opus-4-8", "messages": "hi"},
        {
            "model": "us.anthropic.claude-opus-4-8",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}, {"content": "hi"}],
        },
    ],
)
def test_count_tokens_request_problem_leaves_bodies_bedrock_can_judge_to_bedrock(request_data):
    assert BedrockCountTokensConfig().count_tokens_request_problem(request_data) is None


def test_count_tokens_request_problem_names_tool_calls_instead_of_asking_for_content():
    problem = BedrockCountTokensConfig().count_tokens_request_problem(
        {
            "model": "us.amazon.nova-pro-v1:0",
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "tool_calls": [{"id": "1", "type": "function", "function": {"name": "t", "arguments": "{}"}}],
                },
            ],
        }
    )

    assert problem is not None
    assert "tool_calls" in problem
    assert "must have a 'content' field" not in problem


_GEMINI_THOUGHT_HISTORY: Final = [
    {"role": "user", "content": "Pick a number."},
    {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "plan", "signature": "EqQBCkYIAxgCIkA_anthropic_signed"},
            {"type": "text", "text": "Seven."},
        ],
    },
    {"role": "user", "content": "Double it."},
    {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "Gemini reasoning", "signature": ""},
            {"type": "text", "text": "Fourteen."},
        ],
    },
    {"role": "user", "content": "Halve it."},
]


def _counted_invoke_body(request_model: str, messages: list[object], model: str) -> dict[str, object]:
    transformed: Final = BedrockCountTokensConfig().transform_anthropic_to_bedrock_count_tokens(
        {"model": request_model, "messages": messages}, model
    )
    return json.loads(base64.b64decode(transformed["input"]["invokeModel"]["body"]))


def test_count_tokens_counts_the_thinking_history_bedrock_invoke_sends():
    model: Final = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    messages: Final = _GEMINI_THOUGHT_HISTORY
    sent: Final = AmazonAnthropicClaudeMessagesConfig().transform_anthropic_messages_request(
        model=model,
        messages=messages,
        anthropic_messages_optional_request_params={"max_tokens": 1024},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    transformed: Final = BedrockCountTokensConfig().transform_anthropic_to_bedrock_count_tokens(
        {"model": model, "messages": messages}, model
    )
    counted: Final = json.loads(base64.b64decode(transformed["input"]["invokeModel"]["body"]))

    assert counted["messages"] == sent["messages"]
    assert counted["messages"] != messages


def test_count_tokens_judges_the_deployment_model_and_not_the_alias_the_caller_named():
    counted: Final = _counted_invoke_body(
        "team-sonnet", _GEMINI_THOUGHT_HISTORY, "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    )

    assert counted["messages"] != _GEMINI_THOUGHT_HISTORY


def test_count_tokens_keeps_unsigned_thinking_for_a_model_that_is_not_claude():
    model: Final = "us.amazon.nova-2-lite-v1:0"

    assert _counted_invoke_body(model, _GEMINI_THOUGHT_HISTORY, model)["messages"] == _GEMINI_THOUGHT_HISTORY


def test_count_tokens_leaves_a_message_that_is_not_an_object_for_bedrock_to_judge():
    model: Final = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    messages: Final = ["hi", *_GEMINI_THOUGHT_HISTORY]

    assert _counted_invoke_body(model, messages, model)["messages"] == messages
