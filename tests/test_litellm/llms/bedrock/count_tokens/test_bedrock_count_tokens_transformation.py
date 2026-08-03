import base64
import json
import os
import sys

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path
from litellm.llms.bedrock.count_tokens.transformation import (
    DEFAULT_ANTHROPIC_INVOKE_MODEL_MAX_TOKENS,
    BedrockCountTokensConfig,
)


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


def test_transform_to_invoke_model_format_base64_encodes_body():
    """The CountTokens API expects invokeModel.body as a base64-encoded blob;
    Anthropic Messages bodies additionally need anthropic_version/max_tokens
    to pass Bedrock's InvokeModel schema validation."""
    config = BedrockCountTokensConfig()

    request = {
        "model": "anthropic.claude-3-sonnet-20240229-v1:0",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
    }

    result = config.transform_anthropic_to_bedrock_count_tokens(request)

    body = json.loads(base64.b64decode(result["input"]["invokeModel"]["body"]))
    assert body["messages"] == request["messages"]
    assert "model" not in body
    assert body["anthropic_version"] == "bedrock-2023-05-31"
    assert body["max_tokens"] == DEFAULT_ANTHROPIC_INVOKE_MODEL_MAX_TOKENS


def test_transform_to_invoke_model_format_raw_body_unchanged():
    """Non-messages bodies (e.g. Titan inputText) must not get Anthropic fields."""
    config = BedrockCountTokensConfig()

    result = config.transform_anthropic_to_bedrock_count_tokens(
        {"model": "amazon.titan-text-express-v1", "inputText": "hello"}
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

    result = config.transform_anthropic_to_bedrock_count_tokens(anthropic_request)

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

    result = config.transform_anthropic_to_bedrock_count_tokens(request)

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

    result = config.transform_anthropic_to_bedrock_count_tokens(request)

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

    result = config.transform_anthropic_to_bedrock_count_tokens(request)

    converse = result["input"]["converse"]
    assert "toolConfig" in converse
    tools = converse["toolConfig"]["tools"]
    assert len(tools) == 1
    assert tools[0]["toolSpec"]["name"] == "read_file"
    assert tools[0]["toolSpec"]["description"] == "Read a file"
    assert tools[0]["toolSpec"]["inputSchema"]["json"]["type"] == "object"


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

    result = config.transform_anthropic_to_bedrock_count_tokens(request)

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

    result = config.transform_anthropic_to_bedrock_count_tokens(request)

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

    result = config.transform_anthropic_to_bedrock_count_tokens(request)

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
