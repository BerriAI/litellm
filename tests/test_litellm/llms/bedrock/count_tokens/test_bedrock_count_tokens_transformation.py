import os
import sys

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path
from litellm.llms.bedrock.count_tokens.transformation import BedrockCountTokensConfig


def test_detect_input_type():
    """Test input type detection (converse vs invokeModel)"""
    config = BedrockCountTokensConfig()

    # Test messages format -> converse
    request_with_messages = {"messages": [{"role": "user", "content": "hi"}]}
    assert config._detect_input_type(request_with_messages) == "converse"

    # Test text format -> invokeModel
    request_with_text = {"inputText": "hello"}
    assert config._detect_input_type(request_with_text) == "invokeModel"


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
        "system": [{"type": "text", "text": "Block 1"}, {"type": "text", "text": "Block 2"}],
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
            {"name": "my_tool", "description": "A tool", "input_schema": {"type": "object", "properties": {}}},
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
            {"name": "my-tool!", "description": "A tool", "input_schema": {"type": "object", "properties": {}}},
        ],
    }

    result = config.transform_anthropic_to_bedrock_count_tokens(request)

    tool_name = result["input"]["converse"]["toolConfig"]["tools"][0]["toolSpec"]["name"]
    # Should be sanitized: only [a-zA-Z0-9_]
    assert tool_name == "my_tool_"
