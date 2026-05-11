import os
import sys

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path
from litellm.llms.bedrock.count_tokens.transformation import BedrockCountTokensConfig


def test_detect_input_type():
    """Test input type detection (converse vs invokeModel)"""
    config = BedrockCountTokensConfig()

    # String message content is Converse-compatible -> converse
    request_with_messages = {"messages": [{"role": "user", "content": "hi"}]}
    assert config._detect_input_type(request_with_messages) == "converse"

    # Test text format -> invokeModel
    request_with_text = {"inputText": "hello"}
    assert config._detect_input_type(request_with_text) == "invokeModel"


def test_detect_input_type_anthropic_shape_routes_to_invoke_model():
    """Anthropic-shape content blocks (``{"type": "text"}``) cannot be sent through
    the Converse path because Converse expects ``{"text": ...}`` blocks. Detect
    Anthropic-typed blocks and route to InvokeModel, which preserves the body
    verbatim and lets Bedrock's tokeniser score it correctly.
    """
    config = BedrockCountTokensConfig()

    # Single text block in Anthropic shape
    req = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        ],
    }
    assert config._detect_input_type(req) == "invokeModel"

    # Tool-using turn with tool_use + tool_result blocks
    req = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Reading the file."},
                    {
                        "type": "tool_use",
                        "id": "toolu_01ABC",
                        "name": "read_file",
                        "input": {"path": "main.py"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01ABC",
                        "content": "def main(): pass",
                    }
                ],
            },
        ],
    }
    assert config._detect_input_type(req) == "invokeModel"


def test_detect_input_type_converse_shape_blocks_stay_converse():
    """Converse-shape blocks (``{"text": ...}``, no ``type`` key) should still
    route to converse to preserve backwards-compatible behaviour for callers
    already producing Bedrock-shape input.
    """
    config = BedrockCountTokensConfig()
    req = {
        "messages": [
            {"role": "user", "content": [{"text": "hi"}]},
        ],
    }
    assert config._detect_input_type(req) == "converse"


def test_detect_input_type_skips_non_dict_message_entries():
    """Defensive: a malformed messages list with non-dict entries shouldn't
    crash the detector. Skip the bad entry and decide based on the rest."""
    config = BedrockCountTokensConfig()
    req = {
        "messages": [
            "not-a-dict",  # ignored
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        ],
    }
    assert config._detect_input_type(req) == "invokeModel"

    # All non-dict entries falls through to the default "converse" branch
    req = {"messages": ["not-a-dict", 42]}
    assert config._detect_input_type(req) == "converse"


def test_transform_to_invoke_model_format_base64_encodes_body():
    """The Bedrock CountTokens API spec describes ``invokeModel.body`` as a
    Base64-encoded blob. A raw JSON string fails with ``Unable to parse
    request body`` for non-trivial payloads.
    """
    import base64
    import json

    config = BedrockCountTokensConfig()

    request_data = {
        "model": "anthropic.claude-3-sonnet-20240229-v1:0",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        ],
    }

    result = config._transform_to_invoke_model_format(request_data)

    # Top-level shape: {"input": {"invokeModel": {"body": "<base64>"}}}
    assert "input" in result
    assert "invokeModel" in result["input"]
    body_value = result["input"]["invokeModel"]["body"]

    # Must be Base64; decoding must round-trip to valid JSON
    decoded = base64.b64decode(body_value).decode("utf-8")
    parsed = json.loads(decoded)

    # ``model`` is dropped (it's not part of the model input body)
    assert "model" not in parsed

    # Bedrock InvokeModel requires anthropic_version + max_tokens; transform
    # supplies sensible defaults if the caller omitted them.
    assert parsed["anthropic_version"] == "bedrock-2023-05-31"
    assert "max_tokens" in parsed

    # The original messages are preserved
    assert parsed["messages"] == request_data["messages"]


def test_transform_anthropic_shape_full_request_uses_invoke_model():
    """End-to-end: a request with Anthropic-shape content blocks should land
    in InvokeModel format, base64-encoded, with the original body preserved.
    """
    import base64
    import json

    config = BedrockCountTokensConfig()
    req = {
        "model": "anthropic.claude-3-sonnet-20240229-v1:0",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        ],
        "system": [{"type": "text", "text": "Be helpful"}],
        "tools": [
            {
                "name": "search",
                "description": "Search",
                "input_schema": {"type": "object", "properties": {}},
            },
        ],
    }

    result = config.transform_anthropic_to_bedrock_count_tokens(req)

    assert "invokeModel" in result["input"]
    body = json.loads(
        base64.b64decode(result["input"]["invokeModel"]["body"]).decode("utf-8")
    )
    assert body["messages"] == req["messages"]
    assert body["system"] == req["system"]
    assert body["tools"] == req["tools"]


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
