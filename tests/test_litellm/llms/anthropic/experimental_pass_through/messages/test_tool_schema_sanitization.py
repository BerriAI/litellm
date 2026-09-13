"""Tests for tool ``input_schema`` normalization on the Anthropic /v1/messages route.

Regression coverage for #24121: tools whose ``input_schema`` omits ``type`` (as Claude
Code / MCP clients emit, declaring a JSON-schema ``$schema`` draft reference instead) are
forwarded to Anthropic unchanged on the beta ``/v1/messages`` pass-through and rejected
with::

    tools.0.custom.input_schema.type: Input should be 'object'

The root cause is the missing ``type``, not the ``$schema`` field (Anthropic tolerates
``$schema`` when ``type: object`` is present). The ``/v1/chat/completions`` path already
coerces ``type`` -> ``"object"`` in ``_map_tool_helper``; these tests assert the
pass-through path now does the same (and also drops ``$schema``/``$id`` for parity).
"""

from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)


def _transform(tools):
    return AnthropicMessagesConfig().transform_anthropic_messages_request(
        model="claude-sonnet-4-5",
        messages=[{"role": "user", "content": "Hello"}],
        anthropic_messages_optional_request_params={"max_tokens": 1024, "tools": tools},
        litellm_params={},
        headers={},
    )


def test_missing_type_is_coerced_to_object():
    """THE fix: input_schema without ``type`` must get ``type: "object"`` injected.

    This is exactly the shape that Anthropic rejects with HTTP 400 before the fix.
    """
    tool = {
        "name": "get_weather",
        "description": "Get weather",
        "input_schema": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    }
    input_schema = _transform([tool])["tools"][0]["input_schema"]
    assert input_schema["type"] == "object"
    # $schema/$id are also stripped for parity with the chat path.
    assert "$schema" not in input_schema
    # Legitimate fields preserved.
    assert input_schema["properties"]["location"]["type"] == "string"
    assert input_schema["required"] == ["location"]


def test_non_object_type_is_coerced_and_properties_ensured():
    """A root-level non-object ``type`` is coerced, and ``properties`` is injected."""
    tool = {
        "name": "echo",
        "description": "echo",
        "input_schema": {"type": "string"},
    }
    input_schema = _transform([tool])["tools"][0]["input_schema"]
    assert input_schema["type"] == "object"
    assert input_schema["properties"] == {}


def test_valid_object_schema_is_preserved():
    """A well-formed object schema is left intact (aside from $schema/$id removal)."""
    tool = {
        "name": "ok",
        "description": "ok",
        "input_schema": {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        },
    }
    input_schema = _transform([tool])["tools"][0]["input_schema"]
    assert input_schema["type"] == "object"
    assert input_schema["properties"] == {"x": {"type": "string"}}
    assert input_schema["required"] == ["x"]


def test_schema_refs_stripped_recursively():
    """Nested ``$schema``/``$id`` inside properties are also removed."""
    tool = {
        "name": "nested",
        "description": "nested",
        "input_schema": {
            "type": "object",
            "properties": {
                "inner": {
                    "$id": "https://example.com/x.json",
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                }
            },
        },
    }
    inner = _transform([tool])["tools"][0]["input_schema"]["properties"]["inner"]
    assert "$id" not in inner
    assert inner["properties"]["x"]["type"] == "string"


def test_no_tools_is_a_noop():
    """Requests without tools are unaffected."""
    result = AnthropicMessagesConfig().transform_anthropic_messages_request(
        model="claude-sonnet-4-5",
        messages=[{"role": "user", "content": "Hello"}],
        anthropic_messages_optional_request_params={"max_tokens": 1024},
        litellm_params={},
        headers={},
    )
    assert "tools" not in result or result.get("tools") is None
