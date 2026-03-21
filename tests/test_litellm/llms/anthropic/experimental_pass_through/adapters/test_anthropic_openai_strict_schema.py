import pytest
from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import LiteLLMAnthropicMessagesAdapter

def test_anthropic_openai_full_strict_compliance():
    """Verify recursive injection of additionalProperties: false and full required arrays."""
    adapter = LiteLLMAnthropicMessagesAdapter()

    # 1. Test: Complex Nested Schema + Array items
    output_format = {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {
                "users": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}}
                    }
                }
            }
        }
    }

    result = adapter.translate_anthropic_output_format_to_openai(output_format)
    # FIX: proving result is not None to resolve Pylance
    assert result is not None
    schema = result["json_schema"]["schema"]

    # Check Root
    assert schema["additionalProperties"] is False
    assert "users" in schema["required"]

    # Check Nested Array Object
    user_schema = schema["properties"]["users"]["items"]
    assert user_schema["additionalProperties"] is False
    assert "name" in user_schema["required"]

    # 2. Test: Overriding pre-existing additionalProperties: True
    override_format = {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {"id": {"type": "integer"}},
            "additionalProperties": True 
        }
    }

    override_result = adapter.translate_anthropic_output_format_to_openai(override_format)
    # FIX: proving result is not None to resolve Pylance
    assert override_result is not None
    assert override_result["json_schema"]["schema"]["additionalProperties"] is False

    # 3. Test: Fixing partial required array (The latest feedback fix)
    partial_format = {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {
                "first_name": {"type": "string"},
                "last_name": {"type": "string"}
            },
            "required": ["first_name"]  # "last_name" is missing
        }
    }

    partial_result = adapter.translate_anthropic_output_format_to_openai(partial_format)
    # FIX: proving result is not None to resolve Pylance
    assert partial_result is not None
    final_required = partial_result["json_schema"]["schema"]["required"]
    assert set(final_required) == {"first_name", "last_name"}