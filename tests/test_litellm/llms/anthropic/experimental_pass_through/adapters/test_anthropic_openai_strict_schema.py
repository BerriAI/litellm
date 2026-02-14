from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import LiteLLMAnthropicMessagesAdapter

def test_anthropic_openai_full_strict_compliance():
    """Verify recursive injection of additionalProperties: false and required fields."""
    adapter = LiteLLMAnthropicMessagesAdapter()
    
    # 1. Complex Nested Schema
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
    schema = result["json_schema"]["schema"]
    
    # Check Root Level
    assert schema["additionalProperties"] is False
    assert "users" in schema["required"]
    
    # Check Nested Object Level
    user_schema = schema["properties"]["users"]["items"]
    assert user_schema["additionalProperties"] is False
    assert "name" in user_schema["required"]
    
    # 2. Edge Case: Overriding 'True'
    override_format = {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {"id": {"type": "integer"}},
            "additionalProperties": True 
        }
    }
    
    override_result = adapter.translate_anthropic_output_format_to_openai(override_format)
    override_schema = override_result["json_schema"]["schema"]
    
    assert override_schema["additionalProperties"] is False
    assert "id" in override_schema["required"]
    
    print("\n✅ Success: Full strict mode compliance verified (additionalProperties & required)!")
