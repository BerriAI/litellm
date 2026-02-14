from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import LiteLLMAnthropicMessagesAdapter

def test_anthropic_openai_complex_strict_schema():
    """Verify recursive injection of additionalProperties and required fields in complex schemas."""
    adapter = LiteLLMAnthropicMessagesAdapter()
    
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
    
    # Verify Root Object
    assert schema["additionalProperties"] is False
    assert "users" in schema["required"]
    
    # Verify Nested Object in Array
    user_schema = schema["properties"]["users"]["items"]
    assert user_schema["additionalProperties"] is False
    assert "name" in user_schema["required"]
    print("\n✅ Success: Complex recursive strict schema and required fields verified!")
