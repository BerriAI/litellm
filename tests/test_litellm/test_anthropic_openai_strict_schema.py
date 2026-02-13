from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import LiteLLMAnthropicMessagesAdapter

def test_anthropic_openai_nested_strict_schema():
    """Verify recursive injection of additionalProperties: false in nested schemas."""
    adapter = LiteLLMAnthropicMessagesAdapter()
    
    # A nested schema to test recursion
    output_format = {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}}
                }
            }
        }
    }
    
    result = adapter.translate_anthropic_output_format_to_openai(output_format)
    schema = result["json_schema"]["schema"]
    
    # Check root level
    assert schema["additionalProperties"] is False
    # Check nested object level
    assert schema["properties"]["user"]["additionalProperties"] is False
    print("\n✅ Success: Recursive strict schema verified for nested objects!")
