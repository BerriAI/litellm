import pytest
import os
from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import LiteLLMAnthropicMessagesAdapter

def test_anthropic_to_openai_strict_transformation():
    adapter = LiteLLMAnthropicMessagesAdapter()
    
    # Mock Anthropic output_format
    anthropic_format = {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"}
            },
            "required": ["answer"]
        }
    }
    
    # Run transformation
    openai_format = adapter.translate_anthropic_output_format_to_openai(anthropic_format)
    
    # Assertions
    assert openai_format["json_schema"]["strict"] is True
    # Verify the fix: additionalProperties should have been injected by _ensure_strict_json_schema
    assert "additionalProperties" in openai_format["json_schema"]["schema"]
    assert openai_format["json_schema"]["schema"]["additionalProperties"] is False
    print("\n✅ Success: additionalProperties: False injected into schema!")

if __name__ == "__main__":
    test_anthropic_to_openai_strict_transformation()
