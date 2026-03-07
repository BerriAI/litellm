
import pytest
from litellm.litellm_core_utils.prompt_templates.factory import _get_thought_signature_from_tool


def test_get_thought_signature_from_tool_normalization():
    """
    Test that _get_thought_signature_from_tool correctly handles 'thought_signatures' (list)
    and returns the first element as a string.
    """
    signature = "test_signature_123"
    
    # Case 1: thought_signatures (plural list) - The issue we fixed
    tool_with_plural = {
        "type": "function",
        "function": {
            "name": "test_func",
            "provider_specific_fields": {
                "thought_signatures": [signature]
            }
        }
    }
    
    assert _get_thought_signature_from_tool(tool_with_plural) == signature

    # Case 2: thought_signature (singular string) - The existing behavior
    tool_with_singular = {
        "type": "function",
        "function": {
            "name": "test_func",
            "provider_specific_fields": {
                "thought_signature": signature
            }
        }
    }
    
    assert _get_thought_signature_from_tool(tool_with_singular) == signature

    # Case 3: Both present (singular should take precedence if we kept that logic, 
    # but in our fix we did `get(singular) or get(plural)`. 
    # Let's verify behavior. fix was: `provider_fields.get("thought_signature") or provider_fields.get("thought_signatures")`
    # So singular takes precedence if truthy.
    
    tool_with_both = {
        "type": "function",
        "function": {
            "name": "test_func",
            "provider_specific_fields": {
                "thought_signature": "singular_pref",
                "thought_signatures": ["plural_ignored"]
            }
        }
    }
    assert _get_thought_signature_from_tool(tool_with_both) == "singular_pref"


def test_gemini_transformation_thought_signatures_normalization():
    """
    Test that _gemini_convert_messages_with_history correctly handles 'thought_signatures' 
    in assistant messages (converting list to single thoughtSignature).
    """
    from litellm.llms.vertex_ai.gemini.transformation import _gemini_convert_messages_with_history
    
    signature = "test_signature_abc"
    
    # Input messages simulating an assistant message with thought_signatures (plural)
    messages = [
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant", 
            "content": "Hi there",
            "provider_specific_fields": {
                "thought_signatures": [signature]
            }
        }
    ]
    
    # Run transformation
    contents = _gemini_convert_messages_with_history(messages)
    
    # Verify output
    # content[0] is user, content[1] is model (assistant)
    assert len(contents) == 2
    assert contents[1]["role"] == "model"
    assert len(contents[1]["parts"]) == 1
    part = contents[1]["parts"][0]
    
    # Verify thoughtSignature is present and normalized
    assert part["text"] == "Hi there"
    assert "thoughtSignature" in part
    assert part["thoughtSignature"] == signature

    # Test case 2: Singular 'thought_signature' (backward compatibility)
    messages_singular = [
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant", 
            "content": "Hi there",
            "provider_specific_fields": {
                "thought_signature": signature
            }
        }
    ]
    
    contents_singular = _gemini_convert_messages_with_history(messages_singular)
    part_singular = contents_singular[1]["parts"][0]
    assert part_singular["thoughtSignature"] == signature
