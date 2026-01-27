
import sys
import os
import pytest
sys.path.insert(0, os.path.abspath("."))

from litellm.litellm_core_utils.prompt_templates.factory import anthropic_messages_pt

def test_anthropic_deduplication_logic():
    """
    Verify that anthropic_messages_pt correctly deduplicates tool calls
    when merging consecutive assistant messages.
    
    Scenario:
    - User message
    - Assistant message with tool call A
    - Assistant message (merged) with tool call A (duplicate) and tool call B (new)
    
    Expected Result:
    - Assistant message contains tool call A and tool call B exactly once.
    """
    messages = [
        {"role": "user", "content": "Weather in Paris?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tool_call_unique_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{}"}
                }
            ]
        },
        {
            "role": "assistant", 
            "content": None,
            "tool_calls": [
                {
                    "id": "tool_call_unique_1", # Duplicate! Should be removed
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{}"}
                },
                {
                    "id": "tool_call_unique_2", # New! Should be kept
                    "type": "function",
                    "function": {"name": "get_time", "arguments": "{}"}
                }
            ]
        }
    ]

    # Run transformation
    # We pass dummy model/provider args as they are required but valid for this test
    result = anthropic_messages_pt(
        messages=messages,
        model="claude-3-opus-20240229", 
        llm_provider="anthropic"
    )

    # Inspect results
    # We expect 1 user message and 1 merged assistant message
    assert len(result) == 2
    assert result[0]["role"] == "user"
    assert result[1]["role"] == "assistant"

    assistant_content = result[1]["content"]
    
    # Filter for tool_use blocks
    tool_uses = [
        b for b in assistant_content 
        if isinstance(b, dict) and b.get("type") == "tool_use"
    ]

    # We expect exactly 2 tool uses (one for unique_1, one for unique_2)
    # The duplicate unique_1 should be gone.
    assert len(tool_uses) == 2
    
    ids = [t["id"] for t in tool_uses]
    assert "tool_call_unique_1" in ids
    assert "tool_call_unique_2" in ids
    assert ids.count("tool_call_unique_1") == 1
    assert ids.count("tool_call_unique_2") == 1
