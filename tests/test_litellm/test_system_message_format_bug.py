"""
Test for GitHub issue #11267 - System message format issue with Ollama + tools
"""

from unittest.mock import patch

@patch("litellm.add_function_to_prompt", True)
def test_system_message_format_issue_reproduction():
    """
    Reproduces the system message format bug from GitHub issue #11267.
    """
    from litellm import completion
    
    # Define test data directly from data.jsonl content
    model = "ollama/custom_model_name"  # Use explicit Ollama model
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is the capital of France?"
                }
            ]
        },
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are Claude Code, Anthropic's official CLI for Claude.",
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        }
    ]
    
    temperature = 1
    
    # Add tools to trigger the bug - this is what causes the issue
    tools = [
        {
            "type": "function", 
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }
            }
        }
    ]

    response = completion(
        model=model,
        messages=messages,
        tools=tools,
        temperature=temperature,
        mock_response=True
    )

    assert len(messages[1]["content"]) == 2


if __name__ == "__main__":
    print("Testing system message format issue...")
    test_system_message_format_issue_reproduction()
    print("Tests completed!")