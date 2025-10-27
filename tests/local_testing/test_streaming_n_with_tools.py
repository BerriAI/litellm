"""
Test for streaming with n>1 and tool calls.
Regression test for: https://github.com/BerriAI/litellm/issues/8977
"""
import pytest
import litellm


@pytest.mark.parametrize("model", ["gpt-4o", "gpt-4-turbo"])
@pytest.mark.asyncio
async def test_streaming_tool_calls_with_n_greater_than_1(model):
    """
    Test that the index field in a choice object is correctly populated
    when using streaming mode with n>1 and tool calls.
    
    Regression test for: https://github.com/BerriAI/litellm/issues/8977
    """
    tools = [
        {
            "type": "function",
            "function": {
                "strict": True,
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["location", "unit"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    
    response = litellm.completion(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "What is the weather in San Francisco?",
            },
        ],
        tools=tools,
        stream=True,
        n=3,
    )
    
    # Collect all chunks and their indices
    indices_seen = []
    for chunk in response:
        assert len(chunk.choices) == 1, "Each streaming chunk should have exactly 1 choice"
        assert hasattr(chunk.choices[0], "index"), "Choice should have an index attribute"
        index = chunk.choices[0].index
        indices_seen.append(index)
    
    # Verify that we got chunks with different indices (0, 1, 2 for n=3)
    unique_indices = set(indices_seen)
    assert unique_indices == {0, 1, 2}, f"Should have indices 0, 1, 2 for n=3, got {unique_indices}"
    
    print(f"✓ Test passed: streaming with n=3 and tool calls correctly populates index field")
    print(f"  Indices seen: {indices_seen}")
    print(f"  Unique indices: {unique_indices}")


@pytest.mark.parametrize("model", ["gpt-4o"])
@pytest.mark.asyncio
async def test_streaming_content_with_n_greater_than_1(model):
    """
    Test that the index field is correctly populated for regular content streaming
    (not tool calls) with n>1.
    """
    response = litellm.completion(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Say hello in one word",
            },
        ],
        stream=True,
        n=2,
        max_tokens=10,
    )
    
    # Collect all chunks and their indices
    indices_seen = []
    for chunk in response:
        assert len(chunk.choices) == 1, "Each streaming chunk should have exactly 1 choice"
        assert hasattr(chunk.choices[0], "index"), "Choice should have an index attribute"
        index = chunk.choices[0].index
        indices_seen.append(index)
    
    # Verify that we got chunks with different indices (0, 1 for n=2)
    unique_indices = set(indices_seen)
    assert unique_indices == {0, 1}, f"Should have indices 0, 1 for n=2, got {unique_indices}"
    
    print(f"✓ Test passed: streaming with n=2 and regular content correctly populates index field")
    print(f"  Indices seen: {indices_seen}")
    print(f"  Unique indices: {unique_indices}")

