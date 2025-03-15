import os
import pytest
from openai import OpenAI
from pydantic import BaseModel
import litellm as llm

# Don't hardcode API keys in code - we'll use a fixture to manage this
# and get the key from environment variable

class OutputSchema(BaseModel):
    message: str
    title: str
    model_config = dict(extra="forbid")

@pytest.fixture
def tools():
    return [
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
                    "additionalProperties": False
                },
            },
        }
    ]

@pytest.fixture
def messages():
    return [
        {
            "role": "system",
            "content": f"Use tools to formulate your answer. Later, Please use "
                    f"JSON for the response with the following schema: "
                    f"{OutputSchema.model_json_schema()}",
        },
        {
            "role": "user",
            "content": "What is the weather in San Francisco?",
        },
    ]

@pytest.fixture
def openai_client():
    # Get API key from environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY environment variable not set")
    return OpenAI(api_key=api_key)

def test_tool_call_index_preservation(openai_client, tools, messages):
    """
    Test that LiteLLM properly preserves the index value in streaming tool calls with n>1.
    """
    print("Testing index preservation in tool calls with n=3...")
    
    
    # Get responses from both APIs
    response_oai = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        stream=True,
        n=3
    )
    
    response_litellm = llm.completion(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        stream=True,
        n=3
    )
    
    # Collect and compare indices
    indices_oai = []
    indices_litellm = []
    
    # Only compare the first 20 chunks
    for i, (oai_chunk, litellm_chunk) in enumerate(zip(response_oai, response_litellm)):
        # print(f"\nchunk:", len(oai_chunk.choices), oai_chunk.choices[0].index, litellm_chunk.choices[0].delta.tool_calls[0])
        if i >= 20:
            break
            
        oai_index = oai_chunk.choices[0].index if len(oai_chunk.choices) > 0 else None
        litellm_index = litellm_chunk.choices[0].index if len(litellm_chunk.choices) > 0 else None
        
        indices_oai.append(oai_index)
        indices_litellm.append(litellm_index)
    
    # Count occurrences of each index
    oai_index_counts = {}
    litellm_index_counts = {}
    
    for idx in indices_oai:
        if idx is not None:
            oai_index_counts[idx] = oai_index_counts.get(idx, 0) + 1
            
    for idx in indices_litellm:
        if idx is not None:
            litellm_index_counts[idx] = litellm_index_counts.get(idx, 0) + 1
    
    # Print comparison of index counts
    print("\nIndex counts comparison (OpenAI, LiteLLM):")
    all_indices = sorted(set(oai_index_counts.keys()) | set(litellm_index_counts.keys()))
    for idx in all_indices:
        oai_count = oai_index_counts.get(idx, 0)
        litellm_count = litellm_index_counts.get(idx, 0)
        print(f"Index {idx}: OpenAI={oai_count}, LiteLLM={litellm_count}")
        
        # Assert that counts match for each index
        assert oai_count == litellm_count, f"Mismatch in count for index {idx}: OpenAI={oai_count}, LiteLLM={litellm_count}"
    
    # Print raw indices for debugging
    print("\nRaw indices (OpenAI, LiteLLM):")
    for i, (oai, litellm) in enumerate(zip(indices_oai, indices_litellm)):
        print(f"{oai}, {litellm}")
