from litellm import completion
import os
from dotenv import load_dotenv

# Load the .env.example file
load_dotenv(dotenv_path='.env.example')


def test_nebius_completion_response():
    # Verify that the API key is loaded
    assert "NEBIUS_API_KEY" in os.environ, "API key not found in environment variables"

    # Call the completion function with parameters
    response = completion(
        model="nebius/meta-llama/Meta-Llama-3.1-70B-Instruct",
        messages=[
            {
                "role": "user",
                "content": "Tell me about nebius.ai",
            }
        ],
        max_tokens=100,
        temperature=0.2,
    )

    # Assert that a response was received and has the correct structure
    assert response is not None, "No response received from the API"
    assert "choices" in response, "Response does not contain 'choices' field"
    assert isinstance(response["choices"], list), "'choices' field is not a list"
    assert len(response["choices"]) > 0, "'choices' list is empty"
    assert "message" in response["choices"][0], "First choice does not contain 'message' field"
    print("API response received successfully with expected structure")


def test_nebius_streaming_response():
    # Verify that the API key is loaded
    assert "NEBIUS_API_KEY" in os.environ, "API key not found in environment variables"

    # Call the completion function with stream=True
    response = completion(
        model="nebius/meta-llama/Meta-Llama-3.1-70B-Instruct",
        messages=[
            {
                "role": "user",
                "content": "Tell me about nebius.ai",
            }
        ],
        max_tokens=100,
        temperature=0.2,
        stream=True,
    )

    # Check that response is an iterable (streaming)
    assert hasattr(response, "__iter__"), "Response is not an iterable, expected a streaming response"

    # Iterate over streaming response chunks and verify each chunk
    for chunk in response:
        assert chunk is not None, "Received an empty chunk in streaming response"
        assert "choices" in chunk, "Chunk does not contain 'choices' field"
        assert isinstance(chunk["choices"], list), "'choices' field in chunk is not a list"
        assert len(chunk["choices"]) > 0, "'choices' list in chunk is empty"

    print("Streaming response received successfully with expected structure")
