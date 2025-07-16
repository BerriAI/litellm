import httpx
from openai import OpenAI, BadRequestError
import pytest


def generate_key():
    """Generate a key for testing"""
    url = "http://0.0.0.0:4000/key/generate"
    headers = {
        "Authorization": "Bearer sk-1234",
        "Content-Type": "application/json",
    }
    data = {}

    response = httpx.post(url, headers=headers, json=data)
    if response.status_code != 200:
        raise Exception(f"Key generation failed with status: {response.status_code}")
    return response.json()["key"]


def get_test_client():
    """Create OpenAI client with generated key"""
    key = generate_key()
    return OpenAI(api_key=key, base_url="http://0.0.0.0:4000")


def validate_response(response):
    """
    Validate basic response structure from OpenAI responses API
    """
    assert response is not None
    assert hasattr(response, "choices")
    assert len(response.choices) > 0
    assert hasattr(response.choices[0], "message")
    assert hasattr(response.choices[0].message, "content")
    assert isinstance(response.choices[0].message.content, str)
    assert hasattr(response, "id")
    assert isinstance(response.id, str)
    assert hasattr(response, "model")
    assert isinstance(response.model, str)
    assert hasattr(response, "created")
    assert isinstance(response.created, int)
    assert hasattr(response, "usage")
    assert hasattr(response.usage, "prompt_tokens")
    assert hasattr(response.usage, "completion_tokens")
    assert hasattr(response.usage, "total_tokens")


def validate_stream_chunk(chunk):
    """
    Validate streaming chunk structure from OpenAI responses API
    """
    assert chunk is not None
    assert hasattr(chunk, "choices")
    assert len(chunk.choices) > 0
    assert hasattr(chunk.choices[0], "delta")

    # Some chunks might not have content in the delta
    if (
        hasattr(chunk.choices[0].delta, "content")
        and chunk.choices[0].delta.content is not None
    ):
        assert isinstance(chunk.choices[0].delta.content, str)

    assert hasattr(chunk, "id")
    assert isinstance(chunk.id, str)
    assert hasattr(chunk, "model")
    assert isinstance(chunk.model, str)
    assert hasattr(chunk, "created")
    assert isinstance(chunk.created, int)

@pytest.mark.flaky(retries=3, delay=2)
def test_basic_response():
    client = get_test_client()
    response = client.responses.create(
        model="gpt-4o", input="just respond with the word 'ping'"
    )
    print("basic response=", response)

    # get the response
    response = client.responses.retrieve(response.id)
    print("GET response=", response)


    # delete the response
    delete_response = client.responses.delete(response.id)
    print("DELETE response=", delete_response)

    # expect an error when getting the response again since it was deleted
    with pytest.raises(Exception):
        get_response = client.responses.retrieve(response.id)


def test_streaming_response():
    client = get_test_client()
    stream = client.responses.create(
        model="gpt-4o", input="just respond with the word 'ping'", stream=True
    )

    collected_chunks = []
    for chunk in stream:
        print("stream chunk=", chunk)
        collected_chunks.append(chunk)

    assert len(collected_chunks) > 0


def test_bad_request_error():
    client = get_test_client()
    with pytest.raises(BadRequestError):
        # Trigger error with invalid model name
        client.responses.create(model="non-existent-model", input="This should fail")


def test_bad_request_bad_param_error():
    client = get_test_client()
    with pytest.raises(BadRequestError):
        # Trigger error with invalid model name
        client.responses.create(
            model="gpt-4o", input="This should fail", temperature=2000
        )

def test_anthropic_with_responses_api():
    client = get_test_client()
    response = client.responses.create(
        model="anthropic/claude-3-5-sonnet-20240620", 
        input="just respond with the word 'ping'",
        previous_response_id="hi",
    )
    print("anthropic response=", response)


def test_streaming_id_consistency_across_chunks():
    """Test that streaming chunk IDs are consistent and properly encoded"""
    # Use master key directly instead of generating one
    client = OpenAI(api_key="sk-1234", base_url="http://0.0.0.0:4000")
    
    # Make streaming request
    stream = client.responses.create(
        model="gemini-1.5-flash", 
        input="Write a short story about a robot. Take your time with multiple sentences.",
        stream=True
    )
    
    collected_chunks = []
    chunk_ids = []
    
    for chunk in stream:
        collected_chunks.append(chunk)
        # Extract item_id from different chunk types
        if hasattr(chunk, 'item_id'):
            chunk_ids.append(chunk.item_id)
    
    # Verify we got multiple chunks
    assert len(collected_chunks) > 1, "Should have multiple streaming chunks"
    assert len(chunk_ids) > 0, "Should have chunk IDs"
    
    # Verify all chunk IDs are encoded with resp_ prefix
    for chunk_id in chunk_ids:
        assert chunk_id.startswith("resp_"), f"Chunk ID {chunk_id} should start with resp_"
        assert len(chunk_id) > 10, f"Chunk ID {chunk_id} should be encoded, not raw"
    
    print(f"✓ Verified {len(chunk_ids)} streaming chunks all have encoded IDs")


def test_streaming_response_id_as_previous_response_id():
    """Test that streaming response IDs can be used as previous_response_id"""
    # Use master key directly instead of generating one
    client = OpenAI(api_key="sk-1234", base_url="http://0.0.0.0:4000")
    
    # First streaming request
    stream = client.responses.create(
        model="gemini-1.5-flash",
        input="Say hello",
        stream=True
    )
    
    final_response_id = None
    for chunk in stream:
        if hasattr(chunk, 'response') and chunk.response:
            final_response_id = chunk.response.id
            break
    
    assert final_response_id is not None, "Should get final response ID from streaming"
    # Final response ID is not encoded, only streaming chunk IDs are encoded
    assert len(final_response_id) > 10, "Final response ID should be a valid ID"
    
    # Use the streaming response ID as previous_response_id in follow-up request
    follow_up_response = client.responses.create(
        model="gemini-1.5-flash",
        input="Say goodbye", 
        previous_response_id=final_response_id
    )
    
    # Should succeed without error
    assert len(follow_up_response.id) > 10, "Follow-up response should have a valid ID"
    
    print(f"✓ Successfully used streaming response ID {final_response_id} as previous_response_id")
