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
        model="anthropic/claude-sonnet-4-5-20250929", 
        input="just respond with the word 'ping'",
        previous_response_id="hi",
    )
    print("anthropic response=", response)


def test_cancel_response():
    try:
        client = get_test_client()
        from litellm.types.llms.openai import ResponsesAPIResponse
        response = client.responses.create(
            model="gpt-4o", input="just respond with the word 'ping'", background=True
        )
        print("basic response=", response)

        # cancel the response
        cancel_response = client.responses.cancel(response.id)
        print("CANCEL response=", cancel_response)
        
        # verify cancel response structure
        assert hasattr(cancel_response, "id")
    except Exception as e:
        if "Cannot cancel a completed response" in str(e):
            pass
        else:
            raise e


def test_cancel_streaming_response():
    try:
        client = get_test_client()
        from litellm.types.llms.openai import ResponsesAPIResponse
        stream = client.responses.create(
            model="gpt-4o", input="just respond with the word 'ping'", stream=True, background=True
        )

        collected_chunks = []
        response_id = None
        for chunk in stream:
            print("stream chunk=", chunk)
            collected_chunks.append(chunk)
            # Extract response ID from the first chunk that has it
            if response_id is None and hasattr(chunk, 'response') and hasattr(chunk.response, 'id'):
                response_id = chunk.response.id

        assert len(collected_chunks) > 0
        
        # cancel the response if we got a response ID
        if response_id:
            cancel_response = client.responses.cancel(response_id)
            print("CANCEL streaming response=", cancel_response)
            assert hasattr(cancel_response, "id")
    except Exception as e:
        if "Cannot cancel a completed response" in str(e):
            pass
        else:
            raise e


def test_cancel_invalid_response_id():
    client = get_test_client()
    with pytest.raises(Exception):
        # Try to cancel a non-existent response ID
        client.responses.cancel("invalid_response_id_12345")