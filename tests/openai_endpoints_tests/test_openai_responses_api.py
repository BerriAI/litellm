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
    assert isinstance(response.choices[0].message.content, str)
    assert len(response.choices) > 0


def validate_stream_chunk(chunk):
    """
    Validate streaming chunk structure from OpenAI responses API
    """
    assert chunk is not None
    assert isinstance(chunk.choices[0].delta.content, str)


def test_basic_response():
    client = get_test_client()
    response = client.responses.create(
        model="gpt-4", input="Tell me a three sentence bedtime story about a unicorn."
    )
    validate_response(response)


def test_streaming_response():
    client = get_test_client()
    stream = client.responses.create(
        model="gpt-4", input="Tell me a story", stream=True
    )

    collected_chunks = []
    for chunk in stream:
        validate_stream_chunk(chunk)
        collected_chunks.append(chunk)

    assert len(collected_chunks) > 0


def test_bad_request_error():
    client = get_test_client()
    with pytest.raises(BadRequestError):
        # Trigger error with invalid model name
        client.responses.create(model="non-existent-model", input="This should fail")
