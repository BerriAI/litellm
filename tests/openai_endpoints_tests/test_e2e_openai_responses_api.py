import os
import time
from collections.abc import Iterator
from typing import Final

import httpx
import pytest
from openai import APIStatusError, BadRequestError, NotFoundError, OpenAI, Stream
from openai.types.responses import ResponseStreamEvent

BACKGROUND_STREAM_ADMISSION_DEADLINE_SECONDS: Final = 90


def generate_key():
    """Generate a key for testing"""
    url = "http://0.0.0.0:4000/key/generate"
    headers = {
        "Authorization": f"Bearer {os.environ['LITELLM_MASTER_KEY']}",
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


def test_model_not_found_error():
    client = get_test_client()
    with pytest.raises(NotFoundError):
        client.responses.create(model="non-existent-model", input="This should fail")


def test_bad_request_bad_param_error():
    client = get_test_client()
    with pytest.raises(BadRequestError):
        # Out-of-range temperature on a non-reasoning model, so drop_params forwards it
        client.responses.create(
            model="gpt-4.1", input="This should fail", temperature=2000
        )


def admitted_response_id(chunk: ResponseStreamEvent) -> str | None:
    response: Final = getattr(chunk, "response", None)
    return None if response is None else response.id


def events_until_admission(stream: Stream[ResponseStreamEvent], started: float) -> Iterator[ResponseStreamEvent]:
    for chunk in stream:
        print("stream chunk=", chunk)
        yield chunk
        if admitted_response_id(chunk) is not None:
            return
        if time.monotonic() - started > BACKGROUND_STREAM_ADMISSION_DEADLINE_SECONDS:
            return


def test_cancel_invalid_response_id():
    client = get_test_client()
    with pytest.raises(APIStatusError):
        # Try to cancel a non-existent response ID
        client.responses.cancel("invalid_response_id_12345")
