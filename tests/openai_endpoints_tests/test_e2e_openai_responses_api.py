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
        model="gpt-5.5", input="just respond with the word 'ping'"
    )
    print("basic response=", response)

    # get the response
    response = client.responses.retrieve(response.id)
    print("GET response=", response)

    # delete the response
    delete_response = client.responses.delete(response.id)
    print("DELETE response=", delete_response)

    # expect an error when getting the response again since it was deleted
    with pytest.raises(APIStatusError):
        get_response = client.responses.retrieve(response.id)


def test_streaming_response():
    client = get_test_client()
    stream = client.responses.create(
        model="gpt-5.5", input="just respond with the word 'ping'", stream=True
    )

    collected_chunks = []
    for chunk in stream:
        print("stream chunk=", chunk)
        collected_chunks.append(chunk)

    assert len(collected_chunks) > 0


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
            model="gpt-5.5", input="just respond with the word 'ping'", background=True
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


def test_cancel_streaming_response():
    client: Final = get_test_client()
    started: Final = time.monotonic()
    stream: Final = client.responses.create(
        model="gpt-5.5",
        input="count from 1 to 500, one number per line",
        stream=True,
        background=True,
        timeout=BACKGROUND_STREAM_ADMISSION_DEADLINE_SECONDS,
    )

    with stream:
        events: Final = tuple(events_until_admission(stream, started))

    elapsed: Final = time.monotonic() - started
    keepalive_events: Final = sum(1 for chunk in events if chunk.type == "keepalive")
    response_id: Final = next((rid for rid in map(admitted_response_id, events) if rid is not None), None)
    if response_id is None and keepalive_events:
        pytest.skip(
            f"OpenAI held the background stream in keepalive for {elapsed:.0f}s "
            f"({keepalive_events} keepalive events) without creating the response"
        )
    assert response_id is not None, f"no response event within {elapsed:.0f}s of streaming a background response"

    cancel_response: Final = client.responses.cancel(response_id)
    print("CANCEL streaming response=", cancel_response)
    assert cancel_response.status == "cancelled"


def test_cancel_invalid_response_id():
    client = get_test_client()
    with pytest.raises(APIStatusError):
        # Try to cancel a non-existent response ID
        client.responses.cancel("invalid_response_id_12345")
