"""
Tests that client headers are forwarded to the provider on the OpenAI
text completion path.

Regression tests for https://github.com/BerriAI/litellm/issues/27410
"""


import pytest
import respx
from httpx import Response


import litellm
from litellm import atext_completion, text_completion


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key")


@pytest.fixture
def mock_completions_endpoint():
    return respx.post("https://api.openai.com/v1/completions").mock(
        return_value=Response(
            200,
            json={
                "id": "cmpl-test123",
                "object": "text_completion",
                "created": 1677652288,
                "model": "gpt-3.5-turbo-instruct",
                "choices": [
                    {
                        "text": "hi",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )
    )


@respx.mock
def test_completion_forwards_client_headers_to_provider(mock_completions_endpoint):
    text_completion(
        model="gpt-3.5-turbo-instruct",
        prompt="hello",
        max_tokens=5,
        headers={"x-mycorp-llmcall-id": "abc-123"},
    )

    request_headers = mock_completions_endpoint.calls.last.request.headers
    assert request_headers["x-mycorp-llmcall-id"] == "abc-123"


@respx.mock
def test_completion_forwards_extra_headers_to_provider(mock_completions_endpoint):
    text_completion(
        model="gpt-3.5-turbo-instruct",
        prompt="hello",
        max_tokens=5,
        extra_headers={"x-mycorp-llmcall-id": "abc-123"},
    )

    request_headers = mock_completions_endpoint.calls.last.request.headers
    assert request_headers["x-mycorp-llmcall-id"] == "abc-123"


@respx.mock
async def test_acompletion_forwards_client_headers_to_provider(
    mock_completions_endpoint, monkeypatch
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    await atext_completion(
        model="gpt-3.5-turbo-instruct",
        prompt="hello",
        max_tokens=5,
        headers={"x-mycorp-llmcall-id": "abc-123"},
    )

    request_headers = mock_completions_endpoint.calls.last.request.headers
    assert request_headers["x-mycorp-llmcall-id"] == "abc-123"


@pytest.fixture
def rejected_completions_endpoint():
    return respx.post("https://api.openai.com/v1/completions").mock(
        return_value=Response(
            400,
            json={"error": {"message": "bad prompt", "type": "invalid_request_error"}},
            headers={"x-request-id": "req-1"},
        )
    )


@respx.mock
def test_completion_surfaces_the_status_and_headers_of_a_rejected_request(rejected_completions_endpoint):
    with pytest.raises(litellm.BadRequestError) as rejected:
        text_completion(model="gpt-3.5-turbo-instruct", prompt="hello")

    assert rejected.value.status_code == 400
    assert rejected.value.litellm_response_headers["x-request-id"] == "req-1"


@respx.mock
def test_streaming_completion_surfaces_the_status_and_headers_of_a_rejected_request(rejected_completions_endpoint):
    with pytest.raises(litellm.BadRequestError) as rejected:
        list(text_completion(model="gpt-3.5-turbo-instruct", prompt="hello", stream=True))

    assert rejected.value.status_code == 400
    assert rejected.value.litellm_response_headers["x-request-id"] == "req-1"


@respx.mock
async def test_acompletion_surfaces_the_status_and_headers_of_a_rejected_request(
    rejected_completions_endpoint, monkeypatch
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    with pytest.raises(litellm.BadRequestError) as rejected:
        await atext_completion(model="gpt-3.5-turbo-instruct", prompt="hello")

    assert rejected.value.status_code == 400
    assert rejected.value.litellm_response_headers["x-request-id"] == "req-1"


@respx.mock
async def test_async_streaming_completion_reports_an_error_event_sent_mid_stream(monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx.post("https://api.openai.com/v1/completions").mock(
        return_value=Response(
            200,
            text='data: {"error": {"message": "upstream overloaded", "type": "server_error"}}\n\n',
            headers={"content-type": "text/event-stream"},
        )
    )
    with pytest.raises(litellm.InternalServerError) as failed:
        async for _ in await atext_completion(model="gpt-3.5-turbo-instruct", prompt="hello", stream=True):
            pass

    assert failed.value.status_code == 500
    assert "upstream overloaded" in str(failed.value)
