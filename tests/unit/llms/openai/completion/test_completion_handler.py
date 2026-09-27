"""
Tests that client headers are forwarded to the provider on the OpenAI
text completion path.

Regression tests for https://github.com/BerriAI/litellm/issues/27410
"""


import httpx
import pytest
import respx
from httpx import Response


import litellm
from litellm import atext_completion, text_completion
from litellm.integrations.custom_logger import CustomLogger


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

class _FailureRecorder(CustomLogger):
    def __init__(self):
        self.payloads = []

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self.payloads.append(kwargs["standard_logging_object"])


@pytest.mark.parametrize(
    ("provider_response", "expected_error"),
    [
        pytest.param(httpx.ConnectError("connection refused"), litellm.APIConnectionError, id="connection-refused"),
        pytest.param(Response(500, json={"error": {"message": "boom"}}), litellm.InternalServerError, id="5xx-before-first-byte"),
    ],
)
@respx.mock
async def test_astream_failing_before_first_byte_logs_one_failure(provider_response, expected_error, monkeypatch):
    respx.post("https://api.openai.com/v1/completions").mock(side_effect=provider_response)
    recorder = _FailureRecorder()
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    monkeypatch.setattr(litellm, "_async_failure_callback", [recorder])

    response = await atext_completion(
        model="gpt-3.5-turbo-instruct", prompt="hello", max_tokens=5, stream=True, max_retries=0
    )
    with pytest.raises(expected_error):
        async for _ in response:
            pass

    assert len(recorder.payloads) == 1
    payload = recorder.payloads[0]
    assert payload["status"] == "failure"
    assert payload["custom_llm_provider"] == "text-completion-openai"
    assert payload["model"] == "gpt-3.5-turbo-instruct"
