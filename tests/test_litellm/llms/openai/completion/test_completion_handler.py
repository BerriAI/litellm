"""
Tests that client headers are forwarded to the provider on the OpenAI
text completion path.

Regression tests for https://github.com/BerriAI/litellm/issues/27410
"""

import os
import sys

import pytest
import respx
from httpx import Response

sys.path.insert(0, os.path.abspath("../../../../.."))

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
