"""End-to-end (mocked HTTP) tests for the Consus provider.

Verifies that `litellm.completion(model="consus/...", ...)`:
  1. Hits `https://api.consus.io/v1/chat/completions`
  2. Sends the API key in `x-api-key` (NOT `Authorization`)
  3. Forwards the model name unprefixed (consus/X -> X) to the gateway
  4. Returns a parsed OpenAI-style response
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

import litellm
from litellm.llms.consus.chat.transformation import CONSUS_API_BASE

TEST_API_KEY = "consus-test-key"
TEST_MODEL_NAME = "claude-sonnet-4-5:il5+itar"
TEST_MODEL = f"consus/{TEST_MODEL_NAME}"


@pytest.fixture(autouse=True)
def _clear_consus_env(monkeypatch):
    monkeypatch.delenv("CONSUS_API_KEY", raising=False)
    monkeypatch.delenv("CONSUS_API_BASE", raising=False)
    monkeypatch.setattr(litellm, "consus_key", None, raising=False)
    yield


def _openai_style_response(content: str) -> dict:
    return {
        "id": "chatcmpl-consus-1",
        "object": "chat.completion",
        "created": 1234567890,
        "model": TEST_MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


class TestConsusCompletionMock:
    def test_request_uses_x_api_key_and_unprefixed_model(self, respx_mock):
        route = respx_mock.post(f"{CONSUS_API_BASE}/chat/completions").respond(
            json=_openai_style_response("hello from consus"),
            status_code=200,
        )

        response = litellm.completion(
            model=TEST_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            api_key=TEST_API_KEY,
        )

        assert response.choices[0].message.content == "hello from consus"  # type: ignore

        assert route.called
        request = route.calls[0].request

        assert request.headers["x-api-key"] == TEST_API_KEY
        assert "authorization" not in {k.lower() for k in request.headers.keys()}

        sent = json.loads(request.content)
        # The `consus/` prefix must be stripped before forwarding —
        # but the `:compliance` suffix must be preserved.
        assert sent["model"] == TEST_MODEL_NAME

    def test_request_uses_default_api_base_when_none_given(self, respx_mock):
        route = respx_mock.post(f"{CONSUS_API_BASE}/chat/completions").respond(
            json=_openai_style_response("ok"),
            status_code=200,
        )

        litellm.completion(
            model="consus/gpt-4.1:il5+itar",
            messages=[{"role": "user", "content": "hi"}],
            api_key=TEST_API_KEY,
        )

        assert route.called

    def test_completion_uses_env_api_key(self, monkeypatch, respx_mock):
        monkeypatch.setenv("CONSUS_API_KEY", "env-resolved-key")
        route = respx_mock.post(f"{CONSUS_API_BASE}/chat/completions").respond(
            json=_openai_style_response("ok"),
            status_code=200,
        )

        litellm.completion(
            model=TEST_MODEL,
            messages=[{"role": "user", "content": "hi"}],
        )

        assert route.called
        assert route.calls[0].request.headers["x-api-key"] == "env-resolved-key"
