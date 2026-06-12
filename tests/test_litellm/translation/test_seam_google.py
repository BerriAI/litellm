"""The completion() forks for the google routes: flag-gated v2 execution
with fail-closed fallback to v1.

respx intercepts the vertex/gemini endpoints, the vertex token fetch is
stubbed at ``VertexBase.get_access_token`` (the corpus chokepoint), and both
the v2 pipeline and the v1 fallback run for real against the mocked wire.
"""

import json

import pytest
import respx
from httpx import Response

import litellm

_VERTEX_GEMINI_URL = (
    "https://us-central1-aiplatform.googleapis.com/v1/projects/char-test-project"
    "/locations/us-central1/publishers/google/models/gemini-2.5-pro:generateContent"
)
_STUDIO_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models"
    "/gemini-2.5-flash:generateContent"
)
_VERTEX_CLAUDE_URL = (
    "https://us-east5-aiplatform.googleapis.com/v1/projects/char-test-project"
    "/locations/us-east5/publishers/anthropic/models"
    "/claude-sonnet-4@20250514:rawPredict"
)

_GEMINI_RESPONSE = {
    "candidates": [
        {
            "content": {"role": "model", "parts": [{"text": "Hello from v2."}]},
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 10,
        "candidatesTokenCount": 5,
        "totalTokenCount": 15,
        "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 10}],
        "candidatesTokensDetails": [{"modality": "TEXT", "tokenCount": 5}],
    },
    "modelVersion": "gemini-2.5-pro",
    "responseId": "seam-gemini-0001",
}

_CLAUDE_RESPONSE = {
    "id": "msg_seam_vertex_01",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-20250514",
    "content": [{"type": "text", "text": "Hello from v2."}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 10, "output_tokens": 5},
}


@pytest.fixture(autouse=True)
def _google_flag(monkeypatch, vertex_token_stub):
    monkeypatch.setattr(
        litellm,
        "translation_v2_providers",
        ["vertex_ai", "gemini", "vertex_anthropic"],
    )
    monkeypatch.setenv("VERTEXAI_PROJECT", "char-test-project")
    monkeypatch.setenv("VERTEXAI_LOCATION", "us-central1")
    monkeypatch.setenv("GEMINI_API_KEY", "char-gemini-test-key")


@respx.mock
def test_vertex_gemini_flag_on_serves_through_v2() -> None:
    route = respx.post(_VERTEX_GEMINI_URL).mock(
        return_value=Response(200, json=_GEMINI_RESPONSE)
    )
    response = litellm.completion(
        model="vertex_ai/gemini-2.5-pro",
        max_tokens=32,
        messages=[{"role": "user", "content": "hi"}],
    )
    assert route.called
    request = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer char-vertex-token"
    sent = json.loads(request.content)
    assert sent == {
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "generationConfig": {"max_output_tokens": 32},
    }
    assert response.choices[0].message.content == "Hello from v2."
    assert response.id == "seam-gemini-0001"
    assert response.usage.prompt_tokens == 10
    assert response.usage.total_tokens == 15


@respx.mock
def test_studio_flag_on_serves_through_v2() -> None:
    route = respx.post(_STUDIO_URL).mock(
        return_value=Response(200, json=_GEMINI_RESPONSE)
    )
    response = litellm.completion(
        model="gemini/gemini-2.5-flash",
        max_tokens=32,
        messages=[{"role": "user", "content": "hi"}],
    )
    assert route.called
    request = route.calls.last.request
    assert request.headers["x-goog-api-key"] == "char-gemini-test-key"
    assert response.choices[0].message.content == "Hello from v2."


@respx.mock
def test_vertex_claude_flag_on_serves_through_v2(monkeypatch) -> None:
    monkeypatch.setenv("VERTEXAI_LOCATION", "us-east5")
    route = respx.post(_VERTEX_CLAUDE_URL).mock(
        return_value=Response(200, json=_CLAUDE_RESPONSE)
    )
    response = litellm.completion(
        model="vertex_ai/claude-sonnet-4@20250514",
        max_tokens=32,
        messages=[{"role": "user", "content": "hi"}],
    )
    assert route.called
    request = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer char-vertex-token"
    sent = json.loads(request.content)
    assert sent["anthropic_version"] == "vertex-2023-10-16"
    assert "model" not in sent
    assert "json_mode" not in sent
    assert response.choices[0].message.content == "Hello from v2."
    assert response.model == "claude-sonnet-4@20250514"


@respx.mock
def test_unsupported_shape_falls_back_to_v1() -> None:
    route = respx.post(_VERTEX_GEMINI_URL).mock(
        return_value=Response(200, json=_GEMINI_RESPONSE)
    )
    response = litellm.completion(
        model="vertex_ai/gemini-2.5-pro",
        max_tokens=32,
        # n (candidate_count) is outside the v2 inbound surface: the request
        # must be served (by v1), never rejected or dropped.
        n=2,
        messages=[{"role": "user", "content": "hi"}],
    )
    assert route.called
    sent = json.loads(route.calls.last.request.content)
    assert sent["generationConfig"].get("candidate_count") == 2  # v1 served it
    assert response.choices[0].message.content == "Hello from v2."


@respx.mock
def test_provider_error_raises_contract_not_fallback() -> None:
    respx.post(_VERTEX_GEMINI_URL).mock(
        return_value=Response(429, json={"error": {"message": "quota"}})
    )
    with pytest.raises(litellm.exceptions.RateLimitError):
        litellm.completion(
            model="vertex_ai/gemini-2.5-pro",
            max_tokens=32,
            messages=[{"role": "user", "content": "hi"}],
            num_retries=0,
        )
