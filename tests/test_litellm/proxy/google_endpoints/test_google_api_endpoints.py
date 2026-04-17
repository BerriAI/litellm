#!/usr/bin/env python3
"""
Test to verify the Google GenAI proxy API endpoints
"""
import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path


def _make_app():
    """Create a test FastAPI app with the google router and a ProxyException handler."""
    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        from fastapi.testclient import TestClient

        from litellm.proxy._types import ProxyException, UserAPIKeyAuth
        from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
        from litellm.proxy.google_endpoints.endpoints import router as google_router
    except ImportError as e:
        return None, None

    app = FastAPI()
    # Bypass authentication in tests — user_api_key_auth enforces API key validation
    # which requires a running proxy and database. Override with a no-op stub.
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        api_key="sk-test"
    )
    app.include_router(google_router)

    @app.exception_handler(ProxyException)
    async def proxy_exception_handler(request, exc):
        return JSONResponse(
            status_code=int(exc.code) if exc.code else 500,
            content={
                "error": {
                    "message": exc.message,
                    "type": exc.type,
                    "code": str(exc.code),
                }
            },
        )

    return app, TestClient(app)


# ---------------------------------------------------------------------------
# Success path tests
# ---------------------------------------------------------------------------


def test_google_generate_content_endpoint():
    """Endpoint exists and returns 200 for a valid model response."""
    app, client = _make_app()
    if app is None:
        pytest.skip("Missing dependencies")

    from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

    with patch.object(
        ProxyBaseLLMRequestProcessing,
        "base_process_llm_request",
        new=AsyncMock(return_value={"candidates": []}),
    ):
        response = client.post(
            "/v1beta/models/gemini-2.0-flash:generateContent",
            json={"contents": [{"role": "user", "parts": [{"text": "Hello"}]}]},
        )
        assert response.status_code == 200


def test_google_stream_generate_content_endpoint():
    """Stream endpoint exists and returns 200 for a valid model response."""
    app, client = _make_app()
    if app is None:
        pytest.skip("Missing dependencies")

    from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

    with patch.object(
        ProxyBaseLLMRequestProcessing,
        "base_process_llm_request",
        new=AsyncMock(return_value={"candidates": []}),
    ):
        response = client.post(
            "/v1beta/models/gemini-2.0-flash:streamGenerateContent",
            json={"contents": [{"role": "user", "parts": [{"text": "Hello"}]}]},
        )
        assert response.status_code == 200


def test_google_generate_content_passes_generationConfig_as_config():
    """generationConfig from the request body is transformed to 'config' before routing."""
    app, client = _make_app()
    if app is None:
        pytest.skip("Missing dependencies")

    from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

    captured = {}

    async def capture_call(self, **kwargs):
        captured.update(self.data)
        return {"candidates": []}

    with patch.object(
        ProxyBaseLLMRequestProcessing,
        "base_process_llm_request",
        capture_call,
    ):
        client.post(
            "/v1beta/models/gemini-2.0-flash:generateContent",
            json={
                "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
                "generationConfig": {"temperature": 0.5, "maxOutputTokens": 100},
            },
        )

    assert "config" in captured
    assert captured["config"]["temperature"] == 0.5
    assert "generationConfig" not in captured


def test_google_generate_content_with_system_instruction():
    """systemInstruction is forwarded to the processor unchanged."""
    app, client = _make_app()
    if app is None:
        pytest.skip("Missing dependencies")

    from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

    captured = {}

    async def capture_call(self, **kwargs):
        captured.update(self.data)
        return {"candidates": []}

    with patch.object(
        ProxyBaseLLMRequestProcessing,
        "base_process_llm_request",
        capture_call,
    ):
        system_instruction = {"parts": [{"text": "Your name is Doodle."}]}
        client.post(
            "/v1beta/models/gemini-2.5-pro:generateContent",
            json={
                "systemInstruction": system_instruction,
                "contents": [
                    {"parts": [{"text": "What is your name?"}], "role": "user"}
                ],
            },
        )

    assert "systemInstruction" in captured
    assert captured["systemInstruction"] == system_instruction


def test_google_stream_generate_content_sets_stream_true():
    """stream=True is enforced for the stream endpoint."""
    app, client = _make_app()
    if app is None:
        pytest.skip("Missing dependencies")

    from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

    captured = {}

    async def capture_call(self, **kwargs):
        captured.update(self.data)
        return {"candidates": []}

    with patch.object(
        ProxyBaseLLMRequestProcessing,
        "base_process_llm_request",
        capture_call,
    ):
        client.post(
            "/v1beta/models/gemini-2.0-flash:streamGenerateContent",
            json={"contents": [{"role": "user", "parts": [{"text": "Hello"}]}]},
        )

    assert captured.get("stream") is True


# ---------------------------------------------------------------------------
# Error path tests — the bug fix
# ---------------------------------------------------------------------------


def test_google_generate_content_returns_400_for_nonexistent_model():
    """
    When the router raises BadRequestError for a non-existent model, the endpoint
    must return 400 (not 500).

    Regression test for: /v1beta/models/{model}:generateContent returning 500
    instead of 400/404 for models not in the router's model list.

    _handle_llm_api_exception is NOT mocked here — the real conversion logic
    (BadRequestError.status_code=400 → ProxyException(code=400)) is exercised
    end-to-end through the HTTP response.
    """
    from unittest.mock import AsyncMock, MagicMock

    app, client = _make_app()
    if app is None:
        pytest.skip("Missing dependencies")

    import litellm
    from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

    bad_request_error = litellm.BadRequestError(
        message="litellm.BadRequestError: You passed in model=nonexistent-model. "
        "There are no healthy deployments for this model.",
        model="nonexistent-model",
        llm_provider="",
    )

    mock_logging = MagicMock()
    mock_logging.post_call_failure_hook = AsyncMock(return_value=None)
    mock_logging.post_call_response_headers_hook = AsyncMock(return_value=None)

    with patch.object(
        ProxyBaseLLMRequestProcessing,
        "base_process_llm_request",
        side_effect=bad_request_error,
    ), patch("litellm.proxy.proxy_server.proxy_logging_obj", mock_logging):
        response = client.post(
            "/v1beta/models/nonexistent-model:generateContent",
            json={"contents": [{"role": "user", "parts": [{"text": "hello"}]}]},
        )

    assert response.status_code == 400, (
        f"Expected 400 Bad Request, got {response.status_code}. "
        "Non-existent model should be a client error, not a server error."
    )


def test_google_stream_generate_content_returns_400_for_nonexistent_model():
    """
    Stream endpoint must also return 400 (not 500) for a non-existent model.
    _handle_llm_api_exception is NOT mocked — the real conversion logic is exercised.
    """
    from unittest.mock import AsyncMock, MagicMock

    app, client = _make_app()
    if app is None:
        pytest.skip("Missing dependencies")

    import litellm
    from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

    bad_request_error = litellm.BadRequestError(
        message="litellm.BadRequestError: You passed in model=nonexistent-model. "
        "There are no healthy deployments for this model.",
        model="nonexistent-model",
        llm_provider="",
    )

    mock_logging = MagicMock()
    mock_logging.post_call_failure_hook = AsyncMock(return_value=None)
    mock_logging.post_call_response_headers_hook = AsyncMock(return_value=None)

    with patch.object(
        ProxyBaseLLMRequestProcessing,
        "base_process_llm_request",
        side_effect=bad_request_error,
    ), patch("litellm.proxy.proxy_server.proxy_logging_obj", mock_logging):
        response = client.post(
            "/v1beta/models/nonexistent-model:streamGenerateContent",
            json={"contents": [{"role": "user", "parts": [{"text": "hello"}]}]},
        )

    assert response.status_code == 400, (
        f"Expected 400 Bad Request, got {response.status_code}. "
        "Non-existent model should be a client error, not a server error."
    )
