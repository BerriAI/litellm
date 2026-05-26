#!/usr/bin/env python3
"""
Test to verify the Google GenAI proxy API endpoints
"""
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path


def _build_test_client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from litellm.proxy.google_endpoints.endpoints import router as google_router

    app = FastAPI()
    app.include_router(google_router)
    return TestClient(app)


def _patch_base_process(return_value=None):
    """Patch ProxyBaseLLMRequestProcessing.base_process_llm_request so endpoint
    tests don't run the full pipeline. Returns the AsyncMock so callers can
    inspect call args."""
    if return_value is None:
        return_value = {"test": "response"}
    return patch(
        "litellm.proxy.google_endpoints.endpoints.ProxyBaseLLMRequestProcessing.base_process_llm_request",
        new_callable=AsyncMock,
        return_value=return_value,
    )


def test_google_generate_content_endpoint():
    """generateContent routes through ProxyBaseLLMRequestProcessing with the
    agenerate_content route_type — that pipeline runs pre_call_hook +
    during_call_hook + post_call_success_hook for every guardrail callback."""
    try:
        client = _build_test_client()
    except ImportError as e:
        pytest.skip(f"Skipping test due to missing dependency: {e}")

    with _patch_base_process() as mock_base:
        response = client.post(
            "/v1beta/models/test-model:generateContent",
            json={"contents": [{"role": "user", "parts": [{"text": "Hello"}]}]},
        )

        assert response.status_code == 200
        mock_base.assert_called_once()
        kwargs = mock_base.call_args.kwargs
        assert kwargs["route_type"] == "agenerate_content"
        assert kwargs["model"] == "test-model"


def test_google_stream_generate_content_endpoint():
    """streamGenerateContent must route through the same processor with the
    streaming route_type so the guardrail pipeline runs."""
    try:
        client = _build_test_client()
    except ImportError as e:
        pytest.skip(f"Skipping test due to missing dependency: {e}")

    with (
        _patch_base_process() as mock_base,
        patch(
            "litellm.proxy.google_endpoints.endpoints.ProxyBaseLLMRequestProcessing.__init__",
            return_value=None,
        ) as mock_init,
    ):
        response = client.post(
            "/v1beta/models/test-model:streamGenerateContent",
            json={"contents": [{"role": "user", "parts": [{"text": "Hello"}]}]},
        )

        assert response.status_code == 200
        mock_base.assert_called_once()
        kwargs = mock_base.call_args.kwargs
        assert kwargs["route_type"] == "agenerate_content_stream"
        assert kwargs["model"] == "test-model"

        # stream=True must be forced into the data the processor receives.
        init_kwargs = mock_init.call_args.kwargs
        assert init_kwargs["data"]["stream"] is True
        assert init_kwargs["data"]["model"] == "test-model"
        assert init_kwargs["data"]["contents"] == [
            {"role": "user", "parts": [{"text": "Hello"}]}
        ]


def test_google_generate_content_data_flows_through_processor():
    """The body the client sends must reach ProxyBaseLLMRequestProcessing
    intact so the pipeline can apply guardrails to it."""
    try:
        client = _build_test_client()
    except ImportError as e:
        pytest.skip(f"Skipping test due to missing dependency: {e}")

    with (
        _patch_base_process(),
        patch(
            "litellm.proxy.google_endpoints.endpoints.ProxyBaseLLMRequestProcessing.__init__",
            return_value=None,
        ) as mock_init,
    ):
        client.post(
            "/v1beta/models/test-model:generateContent",
            json={
                "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
                "systemInstruction": {"parts": [{"text": "Your name is Doodle."}]},
                "generationConfig": {
                    "responseModalities": ["TEXT", "IMAGE"],
                    "imageConfig": {"aspectRatio": "9:16", "imageSize": "4K"},
                },
            },
        )

        data = mock_init.call_args.kwargs["data"]
        assert data["model"] == "test-model"
        assert data["contents"] == [{"role": "user", "parts": [{"text": "Hello"}]}]
        assert data["systemInstruction"] == {
            "parts": [{"text": "Your name is Doodle."}]
        }
        # generationConfig arrives intact here; the rename to `config` is
        # done downstream in route_request (see test_route_llm_request).
        assert data["generationConfig"]["responseModalities"] == ["TEXT", "IMAGE"]
        assert data["generationConfig"]["imageConfig"]["aspectRatio"] == "9:16"


def test_google_generate_content_forwards_call_id_header():
    """The endpoint must forward the x-litellm-call-id header to the processor
    so the helper can stamp it on the logging object. Trace continuity from
    client → callbacks (S3, Langfuse, etc.) depends on this header surviving
    the hop through these endpoints."""
    try:
        client = _build_test_client()
    except ImportError as e:
        pytest.skip(f"Skipping test due to missing dependency: {e}")

    with _patch_base_process() as mock_base:
        client.post(
            "/v1beta/models/test-model:generateContent",
            json={"contents": [{"role": "user", "parts": [{"text": "Hello"}]}]},
            headers={"x-litellm-call-id": "trace-abc-123"},
        )

        forwarded_request = mock_base.call_args.kwargs["request"]
        assert forwarded_request.headers.get("x-litellm-call-id") == "trace-abc-123"


def test_google_count_tokens_unchanged():
    """countTokens has its own path and isn't affected by the pipeline change."""
    try:
        client = _build_test_client()
    except ImportError as e:
        pytest.skip(f"Skipping test due to missing dependency: {e}")

    fake_response = MagicMock()
    fake_response.original_response = {
        "totalTokens": 7,
        "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 7}],
    }
    fake_response.total_tokens = 7

    with patch(
        "litellm.proxy.proxy_server.token_counter",
        new_callable=AsyncMock,
        return_value=fake_response,
    ):
        response = client.post(
            "/v1beta/models/test-model:countTokens",
            json={"contents": [{"role": "user", "parts": [{"text": "Hello"}]}]},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["totalTokens"] == 7
