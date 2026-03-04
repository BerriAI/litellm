"""
Regression tests for Google Interactions endpoint parameter handling.

These tests ensure `agent` is NOT remapped into `model` for create interaction calls.
"""

import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../.."))


def _build_test_client():
    try:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from litellm.proxy.google_endpoints.endpoints import router as google_router
    except ImportError as e:
        pytest.skip(f"Skipping test due to missing dependency: {e}")

    app = FastAPI()
    app.include_router(google_router)
    return TestClient(app)


def _patch_proxy_server_dependencies():
    """
    Patch proxy_server globals imported inside create_interaction endpoint.

    We patch only to make endpoint invocation deterministic for unit tests.
    """
    return patch.multiple(
        "litellm.proxy.proxy_server",
        general_settings={},
        llm_router=object(),
        proxy_config=object(),
        proxy_logging_obj=object(),
        select_data_generator=None,
        user_api_base=None,
        user_max_tokens=None,
        user_model=None,
        user_request_timeout=None,
        user_temperature=None,
        version="test-version",
    )


def test_interactions_agent_only_keeps_model_none():
    """
    If request contains only `agent`, endpoint must pass `model=None` to processing.

    This prevents accidental payload translation to `{"model": "deep-research-..."}`.
    """
    client = _build_test_client()

    with _patch_proxy_server_dependencies(), patch(
        "litellm.proxy.common_request_processing.ProxyBaseLLMRequestProcessing.base_process_llm_request",
        new_callable=AsyncMock,
    ) as mock_base_process:
        mock_base_process.return_value = {"id": "int_123", "status": "created"}

        response = client.post(
            "/v1beta/interactions",
            json={
                "agent": "deep-research-pro-preview-12-2025",
                "input": "Research quantum computing",
                "background": True,
            },
            headers={"Authorization": "Bearer sk-test-key"},
        )

        assert response.status_code == 200
        assert response.json() == {"id": "int_123", "status": "created"}

        call_kwargs = mock_base_process.call_args.kwargs

        assert call_kwargs["route_type"] == "acreate_interaction"
        assert call_kwargs["model"] is None


def test_interactions_model_is_forwarded_when_provided():
    """If request contains `model`, endpoint must forward it as processing model."""
    client = _build_test_client()

    with _patch_proxy_server_dependencies(), patch(
        "litellm.proxy.common_request_processing.ProxyBaseLLMRequestProcessing.base_process_llm_request",
        new_callable=AsyncMock,
    ) as mock_base_process:
        mock_base_process.return_value = {"id": "int_456", "status": "created"}

        response = client.post(
            "/v1beta/interactions",
            json={
                "model": "gemini/gemini-2.5-flash",
                "input": "Hello world",
            },
            headers={"Authorization": "Bearer sk-test-key"},
        )

        assert response.status_code == 200
        assert response.json() == {"id": "int_456", "status": "created"}

        call_kwargs = mock_base_process.call_args.kwargs
        assert call_kwargs["model"] == "gemini/gemini-2.5-flash"


def test_interactions_model_takes_precedence_when_both_present():
    """If both model and agent are provided, model is the routing model argument."""
    client = _build_test_client()

    with _patch_proxy_server_dependencies(), patch(
        "litellm.proxy.common_request_processing.ProxyBaseLLMRequestProcessing.base_process_llm_request",
        new_callable=AsyncMock,
    ) as mock_base_process:
        mock_base_process.return_value = {"id": "int_789", "status": "created"}

        response = client.post(
            "/v1beta/interactions",
            json={
                "model": "gemini/gemini-2.5-flash",
                "agent": "deep-research-pro-preview-12-2025",
                "input": "Test both",
            },
            headers={"Authorization": "Bearer sk-test-key"},
        )

        assert response.status_code == 200
        assert response.json() == {"id": "int_789", "status": "created"}

        call_kwargs = mock_base_process.call_args.kwargs

        assert call_kwargs["model"] == "gemini/gemini-2.5-flash"
