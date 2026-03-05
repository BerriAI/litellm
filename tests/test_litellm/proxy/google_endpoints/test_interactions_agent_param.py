"""
Regression tests for Google Interactions endpoint parameter handling.

Tests that agent and model are handled correctly for create interaction
calls. When both are provided, agent takes precedence (non-breaking).
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))


def _build_test_client():
    try:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth
        from litellm.proxy.google_endpoints.endpoints import router as google_router
    except ImportError as e:
        pytest.skip(f"Skipping test due to missing dependency: {e}")

    app = FastAPI()
    app.include_router(google_router)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(api_key="fake-key")
    return TestClient(app)


def _patch_endpoint_dependencies():
    """Patch proxy_server globals on the endpoints module where names are read."""
    return patch.multiple(
        "litellm.proxy.google_endpoints.endpoints",
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


class TestInteractionsAgentParameter:

    @pytest.mark.parametrize(
        "request_body, expected_model",
        [
            pytest.param({"agent": "deep-research-pro-preview-12-2025", "input": "Research", "background": True}, "deep-research-pro-preview-12-2025", id="agent-only"),
            pytest.param({"model": "gemini/gemini-2.5-flash", "input": "Hello"}, "gemini/gemini-2.5-flash", id="model-only"),
            pytest.param({"input": "Test"}, None, id="neither"),
        ],
    )
    def test_interactions_model_routing(self, request_body, expected_model):
        client = _build_test_client()
        with _patch_endpoint_dependencies(), patch(
            "litellm.proxy.common_request_processing.ProxyBaseLLMRequestProcessing.base_process_llm_request",
            new_callable=AsyncMock,
        ) as mock:
            mock.return_value = {"id": "int_test", "status": "created"}
            response = client.post("/v1beta/interactions", json=request_body, headers={"Authorization": "Bearer sk-test"})
            assert response.status_code == 200
            assert mock.call_args.kwargs["model"] == expected_model

    def test_both_agent_and_model_agent_takes_precedence(self):
        """When both agent and model are provided, agent wins (non-breaking)."""
        client = _build_test_client()
        with _patch_endpoint_dependencies(), patch(
            "litellm.proxy.common_request_processing.ProxyBaseLLMRequestProcessing.base_process_llm_request",
            new_callable=AsyncMock,
        ) as mock:
            mock.return_value = {"id": "int_test", "status": "created"}
            response = client.post(
                "/v1beta/interactions",
                json={"model": "gemini/gemini-2.5-flash", "agent": "deep-research-pro-preview-12-2025", "input": "Test"},
                headers={"Authorization": "Bearer sk-test"},
            )
            assert response.status_code == 200
            assert mock.call_args.kwargs["model"] == "deep-research-pro-preview-12-2025"
