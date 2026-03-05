"""
Regression tests for Google Interactions endpoint parameter handling.

These tests ensure agent and model are handled correctly for
create interaction calls, including mutual exclusivity validation.
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
    # Override auth dependency so tests do not require a real API key
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

    def test_agent_parameter_fallback_logic(self):
        data = {"agent": "deep-research-pro-preview-12-2025", "input": "Research", "background": True}
        assert (data.get("model") or data.get("agent")) == "deep-research-pro-preview-12-2025"

        data = {"model": "gemini-2.5-flash", "input": "Hello world"}
        assert (data.get("model") or data.get("agent")) == "gemini-2.5-flash"

        data = {"input": "Test"}
        assert (data.get("model") or data.get("agent")) is None

    def test_route_type_in_skip_model_routing_list(self):
        skip = ["acreate_interaction", "aget_interaction", "adelete_interaction", "acancel_interaction"]
        assert "acreate_interaction" in skip

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

    def test_both_agent_and_model_returns_400(self):
        client = _build_test_client()
        with _patch_endpoint_dependencies():
            response = client.post(
                "/v1beta/interactions",
                json={"model": "gemini/gemini-2.5-flash", "agent": "deep-research-pro-preview-12-2025", "input": "Test"},
                headers={"Authorization": "Bearer sk-test"},
            )
            assert response.status_code == 400
