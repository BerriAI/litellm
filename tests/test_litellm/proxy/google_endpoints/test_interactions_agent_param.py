"""
Regression tests for Google Interactions endpoint parameter handling.

These tests ensure `agent` is NOT remapped into `model` for create interaction calls.
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


@pytest.mark.parametrize(
    "request_body, expected_model",
    [
        pytest.param(
            {"agent": "deep-research-pro-preview-12-2025", "input": "Research quantum computing", "background": True},
            None,
            id="agent-only-model-none",
        ),
        pytest.param(
            {"model": "gemini/gemini-2.5-flash", "input": "Hello world"},
            "gemini/gemini-2.5-flash",
            id="model-only-forwarded",
        ),
        pytest.param(
            {"model": "gemini/gemini-2.5-flash", "agent": "deep-research-pro-preview-12-2025", "input": "Test both"},
            "gemini/gemini-2.5-flash",
            id="both-model-takes-precedence",
        ),
        pytest.param(
            {"input": "Test with no model or agent"},
            None,
            id="neither-model-is-none",
        ),
    ],
)
def test_interactions_model_routing(request_body, expected_model):
    """Verify model kwarg passed to base_process_llm_request for various inputs."""
    client = _build_test_client()

    with _patch_proxy_server_dependencies(), patch(
        "litellm.proxy.common_request_processing.ProxyBaseLLMRequestProcessing.base_process_llm_request",
        new_callable=AsyncMock,
    ) as mock_base_process:
        mock_base_process.return_value = {"id": "int_test", "status": "created"}

        response = client.post(
            "/v1beta/interactions",
            json=request_body,
            headers={"Authorization": "Bearer sk-test-key"},
        )

        assert response.status_code == 200
        assert mock_base_process.call_args.kwargs["model"] == expected_model
