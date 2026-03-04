"""
Tests for batch output_expires_after passthrough and team-level expiry enforcement.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.proxy_server import app
from litellm.proxy.utils import ProxyLogging
from litellm.router import Router
from litellm.types.utils import LiteLLMBatch

from fastapi.testclient import TestClient

client = TestClient(app)

TEAM_EXPIRY = {"anchor": "created_at", "seconds": 3600}
CALLER_EXPIRY = {"anchor": "created_at", "seconds": 86400}


@pytest.fixture
def llm_router() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "openai/gpt-3.5-turbo",
                    "api_key": "test-key",
                },
                "model_info": {"id": "gpt-3.5-turbo-id"},
            },
        ]
    )


def _setup_proxy(monkeypatch, llm_router: Router):
    proxy_logging_obj = ProxyLogging(
        user_api_key_cache=DualCache(default_in_memory_ttl=1)
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", llm_router)
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging_obj
    )


def _make_batch_response() -> LiteLLMBatch:
    return LiteLLMBatch(
        id="batch_abc123",
        completion_window="24h",
        created_at=1234567890,
        endpoint="/v1/chat/completions",
        input_file_id="file-abc123",
        object="batch",
        status="validating",
    )


def test_output_expires_after_passthrough():
    """output_expires_after flows through create_batch to the provider."""
    captured = {}

    def capturing_create(**kwargs):
        captured.update(kwargs)
        mock_response = MagicMock()
        mock_response.id = "batch_123"
        return mock_response

    with patch("litellm.batches.main.openai_batches_instance") as mock_instance:
        mock_instance.create_batch.side_effect = capturing_create
        litellm.create_batch(
            completion_window="24h",
            endpoint="/v1/chat/completions",
            input_file_id="file-abc123",
            output_expires_after=CALLER_EXPIRY,
            custom_llm_provider="openai",
        )

    assert captured["create_batch_data"]["output_expires_after"] == CALLER_EXPIRY


class TestBatchEndpointTeamOverride:
    """Verify team-level enforced_batch_output_expires_after in the proxy endpoint."""

    def _post_batch(
        self,
        monkeypatch,
        llm_router: Router,
        team_metadata: dict,
        request_body: dict,
    ) -> dict:
        """POST /v1/batches with given team_metadata and body, return captured kwargs."""
        _setup_proxy(monkeypatch, llm_router)

        user_key = UserAPIKeyAuth(
            api_key="test-key",
            team_metadata=team_metadata,
        )
        app.dependency_overrides[user_api_key_auth] = lambda: user_key

        captured_kwargs = {}

        async def mock_acreate_batch(**kwargs):
            captured_kwargs.update(kwargs)
            return _make_batch_response()

        monkeypatch.setattr(litellm, "acreate_batch", mock_acreate_batch)

        try:
            response = client.post(
                "/v1/batches",
                json=request_body,
                headers={"Authorization": "Bearer test-key"},
            )
            assert response.status_code == 200
        finally:
            app.dependency_overrides.clear()

        return captured_kwargs

    def test_team_override_overrides_caller(self, monkeypatch, llm_router):
        """Team enforcement wins over caller-provided value."""
        kwargs = self._post_batch(
            monkeypatch,
            llm_router,
            team_metadata={
                "enforced_batch_output_expires_after": TEAM_EXPIRY,
            },
            request_body={
                "input_file_id": "file-abc123",
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
                "output_expires_after": CALLER_EXPIRY,
            },
        )
        assert kwargs["output_expires_after"] == TEAM_EXPIRY

    def test_no_team_setting_preserves_caller(self, monkeypatch, llm_router):
        """No team setting = caller value passes through."""
        kwargs = self._post_batch(
            monkeypatch,
            llm_router,
            team_metadata={},
            request_body={
                "input_file_id": "file-abc123",
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
                "output_expires_after": CALLER_EXPIRY,
            },
        )
        assert kwargs["output_expires_after"] == CALLER_EXPIRY
