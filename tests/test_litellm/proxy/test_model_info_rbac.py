"""
Tests for RBAC enforcement on /v2/model/info endpoint.

These tests verify that non-admin users can only see their accessible models
and cannot access the full model list without RBAC filtering, regardless of
whether they pass user_models_only=false explicitly.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system-path

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.proxy_server import app
import litellm.proxy.proxy_server as ps


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_router():
    """Create a mock LLM router with a sample model list."""
    router = MagicMock()
    router.model_list = [
        {
            "model_name": "gpt-4",
            "litellm_params": {"model": "gpt-4"},
            "model_info": {"id": "model-1"},
        },
        {
            "model_name": "claude-3",
            "litellm_params": {"model": "claude-3"},
            "model_info": {"id": "model-2"},
        },
    ]
    router.get_model_list = MagicMock(return_value=router.model_list)
    return router


@pytest.fixture
def mock_prisma():
    """Create a mock prisma client (non-None so the None-guard passes)."""
    return MagicMock()


class TestModelInfoV2RBAC:
    """Test suite for RBAC enforcement on /v2/model/info."""

    def _setup_common_mocks(
        self, monkeypatch, mock_router, mock_prisma, user_role, user_id="test-user"
    ):
        """Override auth and proxy globals for a single test."""
        app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
            user_role=user_role,
            user_id=user_id,
        )
        monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", mock_router)
        monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)
        monkeypatch.setattr("litellm.proxy.proxy_server.user_model", None)
        monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {})

    def teardown_method(self):
        """Remove dependency overrides after each test."""
        app.dependency_overrides.pop(ps.user_api_key_auth, None)

    # ------------------------------------------------------------------
    # Helpers shared across test cases
    # ------------------------------------------------------------------

    @staticmethod
    def _make_patches(non_admin_mock, apply_search_mock, append_agents_mock):
        proxy_config_mock = MagicMock()
        proxy_config_mock.get_config = AsyncMock(return_value=None)
        return (
            patch("litellm.proxy.proxy_server.non_admin_all_models", non_admin_mock),
            patch("litellm.proxy.proxy_server.proxy_config", proxy_config_mock),
            patch(
                "litellm.proxy.proxy_server._apply_search_filter_to_models",
                apply_search_mock,
            ),
            patch(
                "litellm.proxy.agent_endpoints.model_list_helpers.append_agents_to_model_info",
                append_agents_mock,
            ),
        )

    # ------------------------------------------------------------------
    # Test cases
    # ------------------------------------------------------------------

    def test_non_admin_user_models_only_enforced(
        self, client, mock_router, mock_prisma, monkeypatch
    ):
        """
        Non-admin users must always have user_models_only=True enforced, even
        when they do not pass user_models_only=true explicitly.
        """
        self._setup_common_mocks(
            monkeypatch, mock_router, mock_prisma, LitellmUserRoles.INTERNAL_USER
        )

        non_admin_mock = AsyncMock(return_value=[])
        apply_search_mock = AsyncMock(return_value=([], 0))
        append_agents_mock = AsyncMock(return_value=[])

        with (
            patch("litellm.proxy.proxy_server.non_admin_all_models", non_admin_mock),
            patch(
                "litellm.proxy.proxy_server.proxy_config",
                MagicMock(get_config=AsyncMock(return_value=None)),
            ),
            patch(
                "litellm.proxy.proxy_server._apply_search_filter_to_models",
                apply_search_mock,
            ),
            patch(
                "litellm.proxy.agent_endpoints.model_list_helpers.append_agents_to_model_info",
                append_agents_mock,
            ),
        ):
            response = client.get(
                "/v2/model/info",
                headers={"Authorization": "Bearer sk-test"},
            )

        assert response.status_code == 200
        # RBAC must have triggered non_admin_all_models
        non_admin_mock.assert_called_once()

    def test_admin_user_models_only_not_enforced(
        self, client, mock_router, mock_prisma, monkeypatch
    ):
        """
        Admin users are NOT forced into user_models_only=True; they can see
        all models without RBAC filtering.
        """
        self._setup_common_mocks(
            monkeypatch, mock_router, mock_prisma, LitellmUserRoles.PROXY_ADMIN
        )

        non_admin_mock = AsyncMock(return_value=[])
        apply_search_mock = AsyncMock(return_value=([], 0))
        append_agents_mock = AsyncMock(return_value=[])

        with (
            patch("litellm.proxy.proxy_server.non_admin_all_models", non_admin_mock),
            patch(
                "litellm.proxy.proxy_server.proxy_config",
                MagicMock(get_config=AsyncMock(return_value=None)),
            ),
            patch(
                "litellm.proxy.proxy_server._apply_search_filter_to_models",
                apply_search_mock,
            ),
            patch(
                "litellm.proxy.agent_endpoints.model_list_helpers.append_agents_to_model_info",
                append_agents_mock,
            ),
        ):
            response = client.get(
                "/v2/model/info",
                headers={"Authorization": "Bearer sk-test"},
            )

        assert response.status_code == 200
        # Admin users must NOT be filtered through non_admin_all_models
        non_admin_mock.assert_not_called()

    def test_non_admin_explicit_false_still_enforced(
        self, client, mock_router, mock_prisma, monkeypatch
    ):
        """
        A non-admin user who explicitly passes user_models_only=false must
        still have the RBAC guard override it to True. The query parameter
        cannot be used to bypass the role-based filter.
        """
        self._setup_common_mocks(
            monkeypatch, mock_router, mock_prisma, LitellmUserRoles.INTERNAL_USER
        )

        non_admin_mock = AsyncMock(return_value=[])
        apply_search_mock = AsyncMock(return_value=([], 0))
        append_agents_mock = AsyncMock(return_value=[])

        with (
            patch("litellm.proxy.proxy_server.non_admin_all_models", non_admin_mock),
            patch(
                "litellm.proxy.proxy_server.proxy_config",
                MagicMock(get_config=AsyncMock(return_value=None)),
            ),
            patch(
                "litellm.proxy.proxy_server._apply_search_filter_to_models",
                apply_search_mock,
            ),
            patch(
                "litellm.proxy.agent_endpoints.model_list_helpers.append_agents_to_model_info",
                append_agents_mock,
            ),
        ):
            response = client.get(
                "/v2/model/info",
                params={"user_models_only": "false"},
                headers={"Authorization": "Bearer sk-test"},
            )

        assert response.status_code == 200
        # Even with user_models_only=false, RBAC enforcement must apply
        non_admin_mock.assert_called_once()
