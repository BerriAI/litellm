"""
Tests for graceful handling of empty model list scenarios.

These tests verify that /v2/model/info and /model_group/info endpoints
return empty data arrays instead of 500 errors when no models are configured.
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


@pytest.fixture(autouse=True)
def mock_auth():
    """Override auth dependency for all tests."""
    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="test-user"
    )
    yield
    app.dependency_overrides.pop(ps.user_api_key_auth, None)


class TestEmptyModelListHandling:
    """Test suite for empty model list scenarios."""

    def test_v2_model_info_returns_empty_data_when_router_is_none(
        self, client, monkeypatch
    ):
        """
        Test that /v2/model/info returns paginated empty response instead of 500
        when llm_router is None.
        """
        monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None)
        monkeypatch.setattr("litellm.proxy.proxy_server.llm_model_list", None)
        monkeypatch.setattr("litellm.proxy.proxy_server.user_model", None)
        monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {})

        response = client.get(
            "/v2/model/info",
            headers={"Authorization": "Bearer sk-test"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["data"] == []
        assert data["total_count"] == 0
        assert data["current_page"] == 1
        assert data["total_pages"] == 0
        assert data["size"] == 50  # default page size

    def test_v2_model_info_returns_empty_data_when_model_list_empty(
        self, client, monkeypatch
    ):
        """
        Test that /v2/model/info returns paginated empty response instead of 500
        when llm_router exists but model_list is empty.
        """
        mock_router = MagicMock()
        mock_router.model_list = []

        monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", mock_router)
        monkeypatch.setattr("litellm.proxy.proxy_server.llm_model_list", [])
        monkeypatch.setattr("litellm.proxy.proxy_server.user_model", None)
        monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {})

        response = client.get(
            "/v2/model/info",
            headers={"Authorization": "Bearer sk-test"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["data"] == []
        assert data["total_count"] == 0
        assert data["current_page"] == 1
        assert data["total_pages"] == 0
        assert data["size"] == 50  # default page size

    def test_v2_model_info_pagination_with_empty_results(
        self, client, monkeypatch
    ):
        """
        Test that /v2/model/info pagination parameters work correctly
        when there are no models (empty results).
        """
        mock_router = MagicMock()
        mock_router.model_list = []

        monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", mock_router)
        monkeypatch.setattr("litellm.proxy.proxy_server.llm_model_list", [])
        monkeypatch.setattr("litellm.proxy.proxy_server.user_model", None)
        monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {})

        # Test with custom pagination parameters
        response = client.get(
            "/v2/model/info",
            params={"page": 2, "size": 25},
            headers={"Authorization": "Bearer sk-test"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["data"] == []
        assert data["total_count"] == 0
        assert data["current_page"] == 2  # Should respect the page parameter
        assert data["total_pages"] == 0
        assert data["size"] == 25  # Should respect the size parameter

    def test_model_group_info_returns_empty_data_when_model_list_none(
        self, client, monkeypatch
    ):
        """
        Test that /model_group/info returns {"data": []} instead of 500
        when llm_model_list is None.
        """
        monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None)
        monkeypatch.setattr("litellm.proxy.proxy_server.llm_model_list", None)
        monkeypatch.setattr("litellm.proxy.proxy_server.user_model", None)
        monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {})

        response = client.get(
            "/model_group/info",
            headers={"Authorization": "Bearer sk-test"},
        )

        assert response.status_code == 200
        assert response.json() == {"data": []}

    def test_model_group_info_returns_empty_data_when_model_list_empty(
        self, client, monkeypatch
    ):
        """
        Test that /model_group/info returns {"data": []} instead of 500
        when llm_model_list is empty.
        """
        mock_router = MagicMock()
        mock_router.model_list = []

        monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", mock_router)
        monkeypatch.setattr("litellm.proxy.proxy_server.llm_model_list", [])
        monkeypatch.setattr("litellm.proxy.proxy_server.user_model", None)
        monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {})

        response = client.get(
            "/model_group/info",
            headers={"Authorization": "Bearer sk-test"},
        )

        assert response.status_code == 200
        assert response.json() == {"data": []}
