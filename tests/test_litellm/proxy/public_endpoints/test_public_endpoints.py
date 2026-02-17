import os
import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)

from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.public_endpoints import router
from litellm.types.proxy.management_endpoints.model_management_endpoints import (
    ModelGroupInfoProxy,
)
from litellm.types.utils import LlmProviders


def test_get_supported_providers_returns_enum_values():
    app_instance = FastAPI()
    app_instance.include_router(router)
    client = TestClient(app_instance)

    response = client.get("/public/providers")

    assert response.status_code == 200
    expected_providers = sorted(provider.value for provider in LlmProviders)
    assert response.json() == expected_providers


def test_get_provider_create_fields():
    app_instance = FastAPI()
    app_instance.include_router(router)
    client = TestClient(app_instance)

    response = client.get("/public/providers/fields")

    assert response.status_code == 200

    response_data = response.json()

    assert isinstance(response_data, list)

    assert len(response_data) > 0

    first_provider = response_data[0]
    assert "provider" in first_provider
    assert "provider_display_name" in first_provider
    assert "litellm_provider" in first_provider
    assert "credential_fields" in first_provider

    assert isinstance(first_provider["credential_fields"], list)

    has_detailed_fields = any(
        provider.get("credential_fields") and len(provider.get("credential_fields", [])) > 0
        for provider in response_data
    )
    assert has_detailed_fields, "Expected at least one provider to have detailed credential fields"


def test_get_litellm_model_cost_map_returns_cost_map():
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    response = client.get("/public/litellm_model_cost_map")

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, dict)
    assert len(payload) > 0, "Expected model cost map to contain at least one model"

    # Verify the structure contains expected keys for at least one model
    # Check for a common model like gpt-4 or gpt-3.5-turbo
    model_keys = list(payload.keys())
    assert len(model_keys) > 0

    # Verify at least one model has expected cost fields
    sample_model = model_keys[0]
    sample_model_data = payload[sample_model]
    assert isinstance(sample_model_data, dict)
    # Check for common cost fields that should be present
    assert "input_cost_per_token" in sample_model_data or "output_cost_per_token" in sample_model_data


def test_watsonx_provider_fields():
    """Test that Watsonx provider has all required credential fields including multiple auth options."""
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    response = client.get("/public/providers/fields")
    providers = response.json()

    watsonx = next((p for p in providers if p["provider"] == "WATSONX"), None)
    assert watsonx is not None

    field_keys = [f["key"] for f in watsonx["credential_fields"]]
    # Core fields
    assert "api_base" in field_keys
    assert "project_id" in field_keys
    assert "space_id" in field_keys
    # Multiple auth methods supported
    assert "api_key" in field_keys
    assert "token" in field_keys
    assert "zen_api_key" in field_keys


def test_public_model_hub_with_healthy_model():
    """Test that health information is populated for a healthy model"""
    app = FastAPI()
    app.include_router(router)
    # Override auth dependency
    app.dependency_overrides[user_api_key_auth] = lambda: MagicMock()
    client = TestClient(app)

    # Create mock model groups
    mock_model_group = ModelGroupInfoProxy(
        model_group="gpt-3.5-turbo",
        providers=["openai"],
        is_public_model_group=True,
    )

    # Create mock health check
    mock_health_check = MagicMock()
    mock_health_check.model_id = None
    mock_health_check.model_name = "gpt-3.5-turbo"
    mock_health_check.status = "healthy"
    mock_health_check.response_time_ms = 150.5
    mock_health_check.checked_at = datetime.now(timezone.utc)

    mock_llm_router = MagicMock()
    mock_prisma = MagicMock()
    mock_prisma.get_all_latest_health_checks = AsyncMock(
        return_value=[mock_health_check]
    )

    with patch("litellm.public_model_groups", ["gpt-3.5-turbo"]), \
         patch("litellm.proxy.proxy_server._get_model_group_info") as mock_get_info, \
         patch("litellm.proxy.proxy_server.llm_router", mock_llm_router), \
         patch("litellm.proxy.proxy_server.prisma_client", mock_prisma), \
         patch("litellm.proxy.health_endpoints._health_endpoints._convert_health_check_to_dict") as mock_convert:
        
        mock_get_info.return_value = [mock_model_group]
        mock_convert.return_value = {
            "status": "healthy",
            "response_time_ms": 150.5,
            "checked_at": mock_health_check.checked_at.isoformat(),
        }

        response = client.get(
            "/public/model_hub",
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["model_group"] == "gpt-3.5-turbo"
        assert data[0]["health_status"] == "healthy"
        assert data[0]["health_response_time"] == 150.5
        assert data[0]["health_checked_at"] is not None
    app.dependency_overrides.clear()


def test_public_model_hub_with_unhealthy_model():
    """Test that health information is populated for an unhealthy model"""
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[user_api_key_auth] = lambda: MagicMock()
    client = TestClient(app)

    mock_model_group = ModelGroupInfoProxy(
        model_group="gpt-4",
        providers=["openai"],
        is_public_model_group=True,
    )

    mock_health_check = MagicMock()
    mock_health_check.model_id = None
    mock_health_check.model_name = "gpt-4"
    mock_health_check.status = "unhealthy"
    mock_health_check.response_time_ms = None
    mock_health_check.checked_at = datetime.now(timezone.utc)

    mock_llm_router = MagicMock()
    mock_prisma = MagicMock()
    mock_prisma.get_all_latest_health_checks = AsyncMock(
        return_value=[mock_health_check]
    )

    with patch("litellm.public_model_groups", ["gpt-4"]), \
         patch("litellm.proxy.proxy_server._get_model_group_info") as mock_get_info, \
         patch("litellm.proxy.proxy_server.llm_router", mock_llm_router), \
         patch("litellm.proxy.proxy_server.prisma_client", mock_prisma), \
         patch("litellm.proxy.health_endpoints._health_endpoints._convert_health_check_to_dict") as mock_convert:
        
        mock_get_info.return_value = [mock_model_group]
        mock_convert.return_value = {
            "status": "unhealthy",
            "response_time_ms": None,
            "checked_at": mock_health_check.checked_at.isoformat(),
        }

        response = client.get(
            "/public/model_hub",
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["model_group"] == "gpt-4"
        assert data[0]["health_status"] == "unhealthy"
        assert data[0]["health_response_time"] is None
        assert data[0]["health_checked_at"] is not None
    app.dependency_overrides.clear()


def test_public_model_hub_without_health_check():
    """Test that health information is null when no health check exists"""
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[user_api_key_auth] = lambda: MagicMock()
    client = TestClient(app)

    mock_model_group = ModelGroupInfoProxy(
        model_group="claude-3",
        providers=["anthropic"],
        is_public_model_group=True,
    )

    mock_llm_router = MagicMock()
    mock_prisma = MagicMock()
    mock_prisma.get_all_latest_health_checks = AsyncMock(return_value=[])

    with patch("litellm.public_model_groups", ["claude-3"]), \
         patch("litellm.proxy.proxy_server._get_model_group_info") as mock_get_info, \
         patch("litellm.proxy.proxy_server.llm_router", mock_llm_router), \
         patch("litellm.proxy.proxy_server.prisma_client", mock_prisma):
        
        mock_get_info.return_value = [mock_model_group]

        response = client.get(
            "/public/model_hub",
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["model_group"] == "claude-3"
        assert data[0]["health_status"] is None
        assert data[0]["health_response_time"] is None
        assert data[0]["health_checked_at"] is None
    app.dependency_overrides.clear()


def test_public_model_hub_mixed_health_statuses():
    """Test multiple models with different health statuses"""
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[user_api_key_auth] = lambda: MagicMock()
    client = TestClient(app)

    healthy_model = ModelGroupInfoProxy(
        model_group="gpt-3.5-turbo",
        providers=["openai"],
        is_public_model_group=True,
    )
    unhealthy_model = ModelGroupInfoProxy(
        model_group="gpt-4",
        providers=["openai"],
        is_public_model_group=True,
    )
    no_health_model = ModelGroupInfoProxy(
        model_group="claude-3",
        providers=["anthropic"],
        is_public_model_group=True,
    )

    healthy_check = MagicMock()
    healthy_check.model_id = None
    healthy_check.model_name = "gpt-3.5-turbo"
    healthy_check.status = "healthy"
    healthy_check.response_time_ms = 120.0
    healthy_check.checked_at = datetime.now(timezone.utc)

    unhealthy_check = MagicMock()
    unhealthy_check.model_id = None
    unhealthy_check.model_name = "gpt-4"
    unhealthy_check.status = "unhealthy"
    unhealthy_check.response_time_ms = None
    unhealthy_check.checked_at = datetime.now(timezone.utc)

    mock_llm_router = MagicMock()
    mock_prisma = MagicMock()
    mock_prisma.get_all_latest_health_checks = AsyncMock(
        return_value=[healthy_check, unhealthy_check]
    )

    def convert_side_effect(check):
        if check.model_name == "gpt-3.5-turbo":
            return {
                "status": "healthy",
                "response_time_ms": 120.0,
                "checked_at": check.checked_at.isoformat(),
            }
        elif check.model_name == "gpt-4":
            return {
                "status": "unhealthy",
                "response_time_ms": None,
                "checked_at": check.checked_at.isoformat(),
            }
        return {}

    with patch("litellm.public_model_groups", ["gpt-3.5-turbo", "gpt-4", "claude-3"]), \
         patch("litellm.proxy.proxy_server._get_model_group_info") as mock_get_info, \
         patch("litellm.proxy.proxy_server.llm_router", mock_llm_router), \
         patch("litellm.proxy.proxy_server.prisma_client", mock_prisma), \
         patch("litellm.proxy.health_endpoints._health_endpoints._convert_health_check_to_dict") as mock_convert:
        
        mock_get_info.return_value = [
            healthy_model,
            unhealthy_model,
            no_health_model,
        ]
        mock_convert.side_effect = convert_side_effect

        response = client.get(
            "/public/model_hub",
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3

        # Find each model and verify health status
        gpt35 = next(m for m in data if m["model_group"] == "gpt-3.5-turbo")
        assert gpt35["health_status"] == "healthy"
        assert gpt35["health_response_time"] == 120.0
        assert gpt35["health_checked_at"] is not None

        gpt4 = next(m for m in data if m["model_group"] == "gpt-4")
        assert gpt4["health_status"] == "unhealthy"
        assert gpt4["health_response_time"] is None
        assert gpt4["health_checked_at"] is not None

        claude = next(m for m in data if m["model_group"] == "claude-3")
        assert claude["health_status"] is None
        assert claude["health_response_time"] is None
        assert claude["health_checked_at"] is None
    app.dependency_overrides.clear()

