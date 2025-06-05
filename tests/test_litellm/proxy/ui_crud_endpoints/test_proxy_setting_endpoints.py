import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.proxy._types import DefaultInternalUserParams, LitellmUserRoles
from litellm.proxy.proxy_server import app
from litellm.types.proxy.management_endpoints.ui_sso import DefaultTeamSSOParams, SSOConfig

client = TestClient(app)


@pytest.fixture
def mock_proxy_config(monkeypatch):
    """Mock the proxy_config to avoid actual file operations during tests"""
    mock_config = {
        "litellm_settings": {
            "default_internal_user_params": {
                "user_role": LitellmUserRoles.INTERNAL_USER,
                "max_budget": 100.0,
                "budget_duration": "30d",
                "models": ["gpt-3.5-turbo", "gpt-4"],
            },
            "default_team_params": {
                "models": ["gpt-3.5-turbo"],
                "max_budget": 50.0,
                "budget_duration": "14d",
                "tpm_limit": 100,
                "rpm_limit": 10,
            },
        },
        "general_settings": {
            "proxy_admin_email": "admin@example.com"
        },
        "environment_variables": {
            "GOOGLE_CLIENT_ID": "test_google_client_id",
            "GOOGLE_CLIENT_SECRET": "test_google_client_secret",
            "MICROSOFT_CLIENT_ID": "test_microsoft_client_id",
            "MICROSOFT_CLIENT_SECRET": "test_microsoft_client_secret",
            "PROXY_BASE_URL": "https://example.com"
        }
    }

    async def mock_get_config():
        return mock_config

    # Add a counter to track save_config calls
    save_config_call_count = 0

    async def mock_save_config(new_config=None):
        nonlocal mock_config, save_config_call_count
        save_config_call_count += 1
        if new_config:
            mock_config = new_config
        return mock_config

    from litellm.proxy.proxy_server import proxy_config

    monkeypatch.setattr(proxy_config, "get_config", mock_get_config)
    monkeypatch.setattr(proxy_config, "save_config", mock_save_config)

    # Return both the config and the call counter
    return {"config": mock_config, "save_call_count": lambda: save_config_call_count}


@pytest.fixture
def mock_auth(monkeypatch):
    """Mock the authentication to bypass auth checks"""

    async def mock_user_api_key_auth():
        return {"user_id": "test_user"}

    from litellm.proxy.ui_crud_endpoints.proxy_setting_endpoints import (
        user_api_key_auth,
    )

    monkeypatch.setattr(
        "litellm.proxy.ui_crud_endpoints.proxy_setting_endpoints.user_api_key_auth",
        mock_user_api_key_auth,
    )


class TestProxySettingEndpoints:
    def test_get_internal_user_settings(self, mock_proxy_config, mock_auth):
        """Test getting the internal user settings"""
        response = client.get("/get/internal_user_settings")

        assert response.status_code == 200
        data = response.json()

        # Check structure of response (updated to use field_schema)
        assert "values" in data
        assert "field_schema" in data

        # Check values match our mock config
        values = data["values"]
        mock_params = mock_proxy_config["config"]["litellm_settings"][
            "default_internal_user_params"
        ]
        assert values["user_role"] == mock_params["user_role"]
        assert values["max_budget"] == mock_params["max_budget"]
        assert values["budget_duration"] == mock_params["budget_duration"]
        assert values["models"] == mock_params["models"]

        # Check field_schema contains descriptions (updated from schema to field_schema)
        assert "properties" in data["field_schema"]
        assert "user_role" in data["field_schema"]["properties"]
        assert "description" in data["field_schema"]["properties"]["user_role"]

    def test_update_internal_user_settings(
        self, mock_proxy_config, mock_auth, monkeypatch
    ):
        """Test updating the internal user settings"""
        # Mock litellm.default_internal_user_params
        import litellm

        monkeypatch.setattr(litellm, "default_internal_user_params", {})

        # New settings to update
        new_settings = {
            "user_role": LitellmUserRoles.PROXY_ADMIN,
            "max_budget": 200.0,
            "budget_duration": "7d",
            "models": ["gpt-4", "claude-3"],
        }

        response = client.patch("/update/internal_user_settings", json=new_settings)

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert data["status"] == "success"
        assert "settings" in data

        # Verify settings were updated
        settings = data["settings"]
        assert settings["user_role"] == new_settings["user_role"]
        assert settings["max_budget"] == new_settings["max_budget"]
        assert settings["budget_duration"] == new_settings["budget_duration"]
        assert settings["models"] == new_settings["models"]

        # Verify the config was updated
        updated_config = mock_proxy_config["config"]["litellm_settings"][
            "default_internal_user_params"
        ]
        assert updated_config["user_role"] == new_settings["user_role"]
        assert updated_config["max_budget"] == new_settings["max_budget"]

        # Verify save_config was called exactly once
        assert mock_proxy_config["save_call_count"]() == 1

    def test_get_default_team_settings(self, mock_proxy_config, mock_auth):
        """Test getting the default team settings"""
        response = client.get("/get/default_team_settings")

        assert response.status_code == 200
        data = response.json()

        # Check structure of response (updated to use field_schema)
        assert "values" in data
        assert "field_schema" in data

        # Check values match our mock config
        values = data["values"]
        mock_params = mock_proxy_config["config"]["litellm_settings"][
            "default_team_params"
        ]
        assert values["models"] == mock_params["models"]
        assert values["max_budget"] == mock_params["max_budget"]
        assert values["budget_duration"] == mock_params["budget_duration"]
        assert values["tpm_limit"] == mock_params["tpm_limit"]
        assert values["rpm_limit"] == mock_params["rpm_limit"]

        # Check field_schema contains descriptions (updated from schema to field_schema)
        assert "properties" in data["field_schema"]
        assert "models" in data["field_schema"]["properties"]
        assert "description" in data["field_schema"]["properties"]["models"]

    def test_update_default_team_settings(
        self, mock_proxy_config, mock_auth, monkeypatch
    ):
        """Test updating the default team settings"""
        # Mock litellm.default_team_params
        import litellm

        monkeypatch.setattr(litellm, "default_team_params", {})

        # New settings to update
        new_settings = {
            "models": ["gpt-4", "claude-3"],
            "max_budget": 150.0,
            "budget_duration": "30d",
            "tpm_limit": 200,
            "rpm_limit": 20,
        }

        response = client.patch("/update/default_team_settings", json=new_settings)

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert data["status"] == "success"
        assert "settings" in data

        # Verify settings were updated
        settings = data["settings"]
        assert settings["models"] == new_settings["models"]
        assert settings["max_budget"] == new_settings["max_budget"]
        assert settings["budget_duration"] == new_settings["budget_duration"]
        assert settings["tpm_limit"] == new_settings["tpm_limit"]
        assert settings["rpm_limit"] == new_settings["rpm_limit"]

        # Verify the config was updated
        updated_config = mock_proxy_config["config"]["litellm_settings"][
            "default_team_params"
        ]
        assert updated_config["models"] == new_settings["models"]
        assert updated_config["max_budget"] == new_settings["max_budget"]
        assert updated_config["tpm_limit"] == new_settings["tpm_limit"]

        # Verify save_config was called exactly once
        assert mock_proxy_config["save_call_count"]() == 1

    def test_get_sso_settings(self, mock_proxy_config, mock_auth):
        """Test getting the SSO settings"""
        response = client.get("/get/sso_settings")

        assert response.status_code == 200
        data = response.json()

        # Check structure of response
        assert "values" in data
        assert "field_schema" in data

        # Check values contain SSO configuration
        values = data["values"]
        assert "google_client_id" in values
        assert "google_client_secret" in values
        assert "microsoft_client_id" in values
        assert "microsoft_client_secret" in values
        assert "proxy_base_url" in values
        assert "user_email" in values

        # Verify values match our mock config
        assert values["google_client_id"] == "test_google_client_id"
        assert values["google_client_secret"] == "test_google_client_secret"
        assert values["microsoft_client_id"] == "test_microsoft_client_id"
        assert values["microsoft_client_secret"] == "test_microsoft_client_secret"
        assert values["proxy_base_url"] == "https://example.com"
        assert values["user_email"] == "admin@example.com"

        # Check field_schema contains descriptions
        assert "properties" in data["field_schema"]
        assert "google_client_id" in data["field_schema"]["properties"]
        assert "description" in data["field_schema"]["properties"]["google_client_id"]

    def test_update_sso_settings(self, mock_proxy_config, mock_auth):
        """Test updating the SSO settings"""
        # New SSO settings to update
        new_sso_settings = {
            "google_client_id": "new_google_client_id",
            "google_client_secret": "new_google_client_secret",
            "microsoft_client_id": "new_microsoft_client_id",
            "microsoft_client_secret": "new_microsoft_client_secret",
            "proxy_base_url": "https://newexample.com",
            "user_email": "newadmin@example.com"
        }

        response = client.patch("/update/sso_settings", json=new_sso_settings)

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert data["status"] == "success"
        assert "settings" in data

        # Verify settings were updated
        settings = data["settings"]
        assert settings["google_client_id"] == new_sso_settings["google_client_id"]
        assert settings["google_client_secret"] == new_sso_settings["google_client_secret"]
        assert settings["microsoft_client_id"] == new_sso_settings["microsoft_client_id"]
        assert settings["microsoft_client_secret"] == new_sso_settings["microsoft_client_secret"]
        assert settings["proxy_base_url"] == new_sso_settings["proxy_base_url"]
        assert settings["user_email"] == new_sso_settings["user_email"]

        # Verify the config was updated
        updated_config = mock_proxy_config["config"]
        assert updated_config["environment_variables"]["GOOGLE_CLIENT_ID"] == new_sso_settings["google_client_id"]
        assert updated_config["environment_variables"]["GOOGLE_CLIENT_SECRET"] == new_sso_settings["google_client_secret"]
        assert updated_config["general_settings"]["proxy_admin_email"] == new_sso_settings["user_email"]

        # Verify save_config was called exactly once
        assert mock_proxy_config["save_call_count"]() == 1
