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
from litellm.types.proxy.management_endpoints.ui_sso import (
    DefaultTeamSSOParams,
    SSOConfig,
)

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
        "general_settings": {"proxy_admin_email": "admin@example.com"},
        "environment_variables": {
            "GOOGLE_CLIENT_ID": "test_google_client_id",
            "GOOGLE_CLIENT_SECRET": "test_google_client_secret",
            "MICROSOFT_CLIENT_ID": "test_microsoft_client_id",
            "MICROSOFT_CLIENT_SECRET": "test_microsoft_client_secret",
            "PROXY_BASE_URL": "https://example.com",
        },
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

        monkeypatch.setattr("litellm.proxy.proxy_server.store_model_in_db", True)
        monkeypatch.setattr(litellm, "default_internal_user_params", {})

        # New settings to update
        new_settings = {
            "user_role": LitellmUserRoles.PROXY_ADMIN,
            "max_budget": 200.0,
            "budget_duration": "7d",
            "models": ["gpt-4", "claude-3"],
        }

        response = client.patch("/update/internal_user_settings", json=new_settings)

        print(response.text)
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

        monkeypatch.setattr("litellm.proxy.proxy_server.store_model_in_db", True)
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

    def test_update_sso_settings(self, mock_proxy_config, mock_auth, monkeypatch):
        monkeypatch.setenv("LITELLM_SALT_KEY", "test_salt_key")
        """Test updating the SSO settings"""
        # New SSO settings to update
        new_sso_settings = {
            "google_client_id": "new_google_client_id",
            "google_client_secret": "new_google_client_secret",
            "microsoft_client_id": "new_microsoft_client_id",
            "microsoft_client_secret": "new_microsoft_client_secret",
            "proxy_base_url": "https://newexample.com",
            "user_email": "newadmin@example.com",
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
        assert (
            settings["google_client_secret"] == new_sso_settings["google_client_secret"]
        )
        assert (
            settings["microsoft_client_id"] == new_sso_settings["microsoft_client_id"]
        )
        assert (
            settings["microsoft_client_secret"]
            == new_sso_settings["microsoft_client_secret"]
        )
        assert settings["proxy_base_url"] == new_sso_settings["proxy_base_url"]
        assert settings["user_email"] == new_sso_settings["user_email"]

        # Verify the config was updated
        updated_config = mock_proxy_config["config"]
        assert (
            updated_config["environment_variables"]["GOOGLE_CLIENT_ID"]
            != new_sso_settings["google_client_id"]
        )
        assert (
            updated_config["environment_variables"]["GOOGLE_CLIENT_SECRET"]
            != new_sso_settings["google_client_secret"]
        )
        assert (
            updated_config["general_settings"]["proxy_admin_email"]
            == new_sso_settings["user_email"]
        )

        # Verify save_config was called exactly once
        assert mock_proxy_config["save_call_count"]() == 1

    def test_update_sso_settings_with_null_values_clears_env_vars(
        self, mock_proxy_config, mock_auth, monkeypatch
    ):
        """Test that updating SSO settings with null values clears environment variables"""
        monkeypatch.setenv("LITELLM_SALT_KEY", "test_salt_key")

        # First, verify we have existing environment variables
        initial_config = mock_proxy_config["config"]
        assert "GOOGLE_CLIENT_ID" in initial_config["environment_variables"]
        assert "MICROSOFT_CLIENT_ID" in initial_config["environment_variables"]

        # Set some initial environment variables for runtime testing
        monkeypatch.setenv("GOOGLE_CLIENT_ID", "test_existing_google_id")
        monkeypatch.setenv("MICROSOFT_CLIENT_ID", "test_existing_microsoft_id")

        # Send SSO settings with null values to clear them
        clear_sso_settings = {
            "google_client_id": None,
            "google_client_secret": None,
            "microsoft_client_id": None,
            "microsoft_client_secret": None,
            "microsoft_tenant": None,
            "generic_client_id": None,
            "generic_client_secret": None,
            "generic_authorization_endpoint": None,
            "generic_token_endpoint": None,
            "generic_userinfo_endpoint": None,
            "proxy_base_url": None,
            "user_email": None,
            "sso_provider": None,
        }

        response = client.patch("/update/sso_settings", json=clear_sso_settings)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

        # Verify that environment variables were cleared from config
        updated_config = mock_proxy_config["config"]

        # These should be removed from environment_variables
        assert "GOOGLE_CLIENT_ID" not in updated_config["environment_variables"]
        assert "GOOGLE_CLIENT_SECRET" not in updated_config["environment_variables"]
        assert "MICROSOFT_CLIENT_ID" not in updated_config["environment_variables"]
        assert "MICROSOFT_CLIENT_SECRET" not in updated_config["environment_variables"]
        assert "MICROSOFT_TENANT" not in updated_config["environment_variables"]
        assert "PROXY_BASE_URL" not in updated_config["environment_variables"]

        # Verify that runtime environment variables were cleared
        assert "GOOGLE_CLIENT_ID" not in os.environ
        assert "MICROSOFT_CLIENT_ID" not in os.environ

        # Verify user_email was cleared from general_settings
        assert updated_config["general_settings"].get("proxy_admin_email") is None

        # Verify save_config was called
        assert mock_proxy_config["save_call_count"]() == 1

    def test_update_sso_settings_with_empty_strings_clears_env_vars(
        self, mock_proxy_config, mock_auth, monkeypatch
    ):
        """Test that updating SSO settings with empty strings also clears environment variables"""
        monkeypatch.setenv("LITELLM_SALT_KEY", "test_salt_key")

        # Set some initial environment variables for runtime testing
        monkeypatch.setenv("GOOGLE_CLIENT_ID", "test_existing_google_id")
        monkeypatch.setenv("MICROSOFT_CLIENT_SECRET", "test_existing_microsoft_secret")

        # Send SSO settings with empty strings to clear them
        clear_sso_settings = {
            "google_client_id": "",
            "google_client_secret": "",
            "microsoft_client_secret": "",
            "proxy_base_url": "",
            "user_email": "",
        }

        response = client.patch("/update/sso_settings", json=clear_sso_settings)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

        # Verify that environment variables with empty strings were cleared from config
        updated_config = mock_proxy_config["config"]
        assert "GOOGLE_CLIENT_ID" not in updated_config["environment_variables"]
        assert "GOOGLE_CLIENT_SECRET" not in updated_config["environment_variables"]
        assert "MICROSOFT_CLIENT_SECRET" not in updated_config["environment_variables"]
        assert "PROXY_BASE_URL" not in updated_config["environment_variables"]

        # Verify that runtime environment variables were cleared
        assert "GOOGLE_CLIENT_ID" not in os.environ
        assert "MICROSOFT_CLIENT_SECRET" not in os.environ

        # Verify user_email was cleared from general_settings
        assert updated_config["general_settings"].get("proxy_admin_email") is None

        # Verify save_config was called
        assert mock_proxy_config["save_call_count"]() == 1

    def test_update_sso_settings_mixed_null_and_valid_values(
        self, mock_proxy_config, mock_auth, monkeypatch
    ):
        """Test updating SSO settings with mix of null and valid values"""
        monkeypatch.setenv("LITELLM_SALT_KEY", "test_salt_key")

        # Set some initial environment variables
        monkeypatch.setenv("GOOGLE_CLIENT_ID", "old_google_id")
        monkeypatch.setenv("MICROSOFT_CLIENT_ID", "old_microsoft_id")
        monkeypatch.setenv("PROXY_BASE_URL", "old_proxy_url")

        # Send mixed SSO settings - some null, some valid
        mixed_sso_settings = {
            "google_client_id": "new_google_client_id",  # Valid value
            "google_client_secret": None,  # Null to clear
            "microsoft_client_id": None,  # Null to clear
            "microsoft_client_secret": "new_microsoft_secret",  # Valid value
            "proxy_base_url": "https://newproxy.com",  # Valid value
            "user_email": None,  # Null to clear
        }

        response = client.patch("/update/sso_settings", json=mixed_sso_settings)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

        # Verify the config was updated correctly
        updated_config = mock_proxy_config["config"]

        # Valid values should be set
        assert (
            updated_config["environment_variables"]["GOOGLE_CLIENT_ID"]
            != "new_google_client_id"
        )  # Encrypted
        assert (
            updated_config["environment_variables"]["MICROSOFT_CLIENT_SECRET"]
            != "new_microsoft_secret"
        )  # Encrypted
        assert (
            updated_config["environment_variables"]["PROXY_BASE_URL"]
            != "https://newproxy.com"
        )  # Encrypted

        # Null values should be cleared
        assert "GOOGLE_CLIENT_SECRET" not in updated_config["environment_variables"]
        assert "MICROSOFT_CLIENT_ID" not in updated_config["environment_variables"]

        # Verify runtime environment variables
        assert os.environ.get("GOOGLE_CLIENT_ID") == "new_google_client_id"
        assert os.environ.get("MICROSOFT_CLIENT_SECRET") == "new_microsoft_secret"
        assert "GOOGLE_CLIENT_SECRET" not in os.environ
        assert "MICROSOFT_CLIENT_ID" not in os.environ

        # Verify user_email was cleared from general_settings
        assert updated_config["general_settings"].get("proxy_admin_email") is None

        # Verify save_config was called
        assert mock_proxy_config["save_call_count"]() == 1

    def test_update_sso_settings_ui_access_mode_handling(
        self, mock_proxy_config, mock_auth, monkeypatch
    ):
        """Test that ui_access_mode is handled correctly in general_settings"""
        monkeypatch.setenv("LITELLM_SALT_KEY", "test_salt_key")

        # Test setting ui_access_mode
        sso_settings_with_ui_mode = {
            "ui_access_mode": "admin_only",
            "user_email": "admin@test.com",
        }

        response = client.patch("/update/sso_settings", json=sso_settings_with_ui_mode)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

        # Verify ui_access_mode was set in general_settings (not environment_variables)
        updated_config = mock_proxy_config["config"]
        assert updated_config["general_settings"]["ui_access_mode"] == "admin_only"
        assert (
            updated_config["general_settings"]["proxy_admin_email"] == "admin@test.com"
        )

        # Verify ui_access_mode is NOT in environment_variables
        assert "ui_access_mode" not in updated_config["environment_variables"]

        # Test clearing ui_access_mode
        clear_ui_mode = {"ui_access_mode": None, "user_email": None}

        response = client.patch("/update/sso_settings", json=clear_ui_mode)

        assert response.status_code == 200

        # Verify ui_access_mode and user_email were cleared
        updated_config = mock_proxy_config["config"]
        assert updated_config["general_settings"].get("ui_access_mode") is None
        assert updated_config["general_settings"].get("proxy_admin_email") is None

        # Verify save_config was called twice (once for each update)
        assert mock_proxy_config["save_call_count"]() == 2
