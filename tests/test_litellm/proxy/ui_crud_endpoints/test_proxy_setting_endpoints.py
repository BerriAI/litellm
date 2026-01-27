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

    def test_get_sso_settings(self, mock_proxy_config, mock_auth, monkeypatch):
        """Test getting the SSO settings from the dedicated database table"""
        from unittest.mock import AsyncMock, MagicMock

        # Mock the prisma client with database record
        # Note: Prisma returns Json fields as dicts (auto-parsed)
        mock_prisma = MagicMock()
        mock_db_record = MagicMock()
        mock_db_record.sso_settings = {
            "google_client_id": "test_google_client_id",
            "google_client_secret": "test_google_client_secret",
            "microsoft_client_id": "test_microsoft_client_id",
            "microsoft_client_secret": "test_microsoft_client_secret",
            "proxy_base_url": "https://example.com",
            "user_email": "admin@example.com",
        }
        mock_prisma.db.litellm_ssoconfig.find_unique = AsyncMock(return_value=mock_db_record)
        monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)

        # Mock decryption to return the values as-is (simulating decryption)
        from litellm.proxy.proxy_server import proxy_config
        monkeypatch.setattr(
            proxy_config, "_decrypt_and_set_db_env_variables", lambda environment_variables: environment_variables
        )

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
        
        # Verify role_mappings is present in response (can be None if not set)
        assert "role_mappings" in values
        assert values["role_mappings"] is None
        
        # Verify find_unique was called with correct parameters
        mock_prisma.db.litellm_ssoconfig.find_unique.assert_called_once()
        call_args = mock_prisma.db.litellm_ssoconfig.find_unique.call_args
        assert call_args.kwargs["where"]["id"] == "sso_config"

    def test_update_sso_settings(self, mock_proxy_config, mock_auth, monkeypatch):
        """Test updating the SSO settings to the dedicated database table"""
        import json
        from unittest.mock import AsyncMock, MagicMock

        monkeypatch.setenv("LITELLM_SALT_KEY", "test_salt_key")
        monkeypatch.setattr("litellm.proxy.proxy_server.store_model_in_db", True)

        # Mock the prisma client
        mock_prisma = MagicMock()
        mock_prisma.db.litellm_ssoconfig.upsert = AsyncMock()
        mock_prisma.db.litellm_config = MagicMock()
        mock_prisma.db.litellm_config.find_unique = AsyncMock(return_value=None)
        mock_prisma.db.litellm_config.update = AsyncMock()
        monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)

        # Mock encryption to return values as-is
        from litellm.proxy.proxy_server import proxy_config
        monkeypatch.setattr(proxy_config, "_encrypt_env_variables", lambda environment_variables: environment_variables)

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

        # Verify upsert was called with correct parameters
        assert mock_prisma.db.litellm_ssoconfig.upsert.called
        call_args = mock_prisma.db.litellm_ssoconfig.upsert.call_args
        
        # Verify the upsert is using the correct ID
        assert call_args.kwargs["where"]["id"] == "sso_config"
        
        # Verify the data structure for create and update
        create_data = call_args.kwargs["data"]["create"]
        update_data = call_args.kwargs["data"]["update"]
        
        assert create_data["id"] == "sso_config"
        assert "sso_settings" in create_data
        assert "sso_settings" in update_data
        
        # Verify the data is stored as JSON string (as per implementation)
        # The encryption mock returns data as-is, so we verify structure
        create_sso_settings = json.loads(create_data["sso_settings"])
        assert create_sso_settings["google_client_id"] == "new_google_client_id"

    def test_update_sso_settings_with_null_values_clears_env_vars(
        self, mock_proxy_config, mock_auth, monkeypatch
    ):
        """Test that updating SSO settings with null values clears environment variables and updates database"""
        import json
        from unittest.mock import AsyncMock, MagicMock

        monkeypatch.setenv("LITELLM_SALT_KEY", "test_salt_key")
        monkeypatch.setattr("litellm.proxy.proxy_server.store_model_in_db", True)

        # Mock the prisma client
        mock_prisma = MagicMock()
        mock_prisma.db.litellm_ssoconfig.upsert = AsyncMock()
        mock_prisma.db.litellm_config = MagicMock()

        env_var_entry = MagicMock()
        env_var_entry.param_value = json.dumps(
            {
                "GOOGLE_CLIENT_ID": "old_google_id",
                "MICROSOFT_CLIENT_SECRET": "old_secret",
                "PROXY_BASE_URL": "old_proxy_url",
            }
        )
        mock_prisma.db.litellm_config.find_unique = AsyncMock(return_value=env_var_entry)
        mock_prisma.db.litellm_config.update = AsyncMock()
        monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)

        # Mock encryption to return values as-is
        from litellm.proxy.proxy_server import proxy_config
        monkeypatch.setattr(proxy_config, "_encrypt_env_variables", lambda environment_variables: environment_variables)

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

        # Verify that runtime environment variables were cleared
        assert "GOOGLE_CLIENT_ID" not in os.environ
        assert "MICROSOFT_CLIENT_ID" not in os.environ

        # Verify upsert was called with correct parameters
        assert mock_prisma.db.litellm_ssoconfig.upsert.called
        call_args = mock_prisma.db.litellm_ssoconfig.upsert.call_args
        
        # Verify null values are stored in database
        create_data = call_args.kwargs["data"]["create"]
        create_sso_settings = json.loads(create_data["sso_settings"])
        assert create_sso_settings["google_client_id"] is None
        assert create_sso_settings["microsoft_client_id"] is None

    def test_update_sso_settings_with_empty_strings_clears_env_vars(
        self, mock_proxy_config, mock_auth, monkeypatch
    ):
        """Test that updating SSO settings with empty strings also clears environment variables and updates database"""
        import json
        from unittest.mock import AsyncMock, MagicMock

        monkeypatch.setenv("LITELLM_SALT_KEY", "test_salt_key")
        monkeypatch.setattr("litellm.proxy.proxy_server.store_model_in_db", True)

        # Mock the prisma client
        mock_prisma = MagicMock()
        mock_prisma.db.litellm_ssoconfig.upsert = AsyncMock()
        mock_prisma.db.litellm_config = MagicMock()
        env_var_entry = MagicMock()
        env_var_entry.param_value = json.dumps(
            {
                "GOOGLE_CLIENT_ID": "old_google_id",
                "MICROSOFT_CLIENT_SECRET": "old_secret",
                "PROXY_BASE_URL": "old_proxy_url",
            }
        )
        mock_prisma.db.litellm_config.find_unique = AsyncMock(return_value=env_var_entry)
        mock_prisma.db.litellm_config.update = AsyncMock()
        monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)

        # Mock encryption to return values as-is
        from litellm.proxy.proxy_server import proxy_config
        monkeypatch.setattr(proxy_config, "_encrypt_env_variables", lambda environment_variables: environment_variables)

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

        # Verify that runtime environment variables were cleared
        assert "GOOGLE_CLIENT_ID" not in os.environ
        assert "MICROSOFT_CLIENT_SECRET" not in os.environ

        # Verify upsert was called with correct parameters
        assert mock_prisma.db.litellm_ssoconfig.upsert.called
        call_args = mock_prisma.db.litellm_ssoconfig.upsert.call_args
        
        # Verify empty strings are stored in database
        create_data = call_args.kwargs["data"]["create"]
        create_sso_settings = json.loads(create_data["sso_settings"])
        assert create_sso_settings["google_client_id"] == ""
        assert create_sso_settings["microsoft_client_secret"] == ""

    def test_update_sso_settings_mixed_null_and_valid_values(
        self, mock_proxy_config, mock_auth, monkeypatch
    ):
        """Test updating SSO settings with mix of null and valid values - verifies both env vars and database"""
        import json
        from unittest.mock import AsyncMock, MagicMock

        monkeypatch.setenv("LITELLM_SALT_KEY", "test_salt_key")
        monkeypatch.setattr("litellm.proxy.proxy_server.store_model_in_db", True)

        # Mock the prisma client
        mock_prisma = MagicMock()
        mock_prisma.db.litellm_ssoconfig.upsert = AsyncMock()
        mock_prisma.db.litellm_config = MagicMock()

        env_var_entry = MagicMock()
        env_var_entry.param_value = json.dumps(
            {
                "GOOGLE_CLIENT_ID": "test_existing_google_id",
                "MICROSOFT_CLIENT_SECRET": "test_existing_microsoft_secret",
            }
        )
        mock_prisma.db.litellm_config.find_unique = AsyncMock(return_value=env_var_entry)
        mock_prisma.db.litellm_config.update = AsyncMock()
        monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)

        # Mock encryption to return values as-is
        from litellm.proxy.proxy_server import proxy_config
        monkeypatch.setattr(proxy_config, "_encrypt_env_variables", lambda environment_variables: environment_variables)

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

        # Verify runtime environment variables
        assert os.environ.get("GOOGLE_CLIENT_ID") == "new_google_client_id"
        assert os.environ.get("MICROSOFT_CLIENT_SECRET") == "new_microsoft_secret"
        assert "GOOGLE_CLIENT_SECRET" not in os.environ
        assert "MICROSOFT_CLIENT_ID" not in os.environ

        # Verify upsert was called with correct parameters
        assert mock_prisma.db.litellm_ssoconfig.upsert.called
        call_args = mock_prisma.db.litellm_ssoconfig.upsert.call_args
        
        # Verify the mixed values are stored correctly in database
        create_data = call_args.kwargs["data"]["create"]
        create_sso_settings = json.loads(create_data["sso_settings"])
        assert create_sso_settings["google_client_id"] == "new_google_client_id"
        assert create_sso_settings["google_client_secret"] is None
        assert create_sso_settings["microsoft_client_id"] is None
        assert create_sso_settings["microsoft_client_secret"] == "new_microsoft_secret"
        assert create_sso_settings["proxy_base_url"] == "https://newproxy.com"

    def test_update_sso_settings_ui_access_mode_handling(
        self, mock_proxy_config, mock_auth, monkeypatch
    ):
        """Test that ui_access_mode is handled correctly and stored in database"""
        import json
        from unittest.mock import AsyncMock, MagicMock

        monkeypatch.setenv("LITELLM_SALT_KEY", "test_salt_key")
        monkeypatch.setattr("litellm.proxy.proxy_server.store_model_in_db", True)

        # Mock the prisma client
        mock_prisma = MagicMock()
        mock_prisma.db.litellm_ssoconfig.upsert = AsyncMock()
        mock_prisma.db.litellm_config = MagicMock()
        mock_prisma.db.litellm_config.find_unique = AsyncMock(return_value=None)
        mock_prisma.db.litellm_config.update = AsyncMock()
        monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)

        # Mock encryption to return values as-is
        from litellm.proxy.proxy_server import proxy_config
        monkeypatch.setattr(proxy_config, "_encrypt_env_variables", lambda environment_variables: environment_variables)

        # Test setting ui_access_mode
        sso_settings_with_ui_mode = {
            "ui_access_mode": "admin_only",
            "user_email": "admin@test.com",
        }

        response = client.patch("/update/sso_settings", json=sso_settings_with_ui_mode)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

        # Verify upsert was called with correct data
        assert mock_prisma.db.litellm_ssoconfig.upsert.called
        call_args = mock_prisma.db.litellm_ssoconfig.upsert.call_args
        create_data = call_args.kwargs["data"]["create"]
        create_sso_settings = json.loads(create_data["sso_settings"])
        assert create_sso_settings["ui_access_mode"] == "admin_only"
        assert create_sso_settings["user_email"] == "admin@test.com"

        # Test clearing ui_access_mode
        clear_ui_mode = {"ui_access_mode": None, "user_email": None}

        response = client.patch("/update/sso_settings", json=clear_ui_mode)

        assert response.status_code == 200

        # Verify upsert was called again with null values
        assert mock_prisma.db.litellm_ssoconfig.upsert.call_count == 2
        call_args = mock_prisma.db.litellm_ssoconfig.upsert.call_args
        create_data = call_args.kwargs["data"]["create"]
        create_sso_settings = json.loads(create_data["sso_settings"])
        assert create_sso_settings["ui_access_mode"] is None
        assert create_sso_settings["user_email"] is None

    def test_get_ui_theme_settings(self, mock_proxy_config):
        """Test getting UI theme settings without authentication"""
        response = client.get("/get/ui_theme_settings")

        assert response.status_code == 200
        data = response.json()

        assert "values" in data
        assert "field_schema" in data

    def test_update_ui_theme_settings(self, mock_proxy_config, mock_auth, monkeypatch):
        """Test updating UI theme settings"""
        monkeypatch.setenv("LITELLM_SALT_KEY", "test_salt_key")
        monkeypatch.setattr("litellm.proxy.proxy_server.store_model_in_db", True)

        new_theme = {"logo_url": "https://example.com/new-logo.png"}

        response = client.patch("/update/ui_theme_settings", json=new_theme)

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "success"
        assert data["theme_config"]["logo_url"] == "https://example.com/new-logo.png"

        # Verify config was updated
        updated_config = mock_proxy_config["config"]
        assert "UI_LOGO_PATH" in updated_config["environment_variables"]
        assert mock_proxy_config["save_call_count"]() == 1

    def test_get_ui_settings(self, mock_auth, monkeypatch):
        """Test retrieving UI settings with allowlist sanitization"""
        from unittest.mock import AsyncMock, MagicMock

        mock_prisma = MagicMock()
        mock_db_record = MagicMock()
        mock_db_record.ui_settings = {
            "disable_model_add_for_internal_users": True,
            "unexpected_flag": True,
        }
        mock_prisma.db.litellm_uisettings.find_unique = AsyncMock(return_value=mock_db_record)
        monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)

        response = client.get("/get/ui_settings")

        assert response.status_code == 200
        data = response.json()
        assert data["values"]["disable_model_add_for_internal_users"] is True
        assert "unexpected_flag" not in data["values"]
        assert "disable_model_add_for_internal_users" in data["field_schema"]["properties"]
        mock_prisma.db.litellm_uisettings.find_unique.assert_called_once_with(
            where={"id": "ui_settings"}
        )

    @pytest.mark.parametrize(
        "user_role",
        [
            LitellmUserRoles.INTERNAL_USER,
            LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
        ],
    )
    def test_get_ui_settings_allows_internal_roles(self, monkeypatch, user_role):
        """Ensure internal users and viewers can fetch UI settings"""
        from unittest.mock import AsyncMock, MagicMock
        from litellm.proxy.ui_crud_endpoints import proxy_setting_endpoints

        mock_prisma = MagicMock()
        mock_db_record = MagicMock()
        mock_db_record.ui_settings = {"disable_model_add_for_internal_users": False}
        mock_prisma.db.litellm_uisettings.find_unique = AsyncMock(
            return_value=mock_db_record
        )
        monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)

        class MockUser:
            def __init__(self, role):
                self.user_role = role
                self.team_id = "litellm-dashboard"
                self.allowed_routes = []

        async def mock_user_api_key_auth():
            return MockUser(user_role)

        app.dependency_overrides[
            proxy_setting_endpoints.user_api_key_auth
        ] = mock_user_api_key_auth

        try:
            response = client.get("/get/ui_settings")
        finally:
            app.dependency_overrides.pop(
                proxy_setting_endpoints.user_api_key_auth, None
            )

        assert response.status_code == 200
        data = response.json()
        assert data["values"]["disable_model_add_for_internal_users"] is False
        mock_prisma.db.litellm_uisettings.find_unique.assert_called_once_with(
            where={"id": "ui_settings"}
        )

    def test_update_ui_settings_allowlisted_value(
        self, mock_auth, monkeypatch
    ):
        """Test updating UI settings with an allowlisted field"""
        from unittest.mock import AsyncMock, MagicMock
        from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
        from litellm.proxy._types import UserAPIKeyAuth

        # Override the FastAPI dependency with a proper mock
        mock_user_auth = UserAPIKeyAuth(
            user_id="test-user-123",
            user_role=LitellmUserRoles.PROXY_ADMIN,
        )
        app.dependency_overrides[user_api_key_auth] = lambda: mock_user_auth

        monkeypatch.setattr("litellm.proxy.proxy_server.store_model_in_db", True)
        mock_prisma = MagicMock()
        mock_prisma.db.litellm_uisettings.upsert = AsyncMock()
        monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)

        payload = {"disable_model_add_for_internal_users": True}

        try:
            response = client.patch("/update/ui_settings", json=payload)
        finally:
            # Clean up the dependency override
            app.dependency_overrides.clear()

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["settings"]["disable_model_add_for_internal_users"] is True

        assert mock_prisma.db.litellm_uisettings.upsert.called
        call_args = mock_prisma.db.litellm_uisettings.upsert.call_args
        assert call_args.kwargs["where"]["id"] == "ui_settings"
        create_data = call_args.kwargs["data"]["create"]
        stored_settings = json.loads(create_data["ui_settings"])
        assert stored_settings["disable_model_add_for_internal_users"] is True

    def test_update_ui_settings_ignores_non_allowlisted_value(
        self, mock_auth, monkeypatch
    ):
        """Test non-allowlisted UI settings are ignored on update"""
        from unittest.mock import AsyncMock, MagicMock
        from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
        from litellm.proxy._types import UserAPIKeyAuth

        # Override the FastAPI dependency with a proper mock
        mock_user_auth = UserAPIKeyAuth(
            user_id="test-user-123",
            user_role=LitellmUserRoles.PROXY_ADMIN,
        )
        app.dependency_overrides[user_api_key_auth] = lambda: mock_user_auth

        monkeypatch.setattr("litellm.proxy.proxy_server.store_model_in_db", True)
        mock_prisma = MagicMock()
        mock_prisma.db.litellm_uisettings.upsert = AsyncMock()
        monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)

        payload = {
            "disable_model_add_for_internal_users": False,
            "unsupported_flag": True,
        }

        try:
            response = client.patch("/update/ui_settings", json=payload)
        finally:
            # Clean up the dependency override
            app.dependency_overrides.clear()

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "unsupported_flag" not in data["settings"]
        assert data["settings"]["disable_model_add_for_internal_users"] is False

        assert mock_prisma.db.litellm_uisettings.upsert.called
        call_args = mock_prisma.db.litellm_uisettings.upsert.call_args
        stored_settings = json.loads(call_args.kwargs["data"]["create"]["ui_settings"])
        assert "unsupported_flag" not in stored_settings
        assert stored_settings["disable_model_add_for_internal_users"] is False

    def test_get_sso_settings_from_database(self, mock_proxy_config, mock_auth, monkeypatch):
        """Test getting SSO settings from the dedicated database table"""
        import json
        from unittest.mock import AsyncMock, MagicMock

        # Mock the prisma client
        mock_prisma = MagicMock()
        mock_db_record = MagicMock()
        
        # Simulate encrypted data from database
        mock_sso_settings = {
            "google_client_id": "encrypted_google_id",
            "google_client_secret": "encrypted_google_secret",
            "microsoft_client_id": "encrypted_microsoft_id",
            "proxy_base_url": "encrypted_proxy_url",
        }
        
        mock_db_record.sso_settings = mock_sso_settings
        mock_prisma.db.litellm_ssoconfig.find_unique = AsyncMock(return_value=mock_db_record)
        
        monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)
        
        # Mock the decryption method to return decrypted values
        def mock_decrypt_and_set(environment_variables):
            return {
                "google_client_id": "decrypted_google_id",
                "google_client_secret": "decrypted_google_secret",
                "microsoft_client_id": "decrypted_microsoft_id",
                "proxy_base_url": "https://decrypted.example.com",
            }
        
        from litellm.proxy.proxy_server import proxy_config
        monkeypatch.setattr(
            proxy_config, "_decrypt_and_set_db_env_variables", mock_decrypt_and_set
        )
        
        response = client.get("/get/sso_settings")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify structure
        assert "values" in data
        assert "field_schema" in data
        
        # Verify decrypted values are returned
        values = data["values"]
        assert values["google_client_id"] == "decrypted_google_id"
        assert values["google_client_secret"] == "decrypted_google_secret"
        assert values["microsoft_client_id"] == "decrypted_microsoft_id"
        assert values["proxy_base_url"] == "https://decrypted.example.com"
        
        # Verify role_mappings is present in response (can be None if not set)
        assert "role_mappings" in values
        assert values["role_mappings"] is None

    def test_update_sso_settings_to_database(self, mock_proxy_config, mock_auth, monkeypatch):
        """Test updating SSO settings saves to the dedicated database table"""
        import json
        from unittest.mock import AsyncMock, MagicMock

        monkeypatch.setenv("LITELLM_SALT_KEY", "test_salt_key")
        
        # Mock the prisma client
        mock_prisma = MagicMock()
        upsert_mock = AsyncMock()
        mock_prisma.db.litellm_ssoconfig.upsert = upsert_mock
        mock_prisma.db.litellm_config = MagicMock()
        mock_prisma.db.litellm_config.find_unique = AsyncMock(return_value=None)
        mock_prisma.db.litellm_config.update = AsyncMock()
        
        monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)
        monkeypatch.setattr("litellm.proxy.proxy_server.store_model_in_db", True)
        
        # Track what was encrypted
        encrypted_data = {}
        
        def mock_encrypt(environment_variables):
            # Simulate encryption by adding prefix
            encrypted = {
                k: f"encrypted_{v}" if v else v 
                for k, v in environment_variables.items()
            }
            encrypted_data.update(encrypted)
            return encrypted
        
        from litellm.proxy.proxy_server import proxy_config
        monkeypatch.setattr(proxy_config, "_encrypt_env_variables", mock_encrypt)
        
        # New SSO settings to save
        new_sso_settings = {
            "google_client_id": "new_google_id",
            "google_client_secret": "new_google_secret",
            "microsoft_client_id": "new_microsoft_id",
            "proxy_base_url": "https://new.example.com",
        }
        
        response = client.patch("/update/sso_settings", json=new_sso_settings)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "success"
        assert data["settings"]["google_client_id"] == "new_google_id"
        
        # Verify upsert was called
        assert upsert_mock.called
        call_args = upsert_mock.call_args
        
        # Verify it's using the correct ID
        assert call_args.kwargs["where"]["id"] == "sso_config"
        
        # Verify encrypted data was saved
        create_data = call_args.kwargs["data"]["create"]
        update_data = call_args.kwargs["data"]["update"]
        
        assert create_data["id"] == "sso_config"
        # The sso_settings should be JSON string of encrypted data
        assert "sso_settings" in create_data
        assert "sso_settings" in update_data
        
        # Verify the encrypted data is correctly stored
        create_sso_settings = json.loads(create_data["sso_settings"])
        assert create_sso_settings["google_client_id"] == "encrypted_new_google_id"
        assert create_sso_settings["google_client_secret"] == "encrypted_new_google_secret"
        assert create_sso_settings["proxy_base_url"] == "encrypted_https://new.example.com"

    def test_update_sso_settings_removes_sso_env_vars_from_config(
        self, mock_proxy_config, mock_auth, monkeypatch
    ):
        """Ensure SSO-related env vars are deleted from stored config"""
        import json
        from unittest.mock import AsyncMock, MagicMock

        monkeypatch.setenv("LITELLM_SALT_KEY", "test_salt_key")
        monkeypatch.setattr("litellm.proxy.proxy_server.store_model_in_db", True)

        mock_prisma = MagicMock()
        mock_prisma.db = MagicMock()
        mock_prisma.db.litellm_ssoconfig = MagicMock()
        mock_prisma.db.litellm_ssoconfig.upsert = AsyncMock()

        env_var_entry = MagicMock()
        env_var_entry.param_value = json.dumps(
            {
                "GOOGLE_CLIENT_ID": "old_google_id",
                "GENERIC_TOKEN_ENDPOINT": "old_endpoint",
                "UNCHANGED_ENV": "keep_me",
            }
        )
        mock_prisma.db.litellm_config = MagicMock()
        mock_prisma.db.litellm_config.find_unique = AsyncMock(return_value=env_var_entry)
        mock_prisma.db.litellm_config.update = AsyncMock()
        monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)

        from litellm.proxy.proxy_server import proxy_config

        monkeypatch.setattr(
            proxy_config,
            "_encrypt_env_variables",
            lambda environment_variables: environment_variables,
        )

        response = client.patch(
            "/update/sso_settings", json={"google_client_id": "new_google_id"}
        )

        assert response.status_code == 200
        mock_prisma.db.litellm_config.find_unique.assert_called_once()
        mock_prisma.db.litellm_config.update.assert_called_once()
        update_call = mock_prisma.db.litellm_config.update.call_args
        updated_env_vars = json.loads(update_call.kwargs["data"]["param_value"])
        assert "GOOGLE_CLIENT_ID" not in updated_env_vars
        assert "GENERIC_TOKEN_ENDPOINT" not in updated_env_vars
        assert updated_env_vars["UNCHANGED_ENV"] == "keep_me"

    def test_update_sso_settings_preserves_non_sso_env_vars(
        self, mock_proxy_config, mock_auth, monkeypatch
    ):
        """Ensure env vars outside SSO mapping remain unchanged"""
        import json
        from unittest.mock import AsyncMock, MagicMock

        monkeypatch.setenv("LITELLM_SALT_KEY", "test_salt_key")
        monkeypatch.setattr("litellm.proxy.proxy_server.store_model_in_db", True)

        mock_prisma = MagicMock()
        mock_prisma.db = MagicMock()
        mock_prisma.db.litellm_ssoconfig = MagicMock()
        mock_prisma.db.litellm_ssoconfig.upsert = AsyncMock()

        env_var_entry = MagicMock()
        env_var_entry.param_value = {
            "UNRELATED_ENV": "keep_this",
            "ANOTHER_ENV": "also_keep",
        }
        mock_prisma.db.litellm_config = MagicMock()
        mock_prisma.db.litellm_config.find_unique = AsyncMock(return_value=env_var_entry)
        mock_prisma.db.litellm_config.update = AsyncMock()
        monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)

        from litellm.proxy.proxy_server import proxy_config

        monkeypatch.setattr(
            proxy_config,
            "_encrypt_env_variables",
            lambda environment_variables: environment_variables,
        )

        response = client.patch(
            "/update/sso_settings", json={"microsoft_client_id": "new_microsoft_id"}
        )

        assert response.status_code == 200
        mock_prisma.db.litellm_config.find_unique.assert_called_once()
        mock_prisma.db.litellm_config.update.assert_called_once()
        update_call = mock_prisma.db.litellm_config.update.call_args
        updated_env_vars = json.loads(update_call.kwargs["data"]["param_value"])
        assert updated_env_vars == env_var_entry.param_value

    def test_get_sso_settings_empty_database(self, mock_proxy_config, mock_auth, monkeypatch):
        """Test getting SSO settings when database table is empty"""
        from unittest.mock import AsyncMock, MagicMock

        # Mock the prisma client to return None (no record found)
        mock_prisma = MagicMock()
        mock_prisma.db.litellm_ssoconfig.find_unique = AsyncMock(return_value=None)
        
        monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)
        
        # Mock the decryption method
        def mock_decrypt_and_set(environment_variables):
            # Should receive empty dict
            return environment_variables
        
        from litellm.proxy.proxy_server import proxy_config
        monkeypatch.setattr(
            proxy_config, "_decrypt_and_set_db_env_variables", mock_decrypt_and_set
        )
        
        response = client.get("/get/sso_settings")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify structure is still correct with empty values
        assert "values" in data
        assert "field_schema" in data
        
        # All values should be None
        values = data["values"]
        assert values.get("google_client_id") is None
        assert values.get("google_client_secret") is None
        assert values.get("microsoft_client_id") is None
        assert values.get("role_mappings") is None

    def test_update_sso_settings_no_database_connection(self, mock_proxy_config, mock_auth, monkeypatch):
        """Test updating SSO settings when database is not connected"""
        monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
        
        new_sso_settings = {
            "google_client_id": "new_google_id",
        }
        
        response = client.patch("/update/sso_settings", json=new_sso_settings)
        
        assert response.status_code == 500
        data = response.json()
        assert "error" in data["detail"]
        assert "Database not connected" in data["detail"]["error"]

    def test_get_sso_settings_no_database_connection(self, mock_proxy_config, mock_auth, monkeypatch):
        """Test getting SSO settings when database is not connected"""
        monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
        
        response = client.get("/get/sso_settings")
        
        assert response.status_code == 500
        data = response.json()
        assert "error" in data["detail"]
        assert "Database not connected" in data["detail"]["error"]

    def test_get_sso_settings_with_role_mappings(self, mock_proxy_config, mock_auth, monkeypatch):
        """Test getting SSO settings when role_mappings is present in database"""
        from unittest.mock import AsyncMock, MagicMock
        from litellm.proxy._types import LitellmUserRoles

        # Mock the prisma client with database record containing role_mappings
        mock_prisma = MagicMock()
        mock_db_record = MagicMock()
        mock_db_record.sso_settings = {
            "google_client_id": "test_google_client_id",
            "role_mappings": {
                "provider": "google",
                "group_claim": "groups",
                "default_role": LitellmUserRoles.INTERNAL_USER,
                "roles": {
                    LitellmUserRoles.PROXY_ADMIN: ["admin-group"],
                },
            },
        }
        mock_prisma.db.litellm_ssoconfig.find_unique = AsyncMock(return_value=mock_db_record)
        monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)

        # Mock decryption to return the values as-is (role_mappings should not be passed to decryption)
        from litellm.proxy.proxy_server import proxy_config
        def mock_decrypt(environment_variables):
            # role_mappings should not be in environment_variables since it's extracted before decryption
            assert "role_mappings" not in environment_variables
            return environment_variables
        
        monkeypatch.setattr(
            proxy_config, "_decrypt_and_set_db_env_variables", mock_decrypt
        )

        response = client.get("/get/sso_settings")

        assert response.status_code == 200
        data = response.json()

        # Verify role_mappings is returned correctly
        values = data["values"]
        assert "role_mappings" in values
        assert values["role_mappings"] is not None
        assert values["role_mappings"]["provider"] == "google"
        assert values["role_mappings"]["group_claim"] == "groups"
        assert values["role_mappings"]["default_role"] == LitellmUserRoles.INTERNAL_USER
        assert values["role_mappings"]["roles"][LitellmUserRoles.PROXY_ADMIN] == ["admin-group"]

    def test_role_mappings_stored_and_retrieved(self, mock_proxy_config, mock_auth, monkeypatch):
        """Test that role_mappings is properly stored and retrieved from SSO settings"""
        import json
        from unittest.mock import AsyncMock, MagicMock
        from litellm.proxy._types import LitellmUserRoles

        monkeypatch.setenv("LITELLM_SALT_KEY", "test_salt_key")
        monkeypatch.setattr("litellm.proxy.proxy_server.store_model_in_db", True)

        # Mock the prisma client
        mock_prisma = MagicMock()
        mock_prisma.db.litellm_ssoconfig.upsert = AsyncMock()
        mock_prisma.db.litellm_config = MagicMock()
        mock_prisma.db.litellm_config.find_unique = AsyncMock(return_value=None)
        mock_prisma.db.litellm_config.update = AsyncMock()
        monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)

        # Mock encryption to return values as-is
        from litellm.proxy.proxy_server import proxy_config
        monkeypatch.setattr(proxy_config, "_encrypt_env_variables", lambda environment_variables: environment_variables)

        # SSO settings with role_mappings
        role_mappings_data = {
            "provider": "google",
            "group_claim": "groups",
            "default_role": LitellmUserRoles.INTERNAL_USER,
            "roles": {
                LitellmUserRoles.PROXY_ADMIN: ["admin-group"],
                LitellmUserRoles.INTERNAL_USER: ["user-group"],
            },
        }

        new_sso_settings = {
            "google_client_id": "test_google_id",
            "role_mappings": role_mappings_data,
        }

        response = client.patch("/update/sso_settings", json=new_sso_settings)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "role_mappings" in data["settings"]
        
        # Verify role_mappings structure in response
        returned_role_mappings = data["settings"]["role_mappings"]
        assert returned_role_mappings["provider"] == "google"
        assert returned_role_mappings["group_claim"] == "groups"
        assert returned_role_mappings["default_role"] == LitellmUserRoles.INTERNAL_USER
        assert returned_role_mappings["roles"][LitellmUserRoles.PROXY_ADMIN] == ["admin-group"]

        # Verify upsert was called with role_mappings in the data
        assert mock_prisma.db.litellm_ssoconfig.upsert.called
        call_args = mock_prisma.db.litellm_ssoconfig.upsert.call_args
        create_data = call_args.kwargs["data"]["create"]
        stored_sso_settings = json.loads(create_data["sso_settings"])
        assert "role_mappings" in stored_sso_settings
        assert stored_sso_settings["role_mappings"]["provider"] == "google"

        # Now test retrieving role_mappings
        mock_db_record = MagicMock()
        mock_db_record.sso_settings = stored_sso_settings
        mock_prisma.db.litellm_ssoconfig.find_unique = AsyncMock(return_value=mock_db_record)
        monkeypatch.setattr(
            proxy_config, "_decrypt_and_set_db_env_variables", lambda environment_variables: environment_variables
        )

        get_response = client.get("/get/sso_settings")
        assert get_response.status_code == 200
        get_data = get_response.json()
        
        # Verify role_mappings is returned correctly
        assert "role_mappings" in get_data["values"]
        retrieved_role_mappings = get_data["values"]["role_mappings"]
        assert retrieved_role_mappings is not None
        assert retrieved_role_mappings["provider"] == "google"
        assert retrieved_role_mappings["group_claim"] == "groups"
        assert retrieved_role_mappings["default_role"] == LitellmUserRoles.INTERNAL_USER

    def test_setup_role_mappings_custom_logic_with_env_vars(self, monkeypatch):
        """Test the _setup_role_mappings function directly with custom role mapping logic from environment variables"""
        import asyncio
        import os
        from litellm.proxy.management_endpoints.ui_sso import _setup_role_mappings
        from litellm.proxy._types import LitellmUserRoles

        # Set up environment variables for custom role mappings using valid Python dict format
        monkeypatch.setenv("GENERIC_ROLE_MAPPINGS_ROLES", "{'proxy_admin': ['custom-admin-group'], 'internal_user': ['custom-user-group'], 'proxy_admin_viewer': ['custom-viewer-group']}")
        monkeypatch.setenv("GENERIC_ROLE_MAPPINGS_GROUP_CLAIM", "custom-groups")
        monkeypatch.setenv("GENERIC_ROLE_MAPPINGS_DEFAULT_ROLE", "internal_user_viewer")

        # Debug: Print environment variables
        print("GENERIC_ROLE_MAPPINGS_ROLES:", os.getenv("GENERIC_ROLE_MAPPINGS_ROLES"))
        print("GENERIC_ROLE_MAPPINGS_GROUP_CLAIM:", os.getenv("GENERIC_ROLE_MAPPINGS_GROUP_CLAIM"))
        print("GENERIC_ROLE_MAPPINGS_DEFAULT_ROLE:", os.getenv("GENERIC_ROLE_MAPPINGS_DEFAULT_ROLE"))

        # Run the async function
        role_mappings = asyncio.run(_setup_role_mappings())
        
        # Debug: Print result
        print("role_mappings result:", role_mappings)

        # Verify role_mappings is returned correctly from environment variables
        assert role_mappings is not None
        assert role_mappings.provider == "generic"
        assert role_mappings.group_claim == "custom-groups"
        assert role_mappings.default_role == LitellmUserRoles.INTERNAL_USER_VIEW_ONLY
        assert role_mappings.roles[LitellmUserRoles.PROXY_ADMIN] == ["custom-admin-group"]
        assert role_mappings.roles[LitellmUserRoles.INTERNAL_USER] == ["custom-user-group"]
        assert role_mappings.roles[LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY] == ["custom-viewer-group"]

    def test_setup_role_mappings_custom_logic_with_no_config(self, monkeypatch):
        """Test the _setup_role_mappings function returns None when no configuration is available"""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        from litellm.proxy.management_endpoints.ui_sso import _setup_role_mappings

        # Ensure environment variables are not set
        monkeypatch.delenv("GENERIC_ROLE_MAPPINGS_ROLES", raising=False)
        monkeypatch.delenv("GENERIC_ROLE_MAPPINGS_GROUP_CLAIM", raising=False)
        monkeypatch.delenv("GENERIC_ROLE_MAPPINGS_DEFAULT_ROLE", raising=False)

        # Mock the prisma client to return None (no database record)
        mock_prisma = MagicMock()
        mock_prisma.db.litellm_ssoconfig.find_unique = AsyncMock(return_value=None)
        # Run the async function
        role_mappings = asyncio.run(_setup_role_mappings())

        # Should return None when no configuration is available
        assert role_mappings is None

    def test_get_sso_settings_with_env_role_mappings(self, mock_proxy_config, mock_auth, monkeypatch):
        import json
        from unittest.mock import AsyncMock, MagicMock
        from litellm.proxy._types import LitellmUserRoles
        
        monkeypatch.setenv("GENERIC_ROLE_MAPPINGS_ROLES", '{"proxy_admin": ["custom-admin-group"], "internal_user": ["custom-user-group"], "proxy_admin_viewer": ["custom-viewer-group"]}')
        monkeypatch.setenv("GENERIC_ROLE_MAPPINGS_GROUP_CLAIM", "custom-groups")
        monkeypatch.setenv("GENERIC_ROLE_MAPPINGS_DEFAULT_ROLE", "internal_user_viewer")
        
        mock_prisma = MagicMock()
        mock_db_record = MagicMock()
        mock_db_record.sso_settings = {
            "google_client_id": "test_google_client_id",
            "role_mappings": {
                "provider": "google",
                "group_claim": "db-groups",
                "default_role": "proxy_admin",
                "roles": {
                    "proxy_admin": ["db-admin-group"],
                },
            },
        }
        mock_prisma.db.litellm_ssoconfig.find_unique = AsyncMock(return_value=mock_db_record)
        monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)
        
        from litellm.proxy.proxy_server import proxy_config
        monkeypatch.setattr(
            proxy_config, "_decrypt_and_set_db_env_variables", lambda environment_variables: environment_variables
        )
        
        response = client.get("/get/sso_settings")
        
        assert response.status_code == 200
        data = response.json()
        
        values = data["values"]
        assert "role_mappings" in values
        assert values["role_mappings"] is not None
        
        # The database values shoeld override the environment variables
        assert values["role_mappings"]["provider"] == "google"
        assert values["role_mappings"]["group_claim"] == "db-groups"
        assert values["role_mappings"]["default_role"] == LitellmUserRoles.PROXY_ADMIN
        assert values["role_mappings"]["roles"][LitellmUserRoles.PROXY_ADMIN] == ["db-admin-group"]
        
        # Verify that the database was checked but environment variables took priority
        mock_prisma.db.litellm_ssoconfig.find_unique.assert_called_once_with(
            where={"id": "sso_config"}
        )
        
        # Verify other SSO settings are still correctly returned
        assert values["google_client_id"] == "test_google_client_id"
        
        # Verify field_schema is still present
        assert "field_schema" in data
        assert "properties" in data["field_schema"]
        assert "role_mappings" in data["field_schema"]["properties"]
