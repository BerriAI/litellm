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
            }
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

        # Check structure of response
        assert "values" in data
        assert "schema" in data

        # Check values match our mock config
        values = data["values"]
        mock_params = mock_proxy_config["config"]["litellm_settings"][
            "default_internal_user_params"
        ]
        assert values["user_role"] == mock_params["user_role"]
        assert values["max_budget"] == mock_params["max_budget"]
        assert values["budget_duration"] == mock_params["budget_duration"]
        assert values["models"] == mock_params["models"]

        # Check schema contains descriptions
        assert "properties" in data["schema"]
        assert "user_role" in data["schema"]["properties"]
        assert "description" in data["schema"]["properties"]["user_role"]

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
