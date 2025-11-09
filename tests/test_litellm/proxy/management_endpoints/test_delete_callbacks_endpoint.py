import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)

from litellm.proxy._types import (
    CallbackDelete,
    ConfigYAML,
    LitellmUserRoles,
    UserAPIKeyAuth,
)
from litellm.proxy.proxy_server import app

client = TestClient(app)


class MockPrismaClient:
    def __init__(self):
        self.db = MagicMock()
        self.config_data = {
            "litellm_settings": {"success_callback": ["langfuse"]},
            "environment_variables": {
                "LANGFUSE_PUBLIC_KEY": "any-public-key",
                "LANGFUSE_SECRET_KEY": "any-secret-key", 
                "LANGFUSE_HOST": "https://exampleopenaiendpoint-production-c715.up.railway.app",
            },
        }
        
        # Mock the config update/upsert
        self.db.litellm_config.upsert = AsyncMock()
        
        # Mock config retrieval for get_config/callbacks
        self.db.litellm_config.find_first = AsyncMock(
            side_effect=self._mock_find_first
        )
        
        # Mock for get_generic_data
        self.get_generic_data = AsyncMock(side_effect=self._mock_get_generic_data)
        
        # Mock insert_data method (required by delete_callback endpoint)
        self.insert_data = AsyncMock(return_value=MagicMock())
        
        # Mock jsonify_object method (required by config endpoints)
        self.jsonify_object = lambda obj: obj

    async def _mock_find_first(self, where=None):
        """Mock find_first to return config data based on param_name"""
        if where and "param_name" in where:
            param_name = where["param_name"]
            if param_name == "litellm_settings":
                return MagicMock(
                    param_name="litellm_settings",
                    param_value=self.config_data["litellm_settings"]
                )
            elif param_name == "environment_variables":
                return MagicMock(
                    param_name="environment_variables", 
                    param_value=self.config_data["environment_variables"]
                )
        return None

    async def _mock_get_generic_data(self, key=None, value=None, table_name=None):
        """Mock get_generic_data for _update_config_from_db"""
        if key == "param_name" and table_name == "config":
            if value == "litellm_settings":
                return MagicMock(
                    param_name="litellm_settings",
                    param_value=self.config_data["litellm_settings"]
                )
            elif value == "environment_variables":
                return MagicMock(
                    param_name="environment_variables",
                    param_value=self.config_data["environment_variables"]
                )
            elif value in ["general_settings", "router_settings"]:
                return None
        return None

    def remove_callback_from_config(self, callback_name):
        """Remove callback from the mock config"""
        if "success_callback" in self.config_data["litellm_settings"]:
            callbacks = self.config_data["litellm_settings"]["success_callback"]
            if callback_name in callbacks:
                callbacks.remove(callback_name)


@pytest.fixture
def mock_auth():
    """Mock admin user authentication"""
    return UserAPIKeyAuth(
        user_id="test_admin",
        user_role=LitellmUserRoles.PROXY_ADMIN,
        api_key="sk-1234"
    )


@pytest.fixture
def mock_prisma():
    """Mock prisma client"""
    return MockPrismaClient()


def mock_encrypt_value_helper(value, key=None, new_encryption_key=None):
    """Mock encryption - just return the value as-is for testing"""
    return value

def mock_decrypt_value_helper(value, key=None, return_original_value=False):
    """Mock decryption - just return the value as-is for testing"""
    return value
