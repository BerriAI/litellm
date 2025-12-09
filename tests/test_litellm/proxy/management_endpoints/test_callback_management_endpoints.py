import json
import os
import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)  #

from typing import cast

from fastapi import FastAPI

import litellm
from litellm.integrations.datadog.datadog import DataDogLogger
from litellm.integrations.langfuse.langfuse import LangFuseLogger
from litellm.proxy.management_endpoints.callback_management_endpoints import router
from litellm.proxy.proxy_server import app


@pytest.fixture(autouse=True, scope="session")
def clear_existing_callbacks():
    litellm.logging_callback_manager._reset_all_callbacks()

class TestCallbackManagementEndpoints:
    """Test suite for callback management endpoints"""
    
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test"""
        # Reset callbacks before each test
        litellm.success_callback = []
        litellm.failure_callback = []
        litellm._async_success_callback = []
        litellm._async_failure_callback = []
        litellm.callbacks = []
        
        yield
        
        # Clean up after each test
        litellm.success_callback = []
        litellm.failure_callback = []
        litellm._async_success_callback = []
        litellm._async_failure_callback = []
        litellm.callbacks = []

    def test_alist_callbacks_no_active_callbacks(self):
        """Test /callbacks/list endpoint with no active callbacks"""
        # Setup test client
        client = TestClient(app)
        
        # Make request to list callbacks endpoint
        response = client.get(
            "/callbacks/list",
            headers={"Authorization": "Bearer sk-1234"}
        )
        
        # Verify response
        assert response.status_code == 200
        
        response_data = response.json()
        assert "success" in response_data
        assert "failure" in response_data
        assert "success_and_failure" in response_data
        
        # All lists should be empty
        assert response_data["success"] == []
        assert response_data["failure"] == []
        assert response_data["success_and_failure"] == []

    @patch.dict(os.environ, {
        "LANGFUSE_PUBLIC_KEY": "test_public_key",
        "LANGFUSE_SECRET_KEY": "test_secret_key",
        "LANGFUSE_HOST": "https://test.langfuse.com"
    })
    def test_alist_callbacks_with_langfuse_logger(self):
        """Test /callbacks/list endpoint with real Langfuse logger initialized"""
        # Setup test client
        client = TestClient(app)
        
        # Initialize Langfuse logger and add to callbacks
        with patch('litellm.integrations.langfuse.langfuse.Langfuse') as mock_langfuse:
            # Mock the Langfuse client initialization
            mock_langfuse_client = MagicMock()
            mock_langfuse.return_value = mock_langfuse_client
            

            # Add string representation to callback lists (this is how the system typically works)
            litellm.success_callback.append("langfuse")
            litellm._async_success_callback.append("langfuse")
            
            # Make request to list callbacks endpoint
            response = client.get(
                "/callbacks/list",
                headers={"Authorization": "Bearer sk-1234"}
            )
            
            # Verify response
            assert response.status_code == 200
            
            response_data = response.json()
            
            # Verify langfuse appears in success callbacks
            assert "langfuse" in response_data["success"]
            assert response_data["failure"] == []
            assert response_data["success_and_failure"] == []
            
            # Verify the response structure is correct
            assert isinstance(response_data["success"], list)
            assert isinstance(response_data["failure"], list)
            assert isinstance(response_data["success_and_failure"], list)

    def test_alist_callbacks_with_datadog_logger(self):
        """Test /callbacks/list endpoint with DataDog logger configuration"""
        # Setup test client
        client = TestClient(app)
        
        # Test with datadog callbacks added directly (without initializing the logger to avoid async issues)
        # Add string representations to different callback types to test comprehensive categorization
        litellm.success_callback.append("datadog")
        litellm.failure_callback.append("datadog")
        litellm.callbacks.append("datadog")
        
        # Make request to list callbacks endpoint
        response = client.get(
            "/callbacks/list",
            headers={"Authorization": "Bearer sk-1234"}
        )
        
        # Verify response
        assert response.status_code == 200
        
        response_data = response.json()
        
        # Verify datadog appears in the correct categorization
        # Since datadog is in both success and failure, it should appear in success_and_failure
        assert "datadog" in response_data["success_and_failure"]
        
        # The categorization logic should deduplicate properly
        assert len([cb for cb in response_data["success"] if cb == "datadog"]) <= 1
        assert len([cb for cb in response_data["failure"] if cb == "datadog"]) <= 1
        assert len([cb for cb in response_data["success_and_failure"] if cb == "datadog"]) <= 1
        
        # Verify the response structure is correct
        assert isinstance(response_data["success"], list)
        assert isinstance(response_data["failure"], list)
        assert isinstance(response_data["success_and_failure"], list)

    def test_alist_callbacks_mixed_callback_types(self):
        """Test /callbacks/list endpoint with mixed callback types (string and logger instances)"""
        # Setup test client  
        client = TestClient(app)
        
        # Setup mixed callbacks
        litellm.success_callback.append("langfuse")
        litellm.failure_callback.append("datadog")
        litellm.callbacks.append("prometheus")
        
        # Make request to list callbacks endpoint
        response = client.get(
            "/callbacks/list",
            headers={"Authorization": "Bearer sk-1234"}
        )
        
        # Verify response
        assert response.status_code == 200
        
        response_data = response.json()
        
        # Filter out any proxy-specific callbacks that might be present from parallel test runs
        # These are internal callbacks that can persist when tests run in parallel
        proxy_internal_callbacks = ["_PROXY_VirtualKeyModelMaxBudgetLimiter"]
        
        response_data["success_and_failure"] = [
            cb for cb in response_data["success_and_failure"] 
            if cb not in proxy_internal_callbacks
        ]
        
        # Verify callbacks are properly categorized
        assert "prometheus" in response_data["success_and_failure"]  # callbacks list items go to success_and_failure
        assert "langfuse" in response_data["success"]
        assert "datadog" in response_data["failure"]
        
        # Verify no duplicates
        all_callbacks = (
            response_data["success"] + 
            response_data["failure"] + 
            response_data["success_and_failure"]
        )
        assert len(set(all_callbacks)) == len(all_callbacks)


    def test_alist_callbacks_empty_response_structure(self):
        """Test that response always has correct structure even with no callbacks"""
        # Setup test client
        client = TestClient(app)
        
        # Make request to list callbacks endpoint
        response = client.get(
            "/callbacks/list",
            headers={"Authorization": "Bearer sk-1234"}
        )
        
        # Verify response structure
        assert response.status_code == 200
        response_data = response.json()
        
        # Verify all required keys are present
        required_keys = ["success", "failure", "success_and_failure"]
        for key in required_keys:
            assert key in response_data
            assert isinstance(response_data[key], list)

    def test_get_callback_configs(self):
        """Test /callbacks/configs endpoint returns callback configuration JSON"""
        # Setup test client
        client = TestClient(app)
        
        # Make request to get callback configs endpoint
        response = client.get(
            "/callbacks/configs",
            headers={"Authorization": "Bearer sk-1234"}
        )
        
        # Verify response
        assert response.status_code == 200
        
        response_data = response.json()
        
        # Verify response is a list
        assert isinstance(response_data, list)
        
        # Verify it contains callback configurations
        assert len(response_data) > 0
        
        # Verify structure of first callback config
        first_config = response_data[0]
        assert "id" in first_config
        assert "displayName" in first_config
        assert "logo" in first_config
        assert "supports_key_team_logging" in first_config
        assert "dynamic_params" in first_config
        assert "description" in first_config
        
        # Verify dynamic_params structure
        assert isinstance(first_config["dynamic_params"], dict)
        
        # Check if at least one callback has detailed parameter configuration
        has_detailed_params = any(
            config.get("dynamic_params") and len(config.get("dynamic_params", {})) > 0
            for config in response_data
        )
        assert has_detailed_params, "Expected at least one callback to have detailed parameter configuration"

