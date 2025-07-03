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

import litellm
from litellm.integrations.datadog.datadog import DataDogLogger
from litellm.integrations.langfuse.langfuse import LangFuseLogger
from litellm.proxy.management_endpoints.callback_management_endpoints import router
from litellm.proxy.proxy_server import app


class TestCallbackManagementEndpoints:
    """Test suite for callback management endpoints"""
    
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test"""
        # Reset callbacks before each test using the callback manager
        litellm.logging_callback_manager._reset_all_callbacks()
        
        yield
        
        # Clean up after each test
        litellm.logging_callback_manager._reset_all_callbacks()

    def test_list_callbacks_no_active_callbacks(self):
        """Test /callbacks/list endpoint with no active callbacks"""
        from unittest.mock import patch
        client = TestClient(app)
        
        # Create fresh callback lists for this test
        test_success_callback = []
        test_failure_callback = []
        test_callbacks = []
        test_async_success_callback = []
        test_async_failure_callback = []

        with patch("litellm.success_callback", test_success_callback), \
             patch("litellm.failure_callback", test_failure_callback), \
             patch("litellm.callbacks", test_callbacks), \
             patch("litellm._async_success_callback", test_async_success_callback), \
             patch("litellm._async_failure_callback", test_async_failure_callback), \
             patch("litellm.logging_callback_manager._get_all_callbacks") as mock_get_all:
            
            # Mock the callback manager to return our test lists
            def mock_get_all_callbacks():
                return test_callbacks + test_success_callback + test_failure_callback + test_async_success_callback + test_async_failure_callback
            
            mock_get_all.side_effect = mock_get_all_callbacks
            
            response = client.get(
                "/callbacks/list",
                headers={"Authorization": "Bearer sk-1234"}
            )
            assert response.status_code == 200
            response_data = response.json()
            assert "success" in response_data
            assert "failure" in response_data
            assert "success_and_failure" in response_data
            
            # All lists should be empty (excluding system callbacks that start with _PROXY_)
            non_system_success = [cb for cb in response_data["success"] if not cb.startswith("_PROXY_")]
            non_system_failure = [cb for cb in response_data["failure"] if not cb.startswith("_PROXY_")]
            non_system_success_and_failure = [cb for cb in response_data["success_and_failure"] if not cb.startswith("_PROXY_")]
            
            assert non_system_success == []
            assert non_system_failure == []
            assert non_system_success_and_failure == []

    @patch.dict(os.environ, {
        "LANGFUSE_PUBLIC_KEY": "test_public_key",
        "LANGFUSE_SECRET_KEY": "test_secret_key",
        "LANGFUSE_HOST": "https://test.langfuse.com"
    })
    def test_list_callbacks_with_langfuse_logger(self):
        """Test /callbacks/list endpoint with real Langfuse logger initialized"""
        from unittest.mock import patch
        client = TestClient(app)

        # Create fresh callback lists for this test
        test_success_callback = []
        test_failure_callback = []
        test_callbacks = []
        test_async_success_callback = []
        test_async_failure_callback = []

        with patch("litellm.success_callback", test_success_callback), \
             patch("litellm.failure_callback", test_failure_callback), \
             patch("litellm.callbacks", test_callbacks), \
             patch("litellm._async_success_callback", test_async_success_callback), \
             patch("litellm._async_failure_callback", test_async_failure_callback), \
             patch("litellm.logging_callback_manager._get_all_callbacks") as mock_get_all:
            
            # Mock the callback manager to return our test lists
            def mock_get_all_callbacks():
                return test_callbacks + test_success_callback + test_failure_callback + test_async_success_callback + test_async_failure_callback
            
            mock_get_all.side_effect = mock_get_all_callbacks
            
            # Initialize Langfuse logger and add to callbacks
            with patch('litellm.integrations.langfuse.langfuse.Langfuse') as mock_langfuse:
                # Mock the Langfuse client initialization
                mock_langfuse_client = MagicMock()
                mock_langfuse.return_value = mock_langfuse_client
                
                # Add string representation to callback lists (this is how the system typically works)
                test_success_callback.append("langfuse")
                test_async_success_callback.append("langfuse")
                
                response = client.get(
                    "/callbacks/list",
                    headers={"Authorization": "Bearer sk-1234"}
                )
                assert response.status_code == 200
                response_data = response.json()
                
                # Verify langfuse appears in success callbacks
                assert "langfuse" in response_data["success"]
                # Filter out system callbacks for failure and success_and_failure checks
                non_system_failure = [cb for cb in response_data["failure"] if not cb.startswith("_PROXY_")]
                non_system_success_and_failure = [cb for cb in response_data["success_and_failure"] if not cb.startswith("_PROXY_")]
                assert non_system_failure == []
                assert non_system_success_and_failure == []
                
                # Verify the response structure is correct
                assert isinstance(response_data["success"], list)
                assert isinstance(response_data["failure"], list)
                assert isinstance(response_data["success_and_failure"], list)

    def test_list_callbacks_with_datadog_logger(self):
        """Test /callbacks/list endpoint with DataDog logger configuration"""
        from unittest.mock import patch
        client = TestClient(app)

        # Create fresh callback lists for this test
        test_success_callback = []
        test_failure_callback = []
        test_callbacks = []
        test_async_success_callback = []
        test_async_failure_callback = []

        with patch("litellm.success_callback", test_success_callback), \
             patch("litellm.failure_callback", test_failure_callback), \
             patch("litellm.callbacks", test_callbacks), \
             patch("litellm._async_success_callback", test_async_success_callback), \
             patch("litellm._async_failure_callback", test_async_failure_callback), \
             patch("litellm.logging_callback_manager._get_all_callbacks") as mock_get_all:
            
            # Mock the callback manager to return our test lists
            def mock_get_all_callbacks():
                return test_callbacks + test_success_callback + test_failure_callback + test_async_success_callback + test_async_failure_callback
            
            mock_get_all.side_effect = mock_get_all_callbacks
            
            # Test with datadog callbacks added directly (without initializing the logger to avoid async issues)
            # Add string representations to different callback types to test comprehensive categorization
            test_success_callback.append("datadog")
            test_failure_callback.append("datadog")
            test_callbacks.append("datadog")

            response = client.get(
                "/callbacks/list",
                headers={"Authorization": "Bearer sk-1234"}
            )
            assert response.status_code == 200
            response_data = response.json()

            # Verify datadog appears in the correct categorization
            # Since datadog is in both success and failure, it should appear in success_and_failure
            assert "datadog" in response_data["success_and_failure"]
            
            # The categorization logic should deduplicate properly
            # Filter out system callbacks for duplicate checks
            non_system_success = [cb for cb in response_data["success"] if not cb.startswith("_PROXY_")]
            non_system_failure = [cb for cb in response_data["failure"] if not cb.startswith("_PROXY_")]
            non_system_success_and_failure = [cb for cb in response_data["success_and_failure"] if not cb.startswith("_PROXY_")]
            
            assert len([cb for cb in non_system_success if cb == "datadog"]) <= 1
            assert len([cb for cb in non_system_failure if cb == "datadog"]) <= 1
            assert len([cb for cb in non_system_success_and_failure if cb == "datadog"]) <= 1
            
            # Verify the response structure is correct
            assert isinstance(response_data["success"], list)
            assert isinstance(response_data["failure"], list)
            assert isinstance(response_data["success_and_failure"], list)

    def test_list_callbacks_mixed_callback_types(self):
        """Test /callbacks/list endpoint with mixed callback types (string and logger instances)"""
        from unittest.mock import patch
        client = TestClient(app)

        # Create fresh callback lists for this test
        test_success_callback = []
        test_failure_callback = []
        test_callbacks = []
        test_async_success_callback = []
        test_async_failure_callback = []

        with patch("litellm.success_callback", test_success_callback), \
             patch("litellm.failure_callback", test_failure_callback), \
             patch("litellm.callbacks", test_callbacks), \
             patch("litellm._async_success_callback", test_async_success_callback), \
             patch("litellm._async_failure_callback", test_async_failure_callback), \
             patch("litellm.logging_callback_manager._get_all_callbacks") as mock_get_all:
            
            # Mock the callback manager to return our test lists
            def mock_get_all_callbacks():
                return test_callbacks + test_success_callback + test_failure_callback + test_async_success_callback + test_async_failure_callback
            
            mock_get_all.side_effect = mock_get_all_callbacks
            
            # Add callbacks directly to our test lists
            test_success_callback.append("langfuse")
            test_failure_callback.append("datadog")
            test_callbacks.append("prometheus")

            response = client.get(
                "/callbacks/list",
                headers={"Authorization": "Bearer sk-1234"}
            )
            assert response.status_code == 200
            response_data = response.json()
            
            # Verify callbacks are properly categorized
            non_system_success_and_failure = [cb for cb in response_data["success_and_failure"] if not cb.startswith("_PROXY_")]
            non_system_success = [cb for cb in response_data["success"] if not cb.startswith("_PROXY_")]
            non_system_failure = [cb for cb in response_data["failure"] if not cb.startswith("_PROXY_")]
            
            assert "prometheus" in non_system_success_and_failure  # callbacks list items go to success_and_failure
            assert "langfuse" in non_system_success
            assert "datadog" in non_system_failure
            
            # Verify no duplicates (excluding system callbacks)
            all_callbacks = (
                response_data["success"] + 
                response_data["failure"] + 
                response_data["success_and_failure"]
            )
            non_system_callbacks = [cb for cb in all_callbacks if not cb.startswith("_PROXY_")]
            assert len(set(non_system_callbacks)) == len(non_system_callbacks)


    def test_list_callbacks_empty_response_structure(self):
        """Test that response always has correct structure even with no callbacks"""
        from unittest.mock import patch
        client = TestClient(app)
        
        # Create fresh callback lists for this test
        test_success_callback = []
        test_failure_callback = []
        test_callbacks = []
        test_async_success_callback = []
        test_async_failure_callback = []

        with patch("litellm.success_callback", test_success_callback), \
             patch("litellm.failure_callback", test_failure_callback), \
             patch("litellm.callbacks", test_callbacks), \
             patch("litellm._async_success_callback", test_async_success_callback), \
             patch("litellm._async_failure_callback", test_async_failure_callback), \
             patch("litellm.logging_callback_manager._get_all_callbacks") as mock_get_all:
            
            # Mock the callback manager to return our test lists
            def mock_get_all_callbacks():
                return test_callbacks + test_success_callback + test_failure_callback + test_async_success_callback + test_async_failure_callback
            
            mock_get_all.side_effect = mock_get_all_callbacks
            
            response = client.get(
                "/callbacks/list",
                headers={"Authorization": "Bearer sk-1234"}
            )
            assert response.status_code == 200
            response_data = response.json()
            
            # Verify all required keys are present
            required_keys = ["success", "failure", "success_and_failure"]
            for key in required_keys:
                assert key in response_data
                assert isinstance(response_data[key], list)

