"""
Tests for auth time tracking.

Verifies that auth_time_ms is correctly calculated and stored in request metadata.
"""

import asyncio
import os
import sys
import time
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.proxy.auth.user_api_key_auth import _return_user_api_key_auth_obj, _user_api_key_auth_builder
from litellm.proxy._types import UserAPIKeyAuth


class TestAuthTimeTracking:
    """Test suite for auth time tracking."""

    def test_auth_time_is_calculated_and_stored(self):
        """
        Test that auth_time_ms is calculated and stored in UserAPIKeyAuth object
        when authentication completes successfully.
        """
        # Create mock user object with proper attribute values
        # Use spec to ensure only defined attributes exist
        mock_user_obj = Mock(spec=['tpm_limit', 'rpm_limit', 'user_email', 'spend', 'max_budget'])
        mock_user_obj.tpm_limit = 1000
        mock_user_obj.rpm_limit = 100
        mock_user_obj.user_email = "test@example.com"
        # Set actual None values for optional fields
        mock_user_obj.spend = None
        mock_user_obj.max_budget = None
        
        # Create valid_token_dict
        valid_token_dict = {
            "api_key": "sk-test-key",
            "user_id": "user-123",
            "team_id": "team-456",
        }
        
        start_time = datetime.now()
        auth_start_time = time.perf_counter()
        # Simulate some auth processing time
        time.sleep(0.005)  # 5ms delay
        
        # Patch helper functions to avoid Mock attribute issues
        with patch('litellm.proxy.auth.user_api_key_auth._get_user_role', return_value=None), \
             patch('litellm.proxy.auth.user_api_key_auth._is_user_proxy_admin', return_value=False):
            # Call _return_user_api_key_auth_obj using asyncio.run()
            result = asyncio.run(_return_user_api_key_auth_obj(
                user_obj=mock_user_obj,
            api_key="sk-test-key",
            parent_otel_span=None,
            valid_token_dict=valid_token_dict,
            route="/chat/completions",
            start_time=start_time,
            auth_start_time=auth_start_time,
            user_role=None,
            ))
        
        # Verify auth_time_ms is stored in the UserAPIKeyAuth object
        assert hasattr(result, "auth_time_ms"), "auth_time_ms should be an attribute of UserAPIKeyAuth"
        auth_time = result.auth_time_ms
        
        # Verify it's a numeric value
        assert isinstance(auth_time, (int, float)), f"auth_time_ms should be numeric, got {type(auth_time)}"
        
        # Verify it's positive (should be at least 5ms due to our delay)
        assert auth_time > 0, f"auth_time_ms should be positive, got {auth_time}"
        assert auth_time >= 4.0, f"auth_time_ms should be at least 4ms, got {auth_time}"

    def test_auth_time_is_zero_for_instant_auth(self):
        """
        Test that auth_time_ms is still calculated even when
        authentication is very fast (near-zero time).
        """
        # Create mock user object with proper attribute values
        mock_user_obj = Mock(spec=['tpm_limit', 'rpm_limit', 'user_email', 'spend', 'max_budget'])
        mock_user_obj.tpm_limit = 1000
        mock_user_obj.rpm_limit = 100
        mock_user_obj.user_email = "test@example.com"
        mock_user_obj.spend = None
        mock_user_obj.max_budget = None
        
        valid_token_dict = {
            "api_key": "sk-test-key",
            "user_id": "user-123",
        }
        
        start_time = datetime.now()
        auth_start_time = time.perf_counter()
        
        # Patch helper functions to avoid Mock attribute issues
        with patch('litellm.proxy.auth.user_api_key_auth._get_user_role', return_value=None), \
             patch('litellm.proxy.auth.user_api_key_auth._is_user_proxy_admin', return_value=False):
            # Call immediately (no delay)
            result = asyncio.run(_return_user_api_key_auth_obj(
            user_obj=mock_user_obj,
            api_key="sk-test-key",
            parent_otel_span=None,
            valid_token_dict=valid_token_dict,
            route="/chat/completions",
            start_time=start_time,
            auth_start_time=auth_start_time,
            user_role=None,
            ))
        
        # Verify timing is still tracked even for fast operations
        assert hasattr(result, "auth_time_ms"), "auth_time_ms should be an attribute"
        auth_time = result.auth_time_ms
        assert isinstance(auth_time, (int, float))
        # Should be >= 0 (can be very small but should be tracked)
        assert auth_time >= 0, f"auth_time_ms should be >= 0, got {auth_time}"

    def test_auth_time_stored_in_metadata(self):
        """
        Test that auth_time_ms from UserAPIKeyAuth is extracted and stored
        in request metadata by add_user_api_key_auth_to_request_metadata().
        """
        from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup
        
        # Create a UserAPIKeyAuth object with auth_time_ms
        user_api_key_auth = UserAPIKeyAuth(
            api_key="sk-test-key",
            user_id="user-123",
            auth_time_ms=12.5,  # Simulated auth time
        )
        
        # Create data dict with metadata
        data = {
            "metadata": {},
        }
        
        # Call add_user_api_key_auth_to_request_metadata
        result_data = LiteLLMProxyRequestSetup.add_user_api_key_auth_to_request_metadata(
            data=data,
            user_api_key_dict=user_api_key_auth,
            _metadata_variable_name="metadata",
        )
        
        # Verify auth_time_ms is in metadata
        assert "auth_time_ms" in result_data["metadata"], "auth_time_ms should be in metadata"
        assert result_data["metadata"]["auth_time_ms"] == 12.5, f"Expected 12.5, got {result_data['metadata']['auth_time_ms']}"
