"""
Tests for KeyManagementEventHooks.

Validates that email and secret manager operations are independent and non-blocking.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.proxy.hooks.key_management_event_hooks import KeyManagementEventHooks


class TestKeyManagementEventHooksIndependentOperations:
    """Tests that email and secret manager operations are independent."""

    @pytest.mark.asyncio
    async def test_email_failure_does_not_block_secret_manager(self):
        """
        Test that if email sending fails, secret manager operation still runs.

        This validates the independent operation design where one failure
        does not block the other operation.
        """
        secret_manager_called = {"called": False}

        # Mock the email method to raise an exception
        async def mock_send_email_raises(*args, **kwargs):
            raise Exception("Email service unavailable")

        # Mock the secret manager method to track if it was called
        async def mock_store_secret(*args, **kwargs):
            secret_manager_called["called"] = True

        # Create mock objects for the hook parameters
        mock_data = MagicMock()
        mock_data.key_alias = "test-key-alias"

        mock_response = MagicMock()
        mock_response.model_dump.return_value = {"key": "sk-test", "token": "test-token"}
        mock_response.model_dump_json.return_value = '{"key": "sk-test"}'
        mock_response.token_id = "token-123"
        mock_response.key = "sk-test-key"

        mock_user_api_key_dict = MagicMock()
        mock_user_api_key_dict.user_id = "user-123"
        mock_user_api_key_dict.api_key = "api-key-123"

        with patch.object(
            KeyManagementEventHooks,
            "_send_key_created_email",
            side_effect=mock_send_email_raises,
        ), patch.object(
            KeyManagementEventHooks,
            "_store_virtual_key_in_secret_manager",
            side_effect=mock_store_secret,
        ), patch(
            "litellm.store_audit_logs", False
        ), patch(
            "litellm.proxy.hooks.key_management_event_hooks.verbose_proxy_logger"
        ):
            # Should not raise even though email fails
            await KeyManagementEventHooks.async_key_generated_hook(
                data=mock_data,
                response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

        # Secret manager should have been called despite email failure
        assert secret_manager_called["called"] is True

    @pytest.mark.asyncio
    async def test_secret_manager_failure_does_not_block_email(self):
        """
        Test that if secret manager fails, email operation still runs.

        This validates the independent operation design where one failure
        does not block the other operation.
        """
        email_called = {"called": False}

        # Mock the email method to track if it was called
        async def mock_send_email(*args, **kwargs):
            email_called["called"] = True

        # Mock the secret manager method to raise an exception
        async def mock_store_secret_raises(*args, **kwargs):
            raise Exception("Secret manager unavailable")

        # Create mock objects for the hook parameters
        mock_data = MagicMock()
        mock_data.key_alias = "test-key-alias"

        mock_response = MagicMock()
        mock_response.model_dump.return_value = {"key": "sk-test", "token": "test-token"}
        mock_response.model_dump_json.return_value = '{"key": "sk-test"}'
        mock_response.token_id = "token-123"
        mock_response.key = "sk-test-key"

        mock_user_api_key_dict = MagicMock()
        mock_user_api_key_dict.user_id = "user-123"
        mock_user_api_key_dict.api_key = "api-key-123"

        with patch.object(
            KeyManagementEventHooks,
            "_send_key_created_email",
            side_effect=mock_send_email,
        ), patch.object(
            KeyManagementEventHooks,
            "_store_virtual_key_in_secret_manager",
            side_effect=mock_store_secret_raises,
        ), patch(
            "litellm.store_audit_logs", False
        ), patch(
            "litellm.proxy.hooks.key_management_event_hooks.verbose_proxy_logger"
        ):
            # Should not raise even though secret manager fails
            await KeyManagementEventHooks.async_key_generated_hook(
                data=mock_data,
                response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

        # Email should have been called despite secret manager failure
        assert email_called["called"] is True

