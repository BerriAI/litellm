import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from litellm.proxy.hooks.user_management_event_hooks import UserManagementEventHooks
from litellm.proxy.hooks.key_management_event_hooks import KeyManagementEventHooks
from litellm.proxy._types import NewUserRequest, NewUserResponse, GenerateKeyRequest, GenerateKeyResponse, UserAPIKeyAuth
import builtins
import sys
from types import SimpleNamespace

@pytest.mark.asyncio
async def test_v1_user_creation_no_email_when_send_invite_email_false():
    """
    Test that user invitation email is NOT sent when send_invite_email=False
    """
    mock_slack_alerting = MagicMock()
    mock_slack_alerting.send_key_created_or_user_invited_email = AsyncMock()
    mock_proxy_logging_obj = MagicMock()
    mock_proxy_logging_obj.slack_alerting_instance = mock_slack_alerting

    with patch("litellm.logging_callback_manager.get_custom_loggers_for_type", return_value=[]):
        mock_proxy_server = SimpleNamespace(
            general_settings={"alerting": ["email"]},
            proxy_logging_obj=mock_proxy_logging_obj,
            litellm_proxy_admin_name="admin-user",
        )
        with patch.dict(sys.modules, {"litellm.proxy.proxy_server": mock_proxy_server}):
            data = NewUserRequest(
                user_email="test@example.com",
                send_invite_email=False,  # Should NOT send email
            )
            response = NewUserResponse(
                user_id="test-user",
                user_email="test@example.com",
                key="sk-test-key",
            )
            user_api_key_dict = UserAPIKeyAuth(
                user_id="admin-user", api_key="admin-key"
            )
            await UserManagementEventHooks.async_send_user_invitation_email(
                data=data,
                response=response,
                user_api_key_dict=user_api_key_dict,
            )
            mock_slack_alerting.send_key_created_or_user_invited_email.assert_not_called()

@pytest.mark.asyncio
async def test_v1_user_creation_sends_email_when_send_invite_email_true():
    """
    Test that user invitation email IS sent when send_invite_email=True
    """
    mock_slack_alerting = MagicMock()
    mock_slack_alerting.send_key_created_or_user_invited_email = AsyncMock()
    mock_proxy_logging_obj = MagicMock()
    mock_proxy_logging_obj.slack_alerting_instance = mock_slack_alerting

    with patch("litellm.logging_callback_manager.get_custom_loggers_for_type", return_value=[]):
        mock_proxy_server = SimpleNamespace(
            general_settings={"alerting": ["email"]},
            proxy_logging_obj=mock_proxy_logging_obj,
            litellm_proxy_admin_name="admin-user",
        )
        with patch.dict(sys.modules, {"litellm.proxy.proxy_server": mock_proxy_server}):
            data = NewUserRequest(
                user_email="test@example.com",
                send_invite_email=True,  # Should send email
            )
            response = NewUserResponse(
                user_id="test-user",
                user_email="test@example.com",
                key="sk-test-key",
            )
            user_api_key_dict = UserAPIKeyAuth(
                user_id="admin-user", api_key="admin-key"
            )
            await UserManagementEventHooks.async_send_user_invitation_email(
                data=data,
                response=response,
                user_api_key_dict=user_api_key_dict,
            )
            mock_slack_alerting.send_key_created_or_user_invited_email.assert_called_once()

@pytest.mark.asyncio
async def test_v1_key_generation_sends_email_when_send_invite_email_true():
    """
    Test that key generation email IS sent when send_invite_email=True
    """
    mock_send_key_created_email = AsyncMock()
    mock_slack_alerting = MagicMock()
    mock_slack_alerting.send_key_created_or_user_invited_email = AsyncMock()
    mock_proxy_logging_obj = MagicMock()
    mock_proxy_logging_obj.slack_alerting_instance = mock_slack_alerting

    with patch.object(KeyManagementEventHooks, "_send_key_created_email", mock_send_key_created_email):
        with patch("litellm.logging_callback_manager.get_custom_loggers_for_type", return_value=[]):
            mock_proxy_server = SimpleNamespace(
                general_settings={"alerting": ["email"]},
                proxy_logging_obj=mock_proxy_logging_obj,
                litellm_proxy_admin_name="admin-user",
            )
            with patch.dict(sys.modules, {"litellm.proxy.proxy_server": mock_proxy_server}):
                data = GenerateKeyRequest(
                    user_email="test@example.com",
                    send_invite_email=True,  # Should send key email
                )
                response = GenerateKeyResponse(
                    user_email="test@example.com",
                    key="sk-test-key",
                )
                user_api_key_dict = UserAPIKeyAuth(
                    user_id="admin-user", api_key="admin-key"
                )
                await KeyManagementEventHooks.async_key_generated_hook(
                    data=data,
                    response=response,
                    user_api_key_dict=user_api_key_dict,
                )
                mock_send_key_created_email.assert_called_once()

@pytest.mark.asyncio
async def test_v1_key_generation_no_email_when_send_invite_email_false():
    """
    Test that key generation email is NOT sent when send_invite_email=False
    """
    mock_send_key_created_email = AsyncMock()
    mock_slack_alerting = MagicMock()
    mock_slack_alerting.send_key_created_or_user_invited_email = AsyncMock()
    mock_proxy_logging_obj = MagicMock()
    mock_proxy_logging_obj.slack_alerting_instance = mock_slack_alerting

    with patch.object(KeyManagementEventHooks, "_send_key_created_email", mock_send_key_created_email):
        with patch("litellm.logging_callback_manager.get_custom_loggers_for_type", return_value=[]):
            mock_proxy_server = SimpleNamespace(
                general_settings={"alerting": ["email"]},
                proxy_logging_obj=mock_proxy_logging_obj,
                litellm_proxy_admin_name="admin-user",
            )
            with patch.dict(sys.modules, {"litellm.proxy.proxy_server": mock_proxy_server}):
                data = GenerateKeyRequest(
                    user_email="test@example.com",
                    send_invite_email=False,  # Should NOT send key email
                )
                response = GenerateKeyResponse(
                    user_email="test@example.com",
                    key="sk-test-key",
                )
                user_api_key_dict = UserAPIKeyAuth(
                    user_id="admin-user", api_key="admin-key"
                )
                await KeyManagementEventHooks.async_key_generated_hook(
                    data=data,
                    response=response,
                    user_api_key_dict=user_api_key_dict,
                )
                mock_send_key_created_email.assert_not_called()
