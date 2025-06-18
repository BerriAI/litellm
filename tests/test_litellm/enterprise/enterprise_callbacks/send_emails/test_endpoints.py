import json
import os
import sys
import unittest.mock as mock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("../../.."))

from litellm_enterprise.enterprise_callbacks.send_emails.endpoints import (
    _get_email_settings,
    _save_email_settings,
    get_email_event_settings,
    reset_event_settings,
    router,
    update_event_settings,
)
from litellm_enterprise.types.enterprise_callbacks.send_emails import (
    DefaultEmailSettings,
    EmailEvent,
    EmailEventSettings,
    EmailEventSettingsUpdateRequest,
)


# Mock user_api_key_auth dependency
@pytest.fixture
def mock_user_api_key_auth():
    return {"user_id": "test_user"}


# Mock prisma client
@pytest.fixture
def mock_prisma_client():
    mock_client = mock.MagicMock()

    # Setup mock for async methods to work properly
    mock_db = mock.MagicMock()
    mock_config = mock.MagicMock()

    # Make find_unique return a coroutine mock
    async def mock_find_unique(*args, **kwargs):
        return None

    mock_config.find_unique = mock_find_unique

    # Make upsert return a coroutine mock
    async def mock_upsert(*args, **kwargs):
        return None

    mock_config.upsert = mock_upsert

    mock_db.litellm_config = mock_config
    mock_client.db = mock_db

    return mock_client


# Test _get_email_settings helper function
@pytest.mark.asyncio
async def test_get_email_settings_empty_db(mock_prisma_client):
    """Test that default settings are returned when database has no email settings."""

    # Setup mock find_unique to return None
    async def mock_find_unique(*args, **kwargs):
        return None

    mock_prisma_client.db.litellm_config.find_unique = mock_find_unique

    # Call the function
    result = await _get_email_settings(mock_prisma_client)

    # Assert that default settings are returned
    assert result == DefaultEmailSettings.get_defaults()


@pytest.mark.asyncio
async def test_get_email_settings_with_existing_settings(mock_prisma_client):
    """Test that existing email settings are correctly retrieved from the database."""
    # Setup mock find_unique to return existing settings
    mock_settings = {
        "email_settings": {
            EmailEvent.virtual_key_created.value: True,
            EmailEvent.new_user_invitation.value: False,
        }
    }

    mock_entry = mock.MagicMock()
    mock_entry.param_value = json.dumps(mock_settings)

    async def mock_find_unique(*args, **kwargs):
        return mock_entry

    mock_prisma_client.db.litellm_config.find_unique = mock_find_unique

    # Call the function
    result = await _get_email_settings(mock_prisma_client)

    # Assert correct settings are returned
    assert result[EmailEvent.virtual_key_created.value] is True
    assert result[EmailEvent.new_user_invitation.value] is False


@pytest.mark.asyncio
async def test_save_email_settings_new_entry(mock_prisma_client):
    """Test that email settings are properly saved to database when no previous settings exist."""

    # Setup mock find_unique to return None
    async def mock_find_unique(*args, **kwargs):
        return None

    mock_prisma_client.db.litellm_config.find_unique = mock_find_unique

    # Setup mock upsert to return None
    async def mock_upsert(*args, **kwargs):
        return None

    mock_prisma_client.db.litellm_config.upsert = mock_upsert

    # Settings to save
    settings = {
        EmailEvent.virtual_key_created.value: True,
        EmailEvent.new_user_invitation.value: False,
    }

    # Call the function
    await _save_email_settings(mock_prisma_client, settings)

    # Success if no exception was raised


# Test the GET endpoint
@pytest.mark.asyncio
async def test_get_email_event_settings(mock_prisma_client, mock_user_api_key_auth):
    """Test that the GET endpoint returns the correct email event settings."""

    # Mock _get_email_settings to return test data
    async def mock_get_settings(*args, **kwargs):
        return {
            EmailEvent.virtual_key_created.value: True,
            EmailEvent.new_user_invitation.value: False,
        }

    # Setup mocks
    with mock.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client):
        with mock.patch(
            "litellm_enterprise.enterprise_callbacks.send_emails.endpoints._get_email_settings",
            side_effect=mock_get_settings,
        ):
            # Call the endpoint function directly
            response = await get_email_event_settings(
                user_api_key_dict=mock_user_api_key_auth
            )

            # Assert response contains correct settings
            assert isinstance(response.dict(), dict)
            assert "settings" in response.dict()
            settings = response.dict()["settings"]
            assert len(settings) == len(EmailEvent)

            # Find the setting for virtual_key_created and check its value
            virtual_key_setting = next(
                (s for s in settings if s["event"] == EmailEvent.virtual_key_created),
                None,
            )
            assert virtual_key_setting is not None
            assert virtual_key_setting["enabled"] is True


# Test the PATCH endpoint
@pytest.mark.asyncio
async def test_update_event_settings(mock_prisma_client, mock_user_api_key_auth):
    """Test that the PATCH endpoint correctly updates email event settings."""

    # Mock _get_email_settings to return default settings
    async def mock_get_settings(*args, **kwargs):
        return DefaultEmailSettings.get_defaults()

    # Mock _save_email_settings to do nothing
    async def mock_save_settings(*args, **kwargs):
        return None

    # Setup mocks
    with mock.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client):
        with mock.patch(
            "litellm_enterprise.enterprise_callbacks.send_emails.endpoints._get_email_settings",
            side_effect=mock_get_settings,
        ):
            with mock.patch(
                "litellm_enterprise.enterprise_callbacks.send_emails.endpoints._save_email_settings",
                side_effect=mock_save_settings,
            ):
                # Create request with updated settings
                request = EmailEventSettingsUpdateRequest(
                    settings=[
                        EmailEventSettings(
                            event=EmailEvent.virtual_key_created, enabled=True
                        ),
                        EmailEventSettings(
                            event=EmailEvent.new_user_invitation, enabled=False
                        ),
                    ]
                )

                # Call the endpoint function directly
                response = await update_event_settings(
                    request=request, user_api_key_dict=mock_user_api_key_auth
                )

                # Assert response is success
                assert (
                    response["message"] == "Email event settings updated successfully"
                )


# Test the reset endpoint
@pytest.mark.asyncio
async def test_reset_event_settings(mock_prisma_client, mock_user_api_key_auth):
    """Test that the reset endpoint correctly restores default email event settings."""

    # Mock _save_email_settings to do nothing
    async def mock_save_settings(*args, **kwargs):
        return None

    # Setup mocks
    with mock.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client):
        with mock.patch(
            "litellm_enterprise.enterprise_callbacks.send_emails.endpoints._save_email_settings",
            side_effect=mock_save_settings,
        ):
            # Call the endpoint function directly
            response = await reset_event_settings(
                user_api_key_dict=mock_user_api_key_auth
            )

            # Assert response is success
            assert response["message"] == "Email event settings reset to defaults"


# Test handling of prisma client None
@pytest.mark.asyncio
async def test_endpoint_with_no_prisma_client(mock_user_api_key_auth):
    """Test that all endpoints properly handle the case when the database is not connected."""
    # Setup mock to return None for prisma_client
    with mock.patch("litellm.proxy.proxy_server.prisma_client", None):
        # Test get endpoint
        with pytest.raises(HTTPException) as exc_info:
            await get_email_event_settings(user_api_key_dict=mock_user_api_key_auth)
        assert exc_info.value.status_code == 500
        assert "Database not connected" in exc_info.value.detail

        # Test update endpoint
        request = EmailEventSettingsUpdateRequest(settings=[])
        with pytest.raises(HTTPException) as exc_info:
            await update_event_settings(
                request=request, user_api_key_dict=mock_user_api_key_auth
            )
        assert exc_info.value.status_code == 500

        # Test reset endpoint
        with pytest.raises(HTTPException) as exc_info:
            await reset_event_settings(user_api_key_dict=mock_user_api_key_auth)
        assert exc_info.value.status_code == 500
