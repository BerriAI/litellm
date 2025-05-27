import json
import os
import sys
import unittest.mock as mock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("../../.."))

from litellm_enterprise.enterprise_callbacks.send_emails.base_email import (
    BaseEmailLogger,
)
from litellm_enterprise.types.enterprise_callbacks.send_emails import (
    EmailEvent,
    SendKeyCreatedEmailEvent,
)

from litellm.proxy._types import Litellm_EntityType, WebhookEvent


@pytest.fixture
def base_email_logger():
    return BaseEmailLogger()


@pytest.fixture
def mock_send_email():
    with mock.patch.object(BaseEmailLogger, "send_email") as mock_send:
        yield mock_send


@pytest.fixture
def mock_lookup_user_email():
    with mock.patch.object(
        BaseEmailLogger, "_lookup_user_email_from_db"
    ) as mock_lookup:
        yield mock_lookup


def test_format_key_budget(base_email_logger):
    # Test with budget
    assert base_email_logger._format_key_budget(100.0) == "$100.0"

    # Test with no budget
    assert base_email_logger._format_key_budget(None) == "No budget"


@pytest.mark.asyncio
async def test_send_key_created_email(
    base_email_logger, mock_send_email, mock_lookup_user_email
):
    # Setup test data
    event = SendKeyCreatedEmailEvent(
        user_id="test_user",
        user_email="test@example.com",
        virtual_key="test_key",
        max_budget=100.0,
        spend=0.0,
        event_group=Litellm_EntityType.USER,
        event="key_created",
        event_message="Test Key Created",
    )

    # Mock environment variables
    with mock.patch.dict(
        os.environ,
        {
            "EMAIL_LOGO_URL": "https://test-logo.com",
            "EMAIL_SUPPORT_CONTACT": "support@test.com",
            "PROXY_BASE_URL": "http://test.com",
        },
    ):
        # Execute
        await base_email_logger.send_key_created_email(event)

        # Verify
        mock_send_email.assert_called_once()
        call_args = mock_send_email.call_args[1]
        assert call_args["from_email"] == BaseEmailLogger.DEFAULT_LITELLM_EMAIL
        assert call_args["to_email"] == ["test@example.com"]
        assert call_args["subject"] == "LiteLLM: Test Key Created"
        assert "test_key" in call_args["html_body"]
        assert "$100.0" in call_args["html_body"]


@pytest.mark.asyncio
async def test_send_user_invitation_email(
    base_email_logger, mock_send_email, mock_lookup_user_email
):
    # Setup test data
    event = WebhookEvent(
        user_id="test_user",
        user_email="invited@example.com",
        event_group=Litellm_EntityType.USER,
        event="internal_user_created",
        event_message="User Invitation",
        spend=0.0,
    )

    # Mock environment variables
    with mock.patch.dict(
        os.environ,
        {
            "EMAIL_LOGO_URL": "https://test-logo.com",
            "EMAIL_SUPPORT_CONTACT": "support@test.com",
            "PROXY_BASE_URL": "http://test.com",
        },
    ):
        # Execute
        await base_email_logger.send_user_invitation_email(event)

        # Verify
        mock_send_email.assert_called_once()
        call_args = mock_send_email.call_args[1]
        assert call_args["from_email"] == BaseEmailLogger.DEFAULT_LITELLM_EMAIL
        assert call_args["to_email"] == ["invited@example.com"]
        assert call_args["subject"] == "LiteLLM: User Invitation"
        assert "invited@example.com" in call_args["html_body"]


@pytest.mark.asyncio
async def test_send_user_invitation_email_from_db(
    base_email_logger, mock_send_email, mock_lookup_user_email
):
    # Setup test data with no direct email but one in the database
    event = WebhookEvent(
        user_id="test_user",
        event_group=Litellm_EntityType.USER,
        event="internal_user_created",
        event_message="User Invitation",
        spend=0.0,
    )

    # Mock the lookup to return an email
    mock_lookup_user_email.return_value = "db_user@example.com"

    # Mock environment variables
    with mock.patch.dict(
        os.environ,
        {
            "EMAIL_LOGO_URL": "https://test-logo.com",
            "EMAIL_SUPPORT_CONTACT": "support@test.com",
            "PROXY_BASE_URL": "http://test.com",
        },
    ):
        # Execute
        await base_email_logger.send_user_invitation_email(event)

        # Verify
        mock_lookup_user_email.assert_called_once_with(user_id="test_user")
        mock_send_email.assert_called_once()
        call_args = mock_send_email.call_args[1]
        assert call_args["from_email"] == BaseEmailLogger.DEFAULT_LITELLM_EMAIL
        assert call_args["to_email"] == ["db_user@example.com"]
        assert call_args["subject"] == "LiteLLM: User Invitation"
        assert "db_user@example.com" in call_args["html_body"]


@pytest.mark.asyncio
async def test_send_user_invitation_email_no_email(
    base_email_logger, mock_lookup_user_email
):
    # Setup test data with no email
    event = WebhookEvent(
        user_id="test_user",
        event_group=Litellm_EntityType.USER,
        event="internal_user_created",
        event_message="User Invitation",
        spend=0.0,
    )

    # Mock lookup to return None
    mock_lookup_user_email.return_value = None

    # Test that it raises ValueError
    with pytest.raises(ValueError, match="User email not found"):
        await base_email_logger.send_user_invitation_email(event)


@pytest.mark.asyncio
async def test_send_key_created_email_no_email(
    base_email_logger, mock_lookup_user_email
):
    # Setup test data with no email
    event = SendKeyCreatedEmailEvent(
        user_id="test_user",
        user_email=None,
        virtual_key="test_key",
        max_budget=100.0,
        event_message="Test Key Created",
        event_group=Litellm_EntityType.USER,
        event="key_created",
        spend=0.0,
    )

    # Mock lookup to return None
    mock_lookup_user_email.return_value = None

    # Test that it raises ValueError
    with pytest.raises(ValueError, match="User email not found"):
        await base_email_logger.send_key_created_email(event)


@pytest.mark.asyncio
async def test_get_invitation_link(base_email_logger):
    # Mock prisma client and its response
    mock_invitation_row = mock.MagicMock()
    mock_invitation_row.id = "test-invitation-id"

    mock_prisma = mock.MagicMock()

    # Create an async mock for find_many
    async def mock_find_many(*args, **kwargs):
        return [mock_invitation_row]

    mock_prisma.db.litellm_invitationlink.find_many = mock_find_many

    with mock.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma):
        # Test with valid user_id
        result = await base_email_logger._get_invitation_link(
            user_id="test-user", base_url="http://test.com"
        )
        assert result == "http://test.com/ui?invitation_id=test-invitation-id"

        # Test with None user_id
        result = await base_email_logger._get_invitation_link(
            user_id=None, base_url="http://test.com"
        )
        assert result == "http://test.com"

        # Test with no invitation links
        async def mock_find_many_empty(*args, **kwargs):
            return []

        mock_prisma.db.litellm_invitationlink.find_many = mock_find_many_empty
        result = await base_email_logger._get_invitation_link(
            user_id="test-user", base_url="http://test.com"
        )
        assert result == "http://test.com"


def test_construct_invitation_link(base_email_logger):
    # Test invitation link construction
    result = base_email_logger._construct_invitation_link(
        invitation_id="test-id-123", base_url="http://test.com"
    )
    assert result == "http://test.com/ui?invitation_id=test-id-123"


@pytest.mark.asyncio
async def test_get_email_params_user_invitation(
    base_email_logger, mock_lookup_user_email
):
    # Mock environment variables
    with mock.patch.dict(
        os.environ,
        {
            "EMAIL_LOGO_URL": "https://test-logo.com",
            "EMAIL_SUPPORT_CONTACT": "support@test.com",
            "PROXY_BASE_URL": "http://test.com",
        },
    ):
        # Mock invitation link
        with mock.patch.object(
            base_email_logger,
            "_get_invitation_link",
            return_value="http://test.com/ui?invitation_id=test-id",
        ):
            # Test with user invitation event
            result = await base_email_logger._get_email_params(
                email_event=EmailEvent.new_user_invitation,
                user_id="test-user",
                user_email="test@example.com",
            )

            assert result.logo_url == "https://test-logo.com"
            assert result.support_contact == "support@test.com"
            assert result.base_url == "http://test.com/ui?invitation_id=test-id"
            assert result.recipient_email == "test@example.com"
