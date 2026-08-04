import asyncio
import json
import os
import sys
import unittest.mock as mock
from collections.abc import Mapping
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from litellm.caching.caching import DualCache
from litellm_enterprise.enterprise_callbacks.send_emails.base_email import (
    BaseEmailLogger,
)

sys.path.insert(0, os.path.abspath("../../.."))
from litellm_enterprise.types.enterprise_callbacks.send_emails import (
    EmailEvent,
    SendKeyCreatedEmailEvent,
    SendKeyRotatedEmailEvent,
)

from litellm.integrations.email_templates.email_footer import EMAIL_FOOTER
from litellm.proxy._types import CallInfo, Litellm_EntityType, WebhookEvent
from litellm.constants import EMAIL_BUDGET_ALERT_TTL
from litellm.constants import MAX_BUDGET_ALERT_TYPE
from litellm.proxy.db.budget_alert_claim import (
    claim_budget_alert_slot,
    delete_budget_alert_claims,
    release_budget_alert_slot,
)


@pytest.fixture(autouse=True)
def no_invitation_wait(monkeypatch):
    async def _noop(self):
        return None

    monkeypatch.setattr(BaseEmailLogger, "_wait_for_invitation_creation", _noop)


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
            "EMAIL_LOGO_URL": "https://litellm-listing.s3.amazonaws.com/litellm_logo.png",
            "EMAIL_SUPPORT_CONTACT": "support@berri.ai",
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
            "EMAIL_LOGO_URL": "https://litellm-listing.s3.amazonaws.com/litellm_logo.png",
            "EMAIL_SUPPORT_CONTACT": "support@berri.ai",
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
            "EMAIL_LOGO_URL": "https://litellm-listing.s3.amazonaws.com/litellm_logo.png",
            "EMAIL_SUPPORT_CONTACT": "support@berri.ai",
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
async def test_send_key_rotated_email(
    base_email_logger, mock_send_email, mock_lookup_user_email
):
    """
    Test that send_key_rotated_email sends an email with the correct parameters and content
    """
    event = SendKeyRotatedEmailEvent(
        user_id="test_user",
        user_email="test@example.com",
        virtual_key="sk-rotated-key-123",
        key_alias="test-key-alias",
        max_budget=200.0,
        spend=50.0,
        event_group=Litellm_EntityType.KEY,
        event="key_rotated",
        event_message="API Key Rotated",
    )

    with mock.patch.dict(
        os.environ,
        {
            "EMAIL_LOGO_URL": "https://litellm-listing.s3.amazonaws.com/litellm_logo.png",
            "EMAIL_SUPPORT_CONTACT": "support@berri.ai",
            "PROXY_BASE_URL": "http://test.com",
        },
    ):
        await base_email_logger.send_key_rotated_email(event)

        mock_send_email.assert_called_once()
        call_args = mock_send_email.call_args[1]
        assert call_args["from_email"] == BaseEmailLogger.DEFAULT_LITELLM_EMAIL
        assert call_args["to_email"] == ["test@example.com"]
        assert call_args["subject"] == "LiteLLM: API Key Rotated"
        assert "sk-rotated-key-123" in call_args["html_body"]
        assert "$200.0" in call_args["html_body"]
        assert "rotated" in call_args["html_body"].lower()
        assert "Security Best Practices" in call_args["html_body"]


@pytest.mark.asyncio
async def test_send_key_created_email_without_key(
    base_email_logger, mock_send_email, mock_lookup_user_email
):
    """
    Test that send_key_created_email hides the API key when EMAIL_INCLUDE_API_KEY is false
    """
    event = SendKeyCreatedEmailEvent(
        user_id="test_user",
        user_email="test@example.com",
        virtual_key="sk-secret-key-456",
        max_budget=100.0,
        spend=0.0,
        event_group=Litellm_EntityType.USER,
        event="key_created",
        event_message="Test Key Created",
    )

    with mock.patch.dict(
        os.environ,
        {
            "EMAIL_INCLUDE_API_KEY": "false",
            "PROXY_BASE_URL": "http://test.com",
        },
    ):
        await base_email_logger.send_key_created_email(event)

        mock_send_email.assert_called_once()
        call_args = mock_send_email.call_args[1]
        assert "sk-secret-key-456" not in call_args["html_body"]
        assert (
            "[Key hidden for security - retrieve from dashboard]"
            in call_args["html_body"]
        )


@pytest.mark.asyncio
async def test_send_key_rotated_email_without_key(
    base_email_logger, mock_send_email, mock_lookup_user_email
):
    """
    Test that send_key_rotated_email hides the API key when EMAIL_INCLUDE_API_KEY is false
    """
    event = SendKeyRotatedEmailEvent(
        user_id="test_user",
        user_email="test@example.com",
        virtual_key="sk-secret-rotated-789",
        key_alias="test-key-alias",
        max_budget=200.0,
        spend=50.0,
        event_group=Litellm_EntityType.KEY,
        event="key_rotated",
        event_message="API Key Rotated",
    )

    with mock.patch.dict(
        os.environ,
        {
            "EMAIL_INCLUDE_API_KEY": "false",
            "PROXY_BASE_URL": "http://test.com",
        },
    ):
        await base_email_logger.send_key_rotated_email(event)

        mock_send_email.assert_called_once()
        call_args = mock_send_email.call_args[1]
        assert "sk-secret-rotated-789" not in call_args["html_body"]
        assert (
            "[Key hidden for security - retrieve from dashboard]"
            in call_args["html_body"]
        )


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
        assert (
            result == "http://test.com/ui/onboarding?invitation_id=test-invitation-id"
        )

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
    assert result == "http://test.com/ui/onboarding?invitation_id=test-id-123"


@pytest.mark.asyncio
async def test_get_invitation_link_creates_new_when_none_exist(base_email_logger):
    """Test that _get_invitation_link creates a new invitation when none exist"""
    # Mock prisma client with no existing invitation rows
    mock_prisma = mock.MagicMock()

    # Mock find_many to return empty list (no existing invitations)
    async def mock_find_many_empty(*args, **kwargs):
        return []

    mock_prisma.db.litellm_invitationlink.find_many = mock_find_many_empty

    # Mock the create_invitation_for_user function
    mock_created_invitation = mock.MagicMock()
    mock_created_invitation.id = "new-invitation-id"

    with mock.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma):
        with mock.patch(
            "litellm.proxy.management_helpers.user_invitation.create_invitation_for_user",
            return_value=mock_created_invitation,
        ) as mock_create_invitation:
            # Execute
            result = await base_email_logger._get_invitation_link(
                user_id="test-user", base_url="http://test.com"
            )

            # Verify that create_invitation_for_user was called
            mock_create_invitation.assert_called_once()
            call_args = mock_create_invitation.call_args[1]
            assert call_args["data"].user_id == "test-user"
            assert call_args["user_api_key_dict"].user_id == "test-user"

            # Verify the returned link uses the new invitation ID
            assert (
                result
                == "http://test.com/ui/onboarding?invitation_id=new-invitation-id"
            )


@pytest.mark.asyncio
async def test_get_invitation_link_uses_existing_when_available(base_email_logger):
    """Test that _get_invitation_link uses existing invitation when available"""
    # Mock prisma client with existing invitation row
    mock_invitation_row = mock.MagicMock()
    mock_invitation_row.id = "existing-invitation-id"

    mock_prisma = mock.MagicMock()

    # Mock find_many to return existing invitation
    async def mock_find_many_existing(*args, **kwargs):
        return [mock_invitation_row]

    mock_prisma.db.litellm_invitationlink.find_many = mock_find_many_existing

    with mock.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma):
        with mock.patch(
            "litellm.proxy.management_helpers.user_invitation.create_invitation_for_user"
        ) as mock_create_invitation:
            # Execute
            result = await base_email_logger._get_invitation_link(
                user_id="test-user", base_url="http://test.com"
            )

            # Verify that create_invitation_for_user was NOT called
            mock_create_invitation.assert_not_called()

            # Verify the returned link uses the existing invitation ID
            assert (
                result
                == "http://test.com/ui/onboarding?invitation_id=existing-invitation-id"
            )


@pytest.mark.asyncio
async def test_get_invitation_link_creates_new_when_list_is_none(base_email_logger):
    """Test that _get_invitation_link creates a new invitation when invitation_rows is None"""
    # Mock prisma client to return None
    mock_prisma = mock.MagicMock()

    # Mock find_many to return None
    async def mock_find_many_none(*args, **kwargs):
        return None

    mock_prisma.db.litellm_invitationlink.find_many = mock_find_many_none

    # Mock the create_invitation_for_user function
    mock_created_invitation = mock.MagicMock()
    mock_created_invitation.id = "new-invitation-from-none"

    with mock.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma):
        with mock.patch(
            "litellm.proxy.management_helpers.user_invitation.create_invitation_for_user",
            return_value=mock_created_invitation,
        ) as mock_create_invitation:
            # Execute
            result = await base_email_logger._get_invitation_link(
                user_id="test-user", base_url="http://test.com"
            )

            # Verify that create_invitation_for_user was called
            mock_create_invitation.assert_called_once()
            call_args = mock_create_invitation.call_args[1]
            assert call_args["data"].user_id == "test-user"
            assert call_args["user_api_key_dict"].user_id == "test-user"

            # Verify the returned link uses the new invitation ID
            assert (
                result
                == "http://test.com/ui/onboarding?invitation_id=new-invitation-from-none"
            )


@pytest.mark.asyncio
async def test_get_email_params_user_invitation(
    base_email_logger, mock_lookup_user_email
):
    # Mock environment variables
    with mock.patch.dict(
        os.environ,
        {
            "EMAIL_LOGO_URL": "https://litellm-listing.s3.amazonaws.com/litellm_logo.png",
            "EMAIL_SUPPORT_CONTACT": "support@berri.ai",
            "PROXY_BASE_URL": "http://test.com",
        },
    ):
        # Mock invitation link
        with mock.patch.object(
            base_email_logger,
            "_get_invitation_link",
            return_value="http://test.com/ui/onboarding?invitation_id=test-id",
        ):
            # Test with user invitation event
            result = await base_email_logger._get_email_params(
                email_event=EmailEvent.new_user_invitation,
                user_id="test-user",
                user_email="test@example.com",
            )

            assert (
                result.logo_url
                == "https://litellm-listing.s3.amazonaws.com/litellm_logo.png"
            )
            assert result.support_contact == "support@berri.ai"
            assert (
                result.base_url == "http://test.com/ui/onboarding?invitation_id=test-id"
            )
            assert result.recipient_email == "test@example.com"


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up test environment variables"""
    monkeypatch.setenv("EMAIL_LOGO_URL", "https://test-company.com/logo.png")
    monkeypatch.setenv("EMAIL_SUPPORT_CONTACT", "support@test-company.com")
    monkeypatch.setenv("EMAIL_SIGNATURE", "Best regards,\nTest Company Team")
    monkeypatch.setenv("EMAIL_SUBJECT_INVITATION", "Welcome to Test Company!")
    monkeypatch.setenv("EMAIL_SUBJECT_KEY_CREATED", "Your Test Company API Key")
    monkeypatch.setenv("PROXY_BASE_URL", "http://test.com")
    monkeypatch.setenv("PROXY_API_URL", "https://test.com")


@pytest.mark.asyncio
async def test_get_email_params_custom_templates_premium_user(mock_env_vars):
    """Test that _get_email_params returns correct values with custom templates for premium users"""
    # Mock premium_user as True
    with patch("litellm.proxy.proxy_server.premium_user", True):
        email_logger = BaseEmailLogger()

        # Test invitation email params
        invitation_params = await email_logger._get_email_params(
            email_event=EmailEvent.new_user_invitation,
            user_id="testid",
            user_email="test@example.com",
            event_message="New User Invitation",
        )

        assert invitation_params.subject == "Welcome to Test Company!"
        assert invitation_params.signature == "Best regards,\nTest Company Team"
        assert invitation_params.logo_url == "https://test-company.com/logo.png"
        assert invitation_params.support_contact == "support@test-company.com"
        assert invitation_params.base_url == "http://test.com"

        # Test key created email params
        key_params = await email_logger._get_email_params(
            email_event=EmailEvent.virtual_key_created,
            user_id="testid",
            user_email="test@example.com",
            event_message="API Key Created",
        )

        assert key_params.subject == "Your Test Company API Key"
        assert key_params.signature == "Best regards,\nTest Company Team"


@pytest.mark.asyncio
async def test_get_email_params_non_premium_user(mock_env_vars):
    """Test that non-premium users get default templates even when custom ones are provided"""
    # Mock premium_user as False
    with patch("litellm.proxy.proxy_server.premium_user", False):
        email_logger = BaseEmailLogger()

        # Test invitation email params
        email_params = await email_logger._get_email_params(
            email_event=EmailEvent.new_user_invitation,
            user_email="test@example.com",
            event_message="New User Invitation",
        )

        # Should use default values even though custom values are set in env
        assert email_params.subject == "LiteLLM: New User Invitation"
        assert email_params.signature == EMAIL_FOOTER
        assert (
            email_params.logo_url
            == "https://litellm-listing.s3.amazonaws.com/litellm_logo.png"
        )
        assert email_params.support_contact == "support@berri.ai"

        # Test key created email params
        key_params = await email_logger._get_email_params(
            email_event=EmailEvent.virtual_key_created,
            user_email="test@example.com",
            event_message="API Key Created",
        )

        assert key_params.subject == "LiteLLM: API Key Created"
        assert key_params.signature == EMAIL_FOOTER


@pytest.mark.asyncio
async def test_get_email_params_default_templates(monkeypatch):
    """Test that _get_email_params uses default templates when custom ones aren't provided"""
    # Clear any existing environment variables
    monkeypatch.delenv("EMAIL_SUBJECT_INVITATION", raising=False)
    monkeypatch.delenv("EMAIL_SUBJECT_KEY_CREATED", raising=False)
    monkeypatch.delenv("EMAIL_SIGNATURE", raising=False)

    # Mock premium_user as True (shouldn't matter since no custom values are set)
    with patch("litellm.proxy.proxy_server.premium_user", True):
        email_logger = BaseEmailLogger()

        # Test invitation email params with default template
        invitation_params = await email_logger._get_email_params(
            email_event=EmailEvent.new_user_invitation,
            user_email="test@example.com",
            event_message="New User Invitation",
        )

        assert invitation_params.subject == "LiteLLM: New User Invitation"
        assert invitation_params.signature == EMAIL_FOOTER

        # Test key created email params with default template
        key_params = await email_logger._get_email_params(
            email_event=EmailEvent.virtual_key_created,
            user_email="test@example.com",
            event_message="API Key Created",
        )

        assert key_params.subject == "LiteLLM: API Key Created"
        assert key_params.signature == EMAIL_FOOTER


@pytest.mark.asyncio
async def test_send_soft_budget_alert_email(
    base_email_logger, mock_send_email, mock_lookup_user_email
):
    """Test that send_soft_budget_alert_email sends an email with the correct parameters and content"""
    event = WebhookEvent(
        user_id="test_user",
        user_email="test@example.com",
        event_group=Litellm_EntityType.USER,
        event="soft_budget_crossed",
        event_message="Soft Budget Crossed - Total Soft Budget: $100.0",
        spend=105.0,
        max_budget=200.0,
        soft_budget=100.0,
    )

    with mock.patch.dict(
        os.environ,
        {
            "EMAIL_LOGO_URL": "https://litellm-listing.s3.amazonaws.com/litellm_logo.png",
            "EMAIL_SUPPORT_CONTACT": "support@berri.ai",
            "PROXY_BASE_URL": "http://test.com",
        },
    ):
        await base_email_logger.send_soft_budget_alert_email(event)

        mock_send_email.assert_called_once()
        call_args = mock_send_email.call_args[1]
        assert call_args["from_email"] == BaseEmailLogger.DEFAULT_LITELLM_EMAIL
        assert call_args["to_email"] == ["test@example.com"]
        assert (
            call_args["subject"]
            == "LiteLLM: Soft Budget Crossed - Total Soft Budget: $100.0"
        )
        assert "$100.0" in call_args["html_body"]  # soft_budget
        assert "$105.0" in call_args["html_body"]  # spend
        assert "$200.0" in call_args["html_body"]  # max_budget


@pytest.mark.asyncio
async def test_send_soft_budget_alert_email_no_max_budget(
    base_email_logger, mock_send_email, mock_lookup_user_email
):
    """Test that send_soft_budget_alert_email handles missing max_budget correctly"""
    event = WebhookEvent(
        user_id="test_user",
        user_email="test@example.com",
        event_group=Litellm_EntityType.USER,
        event="soft_budget_crossed",
        event_message="Soft Budget Crossed - Total Soft Budget: $100.0",
        spend=105.0,
        max_budget=None,
        soft_budget=100.0,
    )

    with mock.patch.dict(
        os.environ,
        {
            "PROXY_BASE_URL": "http://test.com",
        },
    ):
        await base_email_logger.send_soft_budget_alert_email(event)

        mock_send_email.assert_called_once()
        call_args = mock_send_email.call_args[1]
        assert "$100.0" in call_args["html_body"]  # soft_budget
        assert "$105.0" in call_args["html_body"]  # spend
        assert (
            "Maximum Budget" not in call_args["html_body"]
        )  # max_budget should not be shown


@pytest.mark.asyncio
async def test_budget_alerts_soft_budget_crossed(base_email_logger, mock_send_email):
    """Test that budget_alerts sends email when soft budget is crossed"""
    user_info = CallInfo(
        user_id="test_user",
        user_email="test@example.com",
        spend=105.0,
        max_budget=200.0,
        soft_budget=100.0,
        event_group=Litellm_EntityType.USER,
    )

    # Mock the cache so the claim is won (increment returns 1)
    mock_cache = mock.AsyncMock()
    mock_cache.async_increment_cache = mock.AsyncMock(return_value=1)
    base_email_logger.internal_usage_cache = mock_cache

    with mock.patch.dict(
        os.environ,
        {
            "PROXY_BASE_URL": "http://test.com",
        },
    ):
        await base_email_logger.budget_alerts(type="soft_budget", user_info=user_info)

        # Verify email was sent
        mock_send_email.assert_called_once()
        call_args = mock_send_email.call_args[1]
        assert call_args["to_email"] == ["test@example.com"]

        # Verify the send slot was claimed to prevent duplicate alerts
        mock_cache.async_increment_cache.assert_called_once()
        cache_call_args = mock_cache.async_increment_cache.call_args[1]
        assert (
            cache_call_args["key"]
            == "email_budget_alerts:soft_budget_crossed:test_user"
        )
        assert cache_call_args["value"] == 1
        assert cache_call_args["ttl"] == EMAIL_BUDGET_ALERT_TTL


@pytest.mark.asyncio
async def test_budget_alerts_soft_budget_not_crossed(
    base_email_logger, mock_send_email
):
    """Test that budget_alerts does not send email when soft budget is not crossed"""
    user_info = CallInfo(
        user_id="test_user",
        user_email="test@example.com",
        spend=50.0,
        max_budget=200.0,
        soft_budget=100.0,
        event_group=Litellm_EntityType.USER,
    )

    mock_cache = mock.AsyncMock()
    base_email_logger.internal_usage_cache = mock_cache

    await base_email_logger.budget_alerts(type="soft_budget", user_info=user_info)

    # Verify email was NOT sent
    mock_send_email.assert_not_called()


@pytest.mark.asyncio
async def test_budget_alerts_soft_budget_duplicate_prevention(
    base_email_logger, mock_send_email
):
    """Test that budget_alerts does not send duplicate alerts within TTL period"""
    user_info = CallInfo(
        user_id="test_user",
        user_email="test@example.com",
        spend=105.0,
        max_budget=200.0,
        soft_budget=100.0,
        event_group=Litellm_EntityType.USER,
    )

    # Mock the cache so the slot is already claimed (increment returns > 1)
    mock_cache = mock.AsyncMock()
    mock_cache.async_increment_cache = mock.AsyncMock(return_value=2)
    base_email_logger.internal_usage_cache = mock_cache

    await base_email_logger.budget_alerts(type="soft_budget", user_info=user_info)

    # Verify email was NOT sent (duplicate prevention)
    mock_send_email.assert_not_called()


@pytest.mark.asyncio
async def test_budget_alerts_no_budgets(base_email_logger, mock_send_email):
    """Test that budget_alerts returns early when no budgets are set"""
    user_info = CallInfo(
        user_id="test_user",
        user_email="test@example.com",
        spend=50.0,
        max_budget=None,
        soft_budget=None,
        event_group=Litellm_EntityType.USER,
    )

    await base_email_logger.budget_alerts(type="soft_budget", user_info=user_info)

    # Verify email was NOT sent
    mock_send_email.assert_not_called()


@pytest.mark.asyncio
async def test_budget_alerts_uses_token_for_cache_key(
    base_email_logger, mock_send_email
):
    """Test that budget_alerts uses token for cache key when available"""
    user_info = CallInfo(
        user_id="test_user",
        user_email="test@example.com",
        token="hashed_token_123",
        spend=105.0,
        max_budget=200.0,
        soft_budget=100.0,
        event_group=Litellm_EntityType.KEY,
    )

    # Mock the cache so the claim is won (increment returns 1)
    mock_cache = mock.AsyncMock()
    mock_cache.async_increment_cache = mock.AsyncMock(return_value=1)
    base_email_logger.internal_usage_cache = mock_cache

    with mock.patch.dict(
        os.environ,
        {
            "PROXY_BASE_URL": "http://test.com",
        },
    ):
        await base_email_logger.budget_alerts(type="soft_budget", user_info=user_info)

        # Verify cache key uses token instead of user_id
        mock_cache.async_increment_cache.assert_called_once()
        cache_call_args = mock_cache.async_increment_cache.call_args[1]
        assert (
            cache_call_args["key"]
            == "email_budget_alerts:soft_budget_crossed:hashed_token_123"
        )


@pytest.mark.asyncio
async def test_get_email_params_soft_budget_crossed(
    base_email_logger, mock_lookup_user_email
):
    """Test that _get_email_params handles soft_budget_crossed event correctly"""
    with mock.patch.dict(
        os.environ,
        {
            "PROXY_BASE_URL": "http://test.com",
        },
    ):
        result = await base_email_logger._get_email_params(
            email_event=EmailEvent.soft_budget_crossed,
            user_email="test@example.com",
            event_message="Soft Budget Crossed - Total Soft Budget: $100.0",
        )

        # Should use default subject template for soft_budget_crossed
        assert (
            result.subject == "LiteLLM: Soft Budget Crossed - Total Soft Budget: $100.0"
        )
        assert result.recipient_email == "test@example.com"
        assert result.base_url == "http://test.com"


@pytest.mark.asyncio
async def test_budget_alerts_max_budget_alert_crossed(
    base_email_logger, mock_send_email
):
    """Test that budget_alerts sends email when max budget alert threshold is crossed"""
    user_info = CallInfo(
        user_id="test_user",
        user_email="test@example.com",
        spend=165.0,
        max_budget=200.0,
        event_group=Litellm_EntityType.USER,
    )

    mock_cache = mock.AsyncMock()
    mock_cache.async_increment_cache = mock.AsyncMock(return_value=1)
    base_email_logger.internal_usage_cache = mock_cache

    with mock.patch.dict(
        os.environ,
        {
            "PROXY_BASE_URL": "http://test.com",
        },
    ):
        await base_email_logger.budget_alerts(
            type="max_budget_alert", user_info=user_info
        )

        mock_send_email.assert_called_once()
        call_args = mock_send_email.call_args[1]
        assert call_args["to_email"] == ["test@example.com"]
        assert "Max Budget Alert" in call_args["subject"]

        mock_cache.async_increment_cache.assert_called_once()
        cache_call_args = mock_cache.async_increment_cache.call_args[1]
        assert (
            cache_call_args["key"]
            == "email_budget_alerts:max_budget_alert:test_user:|200.0"
        )
        assert cache_call_args["value"] == 1
        assert cache_call_args["ttl"] == EMAIL_BUDGET_ALERT_TTL


@pytest.mark.asyncio
async def test_multi_threshold_sends_crossed_thresholds(
    base_email_logger, mock_send_email
):
    """Test that multi-threshold path sends emails for all crossed thresholds"""
    user_info = CallInfo(
        token="hashed_key_1",
        user_id="test_user",
        user_email="owner@co.com",
        spend=80.0,
        max_budget=100.0,
        event_group=Litellm_EntityType.KEY,
        max_budget_alert_emails={
            "50": ["finance@co.com"],
            "75": ["finance@co.com", "bu_lead@co.com"],
            "100": ["cto@co.com"],
        },
    )

    mock_cache = mock.AsyncMock()
    mock_cache.async_increment_cache = mock.AsyncMock(return_value=1)
    base_email_logger.internal_usage_cache = mock_cache

    with mock.patch.dict(os.environ, {"PROXY_BASE_URL": "http://test.com"}):
        await base_email_logger.budget_alerts(
            type="max_budget_alert", user_info=user_info
        )

        # spend=80 crosses 50% ($50) and 75% ($75), but not 100% ($100)
        assert mock_send_email.call_count == 2

        # Check cache keys include threshold percentage
        cache_keys = [
            c[1]["key"] for c in mock_cache.async_increment_cache.call_args_list
        ]
        assert "email_budget_alerts:max_budget_alert:50:hashed_key_1:|100.0" in cache_keys
        assert "email_budget_alerts:max_budget_alert:75:hashed_key_1:|100.0" in cache_keys


@pytest.mark.asyncio
async def test_multi_threshold_dedup_cache_prevents_resend(
    base_email_logger, mock_send_email
):
    """Test that cached thresholds are not re-sent"""
    user_info = CallInfo(
        token="hashed_key_1",
        user_id="test_user",
        user_email="owner@co.com",
        spend=80.0,
        max_budget=100.0,
        event_group=Litellm_EntityType.KEY,
        max_budget_alert_emails={
            "50": ["finance@co.com"],
            "75": ["finance@co.com"],
        },
    )

    # Simulate 50% already claimed (increment returns >1), 75% first send (returns 1)
    async def cache_increment(key, value, ttl=None):
        if "50:" in key:
            return 2
        return 1

    mock_cache = mock.AsyncMock()
    mock_cache.async_increment_cache = mock.AsyncMock(side_effect=cache_increment)
    base_email_logger.internal_usage_cache = mock_cache

    with mock.patch.dict(os.environ, {"PROXY_BASE_URL": "http://test.com"}):
        await base_email_logger.budget_alerts(
            type="max_budget_alert", user_info=user_info
        )

        # Only 75% should fire
        assert mock_send_email.call_count == 1
        cache_key = mock_cache.async_increment_cache.call_args[1]["key"]
        assert "75:" in cache_key


@pytest.mark.asyncio
async def test_multi_threshold_owner_email_auto_included(
    base_email_logger, mock_send_email
):
    """Test that the owner email is auto-appended and deduplicated"""
    user_info = CallInfo(
        token="hashed_key_1",
        user_id="test_user",
        user_email="owner@co.com",
        spend=60.0,
        max_budget=100.0,
        event_group=Litellm_EntityType.KEY,
        max_budget_alert_emails={
            "50": ["finance@co.com", "owner@co.com"],  # owner already in list
        },
    )

    mock_cache = mock.AsyncMock()
    mock_cache.async_increment_cache = mock.AsyncMock(return_value=1)
    base_email_logger.internal_usage_cache = mock_cache

    with mock.patch.dict(os.environ, {"PROXY_BASE_URL": "http://test.com"}):
        await base_email_logger.budget_alerts(
            type="max_budget_alert", user_info=user_info
        )

        mock_send_email.assert_called_once()
        to_emails = mock_send_email.call_args[1]["to_email"]
        # owner@co.com should appear exactly once (deduplicated)
        assert sorted(to_emails) == ["finance@co.com", "owner@co.com"]


@pytest.mark.asyncio
async def test_multi_threshold_malformed_keys_skipped(
    base_email_logger, mock_send_email
):
    """Test that non-numeric threshold keys are skipped"""
    user_info = CallInfo(
        token="hashed_key_1",
        user_id="test_user",
        user_email="owner@co.com",
        spend=60.0,
        max_budget=100.0,
        event_group=Litellm_EntityType.KEY,
        max_budget_alert_emails={
            "fifty": ["finance@co.com"],  # invalid
            "50": ["finance@co.com"],  # valid, crossed
        },
    )

    mock_cache = mock.AsyncMock()
    mock_cache.async_increment_cache = mock.AsyncMock(return_value=1)
    base_email_logger.internal_usage_cache = mock_cache

    with mock.patch.dict(os.environ, {"PROXY_BASE_URL": "http://test.com"}):
        await base_email_logger.budget_alerts(
            type="max_budget_alert", user_info=user_info
        )

        # Only the valid "50" threshold should fire
        assert mock_send_email.call_count == 1


@pytest.mark.asyncio
async def test_multi_threshold_empty_emails_only_owner(
    base_email_logger, mock_send_email
):
    """Test that empty email list for a threshold sends only to owner"""
    user_info = CallInfo(
        token="hashed_key_1",
        user_id="test_user",
        user_email="owner@co.com",
        spend=60.0,
        max_budget=100.0,
        event_group=Litellm_EntityType.KEY,
        max_budget_alert_emails={
            "50": [],  # empty list
        },
    )

    mock_cache = mock.AsyncMock()
    mock_cache.async_increment_cache = mock.AsyncMock(return_value=1)
    base_email_logger.internal_usage_cache = mock_cache

    with mock.patch.dict(os.environ, {"PROXY_BASE_URL": "http://test.com"}):
        await base_email_logger.budget_alerts(
            type="max_budget_alert", user_info=user_info
        )

        mock_send_email.assert_called_once()
        to_emails = mock_send_email.call_args[1]["to_email"]
        assert to_emails == ["owner@co.com"]


@pytest.mark.asyncio
async def test_no_map_preserves_old_single_threshold(
    base_email_logger, mock_send_email
):
    """Test that without max_budget_alert_emails, the old 80% single-threshold path works"""
    user_info = CallInfo(
        user_id="test_user",
        user_email="test@example.com",
        spend=165.0,
        max_budget=200.0,
        event_group=Litellm_EntityType.USER,
    )

    mock_cache = mock.AsyncMock()
    mock_cache.async_increment_cache = mock.AsyncMock(return_value=1)
    base_email_logger.internal_usage_cache = mock_cache

    with mock.patch.dict(os.environ, {"PROXY_BASE_URL": "http://test.com"}):
        await base_email_logger.budget_alerts(
            type="max_budget_alert", user_info=user_info
        )

        mock_send_email.assert_called_once()
        call_args = mock_send_email.call_args[1]
        assert call_args["to_email"] == ["test@example.com"]
        # Old path cache key has no threshold percentage
        cache_key = mock_cache.async_increment_cache.call_args[1]["key"]
        assert cache_key == "email_budget_alerts:max_budget_alert:test_user:|200.0"


CUSTOM_SIGNATURE = "<div>Best,<br/>The Acme Platform Team</div>"


@pytest.mark.asyncio
async def test_send_soft_budget_alert_email_uses_custom_signature(
    base_email_logger, mock_send_email, mock_lookup_user_email
):
    """Soft budget alert honors EMAIL_SIGNATURE for premium users."""
    event = WebhookEvent(
        user_id="test_user",
        user_email="test@example.com",
        event_group=Litellm_EntityType.USER,
        event="soft_budget_crossed",
        event_message="Soft Budget Crossed",
        spend=105.0,
        max_budget=200.0,
        soft_budget=100.0,
    )
    with mock.patch.dict(
        os.environ,
        {"PROXY_BASE_URL": "http://test.com", "EMAIL_SIGNATURE": CUSTOM_SIGNATURE},
    ), patch("litellm.proxy.proxy_server.premium_user", True):
        await base_email_logger.send_soft_budget_alert_email(event)

        html_body = mock_send_email.call_args[1]["html_body"]
        assert CUSTOM_SIGNATURE in html_body
        assert "The LiteLLM team" not in html_body


@pytest.mark.asyncio
async def test_send_team_soft_budget_alert_email_uses_custom_signature(
    base_email_logger, mock_send_email, mock_lookup_user_email
):
    """Team soft budget alert honors EMAIL_SIGNATURE for premium users."""
    event = WebhookEvent(
        user_id="test_user",
        event_group=Litellm_EntityType.TEAM,
        event="soft_budget_crossed",
        event_message="Team Soft Budget Crossed",
        spend=105.0,
        max_budget=200.0,
        soft_budget=100.0,
        team_alias="Acme",
        alert_emails=["teamlead@example.com"],
    )
    with mock.patch.dict(
        os.environ,
        {"PROXY_BASE_URL": "http://test.com", "EMAIL_SIGNATURE": CUSTOM_SIGNATURE},
    ), patch("litellm.proxy.proxy_server.premium_user", True):
        await base_email_logger.send_team_soft_budget_alert_email(event)

        html_body = mock_send_email.call_args[1]["html_body"]
        assert CUSTOM_SIGNATURE in html_body
        assert "The LiteLLM team" not in html_body


@pytest.mark.asyncio
async def test_send_max_budget_alert_email_single_recipient_uses_custom_signature(
    base_email_logger, mock_send_email, mock_lookup_user_email
):
    """Max budget alert (single-recipient path) honors EMAIL_SIGNATURE."""
    event = WebhookEvent(
        user_id="test_user",
        user_email="test@example.com",
        event_group=Litellm_EntityType.USER,
        event="max_budget_alert",
        event_message="Max Budget Alert",
        spend=165.0,
        max_budget=200.0,
    )
    with mock.patch.dict(
        os.environ,
        {"PROXY_BASE_URL": "http://test.com", "EMAIL_SIGNATURE": CUSTOM_SIGNATURE},
    ), patch("litellm.proxy.proxy_server.premium_user", True):
        await base_email_logger.send_max_budget_alert_email(event)

        html_body = mock_send_email.call_args[1]["html_body"]
        assert CUSTOM_SIGNATURE in html_body
        assert "The LiteLLM team" not in html_body


@pytest.mark.asyncio
async def test_send_max_budget_alert_email_multi_recipient_uses_custom_signature(
    base_email_logger, mock_send_email, mock_lookup_user_email
):
    """Max budget alert (multi-threshold/recipient path) honors EMAIL_SIGNATURE."""
    event = WebhookEvent(
        user_id="test_user",
        user_email="owner@example.com",
        event_group=Litellm_EntityType.USER,
        event="max_budget_alert",
        event_message="Max Budget Alert",
        spend=165.0,
        max_budget=200.0,
    )
    with mock.patch.dict(
        os.environ,
        {"PROXY_BASE_URL": "http://test.com", "EMAIL_SIGNATURE": CUSTOM_SIGNATURE},
    ), patch("litellm.proxy.proxy_server.premium_user", True):
        await base_email_logger.send_max_budget_alert_email(
            event, threshold_pct=75, recipient_emails=["a@example.com", "b@example.com"]
        )

        html_body = mock_send_email.call_args[1]["html_body"]
        assert CUSTOM_SIGNATURE in html_body
        assert "The LiteLLM team" not in html_body


@pytest.mark.asyncio
async def test_send_soft_budget_alert_email_default_footer_when_no_signature(
    base_email_logger, mock_send_email, mock_lookup_user_email
):
    """Without EMAIL_SIGNATURE, budget alert falls back to the default EMAIL_FOOTER."""
    event = WebhookEvent(
        user_id="test_user",
        user_email="test@example.com",
        event_group=Litellm_EntityType.USER,
        event="soft_budget_crossed",
        event_message="Soft Budget Crossed",
        spend=105.0,
        max_budget=200.0,
        soft_budget=100.0,
    )
    with mock.patch.dict(os.environ, {"PROXY_BASE_URL": "http://test.com"}):
        await base_email_logger.send_soft_budget_alert_email(event)

        html_body = mock_send_email.call_args[1]["html_body"]
        assert EMAIL_FOOTER in html_body


_BUDGET_ALERT_BRANCHES = [
    (
        "multi_threshold",
        "max_budget_alert",
        "send_max_budget_alert_email",
        dict(max_budget=100.0, spend=80.0, max_budget_alert_emails={"50": ["finance@co.com"]}),
    ),
    (
        "single_threshold",
        "max_budget_alert",
        "send_max_budget_alert_email",
        dict(max_budget=100.0, spend=85.0),
    ),
    (
        "soft_budget",
        "soft_budget",
        "send_soft_budget_alert_email",
        dict(soft_budget=50.0, spend=60.0),
    ),
]


def _budget_alert_user_info(extra: dict) -> CallInfo:
    return CallInfo(
        token="hashed_key_1",
        user_id="test_user",
        user_email="owner@co.com",
        event_group=Litellm_EntityType.KEY,
        **extra,
    )


@pytest.mark.parametrize(
    "branch, alert_type, send_method, ci_kwargs",
    _BUDGET_ALERT_BRANCHES,
    ids=[b[0] for b in _BUDGET_ALERT_BRANCHES],
)
@pytest.mark.asyncio
async def test_budget_alert_no_duplicate_on_concurrent_crossing(
    base_email_logger, branch, alert_type, send_method, ci_kwargs
):
    """Regression for LIT-4172: two requests crossing the same threshold at the
    same time must send exactly one email. The old code wrote the dedup marker
    only after the send finished awaiting, so both concurrent tasks passed the
    'already sent' check and both sent. Covers all three send branches."""
    base_email_logger.internal_usage_cache = DualCache()

    sends = []

    async def slow_send(*args, **kwargs):
        sends.append(1)
        await asyncio.sleep(0.05)

    with mock.patch.object(base_email_logger, send_method, side_effect=slow_send):
        with mock.patch.dict(os.environ, {"PROXY_BASE_URL": "http://test.com"}):
            await asyncio.gather(
                base_email_logger.budget_alerts(
                    type=alert_type, user_info=_budget_alert_user_info(ci_kwargs)
                ),
                base_email_logger.budget_alerts(
                    type=alert_type, user_info=_budget_alert_user_info(ci_kwargs)
                ),
            )

    assert len(sends) == 1


@pytest.mark.parametrize(
    "branch, alert_type, send_method, ci_kwargs",
    _BUDGET_ALERT_BRANCHES,
    ids=[b[0] for b in _BUDGET_ALERT_BRANCHES],
)
@pytest.mark.asyncio
async def test_budget_alert_failed_send_releases_claim_for_retry(
    base_email_logger, branch, alert_type, send_method, ci_kwargs
):
    """Claiming the send slot before sending must not swallow the alert forever
    if the send fails; the claim is released so a later request retries. Covers
    all three send branches."""
    base_email_logger.internal_usage_cache = DualCache()

    attempts = []

    async def flaky_send(*args, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise ValueError("transient email backend failure")

    with mock.patch.object(base_email_logger, send_method, side_effect=flaky_send):
        with mock.patch.dict(os.environ, {"PROXY_BASE_URL": "http://test.com"}):
            await base_email_logger.budget_alerts(
                type=alert_type, user_info=_budget_alert_user_info(ci_kwargs)
            )
            await base_email_logger.budget_alerts(
                type=alert_type, user_info=_budget_alert_user_info(ci_kwargs)
            )

    assert len(attempts) == 2


@pytest.mark.asyncio
async def test_budget_alert_release_failure_does_not_propagate(base_email_logger):
    """If the send fails and releasing the claim also fails (transient cache
    error), budget_alerts must swallow it and still log the send failure rather
    than letting the exception escape the fire-and-forget task."""
    mock_cache = mock.AsyncMock()
    mock_cache.async_increment_cache = mock.AsyncMock(return_value=1)
    mock_cache.async_delete_cache = mock.AsyncMock(
        side_effect=RuntimeError("cache backend unavailable")
    )
    base_email_logger.internal_usage_cache = mock_cache

    async def failing_send(*args, **kwargs):
        raise ValueError("smtp backend down")

    with mock.patch.object(
        base_email_logger, "send_max_budget_alert_email", side_effect=failing_send
    ):
        with mock.patch.dict(os.environ, {"PROXY_BASE_URL": "http://test.com"}):
            # Must not raise even though both the send and the release fail.
            await base_email_logger.budget_alerts(
                type="max_budget_alert",
                user_info=_budget_alert_user_info(dict(max_budget=100.0, spend=85.0)),
            )

    mock_cache.async_delete_cache.assert_awaited_once()


class _FakeBudgetAlertTable:
    """In-memory stand-in for the LiteLLM_BudgetAlertSent prisma table.

    Enforces the same unique constraint the real table does, so a claim taken by
    one replica is visible to every other replica sharing this instance.
    """

    _UNIQUE_FIELDS = ("entity_type", "entity_id", "alert_type", "threshold_pct")

    def __init__(self):
        self.rows: list[dict] = []

    @staticmethod
    def _identity(row: Mapping[str, object]) -> tuple:
        return tuple(row[f] for f in _FakeBudgetAlertTable._UNIQUE_FIELDS)

    def _matching(self, where: Mapping[str, object]) -> list[dict]:
        matched = []
        for row in self.rows:
            for field, expected in where.items():
                actual = row.get(field)
                if isinstance(expected, dict) and "not" in expected:
                    if actual == expected["not"]:
                        break
                elif isinstance(expected, dict) and "in" in expected:
                    if actual not in expected["in"]:
                        break
                elif actual != expected:
                    break
            else:
                matched.append(row)
        return matched

    async def create(self, data: Mapping[str, object]) -> dict:
        import prisma

        if any(self._identity(r) == self._identity(data) for r in self.rows):
            raise prisma.errors.UniqueViolationError({"user_facing_error": {"meta": {}}})
        row = dict(data)
        self.rows.append(row)
        return row

    async def update_many(
        self, where: Mapping[str, object], data: Mapping[str, object]
    ) -> int:
        matched = self._matching(where)
        for row in matched:
            row.update(data)
        return len(matched)

    async def delete_many(self, where: Mapping[str, object]) -> int:
        matched = self._matching(where)
        self.rows = [r for r in self.rows if r not in matched]
        return len(matched)


@pytest.fixture
def shared_alert_table(monkeypatch):
    """A single durable claim store shared by every simulated replica."""
    import litellm.proxy.proxy_server as proxy_server

    table = _FakeBudgetAlertTable()
    fake_client = SimpleNamespace(db=SimpleNamespace(litellm_budgetalertsent=table))
    monkeypatch.setattr(proxy_server, "prisma_client", fake_client, raising=False)
    return table


def _replica_logger() -> BaseEmailLogger:
    """A logger with its OWN DualCache, i.e. a separate proxy process with no Redis."""
    logger = BaseEmailLogger()
    logger.internal_usage_cache = DualCache()
    return logger


@pytest.mark.parametrize(
    "branch, alert_type, send_method, ci_kwargs",
    [b for b in _BUDGET_ALERT_BRANCHES if b[1] == "max_budget_alert"],
    ids=[b[0] for b in _BUDGET_ALERT_BRANCHES if b[1] == "max_budget_alert"],
)
@pytest.mark.asyncio
async def test_max_budget_alert_deduped_across_replicas_without_redis(
    shared_alert_table, branch, alert_type, send_method, ci_kwargs
):
    """Regression for LIT-4172 follow-up: two replicas with no Redis must send
    exactly one email for one threshold crossing.

    The in-memory claim added by the first fix is process-local, so each replica
    won its own claim and sent its own copy. The durable claim is what makes the
    two replicas agree.
    """
    sends = []

    async def record_send(*args, **kwargs):
        sends.append(kwargs.get("threshold_pct"))

    for _ in range(2):
        replica = _replica_logger()
        with mock.patch.object(replica, send_method, side_effect=record_send):
            with mock.patch.dict(os.environ, {"PROXY_BASE_URL": "http://test.com"}):
                await replica.budget_alerts(
                    type=alert_type, user_info=_budget_alert_user_info(ci_kwargs)
                )

    assert len(sends) == 1
    assert len(shared_alert_table.rows) == 1


@pytest.mark.asyncio
async def test_max_budget_alert_not_resent_after_replica_restart(shared_alert_table):
    """A restart wipes the in-memory claim. The alert must not fire again, which
    is what made a long-crossed threshold re-alert every restart and every TTL."""
    sends = []

    async def record_send(*args, **kwargs):
        sends.append(1)

    user_info = _budget_alert_user_info(
        dict(max_budget=100.0, spend=80.0, max_budget_alert_emails={"50": []})
    )
    for _ in range(3):
        replica = _replica_logger()
        with mock.patch.object(
            replica, "send_max_budget_alert_email", side_effect=record_send
        ):
            with mock.patch.dict(os.environ, {"PROXY_BASE_URL": "http://test.com"}):
                await replica.budget_alerts(
                    type="max_budget_alert", user_info=user_info
                )

    assert len(sends) == 1


@pytest.mark.parametrize(
    "branch, alert_type, send_method, ci_kwargs",
    [b for b in _BUDGET_ALERT_BRANCHES if b[1] == "max_budget_alert"],
    ids=[b[0] for b in _BUDGET_ALERT_BRANCHES if b[1] == "max_budget_alert"],
)
@pytest.mark.asyncio
async def test_max_budget_alert_rearms_when_budget_window_rolls_over(
    shared_alert_table, branch, alert_type, send_method, ci_kwargs
):
    """A durable claim must not silence the alert forever: once the key's budget
    period resets, the same threshold has to alert again."""
    sends = []

    async def record_send(*args, **kwargs):
        sends.append(1)

    def user_info_for(reset_at):
        return _budget_alert_user_info({**ci_kwargs, "budget_reset_at": reset_at})

    same_window = datetime(2026, 9, 1, tzinfo=timezone.utc)
    next_window = datetime(2026, 10, 1, tzinfo=timezone.utc)
    for window in (same_window, same_window, next_window):
        replica = _replica_logger()
        with mock.patch.object(replica, send_method, side_effect=record_send):
            with mock.patch.dict(os.environ, {"PROXY_BASE_URL": "http://test.com"}):
                await replica.budget_alerts(
                    type=alert_type, user_info=user_info_for(window)
                )

    assert len(sends) == 2
    assert len(shared_alert_table.rows) == 1


@pytest.mark.parametrize(
    "branch, alert_type, send_method, ci_kwargs",
    [b for b in _BUDGET_ALERT_BRANCHES if b[1] == "max_budget_alert"],
    ids=[b[0] for b in _BUDGET_ALERT_BRANCHES if b[1] == "max_budget_alert"],
)
@pytest.mark.asyncio
async def test_max_budget_alert_failed_send_releases_durable_claim(
    shared_alert_table, branch, alert_type, send_method, ci_kwargs
):
    """A send that fails must give the durable claim back, or the whole fleet is
    permanently silenced for that threshold."""
    attempts = []

    async def flaky_send(*args, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise ValueError("transient email backend failure")

    user_info = _budget_alert_user_info(ci_kwargs)
    for _ in range(2):
        replica = _replica_logger()
        with mock.patch.object(replica, send_method, side_effect=flaky_send):
            with mock.patch.dict(os.environ, {"PROXY_BASE_URL": "http://test.com"}):
                await replica.budget_alerts(type=alert_type, user_info=user_info)

    assert len(attempts) == 2
    assert len(shared_alert_table.rows) == 1


@pytest.mark.asyncio
async def test_max_budget_alert_sends_when_no_database_is_configured(monkeypatch):
    """With no prisma client the durable claim cannot be taken. The alert must
    still be sent rather than silently dropped."""
    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr(proxy_server, "prisma_client", None, raising=False)
    sends = []

    async def record_send(*args, **kwargs):
        sends.append(1)

    replica = _replica_logger()
    with mock.patch.object(
        replica, "send_max_budget_alert_email", side_effect=record_send
    ):
        with mock.patch.dict(os.environ, {"PROXY_BASE_URL": "http://test.com"}):
            await replica.budget_alerts(
                type="max_budget_alert",
                user_info=_budget_alert_user_info(
                    dict(max_budget=100.0, spend=80.0, max_budget_alert_emails={"50": []})
                ),
            )

    assert len(sends) == 1


@pytest.mark.parametrize(
    "branch, alert_type, send_method, ci_kwargs",
    [b for b in _BUDGET_ALERT_BRANCHES if b[1] == "max_budget_alert"],
    ids=[b[0] for b in _BUDGET_ALERT_BRANCHES if b[1] == "max_budget_alert"],
)
@pytest.mark.asyncio
async def test_max_budget_alert_rearms_when_max_budget_changes(
    shared_alert_table, branch, alert_type, send_method, ci_kwargs
):
    """A key with no budget_duration never rolls over, so a claim keyed only on the
    reset date would silence the threshold for the life of the key. Changing the
    budget moves every threshold amount and has to re-arm the alerts."""
    sends = []

    async def record_send(*args, **kwargs):
        sends.append(1)

    def user_info_for(max_budget: float):
        return _budget_alert_user_info(
            {**ci_kwargs, "max_budget": max_budget, "budget_reset_at": None}
        )

    for max_budget in (100.0, 100.0, 90.0):
        replica = _replica_logger()
        with mock.patch.object(replica, send_method, side_effect=record_send):
            with mock.patch.dict(os.environ, {"PROXY_BASE_URL": "http://test.com"}):
                await replica.budget_alerts(
                    type=alert_type, user_info=user_info_for(max_budget)
                )

    assert len(sends) == 2
    assert len(shared_alert_table.rows) == 1


@pytest.mark.asyncio
async def test_release_does_not_steal_a_claim_from_a_later_window(shared_alert_table):
    """A slow failing send must only release the claim it actually took, or it
    deletes the row another replica has already taken over for a later window."""
    await claim_budget_alert_slot(
        entity_type="key",
        entity_id="hashed_key_1",
        alert_type=MAX_BUDGET_ALERT_TYPE,
        threshold_pct=50,
        budget_window="later-window",
    )

    await release_budget_alert_slot(
        entity_type="key",
        entity_id="hashed_key_1",
        alert_type=MAX_BUDGET_ALERT_TYPE,
        threshold_pct=50,
        budget_window="earlier-window",
    )

    assert len(shared_alert_table.rows) == 1


@pytest.mark.asyncio
async def test_each_configured_threshold_claims_its_own_slot(shared_alert_table):
    """The claim is per threshold. A key crossing two thresholds at once must send
    both emails; collapsing them onto one claim silently drops the higher one."""
    sends = []

    async def record_send(*args, **kwargs):
        sends.append(kwargs.get("threshold_pct"))

    user_info = _budget_alert_user_info(
        dict(
            max_budget=100.0,
            spend=80.0,
            max_budget_alert_emails={"50": ["finance@co.com"], "75": ["finance@co.com"]},
        )
    )
    replica = _replica_logger()
    with mock.patch.object(
        replica, "send_max_budget_alert_email", side_effect=record_send
    ):
        with mock.patch.dict(os.environ, {"PROXY_BASE_URL": "http://test.com"}):
            await replica.budget_alerts(type="max_budget_alert", user_info=user_info)

    assert sorted(sends) == [50, 75]
    assert len(shared_alert_table.rows) == 2


@pytest.mark.asyncio
async def test_max_budget_alert_sends_when_claim_table_is_unavailable(monkeypatch):
    """Day one of an upgrade the migration may not have run yet. A database error
    that is not "already claimed" must fall through to sending, never silence."""
    import prisma

    import litellm.proxy.proxy_server as proxy_server

    class _BrokenTable:
        async def create(self, data):
            raise prisma.errors.TableNotFoundError({"user_facing_error": {"meta": {}}})

    monkeypatch.setattr(
        proxy_server,
        "prisma_client",
        SimpleNamespace(db=SimpleNamespace(litellm_budgetalertsent=_BrokenTable())),
        raising=False,
    )
    sends = []

    async def record_send(*args, **kwargs):
        sends.append(1)

    for _ in range(2):
        replica = _replica_logger()
        with mock.patch.object(
            replica, "send_max_budget_alert_email", side_effect=record_send
        ):
            with mock.patch.dict(os.environ, {"PROXY_BASE_URL": "http://test.com"}):
                await replica.budget_alerts(
                    type="max_budget_alert",
                    user_info=_budget_alert_user_info(
                        dict(
                            max_budget=100.0,
                            spend=80.0,
                            max_budget_alert_emails={"50": []},
                        )
                    ),
                )

    assert len(sends) == 2


@pytest.mark.asyncio
async def test_deleting_a_key_drops_its_claims_only(shared_alert_table):
    """Claim rows outlive the key unless deletion sweeps them, and a stale row for a
    recycled id would suppress a real alert."""
    for entity_id in ("hashed_key_1", "hashed_key_2"):
        await claim_budget_alert_slot(
            entity_type="key",
            entity_id=entity_id,
            alert_type=MAX_BUDGET_ALERT_TYPE,
            threshold_pct=50,
            budget_window="|100.0",
        )

    await delete_budget_alert_claims(entity_type="key", entity_ids=("hashed_key_1",))

    assert [r["entity_id"] for r in shared_alert_table.rows] == ["hashed_key_2"]


@pytest.mark.asyncio
async def test_in_memory_claim_does_not_outlive_its_budget_window(shared_alert_table):
    """One long-lived replica must re-alert after the window moves. The in-memory
    claim outlives the window by up to EMAIL_BUDGET_ALERT_TTL unless it is scoped
    to the window too."""
    sends = []

    async def record_send(*args, **kwargs):
        sends.append(1)

    replica = _replica_logger()
    with mock.patch.object(
        replica, "send_max_budget_alert_email", side_effect=record_send
    ):
        with mock.patch.dict(os.environ, {"PROXY_BASE_URL": "http://test.com"}):
            for reset_at in (
                datetime(2026, 9, 1, tzinfo=timezone.utc),
                datetime(2026, 9, 1, tzinfo=timezone.utc),
                datetime(2026, 10, 1, tzinfo=timezone.utc),
            ):
                await replica.budget_alerts(
                    type="max_budget_alert",
                    user_info=_budget_alert_user_info(
                        dict(
                            max_budget=100.0,
                            spend=80.0,
                            max_budget_alert_emails={"50": []},
                            budget_reset_at=reset_at,
                        )
                    ),
                )

    assert len(sends) == 2


@pytest.mark.asyncio
async def test_default_threshold_and_configured_threshold_do_not_share_a_claim(
    shared_alert_table,
):
    """The 80% default alert and a configured "80" threshold are different alerts
    with different recipients; one must not consume the other's claim."""
    sends = []

    async def record_send(*args, **kwargs):
        sends.append(kwargs.get("threshold_pct"))

    replica = _replica_logger()
    with mock.patch.object(
        replica, "send_max_budget_alert_email", side_effect=record_send
    ):
        with mock.patch.dict(os.environ, {"PROXY_BASE_URL": "http://test.com"}):
            await replica.budget_alerts(
                type="max_budget_alert",
                user_info=_budget_alert_user_info(dict(max_budget=100.0, spend=85.0)),
            )
            await replica.budget_alerts(
                type="max_budget_alert",
                user_info=_budget_alert_user_info(
                    dict(
                        max_budget=100.0,
                        spend=85.0,
                        max_budget_alert_emails={"80": ["finance@co.com"]},
                    )
                ),
            )

    assert len(sends) == 2
    assert len(shared_alert_table.rows) == 2
