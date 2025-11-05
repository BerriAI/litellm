import json
import os
import sys
import unittest.mock as mock
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from enterprise.litellm_enterprise.enterprise_callbacks.send_emails.base_email import (
    BaseEmailLogger,
)

sys.path.insert(0, os.path.abspath("../../.."))
from litellm_enterprise.types.enterprise_callbacks.send_emails import (
    EmailEvent,
    SendKeyCreatedEmailEvent,
    SendKeyRotatedEmailEvent,
)

from litellm.integrations.email_templates.email_footer import EMAIL_FOOTER
from litellm.proxy._types import Litellm_EntityType, WebhookEvent


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
        assert "[Key hidden for security - retrieve from dashboard]" in call_args["html_body"]


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
        assert "[Key hidden for security - retrieve from dashboard]" in call_args["html_body"]


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
            return_value=mock_created_invitation
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
            assert result == "http://test.com/ui?invitation_id=new-invitation-id"


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
            assert result == "http://test.com/ui?invitation_id=existing-invitation-id"


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
            return_value=mock_created_invitation
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
            assert result == "http://test.com/ui?invitation_id=new-invitation-from-none"


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
            return_value="http://test.com/ui?invitation_id=test-id",
        ):
            # Test with user invitation event
            result = await base_email_logger._get_email_params(
                email_event=EmailEvent.new_user_invitation,
                user_id="test-user",
                user_email="test@example.com",
            )

            assert result.logo_url == "https://litellm-listing.s3.amazonaws.com/litellm_logo.png"
            assert result.support_contact == "support@berri.ai"
            assert result.base_url == "http://test.com/ui?invitation_id=test-id"
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
            event_message="New User Invitation"
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
            event_message="API Key Created"
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
            event_message="New User Invitation"
        )
        
        # Should use default values even though custom values are set in env
        assert email_params.subject == "LiteLLM: New User Invitation"
        assert email_params.signature == EMAIL_FOOTER
        assert email_params.logo_url == "https://litellm-listing.s3.amazonaws.com/litellm_logo.png"
        assert email_params.support_contact == "support@berri.ai"

        
        # Test key created email params
        key_params = await email_logger._get_email_params(
            email_event=EmailEvent.virtual_key_created,
            user_email="test@example.com",
            event_message="API Key Created"
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
            event_message="New User Invitation"
        )
        
        assert invitation_params.subject == "LiteLLM: New User Invitation"
        assert invitation_params.signature == EMAIL_FOOTER
        
        # Test key created email params with default template
        key_params = await email_logger._get_email_params(
            email_event=EmailEvent.virtual_key_created,
            user_email="test@example.com",
            event_message="API Key Created"
        )
        
        assert key_params.subject == "LiteLLM: API Key Created"
        assert key_params.signature == EMAIL_FOOTER 