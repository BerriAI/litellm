import pytest
from unittest.mock import patch, MagicMock
import os
import sys
sys.path.insert(0, os.path.abspath("../.."))

from enterprise.litellm_enterprise.enterprise_callbacks.send_emails.base_email import BaseEmailLogger
from enterprise.litellm_enterprise.types.enterprise_callbacks.send_emails import EmailEvent
from litellm.integrations.email_templates.email_footer import EMAIL_FOOTER

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up test environment variables"""
    monkeypatch.setenv("EMAIL_LOGO_URL", "https://test-company.com/logo.png")
    monkeypatch.setenv("EMAIL_SUPPORT_CONTACT", "support@test-company.com")
    monkeypatch.setenv("EMAIL_SIGNATURE", "Best regards,\nTest Company Team")
    monkeypatch.setenv("EMAIL_SUBJECT_INVITATION", "Welcome to Test Company!")
    monkeypatch.setenv("EMAIL_SUBJECT_KEY_CREATED", "Your Test Company API Key")
    monkeypatch.setenv("PROXY_BASE_URL", "http://test.com")

@pytest.mark.asyncio
async def test_get_email_params_custom_templates(mock_env_vars):
    """Test that _get_email_params returns correct values with custom templates"""
    email_logger = BaseEmailLogger()
    
    # Test invitation email params
    invitation_params = await email_logger._get_email_params(
        email_event=EmailEvent.new_user_invitation,
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
        user_email="test@example.com",
        event_message="API Key Created"
    )
    
    assert key_params.subject == "Your Test Company API Key"
    assert key_params.signature == "Best regards,\nTest Company Team"

@pytest.mark.asyncio
async def test_get_email_params_default_templates(monkeypatch):
    """Test that _get_email_params uses default templates when custom ones aren't provided"""
    # Clear any existing environment variables
    monkeypatch.delenv("EMAIL_SUBJECT_INVITATION", raising=False)
    monkeypatch.delenv("EMAIL_SUBJECT_KEY_CREATED", raising=False)
    monkeypatch.delenv("EMAIL_SIGNATURE", raising=False)
    
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