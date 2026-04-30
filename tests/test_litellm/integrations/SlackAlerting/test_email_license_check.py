import pytest
import os
import asyncio
from unittest.mock import AsyncMock, patch
from litellm.integrations.SlackAlerting.slack_alerting import SlackAlerting
import sys
from litellm.proxy._types import WebhookEvent
import litellm.proxy.proxy_server # Ensure module is loaded for patching

@pytest.mark.asyncio
async def test_send_invite_email_without_license():
    """
    Test that sending an invite email with EMAIL_LOGO_URL set 
    does not fail with a license error when user is not premium.
    
    Reproduction for Issue #19860
    """
    # Setup
    alerting_args = AsyncMock() 
    slack_alerting = SlackAlerting(alerting_args)
    slack_alerting.alerting = ["email"]
    
    webhook_event = WebhookEvent(
        event="internal_user_created",
        event_message="User Invited",
        event_group="team",
        user_email="test@example.com",
        user_id="test_user",
        token="test_token",
        spend=0.0,
    )
    
    # Mock dependencies
    with patch.dict(os.environ, {"EMAIL_LOGO_URL": "https://example.com/logo.png"}), \
         patch("litellm.proxy.proxy_server.premium_user", False), \
         patch("litellm.proxy.utils.send_email", new_callable=AsyncMock) as mock_send_email:
        
        # Action
        try:
            success = await slack_alerting.send_key_created_or_user_invited_email(webhook_event)
        except Exception as e:
            pytest.fail(f"Raised exception: {e}")
            
        # Assert
        assert success is True
        mock_send_email.assert_called_once()

@pytest.mark.asyncio
async def test_check_premium_feature_does_not_raise():
    """
    Verify that checking premium feature for email logo/support contact no longer raises
    an error after the fix for #19860.
    """
    alerting_args = AsyncMock() 
    slack_alerting = SlackAlerting(alerting_args)
    
    with patch("litellm.proxy.proxy_server.premium_user", False):
        try:
            await slack_alerting._check_if_using_premium_email_feature(
                premium_user=False,
                email_logo_url="https://example.com/logo.png"
            )
        except ValueError:
            pytest.fail("Should not raise ValueError for email_logo_url")

