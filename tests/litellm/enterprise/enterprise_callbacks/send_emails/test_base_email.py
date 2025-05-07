import json
import os
import sys
import unittest.mock as mock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("../../.."))

from enterprise.enterprise_callbacks.send_emails.base_email import BaseEmailLogger
from litellm.proxy._types import Litellm_EntityType
from litellm.types.enterprise.enterprise_callbacks.send_emails import (
    SendKeyCreatedEmailEvent,
)


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
