import asyncio
import unittest.mock as mock

import pytest

from litellm_enterprise.enterprise_callbacks.send_emails.smtp_email import (
    SMTPEmailLogger,
)


@pytest.mark.asyncio
async def test_send_email_multiple_recipients_get_separate_messages():
    logger = SMTPEmailLogger()

    with mock.patch("litellm.proxy.utils.send_email", new=mock.AsyncMock()) as smtp_send:
        await logger.send_email(
            from_email="test@example.com",
            to_email=["recipient1@example.com", "recipient2@example.com"],
            subject="Test Subject",
            html_body="<p>Test email body</p>",
        )
        await asyncio.sleep(0)

    receivers_per_message = [
        call.kwargs["receiver_email"] for call in smtp_send.call_args_list
    ]
    assert sorted(receivers_per_message) == [
        "recipient1@example.com",
        "recipient2@example.com",
    ]
