"""
LiteLLM x SendGrid email integration.

Docs: https://docs.sendgrid.com/api-reference/mail-send/mail-send
"""

import os
from typing import List

from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)

from .base_email import BaseEmailLogger


SENDGRID_API_ENDPOINT = "https://api.sendgrid.com/v3/mail/send"


class SendGridEmailLogger(BaseEmailLogger):
    """
    Send emails using SendGrid's Mail Send API.

    Required env vars:
    - SENDGRID_API_KEY
    """

    def __init__(self, internal_usage_cache=None, **kwargs):
        super().__init__(internal_usage_cache=internal_usage_cache, **kwargs)
        self.async_httpx_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback
        )
        self.sendgrid_api_key = os.getenv("SENDGRID_API_KEY")
        self.sendgrid_sender_email = os.getenv("SENDGRID_SENDER_EMAIL")
        verbose_logger.debug("SendGrid Email Logger initialized.")

    async def send_email(
        self,
        from_email: str,
        to_email: List[str],
        subject: str,
        html_body: str,
    ):
        """
        Send an email via SendGrid.
        """
        if not self.sendgrid_api_key:
            raise ValueError("SENDGRID_API_KEY is not set")

        sender_email = self.sendgrid_sender_email or from_email
        verbose_logger.debug(
            f"Sending email via SendGrid from {sender_email} to {to_email} with subject {subject}"
        )

        payload = {
            "from": {"email": sender_email},
            "personalizations": [
                {
                    "to": [{"email": email} for email in to_email],
                    "subject": subject,
                }
            ],
            "content": [
                {
                    "type": "text/html",
                    "value": html_body,
                }
            ],
        }

        response = await self.async_httpx_client.post(
            url=SENDGRID_API_ENDPOINT,
            json=payload,
            headers={"Authorization": f"Bearer {self.sendgrid_api_key}"},
        )

        verbose_logger.debug(
            f"SendGrid response status={response.status_code}, body={response.text}"
        )
        return