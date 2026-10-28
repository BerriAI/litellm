"""
This is the litellm x resend email integration

https://resend.com/docs/api-reference/emails/send-email
"""

import os
from typing import List

from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)

from .base_email import BaseEmailLogger

RESEND_API_ENDPOINT = "https://api.resend.com/emails"


class ResendEmailLogger(BaseEmailLogger):
    """
    Send emails using Resend's API.

    Required env vars:
    - RESEND_API_KEY

    Optional env vars:
    - RESEND_FROM_EMAIL: Override the default sender address. Must be on a
      domain verified in your Resend account. When unset, falls back to the
      `from_email` argument passed by the caller (which defaults to
      `notifications@alerts.litellm.ai` and only works on LiteLLM Cloud).
    """

    def __init__(self, internal_usage_cache=None, **kwargs):
        super().__init__(internal_usage_cache=internal_usage_cache, **kwargs)
        self.async_httpx_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback
        )
        self.resend_api_key = os.getenv("RESEND_API_KEY")
        self.resend_from_email = os.getenv("RESEND_FROM_EMAIL")

    async def send_email(
        self,
        from_email: str,
        to_email: List[str],
        subject: str,
        html_body: str,
    ):
        sender_email = self.resend_from_email or from_email
        verbose_logger.debug(
            f"Sending email from {sender_email} to {to_email} with subject {subject}"
        )
        response = await self.async_httpx_client.post(
            url=RESEND_API_ENDPOINT,
            json={
                "from": sender_email,
                "to": to_email,
                "subject": subject,
                "html": html_body,
            },
            headers={"Authorization": f"Bearer {self.resend_api_key}"},
        )
        verbose_logger.debug(
            f"Email sent with status code {response.status_code}. Got response: {response.json()}"
        )
        return
