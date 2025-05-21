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
    def __init__(self):
        self.async_httpx_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback
        )
        self.resend_api_key = os.getenv("RESEND_API_KEY")

    async def send_email(
        self,
        from_email: str,
        to_email: List[str],
        subject: str,
        html_body: str,
    ):
        verbose_logger.debug(
            f"Sending email from {from_email} to {to_email} with subject {subject}"
        )
        response = await self.async_httpx_client.post(
            url=RESEND_API_ENDPOINT,
            json={
                "from": from_email,
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
