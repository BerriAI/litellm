"""
This is the litellm SMTP email integration
"""
import asyncio
from typing import List

from litellm._logging import verbose_logger

from .base_email import BaseEmailLogger


class SMTPEmailLogger(BaseEmailLogger):
    """
    This is the litellm SMTP email integration

    Required SMTP environment variables:
    - SMTP_HOST
    - SMTP_PORT
    - SMTP_USERNAME
    - SMTP_PASSWORD
    - SMTP_SENDER_EMAIL
    """

    def __init__(self, internal_usage_cache=None, **kwargs):
        super().__init__(internal_usage_cache=internal_usage_cache, **kwargs)
        verbose_logger.debug("SMTP Email Logger initialized....")

    async def send_email(
        self,
        from_email: str,
        to_email: List[str],
        subject: str,
        html_body: str,
    ):
        from litellm.proxy.utils import send_email as send_smtp_email

        verbose_logger.debug(
            f"Sending email from {from_email} to {to_email} with subject {subject}"
        )
        for receiver_email in to_email:
            asyncio.create_task(
                send_smtp_email(
                    receiver_email=receiver_email,
                    subject=subject,
                    html=html_body,
                )
            )
        return
