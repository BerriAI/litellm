"""
Base class for sending emails to user after creating keys or invite links

"""

import json
import os
from typing import List, Optional

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.email_templates.key_created_email import (
    KEY_CREATED_EMAIL_TEMPLATE,
)
from litellm.types.enterprise.enterprise_callbacks.send_emails import (
    SendKeyCreatedEmailEvent,
)


class BaseEmailLogger(CustomLogger):
    DEFAULT_LITELLM_EMAIL = "notifications@alerts.litellm.ai"

    async def send_key_created_email(
        self, send_key_created_email_event: SendKeyCreatedEmailEvent
    ):
        """
        Send email to user after creating key for the user
        """
        email_logo_url = os.getenv(
            "SMTP_SENDER_LOGO", os.getenv("EMAIL_LOGO_URL", None)
        )
        email_support_contact = os.getenv("EMAIL_SUPPORT_CONTACT", None)
        base_url = os.getenv("PROXY_BASE_URL", "http://0.0.0.0:4000")

        recipient_email: Optional[str] = (
            send_key_created_email_event.user_email
            or await self._lookup_user_email_from_db(
                user_id=send_key_created_email_event.user_id
            )
        )
        if recipient_email is None:
            raise ValueError(
                f"User email not found for user_id: {send_key_created_email_event.user_id}. User email is required to send email."
            )

        verbose_proxy_logger.debug(
            f"send_key_created_email_event: {json.dumps(send_key_created_email_event, indent=4, default=str)}"
        )

        email_html_content = KEY_CREATED_EMAIL_TEMPLATE.format(
            email_logo_url=email_logo_url,
            recipient_email=recipient_email,
            key_budget=send_key_created_email_event.max_budget,
            key_token=send_key_created_email_event.virtual_key,
            base_url=base_url,
            email_support_contact=email_support_contact,
        )

        await self.send_email(
            from_email=self.DEFAULT_LITELLM_EMAIL,
            to_email=[recipient_email],
            subject=f"LiteLLM: {send_key_created_email_event.event_message}",
            html_body=email_html_content,
        )
        pass

    async def _lookup_user_email_from_db(self, user_id: Optional[str]) -> Optional[str]:
        """
        Lookup user email from user_id
        """
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            verbose_proxy_logger.debug(
                f"Prisma client not found. Unable to lookup user email for user_id: {user_id}"
            )
            return None

        user_row = await prisma_client.db.litellm_usertable.find_unique(
            where={"user_id": user_id}
        )

        if user_row is not None:
            return user_row.user_email
        return None

    async def send_email(
        self,
        from_email: str,
        to_email: List[str],
        subject: str,
        html_body: str,
    ):
        pass
