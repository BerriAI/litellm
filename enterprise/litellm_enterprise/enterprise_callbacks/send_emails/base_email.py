"""
Base class for sending emails to user after creating keys or invite links

"""

import json
import os
from typing import List, Optional

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.email_templates.email_footer import EMAIL_FOOTER
from litellm.integrations.email_templates.key_created_email import (
    KEY_CREATED_EMAIL_TEMPLATE,
)
from litellm.integrations.email_templates.user_invitation_email import (
    USER_INVITATION_EMAIL_TEMPLATE,
)
from litellm.proxy._types import WebhookEvent
from litellm.types.enterprise.enterprise_callbacks.send_emails import (
    EmailParams,
    SendKeyCreatedEmailEvent,
)
from litellm.types.integrations.slack_alerting import LITELLM_LOGO_URL


class BaseEmailLogger(CustomLogger):
    DEFAULT_LITELLM_EMAIL = "notifications@alerts.litellm.ai"
    DEFAULT_SUPPORT_EMAIL = "support@berri.ai"

    async def send_user_invitation_email(self, event: WebhookEvent):
        """
        Send email to user after inviting them to the team
        """
        email_params = await self._get_email_params(
            user_id=event.user_id, user_email=getattr(event, "user_email", None)
        )
        # Implement invitation email logic using email_params

        verbose_proxy_logger.debug(
            f"send_user_invitation_email_event: {json.dumps(event, indent=4, default=str)}"
        )

        email_html_content = USER_INVITATION_EMAIL_TEMPLATE.format(
            email_logo_url=email_params.logo_url,
            recipient_email=email_params.recipient_email,
            base_url=email_params.base_url,
            email_support_contact=email_params.support_contact,
            email_footer=EMAIL_FOOTER,
        )

        await self.send_email(
            from_email=self.DEFAULT_LITELLM_EMAIL,
            to_email=[email_params.recipient_email],
            subject=f"LiteLLM: {event.event_message}",
            html_body=email_html_content,
        )

        pass

    async def send_key_created_email(
        self, send_key_created_email_event: SendKeyCreatedEmailEvent
    ):
        """
        Send email to user after creating key for the user
        """

        email_params = await self._get_email_params(
            user_id=send_key_created_email_event.user_id,
            user_email=send_key_created_email_event.user_email,
        )

        verbose_proxy_logger.debug(
            f"send_key_created_email_event: {json.dumps(send_key_created_email_event, indent=4, default=str)}"
        )

        email_html_content = KEY_CREATED_EMAIL_TEMPLATE.format(
            email_logo_url=email_params.logo_url,
            recipient_email=email_params.recipient_email,
            key_budget=self._format_key_budget(send_key_created_email_event.max_budget),
            key_token=send_key_created_email_event.virtual_key,
            base_url=email_params.base_url,
            email_support_contact=email_params.support_contact,
            email_footer=EMAIL_FOOTER,
        )

        await self.send_email(
            from_email=self.DEFAULT_LITELLM_EMAIL,
            to_email=[email_params.recipient_email],
            subject=f"LiteLLM: {send_key_created_email_event.event_message}",
            html_body=email_html_content,
        )
        pass

    async def _get_email_params(
        self, user_id: Optional[str] = None, user_email: Optional[str] = None
    ) -> EmailParams:
        """
        Get common email parameters used across different email sending methods

        Returns:
            EmailParams object containing logo_url, support_contact, base_url, and recipient_email
        """
        logo_url = os.getenv("EMAIL_LOGO_URL", None) or LITELLM_LOGO_URL
        support_contact = os.getenv("EMAIL_SUPPORT_CONTACT", self.DEFAULT_SUPPORT_EMAIL)
        base_url = os.getenv("PROXY_BASE_URL", "http://0.0.0.0:4000")

        recipient_email: Optional[
            str
        ] = user_email or await self._lookup_user_email_from_db(user_id=user_id)
        if recipient_email is None:
            raise ValueError(
                f"User email not found for user_id: {user_id}. User email is required to send email."
            )

        return EmailParams(
            logo_url=logo_url,
            support_contact=support_contact,
            base_url=base_url,
            recipient_email=recipient_email,
        )

    def _format_key_budget(self, max_budget: Optional[float]) -> str:
        """
        Format the key budget to be displayed in the email
        """
        if max_budget is None:
            return "No budget"
        return f"${max_budget}"

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
