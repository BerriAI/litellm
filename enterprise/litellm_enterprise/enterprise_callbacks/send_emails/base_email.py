"""
Base class for sending emails to user after creating keys or invite links

"""

import json
import os
from typing import List, Optional

from litellm_enterprise.types.enterprise_callbacks.send_emails import (
    EmailEvent,
    EmailParams,
    SendKeyCreatedEmailEvent,
    SendKeyRotatedEmailEvent,
)

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.email_templates.email_footer import EMAIL_FOOTER
from litellm.integrations.email_templates.key_created_email import (
    KEY_CREATED_EMAIL_TEMPLATE,
)
from litellm.integrations.email_templates.key_rotated_email import (
    KEY_ROTATED_EMAIL_TEMPLATE,
)
from litellm.integrations.email_templates.user_invitation_email import (
    USER_INVITATION_EMAIL_TEMPLATE,
)
from litellm.proxy._types import InvitationNew, UserAPIKeyAuth, WebhookEvent
from litellm.secret_managers.main import get_secret_bool
from litellm.types.integrations.slack_alerting import LITELLM_LOGO_URL


class BaseEmailLogger(CustomLogger):
    DEFAULT_LITELLM_EMAIL = "notifications@alerts.litellm.ai"
    DEFAULT_SUPPORT_EMAIL = "support@berri.ai"
    DEFAULT_SUBJECT_TEMPLATES = {
        EmailEvent.new_user_invitation: "LiteLLM: {event_message}",
        EmailEvent.virtual_key_created: "LiteLLM: {event_message}",
        EmailEvent.virtual_key_rotated: "LiteLLM: {event_message}",
    }

    async def send_user_invitation_email(self, event: WebhookEvent):
        """
        Send email to user after inviting them to the team
        """
        email_params = await self._get_email_params(
            email_event=EmailEvent.new_user_invitation,
            user_id=event.user_id,
            user_email=getattr(event, "user_email", None),
            event_message=event.event_message,
        )

        verbose_proxy_logger.debug(
            f"send_user_invitation_email_event: {json.dumps(event, indent=4, default=str)}"
        )

        email_html_content = USER_INVITATION_EMAIL_TEMPLATE.format(
            email_logo_url=email_params.logo_url,
            recipient_email=email_params.recipient_email,
            base_url=email_params.base_url,
            email_support_contact=email_params.support_contact,
            email_footer=email_params.signature,
        )

        await self.send_email(
            from_email=self.DEFAULT_LITELLM_EMAIL,
            to_email=[email_params.recipient_email],
            subject=email_params.subject,
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
            email_event=EmailEvent.virtual_key_created,
            event_message=send_key_created_email_event.event_message,
        )

        verbose_proxy_logger.debug(
            f"send_key_created_email_event: {json.dumps(send_key_created_email_event, indent=4, default=str)}"
        )

        # Check if API key should be included in email
        include_api_key = get_secret_bool(secret_name="EMAIL_INCLUDE_API_KEY", default_value=True)
        if include_api_key is None:
            include_api_key = True  # Default to True if not set
        key_token_display = send_key_created_email_event.virtual_key if include_api_key else "[Key hidden for security - retrieve from dashboard]"

        email_html_content = KEY_CREATED_EMAIL_TEMPLATE.format(
            email_logo_url=email_params.logo_url,
            recipient_email=email_params.recipient_email,
            key_budget=self._format_key_budget(send_key_created_email_event.max_budget),
            key_token=key_token_display,
            base_url=email_params.base_url,
            email_support_contact=email_params.support_contact,
            email_footer=email_params.signature,
        )

        await self.send_email(
            from_email=self.DEFAULT_LITELLM_EMAIL,
            to_email=[email_params.recipient_email],
            subject=email_params.subject,
            html_body=email_html_content,
        )
        pass

    async def send_key_rotated_email(
        self, send_key_rotated_email_event: SendKeyRotatedEmailEvent
    ):
        """
        Send email to user after rotating key for the user
        """
        email_params = await self._get_email_params(
            user_id=send_key_rotated_email_event.user_id,
            user_email=send_key_rotated_email_event.user_email,
            email_event=EmailEvent.virtual_key_rotated,
            event_message=send_key_rotated_email_event.event_message,
        )

        verbose_proxy_logger.debug(
            f"send_key_rotated_email_event: {json.dumps(send_key_rotated_email_event, indent=4, default=str)}"
        )

        # Check if API key should be included in email
        include_api_key = get_secret_bool(secret_name="EMAIL_INCLUDE_API_KEY", default_value=True)
        if include_api_key is None:
            include_api_key = True  # Default to True if not set
        key_token_display = send_key_rotated_email_event.virtual_key if include_api_key else "[Key hidden for security - retrieve from dashboard]"

        email_html_content = KEY_ROTATED_EMAIL_TEMPLATE.format(
            email_logo_url=email_params.logo_url,
            recipient_email=email_params.recipient_email,
            key_budget=self._format_key_budget(send_key_rotated_email_event.max_budget),
            key_token=key_token_display,
            base_url=email_params.base_url,
            email_support_contact=email_params.support_contact,
            email_footer=email_params.signature,
        )

        await self.send_email(
            from_email=self.DEFAULT_LITELLM_EMAIL,
            to_email=[email_params.recipient_email],
            subject=email_params.subject,
            html_body=email_html_content,
        )
        pass

    async def _get_email_params(
        self,
        email_event: EmailEvent,
        user_id: Optional[str] = None,
        user_email: Optional[str] = None,
        event_message: Optional[str] = None,
    ) -> EmailParams:
        """
        Get common email parameters used across different email sending methods

        Args:
            email_event: Type of email event
            user_id: Optional user ID to look up email
            user_email: Optional direct email address
            event_message: Optional message to include in email subject

        Returns:
            EmailParams object containing logo_url, support_contact, base_url, recipient_email, subject, and signature
        """
        # Get email parameters with premium check for custom values
        custom_logo = os.getenv("EMAIL_LOGO_URL", None)
        custom_support = os.getenv("EMAIL_SUPPORT_CONTACT", None)
        custom_signature = os.getenv("EMAIL_SIGNATURE", None)
        custom_subject_invitation = os.getenv("EMAIL_SUBJECT_INVITATION", None)
        custom_subject_key_created = os.getenv("EMAIL_SUBJECT_KEY_CREATED", None)

        # Track which custom values were not applied
        unused_custom_fields = []

        # Function to safely get custom value or default
        def get_custom_or_default(custom_value: Optional[str], default_value: str, field_name: str) -> str:
            if custom_value is not None:  # Only check premium if trying to use custom value
                from litellm.proxy.proxy_server import premium_user
                if premium_user is not True:
                    unused_custom_fields.append(field_name)
                    return default_value
                return custom_value
            return default_value

        # Get parameters, falling back to defaults if custom values aren't allowed
        logo_url = get_custom_or_default(custom_logo, LITELLM_LOGO_URL, "logo URL")
        support_contact = get_custom_or_default(custom_support, self.DEFAULT_SUPPORT_EMAIL, "support contact")
        base_url = os.getenv("PROXY_BASE_URL", "http://0.0.0.0:4000")  # Not a premium feature
        signature = get_custom_or_default(custom_signature, EMAIL_FOOTER, "email signature")

        # Get custom subject template based on email event type
        if email_event == EmailEvent.new_user_invitation:
            subject_template = get_custom_or_default(
                custom_subject_invitation,
                self.DEFAULT_SUBJECT_TEMPLATES[EmailEvent.new_user_invitation],
                "invitation subject template"
            )
        elif email_event == EmailEvent.virtual_key_created:
            subject_template = get_custom_or_default(
                custom_subject_key_created,
                self.DEFAULT_SUBJECT_TEMPLATES[EmailEvent.virtual_key_created],
                "key created subject template"
            )
        elif email_event == EmailEvent.virtual_key_rotated:
            custom_subject_key_rotated = os.getenv("EMAIL_SUBJECT_KEY_ROTATED", None)
            subject_template = get_custom_or_default(
                custom_subject_key_rotated,
                self.DEFAULT_SUBJECT_TEMPLATES[EmailEvent.virtual_key_rotated],
                "key rotated subject template"
            )
        else:
            subject_template = "LiteLLM: {event_message}"

        subject = subject_template.format(event_message=event_message) if event_message else "LiteLLM Notification"

        recipient_email: Optional[
            str
        ] = user_email or await self._lookup_user_email_from_db(user_id=user_id)
        if recipient_email is None:
            raise ValueError(
                f"User email not found for user_id: {user_id}. User email is required to send email."
            )

        # if user invited event then send invitation link
        if email_event == EmailEvent.new_user_invitation:
            base_url = await self._get_invitation_link(
                user_id=user_id, base_url=base_url
            )

        # If any custom fields were not applied, log a warning
        if unused_custom_fields:
            fields_str = ", ".join(unused_custom_fields)
            warning_msg = (
                f"Email sent with default values instead of custom values for: {fields_str}. "
                "This is an Enterprise feature. To use custom email fields, please upgrade to LiteLLM Enterprise. "
                "Schedule a meeting here: https://calendly.com/d/4mp-gd3-k5k/litellm-1-1-onboarding-chat"
            )
            verbose_proxy_logger.warning(
                f"{warning_msg}"
            )

        return EmailParams(
            logo_url=logo_url,
            support_contact=support_contact,
            base_url=base_url,
            recipient_email=recipient_email,
            subject=subject,
            signature=signature,
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

    async def _get_invitation_link(self, user_id: Optional[str], base_url: str) -> str:
        """
        Get invitation link for the user
        """
        # Early validation
        if not user_id:
            verbose_proxy_logger.debug("No user_id provided for invitation link")
            return base_url
            
        if not await self._is_prisma_client_available():
            return base_url
            
        # Wait for any concurrent invitation creation to complete
        await self._wait_for_invitation_creation()
        
        # Get or create invitation
        invitation = await self._get_or_create_invitation(user_id)
        if not invitation:
            verbose_proxy_logger.warning(f"Failed to get/create invitation for user_id: {user_id}")
            return base_url
            
        return self._construct_invitation_link(invitation.id, base_url)

    async def _is_prisma_client_available(self) -> bool:
        """Check if Prisma client is available"""
        from litellm.proxy.proxy_server import prisma_client
        
        if prisma_client is None:
            verbose_proxy_logger.debug("Prisma client not found. Unable to lookup invitation")
            return False
        return True

    async def _wait_for_invitation_creation(self) -> None:
        """
        Wait for any concurrent invitation creation to complete.
        
        The UI calls /invitation/new to generate the invitation link.
        We wait to ensure any pending invitation creation is completed.
        """
        import asyncio
        await asyncio.sleep(10)

    async def _get_or_create_invitation(self, user_id: str):
        """
        Get existing invitation or create a new one for the user
        
        Returns:
            Invitation object with id attribute, or None if failed
        """
        from litellm.proxy.management_helpers.user_invitation import (
            create_invitation_for_user,
        )
        from litellm.proxy.proxy_server import prisma_client
        
        if prisma_client is None:
            verbose_proxy_logger.error("Prisma client is None in _get_or_create_invitation")
            return None
            
        try:
            # Try to get existing invitation
            existing_invitations = await prisma_client.db.litellm_invitationlink.find_many(
                where={"user_id": user_id},
                order={"created_at": "desc"},
            )
            
            if existing_invitations and len(existing_invitations) > 0:
                verbose_proxy_logger.debug(f"Found existing invitation for user_id: {user_id}")
                return existing_invitations[0]
            
            # Create new invitation if none exists
            verbose_proxy_logger.debug(f"Creating new invitation for user_id: {user_id}")
            return await create_invitation_for_user(
                data=InvitationNew(user_id=user_id),
                user_api_key_dict=UserAPIKeyAuth(user_id=user_id),
            )
            
        except Exception as e:
            verbose_proxy_logger.error(f"Error getting/creating invitation for user_id {user_id}: {e}")
            return None

    def _construct_invitation_link(self, invitation_id: str, base_url: str) -> str:
        """
        Construct invitation link for the user

        # http://localhost:4000/ui?invitation_id=7a096b3a-37c6-440f-9dd1-ba22e8043f6b
        """
        return f"{base_url}/ui?invitation_id={invitation_id}"

    async def send_email(
        self,
        from_email: str,
        to_email: List[str],
        subject: str,
        html_body: str,
    ):
        pass
