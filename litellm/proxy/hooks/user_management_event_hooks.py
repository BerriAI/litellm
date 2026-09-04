"""
Hooks that are triggered when a litellm user event occurs
"""

import asyncio
from datetime import datetime, timezone
from typing import Final

import litellm
from litellm._logging import verbose_proxy_logger
from litellm._uuid import uuid
from litellm.proxy._types import (
    AUDIT_ACTIONS,
    CommonProxyErrors,
    LiteLLM_AuditLogs,
    Litellm_EntityType,
    LitellmTableNames,
    NewUserRequest,
    NewUserResponse,
    UserAPIKeyAuth,
    WebhookEvent,
)
from litellm.proxy.management_helpers.audit_logs import (
    create_audit_log_for_update,
    is_audit_logging_enabled,
)
from litellm.repositories.user_repository import UserRepository


class UserManagementEventHooks:
    @staticmethod
    async def async_user_created_hook(
        data: NewUserRequest,
        response: NewUserResponse,
        user_api_key_dict: UserAPIKeyAuth,
    ):
        """
        This hook is called when a new user is created on litellm

        Handles:
        - Creating an audit log for the user creation
        - Sending a user invitation email to the user
        """
        from litellm.proxy.proxy_server import litellm_proxy_admin_name, prisma_client

        #########################################################
        ########## Send User Invitation Email ################
        #########################################################
        await UserManagementEventHooks.async_send_user_invitation_email(
            data=data,
            response=response,
            user_api_key_dict=user_api_key_dict,
        )

        #########################################################
        ########## CREATE AUDIT LOG ################
        #########################################################
        try:
            if prisma_client is None:
                raise Exception(CommonProxyErrors.db_not_connected_error.value)
            if response.user_id is None:
                raise Exception("no user_id returned for the newly created user")
            user_row: Final = await UserRepository(prisma_client).find_by_id(response.user_id)
            if user_row is None:
                raise Exception(f"no user row found for user_id={response.user_id}")
            asyncio.create_task(
                UserManagementEventHooks.create_internal_user_audit_log(
                    user_id=user_row.user_id,
                    action="created",
                    litellm_changed_by=user_api_key_dict.user_id,
                    user_api_key_dict=user_api_key_dict,
                    litellm_proxy_admin_name=litellm_proxy_admin_name,
                    before_value=None,
                    after_value=user_row.model_dump_json(exclude_none=True),
                )
            )
        except Exception as e:
            verbose_proxy_logger.warning("Unable to create audit log for user on `/user/new` - %s", e)

    @staticmethod
    async def async_send_user_invitation_email(
        data: NewUserRequest,
        response: NewUserResponse,
        user_api_key_dict: UserAPIKeyAuth,
    ):
        """
        Send a user invitation email to the user
        """
        event: Final = WebhookEvent(
            event="internal_user_created",
            event_group=Litellm_EntityType.USER,
            event_message="Welcome to LiteLLM Proxy",
            token=response.token,
            spend=response.spend or 0.0,
            max_budget=response.max_budget,
            user_id=response.user_id,
            user_email=response.user_email,
            team_id=response.team_id,
            key_alias=response.key_alias,
        )

        sent_via_v2: Final = await UserManagementEventHooks._send_v2_user_invitation_emails(
            event=event, send_invite_email=data.send_invite_email
        )

        #########################################################
        ########## LEGACY V1 USER INVITATION EMAIL (FALLBACK) ####
        #########################################################
        if data.send_invite_email is True and not sent_via_v2:
            await UserManagementEventHooks.send_legacy_v1_user_invitation_email(
                data=data,
                response=response,
                user_api_key_dict=user_api_key_dict,
                event=event,
            )

    @staticmethod
    async def _send_v2_user_invitation_emails(event: WebhookEvent, send_invite_email: bool | None) -> bool:
        """
        Send the modern (V2) invitation email via any registered enterprise email logger.

        Returns True if at least one logger delivered, so the caller only falls back to
        the legacy email when V2 did not send (enterprise package absent, no email logger
        configured, or every send raised).
        """
        if send_invite_email is not True:
            return False

        try:
            from litellm_enterprise.enterprise_callbacks.send_emails.base_email import (
                BaseEmailLogger,
            )
        except ImportError:
            verbose_proxy_logger.warning(
                "Defaulting to using Legacy Email Hooks." + CommonProxyErrors.missing_enterprise_package.value
            )
            return False

        email_loggers: Final = tuple(
            email_logger
            for email_logger in litellm.logging_callback_manager.get_custom_loggers_for_type(
                callback_type=BaseEmailLogger
            )
            if isinstance(email_logger, BaseEmailLogger)
        )
        if len(email_loggers) == 0:
            return False

        send_outcomes: Final = await asyncio.gather(
            *(email_logger.send_user_invitation_email(event=event) for email_logger in email_loggers),
            return_exceptions=True,
        )
        for outcome in send_outcomes:
            if isinstance(outcome, BaseException):
                verbose_proxy_logger.error(
                    "Error sending v2 user invitation email for user_id=%s: %s",
                    event.user_id,
                    str(outcome),
                )

        return any(not isinstance(outcome, BaseException) for outcome in send_outcomes)

    @staticmethod
    async def send_legacy_v1_user_invitation_email(
        data: NewUserRequest,
        response: NewUserResponse,
        user_api_key_dict: UserAPIKeyAuth,
        event: WebhookEvent,
    ):
        """
        Send a user invitation email to the user
        """
        from litellm.proxy.proxy_server import general_settings, proxy_logging_obj

        # check if user has setup email alerting
        if "email" not in general_settings.get("alerting", []):
            raise ValueError(
                "Email alerting not setup on config.yaml. Please set `alerting=['email']. \nDocs: https://docs.litellm.ai/docs/proxy/email`"
            )

        # If user configured email alerting - send an Email letting their end-user know the key was created
        asyncio.create_task(
            proxy_logging_obj.slack_alerting_instance.send_key_created_or_user_invited_email(
                webhook_event=event,
            )
        )

    @staticmethod
    async def create_internal_user_audit_log(
        user_id: str,
        action: AUDIT_ACTIONS,
        litellm_changed_by: str | None,
        user_api_key_dict: UserAPIKeyAuth,
        litellm_proxy_admin_name: str | None,
        before_value: str | None = None,
        after_value: str | None = None,
    ):
        """
        Create an audit log for an internal user.

        Parameters:
        - user_id: str - The id of the user to create the audit log for.
        - action: AUDIT_ACTIONS - The action to create the audit log for.
        - user_row: LiteLLM_UserTable - The user row to create the audit log for.
        - litellm_changed_by: Optional[str] - The user id of the user who is changing the user.
        - user_api_key_dict: UserAPIKeyAuth - The user api key dictionary.
        - litellm_proxy_admin_name: Optional[str] - The name of the proxy admin.
        """
        if not is_audit_logging_enabled():
            return

        from litellm.proxy.management_helpers.audit_logs import (
            get_audit_log_changed_by,
        )

        await create_audit_log_for_update(
            request_data=LiteLLM_AuditLogs(
                id=str(uuid.uuid4()),
                updated_at=datetime.now(timezone.utc),
                changed_by=get_audit_log_changed_by(
                    litellm_changed_by=litellm_changed_by,
                    user_api_key_dict=user_api_key_dict,
                    litellm_proxy_admin_name=litellm_proxy_admin_name,
                ),
                changed_by_api_key=user_api_key_dict.api_key,
                table_name=LitellmTableNames.USER_TABLE_NAME,
                object_id=user_id,
                action=action,
                updated_values=after_value,
                before_value=before_value,
            )
        )
