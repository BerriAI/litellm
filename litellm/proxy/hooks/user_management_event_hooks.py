"""
Hooks that are triggered when a litellm user event occurs
"""

import asyncio
from litellm._uuid import uuid
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import (
    AUDIT_ACTIONS,
    CommonProxyErrors,
    LiteLLM_AuditLogs,
    Litellm_EntityType,
    LiteLLM_UserTable,
    LitellmTableNames,
    NewUserRequest,
    NewUserResponse,
    UserAPIKeyAuth,
    WebhookEvent,
)
from litellm.proxy.management_helpers.audit_logs import create_audit_log_for_update


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
            user_row: BaseModel = await prisma_client.db.litellm_usertable.find_first(
                where={"user_id": response.user_id}
            )

            user_row_litellm_typed = LiteLLM_UserTable(
                **user_row.model_dump(exclude_none=True)
            )
            asyncio.create_task(
                UserManagementEventHooks.create_internal_user_audit_log(
                    user_id=user_row_litellm_typed.user_id,
                    action="created",
                    litellm_changed_by=user_api_key_dict.user_id,
                    user_api_key_dict=user_api_key_dict,
                    litellm_proxy_admin_name=litellm_proxy_admin_name,
                    before_value=None,
                    after_value=user_row_litellm_typed.model_dump_json(
                        exclude_none=True
                    ),
                )
            )
        except Exception as e:
            verbose_proxy_logger.warning(
                "Unable to create audit log for user on `/user/new` - {}".format(str(e))
            )
        pass

    @staticmethod
    async def async_send_user_invitation_email(
        data: NewUserRequest,
        response: NewUserResponse,
        user_api_key_dict: UserAPIKeyAuth,
    ):
        """
        Send a user invitation email to the user
        """
        event = WebhookEvent(
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

        #########################################################
        ########## V2 USER INVITATION EMAIL ################
        #########################################################
        try:
            from litellm_enterprise.enterprise_callbacks.send_emails.base_email import (
                BaseEmailLogger,
            )

            use_enterprise_email_hooks = True
        except ImportError:
            verbose_proxy_logger.warning(
                "Defaulting to using Legacy Email Hooks."
                + CommonProxyErrors.missing_enterprise_package.value
            )
            use_enterprise_email_hooks = False

        if use_enterprise_email_hooks and (data.send_invite_email is True):
            initialized_email_loggers = litellm.logging_callback_manager.get_custom_loggers_for_type(
                callback_type=BaseEmailLogger  # type: ignore
            )
            if len(initialized_email_loggers) > 0:
                for email_logger in initialized_email_loggers:
                    if isinstance(email_logger, BaseEmailLogger):  # type: ignore
                        await email_logger.send_user_invitation_email(  # type: ignore
                            event=event,
                        )

        #########################################################
        ########## LEGACY V1 USER INVITATION EMAIL ################
        #########################################################
        if data.send_invite_email is True:
            await UserManagementEventHooks.send_legacy_v1_user_invitation_email(
                data=data,
                response=response,
                user_api_key_dict=user_api_key_dict,
                event=event,
            )

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
        litellm_changed_by: Optional[str],
        user_api_key_dict: UserAPIKeyAuth,
        litellm_proxy_admin_name: Optional[str],
        before_value: Optional[str] = None,
        after_value: Optional[str] = None,
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
        if not litellm.store_audit_logs:
            return

        await create_audit_log_for_update(
            request_data=LiteLLM_AuditLogs(
                id=str(uuid.uuid4()),
                updated_at=datetime.now(timezone.utc),
                changed_by=litellm_changed_by
                or user_api_key_dict.user_id
                or litellm_proxy_admin_name,
                changed_by_api_key=user_api_key_dict.api_key,
                table_name=LitellmTableNames.USER_TABLE_NAME,
                object_id=user_id,
                action=action,
                updated_values=after_value,
                before_value=before_value,
            )
        )
