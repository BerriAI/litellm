import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Optional

import litellm
from litellm.proxy._types import (
    GenerateKeyRequest,
    LiteLLM_AuditLogs,
    LitellmTableNames,
    UserAPIKeyAuth,
    WebhookEvent,
)


class KeyManagementEventHooks:

    @staticmethod
    async def async_key_generated_hook(
        data: GenerateKeyRequest,
        response: dict,
        user_api_key_dict: UserAPIKeyAuth,
        litellm_changed_by: Optional[str] = None,
    ):
        """
        Post /key/generate processing hook

        Handles the following:
        - Sending Email with Key Details
        - Storing Audit Logs for key generation
        - Storing Generated Key in DB
        """
        from litellm.proxy.management_helpers.audit_logs import (
            create_audit_log_for_update,
        )
        from litellm.proxy.proxy_server import (
            general_settings,
            litellm_proxy_admin_name,
            proxy_logging_obj,
        )

        if data.send_invite_email is True:
            await KeyManagementEventHooks._send_key_created_email(response)

        # Enterprise Feature - Audit Logging. Enable with litellm.store_audit_logs = True
        if litellm.store_audit_logs is True:
            _updated_values = json.dumps(response, default=str)
            asyncio.create_task(
                create_audit_log_for_update(
                    request_data=LiteLLM_AuditLogs(
                        id=str(uuid.uuid4()),
                        updated_at=datetime.now(timezone.utc),
                        changed_by=litellm_changed_by
                        or user_api_key_dict.user_id
                        or litellm_proxy_admin_name,
                        changed_by_api_key=user_api_key_dict.api_key,
                        table_name=LitellmTableNames.KEY_TABLE_NAME,
                        object_id=response.get("token_id", ""),
                        action="created",
                        updated_values=_updated_values,
                        before_value=None,
                    )
                )
            )

    @staticmethod
    async def _send_key_created_email(response: dict):
        from litellm.proxy.proxy_server import general_settings, proxy_logging_obj

        if "email" not in general_settings.get("alerting", []):
            raise ValueError(
                "Email alerting not setup on config.yaml. Please set `alerting=['email']. \nDocs: https://docs.litellm.ai/docs/proxy/email`"
            )
        event = WebhookEvent(
            event="key_created",
            event_group="key",
            event_message="API Key Created",
            token=response.get("token", ""),
            spend=response.get("spend", 0.0),
            max_budget=response.get("max_budget", 0.0),
            user_id=response.get("user_id", None),
            team_id=response.get("team_id", "Default Team"),
            key_alias=response.get("key_alias", None),
        )

        # If user configured email alerting - send an Email letting their end-user know the key was created
        asyncio.create_task(
            proxy_logging_obj.slack_alerting_instance.send_key_created_or_user_invited_email(
                webhook_event=event,
            )
        )
