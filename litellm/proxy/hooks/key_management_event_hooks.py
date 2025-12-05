import asyncio
import json
from datetime import datetime, timezone
from typing import Any, List, Optional

import litellm
from litellm._logging import verbose_proxy_logger
from litellm._uuid import uuid
from litellm.proxy._types import (
    GenerateKeyRequest,
    GenerateKeyResponse,
    KeyRequest,
    LiteLLM_AuditLogs,
    Litellm_EntityType,
    LiteLLM_VerificationToken,
    LitellmTableNames,
    RegenerateKeyRequest,
    UpdateKeyRequest,
    UserAPIKeyAuth,
)

# NOTE: This is the prefix for all virtual keys stored in AWS Secrets Manager
LITELLM_PREFIX_STORED_VIRTUAL_KEYS = "litellm/"


class KeyManagementEventHooks:
    @staticmethod
    async def async_key_generated_hook(
        data: GenerateKeyRequest,
        response: GenerateKeyResponse,
        user_api_key_dict: UserAPIKeyAuth,
        litellm_changed_by: Optional[str] = None,
    ):
        """
        Hook that runs after a successful /key/generate request

        Handles the following:
        - Sending Email with Key Details
        - Storing Audit Logs for key generation
        - Storing Generated Key in DB
        """
        from litellm.proxy.management_helpers.audit_logs import (
            create_audit_log_for_update,
        )
        from litellm.proxy.proxy_server import litellm_proxy_admin_name

        # Send email notification - non-blocking, independent operation
        try:
            await KeyManagementEventHooks._send_key_created_email(
                response.model_dump(exclude_none=True)
            )
        except Exception as e:
            verbose_proxy_logger.warning(f"Failed to send key created email: {e}")

        # Enterprise Feature - Audit Logging. Enable with litellm.store_audit_logs = True
        if litellm.store_audit_logs is True:
            _updated_values = response.model_dump_json(exclude_none=True)
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
                        object_id=response.token_id or "",
                        action="created",
                        updated_values=_updated_values,
                        before_value=None,
                    )
                )
            )

        # Store the generated key in the secret manager - non-blocking, independent operation
        try:
            await KeyManagementEventHooks._store_virtual_key_in_secret_manager(
                secret_name=data.key_alias or f"virtual-key-{response.token_id}",
                secret_token=response.key,
            )
        except Exception as e:
            verbose_proxy_logger.warning(
                f"Failed to store virtual key in secret manager: {e}"
            )

    @staticmethod
    async def async_key_updated_hook(
        data: UpdateKeyRequest,
        existing_key_row: Any,
        response: Any,
        user_api_key_dict: UserAPIKeyAuth,
        litellm_changed_by: Optional[str] = None,
    ):
        """
        Post /key/update processing hook

        Handles the following:
        - Storing Audit Logs for key update
        """
        from litellm.proxy.management_helpers.audit_logs import (
            create_audit_log_for_update,
        )
        from litellm.proxy.proxy_server import litellm_proxy_admin_name

        # Enterprise Feature - Audit Logging. Enable with litellm.store_audit_logs = True
        if litellm.store_audit_logs is True:
            _updated_values = json.dumps(data.json(exclude_none=True), default=str)

            _before_value = existing_key_row.json(exclude_none=True)
            _before_value = json.dumps(_before_value, default=str)

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
                        object_id=data.key,
                        action="updated",
                        updated_values=_updated_values,
                        before_value=_before_value,
                    )
                )
            )

    @staticmethod
    async def async_key_rotated_hook(
        data: Optional[RegenerateKeyRequest],
        existing_key_row: LiteLLM_VerificationToken,
        response: GenerateKeyResponse,
        user_api_key_dict: UserAPIKeyAuth,
        litellm_changed_by: Optional[str] = None,
    ):
        from litellm.proxy.management_helpers.audit_logs import (
            create_audit_log_for_update,
        )
        from litellm.proxy.proxy_server import litellm_proxy_admin_name

        # Store the generated key in the secret manager - non-blocking, independent operation
        if data is not None and response.token_id is not None:
            try:
                initial_secret_name = (
                    existing_key_row.key_alias
                    or f"virtual-key-{existing_key_row.token}"
                )
                await KeyManagementEventHooks._rotate_virtual_key_in_secret_manager(
                    current_secret_name=initial_secret_name,
                    new_secret_name=data.key_alias or f"virtual-key-{response.token_id}",
                    new_secret_value=response.key,
                )
            except Exception as e:
                verbose_proxy_logger.warning(
                    f"Failed to rotate virtual key in secret manager: {e}"
                )

        # Send key rotated email if configured - non-blocking, independent operation
        try:
            await KeyManagementEventHooks._send_key_rotated_email(
                response=response.model_dump(exclude_none=True),
                existing_key_alias=existing_key_row.key_alias,
            )
        except Exception as e:
            verbose_proxy_logger.warning(f"Failed to send key rotated email: {e}")

        # store the audit log
        if litellm.store_audit_logs is True and existing_key_row.token is not None:
            asyncio.create_task(
                create_audit_log_for_update(
                    request_data=LiteLLM_AuditLogs(
                        id=str(uuid.uuid4()),
                        updated_at=datetime.now(timezone.utc),
                        changed_by=litellm_changed_by
                        or user_api_key_dict.user_id
                        or litellm_proxy_admin_name,
                        changed_by_api_key=user_api_key_dict.token,
                        table_name=LitellmTableNames.KEY_TABLE_NAME,
                        object_id=existing_key_row.token,
                        action="rotated",
                        updated_values=response.model_dump_json(exclude_none=True),
                        before_value=existing_key_row.model_dump_json(
                            exclude_none=True
                        ),
                    )
                )
            )

    @staticmethod
    async def async_key_deleted_hook(
        data: KeyRequest,
        keys_being_deleted: List[LiteLLM_VerificationToken],
        response: dict,
        user_api_key_dict: UserAPIKeyAuth,
        litellm_changed_by: Optional[str] = None,
    ):
        """
        Post /key/delete processing hook

        Handles the following:
        - Storing Audit Logs for key deletion
        """
        from litellm.proxy.management_helpers.audit_logs import (
            create_audit_log_for_update,
        )
        from litellm.proxy.proxy_server import litellm_proxy_admin_name

        # Enterprise Feature - Audit Logging. Enable with litellm.store_audit_logs = True
        # we do this after the first for loop, since first for loop is for validation. we only want this inserted after validation passes
        if litellm.store_audit_logs is True and data.keys is not None:
            # make an audit log for each key deleted
            for key in keys_being_deleted:
                if key.token is None:
                    continue
                _key_row = key.model_dump_json(exclude_none=True)

                asyncio.create_task(
                    create_audit_log_for_update(
                        request_data=LiteLLM_AuditLogs(
                            id=str(uuid.uuid4()),
                            updated_at=datetime.now(timezone.utc),
                            changed_by=litellm_changed_by
                            or user_api_key_dict.user_id
                            or litellm_proxy_admin_name,
                            changed_by_api_key=user_api_key_dict.token,
                            table_name=LitellmTableNames.KEY_TABLE_NAME,
                            object_id=key.token,
                            action="deleted",
                            updated_values="{}",
                            before_value=_key_row,
                        )
                    )
                )
        # delete the keys from the secret manager
        await KeyManagementEventHooks._delete_virtual_keys_from_secret_manager(
            keys_being_deleted=keys_being_deleted
        )
        pass

    @staticmethod
    async def _store_virtual_key_in_secret_manager(secret_name: str, secret_token: str):
        """
        Store a virtual key in the secret manager

        Args:
            secret_name: Name of the virtual key
            secret_token: Value of the virtual key (example: sk-1234)
        """
        if litellm._key_management_settings is not None:
            if litellm._key_management_settings.store_virtual_keys is True:
                from litellm.secret_managers.base_secret_manager import (
                    BaseSecretManager,
                )

                # store the key in the secret manager
                if isinstance(litellm.secret_manager_client, BaseSecretManager):
                    tags = getattr(litellm._key_management_settings, "tags", None)
                    description = getattr(
                        litellm._key_management_settings, "description", None
                    )
                    verbose_proxy_logger.debug(
                        f"Creating secret with {secret_name} and tags={tags} and description={description}"
                    )

                    await litellm.secret_manager_client.async_write_secret(
                        secret_name=KeyManagementEventHooks._get_secret_name(
                            secret_name
                        ),
                        description=description,
                        secret_value=secret_token,
                        tags=tags
                    )

    @staticmethod
    async def _rotate_virtual_key_in_secret_manager(
        current_secret_name: str, new_secret_name: str, new_secret_value: str
    ):
        """
        Update a virtual key in the secret manager

        Args:
            secret_name: Name of the virtual key
            secret_token: Value of the virtual key (example: sk-1234)
        """
        if litellm._key_management_settings is not None:
            if litellm._key_management_settings.store_virtual_keys is True:
                from litellm.secret_managers.base_secret_manager import (
                    BaseSecretManager,
                )

                # store the key in the secret manager
                if isinstance(litellm.secret_manager_client, BaseSecretManager):
                    await litellm.secret_manager_client.async_rotate_secret(
                        current_secret_name=KeyManagementEventHooks._get_secret_name(
                            current_secret_name
                        ),
                        new_secret_name=KeyManagementEventHooks._get_secret_name(
                            new_secret_name
                        ),
                        new_secret_value=new_secret_value,
                    )

    @staticmethod
    def _get_secret_name(secret_name: str) -> str:
        if litellm._key_management_settings.prefix_for_stored_virtual_keys.endswith(
            "/"
        ):
            return f"{litellm._key_management_settings.prefix_for_stored_virtual_keys}{secret_name}"
        else:
            return f"{litellm._key_management_settings.prefix_for_stored_virtual_keys}/{secret_name}"

    @staticmethod
    async def _delete_virtual_keys_from_secret_manager(
        keys_being_deleted: List[LiteLLM_VerificationToken],
    ):
        """
        Deletes virtual keys from the secret manager

        Args:
            keys_being_deleted: List of keys being deleted, this is passed down from the /key/delete operation
        """
        if litellm._key_management_settings is not None:
            if litellm._key_management_settings.store_virtual_keys is True:
                from litellm.secret_managers.base_secret_manager import (
                    BaseSecretManager,
                )

                if isinstance(litellm.secret_manager_client, BaseSecretManager):
                    for key in keys_being_deleted:
                        if key.key_alias is not None:
                            await litellm.secret_manager_client.async_delete_secret(
                                secret_name=KeyManagementEventHooks._get_secret_name(
                                    key.key_alias
                                )
                            )
                        else:
                            verbose_proxy_logger.warning(
                                f"KeyManagementEventHooks._delete_virtual_key_from_secret_manager: Key alias not found for key {key.token}. Skipping deletion from secret manager."
                            )

    @staticmethod
    def _is_email_sending_enabled() -> bool:
        """
        Check if email sending is enabled via v2 enterprise loggers or v0 alerting config.

        Returns True only if email is actually configured, preventing any email
        processing when the user has not opted in.
        """
        # Check v2 enterprise email loggers
        try:
            from litellm_enterprise.enterprise_callbacks.send_emails.base_email import (
                BaseEmailLogger,
            )

            initialized_email_loggers = (
                litellm.logging_callback_manager.get_custom_loggers_for_type(
                    callback_type=BaseEmailLogger
                )
            )
            if len(initialized_email_loggers) > 0:
                return True
        except ImportError:
            pass

        # Check v0 alerting config
        from litellm.proxy.proxy_server import general_settings

        if "email" in general_settings.get("alerting", []):
            return True

        return False

    @staticmethod
    async def _send_key_created_email(response: dict):
        """
        Send key created email if email sending is enabled.

        This method is non-blocking - it will return silently if email is not
        configured, and will log warnings instead of raising exceptions on failure.
        """
        # Early exit if email is not enabled
        if not KeyManagementEventHooks._is_email_sending_enabled():
            verbose_proxy_logger.debug(
                "Email sending not enabled, skipping key created email"
            )
            return

        from litellm.proxy.proxy_server import general_settings, proxy_logging_obj

        ##########################
        # v2 integration for emails (enterprise)
        ##########################
        try:
            from litellm_enterprise.enterprise_callbacks.send_emails.base_email import (
                BaseEmailLogger,
            )
            from litellm_enterprise.types.enterprise_callbacks.send_emails import (
                SendKeyCreatedEmailEvent,
            )

            initialized_email_loggers = (
                litellm.logging_callback_manager.get_custom_loggers_for_type(
                    callback_type=BaseEmailLogger
                )
            )
            if len(initialized_email_loggers) > 0:
                event = SendKeyCreatedEmailEvent(
                    virtual_key=response.get("key", ""),
                    event="key_created",
                    event_group=Litellm_EntityType.KEY,
                    event_message="API Key Created",
                    token=response.get("token", ""),
                    spend=response.get("spend", 0.0),
                    max_budget=response.get("max_budget", 0.0),
                    user_id=response.get("user_id", None),
                    team_id=response.get("team_id", "Default Team"),
                    key_alias=response.get("key_alias", None),
                )
                for email_logger in initialized_email_loggers:
                    if isinstance(email_logger, BaseEmailLogger):
                        await email_logger.send_key_created_email(
                            send_key_created_email_event=event,
                        )
                return
        except ImportError:
            pass

        ##########################
        # v0 integration for emails
        ##########################
        if "email" in general_settings.get("alerting", []):
            from litellm.proxy._types import WebhookEvent

            event = WebhookEvent(
                event="key_created",
                event_group=Litellm_EntityType.KEY,
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

    @staticmethod
    async def _send_key_rotated_email(response: dict, existing_key_alias: Optional[str]):
        """
        Send key rotated email if email sending is enabled.

        This method is non-blocking - it will return silently if email is not
        configured, and will log warnings instead of raising exceptions on failure.
        """
        # Early exit if email is not enabled
        if not KeyManagementEventHooks._is_email_sending_enabled():
            verbose_proxy_logger.debug(
                "Email sending not enabled, skipping key rotated email"
            )
            return

        try:
            from litellm_enterprise.enterprise_callbacks.send_emails.base_email import (
                BaseEmailLogger,
            )
        except ImportError:
            # Enterprise package not installed - v0 doesn't support key rotated email
            verbose_proxy_logger.debug(
                "Enterprise package not installed, skipping key rotated email"
            )
            return

        try:
            from litellm_enterprise.types.enterprise_callbacks.send_emails import (
                SendKeyRotatedEmailEvent,
            )
        except ImportError:
            verbose_proxy_logger.debug(
                "Enterprise types not available, skipping key rotated email"
            )
            return

        event = SendKeyRotatedEmailEvent(
            virtual_key=response.get("key", ""),
            event="key_rotated",
            event_group=Litellm_EntityType.KEY,
            event_message="API Key Rotated",
            token=response.get("token", ""),
            spend=response.get("spend", 0.0),
            max_budget=response.get("max_budget", 0.0),
            user_id=response.get("user_id", None),
            team_id=response.get("team_id", "Default Team"),
            key_alias=response.get("key_alias", existing_key_alias),
        )

        ##########################
        # v2 integration for emails
        ##########################
        initialized_email_loggers = (
            litellm.logging_callback_manager.get_custom_loggers_for_type(
                callback_type=BaseEmailLogger
            )
        )
        if len(initialized_email_loggers) > 0:
            for email_logger in initialized_email_loggers:
                if isinstance(email_logger, BaseEmailLogger):
                    await email_logger.send_key_rotated_email(
                        send_key_rotated_email_event=event,
                    )
