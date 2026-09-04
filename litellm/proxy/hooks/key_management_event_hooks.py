import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Final

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
from litellm.proxy.utils import _hash_token_if_needed

# NOTE: This is the prefix for all virtual keys stored in AWS Secrets Manager
LITELLM_PREFIX_STORED_VIRTUAL_KEYS: Final = "litellm/"


class KeyManagementEventHooks:
    @staticmethod
    async def async_key_generated_hook(
        data: GenerateKeyRequest,
        response: GenerateKeyResponse,
        user_api_key_dict: UserAPIKeyAuth,
        litellm_changed_by: str | None = None,
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
            get_audit_log_changed_by,
            is_audit_logging_enabled,
        )
        from litellm.proxy.proxy_server import litellm_proxy_admin_name

        # Send email notification - non-blocking, independent operation
        if data.send_invite_email is True:
            try:
                await KeyManagementEventHooks._send_key_created_email(response.model_dump(exclude_none=True))
            except Exception as e:
                verbose_proxy_logger.warning("Failed to send key created email: %s", e)

        if is_audit_logging_enabled():
            _updated_values: Final = response.model_dump_json(exclude_none=True)
            asyncio.create_task(
                create_audit_log_for_update(
                    request_data=LiteLLM_AuditLogs(
                        id=str(uuid.uuid4()),
                        updated_at=datetime.now(timezone.utc),
                        changed_by=get_audit_log_changed_by(
                            litellm_changed_by=litellm_changed_by,
                            user_api_key_dict=user_api_key_dict,
                            litellm_proxy_admin_name=litellm_proxy_admin_name,
                        ),
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
                team_id=data.team_id,
            )
        except Exception as e:
            verbose_proxy_logger.warning("Failed to store virtual key in secret manager: %s", e)

    @staticmethod
    async def async_key_updated_hook(
        data: UpdateKeyRequest,
        existing_key_row: Any,
        response: Any,
        user_api_key_dict: UserAPIKeyAuth,
        litellm_changed_by: str | None = None,
    ):
        """
        Post /key/update processing hook

        Handles the following:
        - Storing Audit Logs for key update
        """
        from litellm.proxy.management_helpers.audit_logs import (
            create_audit_log_for_update,
            get_audit_log_changed_by,
            is_audit_logging_enabled,
        )
        from litellm.proxy.proxy_server import litellm_proxy_admin_name

        if is_audit_logging_enabled():
            _updated_values: Final = json.dumps(data.json(exclude_none=True), default=str)

            _before_value = existing_key_row.json(exclude_none=True)
            _before_value = json.dumps(_before_value, default=str)

            asyncio.create_task(
                create_audit_log_for_update(
                    request_data=LiteLLM_AuditLogs(
                        id=str(uuid.uuid4()),
                        updated_at=datetime.now(timezone.utc),
                        changed_by=get_audit_log_changed_by(
                            litellm_changed_by=litellm_changed_by,
                            user_api_key_dict=user_api_key_dict,
                            litellm_proxy_admin_name=litellm_proxy_admin_name,
                        ),
                        changed_by_api_key=user_api_key_dict.api_key,
                        table_name=LitellmTableNames.KEY_TABLE_NAME,
                        object_id=_hash_token_if_needed(data.key),
                        action="updated",
                        updated_values=_updated_values,
                        before_value=_before_value,
                    )
                )
            )

    @staticmethod
    async def async_key_rotated_hook(
        data: RegenerateKeyRequest | None,
        existing_key_row: LiteLLM_VerificationToken,
        response: GenerateKeyResponse,
        user_api_key_dict: UserAPIKeyAuth,
        litellm_changed_by: str | None = None,
    ):
        from litellm.proxy.management_helpers.audit_logs import (
            create_audit_log_for_update,
            get_audit_log_changed_by,
            is_audit_logging_enabled,
        )
        from litellm.proxy.proxy_server import litellm_proxy_admin_name

        # Store the generated key in the secret manager - non-blocking, independent operation
        if data is not None and response.token_id is not None:
            try:
                initial_secret_name: Final = existing_key_row.key_alias or f"virtual-key-{existing_key_row.token}"
                new_secret_name: Final = response.key_alias or data.key_alias or initial_secret_name
                verbose_proxy_logger.info(
                    "Updating secret in secret manager: secret_name=%s",
                    new_secret_name,
                )
                team_id: Final = getattr(existing_key_row, "team_id", None)
                await KeyManagementEventHooks._rotate_virtual_key_in_secret_manager(
                    current_secret_name=initial_secret_name,
                    new_secret_name=new_secret_name,
                    new_secret_value=response.key,
                    team_id=team_id,
                )
                verbose_proxy_logger.info(
                    "Secret updated in secret manager: secret_name=%s",
                    new_secret_name,
                )
            except Exception as e:
                verbose_proxy_logger.warning("Failed to rotate virtual key in secret manager: %s", e)

        # Send key rotated email if configured - non-blocking, independent operation
        try:
            await KeyManagementEventHooks._send_key_rotated_email(
                response=response.model_dump(exclude_none=True),
                existing_key_alias=existing_key_row.key_alias,
            )
        except Exception as e:
            verbose_proxy_logger.warning("Failed to send key rotated email: %s", e)

        # store the audit log
        if is_audit_logging_enabled() and existing_key_row.token is not None:
            asyncio.create_task(
                create_audit_log_for_update(
                    request_data=LiteLLM_AuditLogs(
                        id=str(uuid.uuid4()),
                        updated_at=datetime.now(timezone.utc),
                        changed_by=get_audit_log_changed_by(
                            litellm_changed_by=litellm_changed_by,
                            user_api_key_dict=user_api_key_dict,
                            litellm_proxy_admin_name=litellm_proxy_admin_name,
                        ),
                        changed_by_api_key=user_api_key_dict.token,
                        table_name=LitellmTableNames.KEY_TABLE_NAME,
                        object_id=existing_key_row.token,
                        action="rotated",
                        updated_values=response.model_dump_json(exclude_none=True),
                        before_value=existing_key_row.model_dump_json(exclude_none=True),
                    )
                )
            )

    @staticmethod
    async def async_key_deleted_hook(
        data: KeyRequest,
        keys_being_deleted: list[LiteLLM_VerificationToken],
        response: dict,
        user_api_key_dict: UserAPIKeyAuth,
        litellm_changed_by: str | None = None,
    ):
        """
        Post /key/delete processing hook

        Handles the following:
        - Storing Audit Logs for key deletion
        """
        from litellm.proxy.management_helpers.audit_logs import (
            create_audit_log_for_update,
            get_audit_log_changed_by,
            is_audit_logging_enabled,
        )
        from litellm.proxy.proxy_server import litellm_proxy_admin_name

        # we do this after the first for loop, since first for loop is for validation. we only want this inserted after validation passes
        if is_audit_logging_enabled() and data.keys is not None:
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
                            changed_by=get_audit_log_changed_by(
                                litellm_changed_by=litellm_changed_by,
                                user_api_key_dict=user_api_key_dict,
                                litellm_proxy_admin_name=litellm_proxy_admin_name,
                            ),
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
        await KeyManagementEventHooks._delete_virtual_keys_from_secret_manager(keys_being_deleted=keys_being_deleted)

    @staticmethod
    async def _store_virtual_key_in_secret_manager(secret_name: str, secret_token: str, team_id: str | None = None):
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
                    tags: Final = getattr(litellm._key_management_settings, "tags", None)
                    description: Final = getattr(litellm._key_management_settings, "description", None)
                    optional_params: Final = await KeyManagementEventHooks._get_secret_manager_optional_params(team_id)
                    verbose_proxy_logger.debug(
                        "Creating secret with %s and tags=%s and description=%s", secret_name, tags, description
                    )

                    await litellm.secret_manager_client.async_write_secret(
                        secret_name=KeyManagementEventHooks._get_secret_name(secret_name),
                        description=description,
                        secret_value=secret_token,
                        tags=tags,
                        optional_params=optional_params,
                    )

    @staticmethod
    async def _rotate_virtual_key_in_secret_manager(
        current_secret_name: str,
        new_secret_name: str,
        new_secret_value: str,
        team_id: str | None = None,
    ):
        """
        Update a virtual key in the secret manager

        Args:
            current_secret_name: Current name of the virtual key
            new_secret_name: New name of the virtual key
            new_secret_value: New value of the virtual key (example: sk-1234)
            team_id: Optional team ID to get team-specific secret manager settings
        """
        if litellm._key_management_settings is not None:
            if litellm._key_management_settings.store_virtual_keys is True:
                from litellm.secret_managers.base_secret_manager import (
                    BaseSecretManager,
                )

                # store the key in the secret manager
                if isinstance(litellm.secret_manager_client, BaseSecretManager):
                    optional_params: Final = await KeyManagementEventHooks._get_secret_manager_optional_params(team_id)
                    await litellm.secret_manager_client.async_rotate_secret(
                        current_secret_name=KeyManagementEventHooks._get_secret_name(current_secret_name),
                        new_secret_name=KeyManagementEventHooks._get_secret_name(new_secret_name),
                        new_secret_value=new_secret_value,
                        optional_params=optional_params,
                    )

    @staticmethod
    def _get_secret_name(secret_name: str) -> str:
        if litellm._key_management_settings.prefix_for_stored_virtual_keys.endswith("/"):
            return f"{litellm._key_management_settings.prefix_for_stored_virtual_keys}{secret_name}"
        else:
            return f"{litellm._key_management_settings.prefix_for_stored_virtual_keys}/{secret_name}"

    @staticmethod
    async def _delete_virtual_keys_from_secret_manager(
        keys_being_deleted: list[LiteLLM_VerificationToken],
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
                    team_settings_cache: Final[dict[str | None, dict | None]] = {}
                    for key in keys_being_deleted:
                        if key.key_alias is not None:
                            team_id = getattr(key, "team_id", None)
                            if team_id not in team_settings_cache:
                                team_settings_cache[
                                    team_id
                                ] = await KeyManagementEventHooks._get_secret_manager_optional_params(team_id)
                            optional_params = team_settings_cache[team_id]
                            await litellm.secret_manager_client.async_delete_secret(
                                secret_name=KeyManagementEventHooks._get_secret_name(key.key_alias),
                                optional_params=optional_params,
                            )
                        else:
                            verbose_proxy_logger.warning(
                                "KeyManagementEventHooks._delete_virtual_key_from_secret_manager: Key alias not found for key %s. Skipping deletion from secret manager.",
                                key.token,
                            )

    @staticmethod
    async def _get_secret_manager_optional_params(
        team_id: str | None,
    ) -> dict | None:
        if team_id is None:
            return None

        try:
            from litellm.proxy import proxy_server as proxy_server_module
        except ImportError:
            return None

        prisma_client: Final = getattr(proxy_server_module, "prisma_client", None)
        user_api_key_cache: Final = getattr(proxy_server_module, "user_api_key_cache", None)

        if prisma_client is None or user_api_key_cache is None:
            return None

        try:
            from litellm.proxy.auth.auth_checks import get_team_object

            team_obj: Final = await get_team_object(
                team_id=team_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            verbose_proxy_logger.debug("Unable to load team metadata for team_id=%s: %s", team_id, exc)
            return None

        metadata = getattr(team_obj, "metadata", None)
        if metadata is None:
            return None

        if hasattr(metadata, "model_dump"):
            metadata = metadata.model_dump()

        if not isinstance(metadata, dict):
            return None

        team_settings: Final = metadata.get("secret_manager_settings")
        if isinstance(team_settings, dict) and team_settings:
            return dict(team_settings)

        return None

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

            initialized_email_loggers: Final = litellm.logging_callback_manager.get_custom_loggers_for_type(
                callback_type=BaseEmailLogger
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
            verbose_proxy_logger.debug("Email sending not enabled, skipping key created email")
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

            initialized_email_loggers: Final = litellm.logging_callback_manager.get_custom_loggers_for_type(
                callback_type=BaseEmailLogger
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
    async def _send_key_rotated_email(response: dict, existing_key_alias: str | None):
        """
        Send key rotated email if email sending is enabled.

        This method is non-blocking - it will return silently if email is not
        configured, and will log warnings instead of raising exceptions on failure.
        """
        # Early exit if email is not enabled
        if not KeyManagementEventHooks._is_email_sending_enabled():
            verbose_proxy_logger.debug("Email sending not enabled, skipping key rotated email")
            return

        try:
            from litellm_enterprise.enterprise_callbacks.send_emails.base_email import (
                BaseEmailLogger,
            )
        except ImportError:
            # Enterprise package not installed - v0 doesn't support key rotated email
            verbose_proxy_logger.debug("Enterprise package not installed, skipping key rotated email")
            return

        try:
            from litellm_enterprise.types.enterprise_callbacks.send_emails import (
                SendKeyRotatedEmailEvent,
            )
        except ImportError:
            verbose_proxy_logger.debug("Enterprise types not available, skipping key rotated email")
            return

        event: Final = SendKeyRotatedEmailEvent(
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
        initialized_email_loggers: Final = litellm.logging_callback_manager.get_custom_loggers_for_type(
            callback_type=BaseEmailLogger
        )
        if len(initialized_email_loggers) > 0:
            for email_logger in initialized_email_loggers:
                if isinstance(email_logger, BaseEmailLogger):
                    await email_logger.send_key_rotated_email(
                        send_key_rotated_email_event=event,
                    )
