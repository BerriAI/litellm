import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any, List, Optional

from fastapi import status

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import (
    GenerateKeyRequest,
    GenerateKeyResponse,
    KeyRequest,
    LiteLLM_AuditLogs,
    LiteLLM_VerificationToken,
    LitellmTableNames,
    ProxyErrorTypes,
    ProxyException,
    RegenerateKeyRequest,
    UpdateKeyRequest,
    UserAPIKeyAuth,
    WebhookEvent,
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

        if data.send_invite_email is True:
            await KeyManagementEventHooks._send_key_created_email(
                response.model_dump(exclude_none=True)
            )

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
        # store the generated key in the secret manager
        await KeyManagementEventHooks._store_virtual_key_in_secret_manager(
            secret_name=data.key_alias or f"virtual-key-{response.token_id}",
            secret_token=response.key,
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
        existing_key_row: Any,
        response: GenerateKeyResponse,
        user_api_key_dict: UserAPIKeyAuth,
        litellm_changed_by: Optional[str] = None,
    ):
        # store the generated key in the secret manager
        if data is not None and response.token_id is not None:
            initial_secret_name = (
                existing_key_row.key_alias or f"virtual-key-{existing_key_row.token}"
            )
            await KeyManagementEventHooks._rotate_virtual_key_in_secret_manager(
                current_secret_name=initial_secret_name,
                new_secret_name=data.key_alias or f"virtual-key-{response.token_id}",
                new_secret_value=response.key,
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
        from litellm.proxy.proxy_server import litellm_proxy_admin_name, prisma_client

        # Enterprise Feature - Audit Logging. Enable with litellm.store_audit_logs = True
        # we do this after the first for loop, since first for loop is for validation. we only want this inserted after validation passes
        if litellm.store_audit_logs is True and data.keys is not None:
            # make an audit log for each team deleted
            for key in data.keys:
                key_row = await prisma_client.get_data(  # type: ignore
                    token=key, table_name="key", query_type="find_unique"
                )

                if key_row is None:
                    raise ProxyException(
                        message=f"Key {key} not found",
                        type=ProxyErrorTypes.bad_request_error,
                        param="key",
                        code=status.HTTP_404_NOT_FOUND,
                    )

                key_row = key_row.json(exclude_none=True)
                _key_row = json.dumps(key_row, default=str)

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
                            object_id=key,
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
                    await litellm.secret_manager_client.async_write_secret(
                        secret_name=KeyManagementEventHooks._get_secret_name(
                            secret_name
                        ),
                        secret_value=secret_token,
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
