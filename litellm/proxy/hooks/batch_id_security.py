"""
Security hook to prevent user B from seeing batches/files from user A.

This hook encrypts batch and file IDs with user/team information,
then validates ownership on retrieve/cancel/list operations.

Pattern follows ResponsesIDSecurity hook.
"""

from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import LitellmUserRoles
from litellm.proxy.common_utils.encrypt_decrypt_utils import (
    decrypt_value_helper,
    encrypt_value_helper,
)
from litellm.types.utils import (
    CallTypesLiteral,
    LiteLLMBatch,
    LLMResponseTypes,
    SpecialEnums,
)

if TYPE_CHECKING:
    from litellm.caching.caching import DualCache
    from litellm.proxy._types import UserAPIKeyAuth


class BatchIDSecurity(CustomLogger):
    """
    Security hook that encrypts batch IDs with user/team info on creation,
    and validates ownership on retrieve/cancel/list operations.
    """

    # Call types this hook handles
    BATCH_CALL_TYPES = {
        "acreate_batch",
        "aretrieve_batch",
        "alist_batches",
    }

    def __init__(self):
        verbose_proxy_logger.info("BatchIDSecurity hook initialized")

    async def async_pre_call_hook(
        self,
        user_api_key_dict: "UserAPIKeyAuth",
        cache: "DualCache",
        data: dict,
        call_type: CallTypesLiteral,
    ) -> Optional[Union[Exception, str, dict]]:
        """
        Decrypt and validate batch IDs on retrieve/cancel operations.
        """
        verbose_proxy_logger.debug(
            f"BatchIDSecurity.async_pre_call_hook called with call_type: {call_type}"
        )

        if call_type not in self.BATCH_CALL_TYPES:
            return None

        if call_type == "aretrieve_batch":
            batch_id = data.get("batch_id")
            verbose_proxy_logger.debug(
                f"BatchIDSecurity: checking batch_id: {batch_id}"
            )
            if batch_id and self._is_encrypted_batch_id(batch_id):
                verbose_proxy_logger.debug(
                    f"BatchIDSecurity: batch_id IS encrypted, decrypting..."
                )
                original_batch_id, user_id, team_id = self._decrypt_batch_id(batch_id)
                verbose_proxy_logger.debug(
                    f"BatchIDSecurity: decrypted to original_id={original_batch_id}, user_id={user_id}, team_id={team_id}"
                )
                self.check_user_access_to_batch_id(
                    user_id, team_id, user_api_key_dict
                )
                data["batch_id"] = original_batch_id
            else:
                verbose_proxy_logger.debug(
                    f"BatchIDSecurity: batch_id is NOT encrypted (raw provider ID)"
                )

        return data

    def check_user_access_to_batch_id(
        self,
        batch_id_user_id: Optional[str],
        batch_id_team_id: Optional[str],
        user_api_key_dict: "UserAPIKeyAuth",
    ) -> bool:
        """
        Check if the user has access to the batch ID.
        Same logic as ResponsesIDSecurity.check_user_access_to_response_id.
        """
        verbose_proxy_logger.debug(
            f"BatchIDSecurity.check_user_access_to_batch_id: "
            f"batch_user={batch_id_user_id}, batch_team={batch_id_team_id}, "
            f"request_user={user_api_key_dict.user_id}, request_team={user_api_key_dict.team_id}"
        )
        from litellm.proxy.proxy_server import general_settings

        # Admin bypass - proxy admins can access any batch
        if (
            user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN.value
            or user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN
        ):
            return True

        # Check user ownership
        if batch_id_user_id and batch_id_user_id != user_api_key_dict.user_id:
            if general_settings.get("disable_batch_id_security", False):
                verbose_proxy_logger.debug(
                    f"Batch ID Security is disabled. User {user_api_key_dict.user_id} is accessing batch from user {batch_id_user_id}."
                )
                return True
            raise HTTPException(
                status_code=403,
                detail="Forbidden. The batch is not associated with the user who this key belongs to. "
                "To disable this security feature, set general_settings::disable_batch_id_security to True in the config.yaml file.",
            )

        # Check team ownership
        if batch_id_team_id and batch_id_team_id != user_api_key_dict.team_id:
            if general_settings.get("disable_batch_id_security", False):
                verbose_proxy_logger.debug(
                    f"Batch ID Security is disabled. Batch belongs to team {batch_id_team_id} but user is accessing with team {user_api_key_dict.team_id}."
                )
                return True
            raise HTTPException(
                status_code=403,
                detail="Forbidden. The batch is not associated with the team who this key belongs to. "
                "To disable this security feature, set general_settings::disable_batch_id_security to True in the config.yaml file.",
            )

        return True

    def _is_encrypted_batch_id(self, batch_id: str) -> bool:
        """
        Check if the batch ID is encrypted with user/team info.
        Encrypted batch IDs have format: batch_{encrypted_value}
        """
        if not batch_id.startswith("batch_"):
            return False

        remaining_string = batch_id[6:]  # Remove "batch_" prefix
        decrypted_value = decrypt_value_helper(
            value=remaining_string, key="batch_id", return_original_value=True
        )

        if decrypted_value is None:
            return False

        if decrypted_value.startswith(SpecialEnums.LITELM_MANAGED_FILE_ID_PREFIX.value):
            return True
        return False

    def _decrypt_batch_id(
        self, batch_id: str
    ) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Decrypt the batch ID and extract user/team info.

        Returns:
            - original_batch_id: the original batch ID from the provider
            - user_id: the user ID who created the batch
            - team_id: the team ID who created the batch
        """
        if not batch_id.startswith("batch_"):
            return batch_id, None, None

        remaining_string = batch_id[6:]  # Remove "batch_" prefix
        decrypted_value = decrypt_value_helper(
            value=remaining_string, key="batch_id", return_original_value=True
        )

        if decrypted_value is None:
            return batch_id, None, None

        if decrypted_value.startswith(SpecialEnums.LITELM_MANAGED_FILE_ID_PREFIX.value):
            # Expected format: "litellm_proxy:batch:batch_id:{batch_id};user_id:{user_id};team_id:{team_id}"
            parts = decrypted_value.split(";")

            if len(parts) >= 3:
                # Extract batch_id from "litellm_proxy:batch:batch_id:{batch_id}"
                batch_id_part = parts[0]
                original_batch_id = batch_id_part.split("batch_id:")[-1]

                # Extract user_id from "user_id:{user_id}"
                user_id_part = parts[1]
                user_id = user_id_part.split("user_id:")[-1]
                user_id = user_id if user_id else None

                # Extract team_id from "team_id:{team_id}"
                team_id_part = parts[2]
                team_id = team_id_part.split("team_id:")[-1]
                team_id = team_id if team_id else None

                return original_batch_id, user_id, team_id
            else:
                return batch_id, None, None

        return batch_id, None, None

    def _get_signing_key(self) -> Optional[str]:
        """Get the signing key for encryption/decryption."""
        import os

        from litellm.proxy.proxy_server import master_key

        salt_key = os.getenv("LITELLM_SALT_KEY", None)
        if salt_key is None:
            salt_key = master_key
        return salt_key

    def _encrypt_batch_id(
        self,
        batch_id: str,
        user_api_key_dict: "UserAPIKeyAuth",
    ) -> str:
        """
        Encrypt the batch ID with user/team information.

        Returns the encrypted batch ID in format: batch_{encrypted_value}
        """
        verbose_proxy_logger.debug(
            f"BatchIDSecurity._encrypt_batch_id called with batch_id: {batch_id}, "
            f"user_id: {user_api_key_dict.user_id}, team_id: {user_api_key_dict.team_id}"
        )

        # Check if signing key is available
        signing_key = self._get_signing_key()
        if signing_key is None:
            verbose_proxy_logger.warning(
                "Batch ID encryption is enabled but no signing key is configured. "
                "Please set LITELLM_SALT_KEY environment variable or configure a master_key. "
                "Skipping batch ID encryption."
            )
            return batch_id

        # Format: litellm_proxy:batch:batch_id:{batch_id};user_id:{user_id};team_id:{team_id}
        encrypted_batch_id = SpecialEnums.LITELLM_MANAGED_BATCH_ID_COMPLETE_STR.value.format(
            batch_id,
            user_api_key_dict.user_id or "",
            user_api_key_dict.team_id or "",
        )

        encoded_value = encrypt_value_helper(value=encrypted_batch_id)
        result = f"batch_{encoded_value}"

        verbose_proxy_logger.debug(
            f"BatchIDSecurity._encrypt_batch_id result: {result[:50]}..."
        )

        # Maintain the 'batch_' prefix
        return result

    def _encrypt_file_id_in_batch(
        self,
        file_id: Optional[str],
        user_api_key_dict: "UserAPIKeyAuth",
    ) -> Optional[str]:
        """
        Encrypt a file ID within a batch response with user/team information.
        """
        if file_id is None:
            return None

        signing_key = self._get_signing_key()
        if signing_key is None:
            return file_id

        # Format: litellm_proxy:file:file_id:{file_id};user_id:{user_id};team_id:{team_id}
        encrypted_file_id = SpecialEnums.LITELLM_MANAGED_FILE_ID_SECURITY_STR.value.format(
            file_id,
            user_api_key_dict.user_id or "",
            user_api_key_dict.team_id or "",
        )

        encoded_value = encrypt_value_helper(value=encrypted_file_id)

        # Maintain the original prefix if present
        if file_id.startswith("file-"):
            return f"file-{encoded_value}"
        return encoded_value

    def _encrypt_batch_response(
        self,
        response: LiteLLMBatch,
        user_api_key_dict: "UserAPIKeyAuth",
    ) -> LiteLLMBatch:
        """
        Encrypt all IDs in a batch response.
        """
        # Encrypt the batch ID
        batch_id = getattr(response, "id", None)
        if batch_id and isinstance(batch_id, str) and batch_id.startswith("batch_"):
            setattr(response, "id", self._encrypt_batch_id(batch_id, user_api_key_dict))

        # Encrypt input_file_id
        input_file_id = getattr(response, "input_file_id", None)
        if input_file_id:
            encrypted_input = self._encrypt_file_id_in_batch(input_file_id, user_api_key_dict)
            if encrypted_input:
                setattr(response, "input_file_id", encrypted_input)

        # Encrypt output_file_id
        output_file_id = getattr(response, "output_file_id", None)
        if output_file_id:
            encrypted_output = self._encrypt_file_id_in_batch(output_file_id, user_api_key_dict)
            if encrypted_output:
                setattr(response, "output_file_id", encrypted_output)

        # Encrypt error_file_id
        error_file_id = getattr(response, "error_file_id", None)
        if error_file_id:
            encrypted_error = self._encrypt_file_id_in_batch(error_file_id, user_api_key_dict)
            if encrypted_error:
                setattr(response, "error_file_id", encrypted_error)

        return response

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: "UserAPIKeyAuth",
        response: LLMResponseTypes,
    ) -> Any:
        """
        Encrypt batch IDs in responses to embed user/team ownership.
        """
        from litellm.proxy.proxy_server import general_settings

        verbose_proxy_logger.info(
            f"BatchIDSecurity.async_post_call_success_hook called with response type: {type(response).__name__}"
        )

        if general_settings.get("disable_batch_id_security", False):
            verbose_proxy_logger.debug("BatchIDSecurity disabled, skipping encryption")
            return response

        # Handle LiteLLMBatch responses (create_batch, retrieve_batch)
        # Check both the type and if it has batch-like attributes
        is_batch = isinstance(response, LiteLLMBatch)
        has_batch_id = hasattr(response, "id") and str(getattr(response, "id", "")).startswith("batch_")
        
        verbose_proxy_logger.info(
            f"BatchIDSecurity: is_batch={is_batch}, has_batch_id={has_batch_id}, "
            f"response_id={getattr(response, 'id', None)}"
        )

        if is_batch or has_batch_id:
            verbose_proxy_logger.info(
                f"Encrypting batch response with id: {getattr(response, 'id', None)}"
            )
            # Cast to LiteLLMBatch - we've verified it has batch-like attributes
            from typing import cast
            batch_response = cast(LiteLLMBatch, response)
            response = self._encrypt_batch_response(batch_response, user_api_key_dict)
            verbose_proxy_logger.info(
                f"After encryption, batch id: {getattr(response, 'id', None)}"
            )

        # Handle list_batches response - filter to only show user's batches
        # List responses have a 'data' attribute with list of batches
        response_data = getattr(response, "data", None)
        if response_data is not None and isinstance(response_data, list):
            verbose_proxy_logger.debug(
                f"BatchIDSecurity: Processing list with {len(response_data)} items"
            )
            filtered_batches: List[Any] = []
            for batch in response_data:
                batch_type = type(batch).__name__
                batch_id = getattr(batch, "id", None)
                verbose_proxy_logger.debug(
                    f"BatchIDSecurity: Item type={batch_type}, id={batch_id}"
                )
                # Check for LiteLLMBatch OR any object with batch-like ID
                is_batch_type = isinstance(batch, LiteLLMBatch)
                has_batch_id = batch_id and str(batch_id).startswith("batch_")
                
                if is_batch_type or has_batch_id:
                    # Check if this is an encrypted batch ID
                    if batch_id and self._is_encrypted_batch_id(batch_id):
                        _, batch_user_id, batch_team_id = self._decrypt_batch_id(batch_id)
                        # Only include batches owned by this user/team
                        if self._user_owns_batch(batch_user_id, batch_team_id, user_api_key_dict):
                            filtered_batches.append(batch)
                        else:
                            verbose_proxy_logger.debug(
                                f"BatchIDSecurity: Filtering out batch - not owned by user"
                            )
                    else:
                        # Non-encrypted batch (created before security was enabled)
                        # FILTER OUT for security - we don't know who owns it
                        verbose_proxy_logger.debug(
                            f"BatchIDSecurity: Filtering out non-encrypted batch {batch_id} - unknown owner"
                        )
                        # Don't include batches without security metadata
                        continue
                else:
                    # Non-batch items in list (shouldn't happen, but handle gracefully)
                    verbose_proxy_logger.debug(
                        f"BatchIDSecurity: Including non-batch item of type {batch_type}"
                    )
                    filtered_batches.append(batch)
            
            verbose_proxy_logger.info(
                f"BatchIDSecurity: Filtered list from {len(response_data)} to {len(filtered_batches)} batches"
            )
            setattr(response, "data", filtered_batches)

        return response

    def _user_owns_batch(
        self,
        batch_user_id: Optional[str],
        batch_team_id: Optional[str],
        user_api_key_dict: "UserAPIKeyAuth",
    ) -> bool:
        """
        Check if the user owns the batch (without raising exception).
        Used for filtering list results.
        """
        # Admin sees all
        if (
            user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN.value
            or user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN
        ):
            return True

        # Check user match
        if batch_user_id and batch_user_id != user_api_key_dict.user_id:
            return False

        # Check team match
        if batch_team_id and batch_team_id != user_api_key_dict.team_id:
            return False

        return True
