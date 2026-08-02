"""
Security hook to prevent user B from seeing response from user A.

This hook uses the DBSpendUpdateWriter to batch-write response IDs to the database
instead of writing immediately on each request.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncGenerator, Optional, Union, cast

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import LitellmUserRoles
from litellm.proxy.common_utils.encrypt_decrypt_utils import (
    decrypt_value_helper,
    encrypt_value_helper,
)
from litellm.types.llms.openai import (
    BaseLiteLLMOpenAIResponseObject,
    ResponsesAPIResponse,
)
from litellm.types.utils import CallTypesLiteral, LLMResponseTypes, SpecialEnums

if TYPE_CHECKING:
    from litellm.caching.caching import DualCache
    from litellm.proxy._types import UserAPIKeyAuth


@dataclass(frozen=True, slots=True)
class DecryptedResponseID:
    response_id: str
    user_id: Optional[str]
    team_id: Optional[str]
    key_hash: Optional[str]


class ResponsesIDSecurity(CustomLogger):
    def __init__(self):
        pass

    async def async_pre_call_hook(
        self,
        user_api_key_dict: "UserAPIKeyAuth",
        cache: "DualCache",
        data: dict,
        call_type: CallTypesLiteral,
    ) -> Optional[Union[Exception, str, dict]]:
        # MAP all the responses api response ids to the encrypted response ids
        responses_api_call_types = {
            "aresponses",
            "aget_responses",
            "adelete_responses",
            "acancel_responses",
            "alist_input_items",
        }
        if call_type not in responses_api_call_types:
            return None
        if call_type == "aresponses":
            # check 'previous_response_id' if present in the data
            previous_response_id = data.get("previous_response_id")
            if previous_response_id:
                data["previous_response_id"] = self._authorize_response_id(previous_response_id, user_api_key_dict)
        elif call_type in {"aget_responses", "adelete_responses", "acancel_responses", "alist_input_items"}:
            response_id = data.get("response_id")

            if response_id:
                data["response_id"] = self._authorize_response_id(response_id, user_api_key_dict)
        return data

    def _authorize_response_id(self, response_id: str, user_api_key_dict: "UserAPIKeyAuth") -> str:
        """
        Returns the provider-side response id the caller is allowed to act on.

        Any id that is not an ownership-bound id issued by this proxy is rejected, unless the caller is a proxy
        admin, the feature is disabled, or no signing key is configured (in which case this proxy never issued
        bound ids in the first place).
        """
        decrypted = self._decrypt_response_id(response_id)
        if decrypted is None:
            self._reject_unverifiable_response_id(user_api_key_dict)
            return response_id

        self.check_user_access_to_response_id(
            decrypted.user_id,
            decrypted.team_id,
            user_api_key_dict,
            response_id_key_hash=decrypted.key_hash,
        )
        return decrypted.response_id

    def _reject_unverifiable_response_id(self, user_api_key_dict: "UserAPIKeyAuth") -> None:
        from litellm.proxy.proxy_server import general_settings

        if self._is_proxy_admin(user_api_key_dict) or general_settings.get("disable_responses_id_security", False):
            return

        if self._get_signing_key() is None:
            verbose_proxy_logger.warning(
                "Responses ID security is enabled but no signing key is configured, so response ids are neither "
                "encrypted nor ownership-checked. Set LITELLM_SALT_KEY or a master_key. "
                "See: https://docs.litellm.ai/docs/proxy/prod#5-set-litellm-salt-key"
            )
            return

        raise HTTPException(
            status_code=403,
            detail="Forbidden. This response id was not issued by this proxy, so its owner cannot be verified. To "
            "disable this security feature, set general_settings::disable_responses_id_security to True in the "
            "config.yaml file.",
        )

    @staticmethod
    def _is_proxy_admin(user_api_key_dict: "UserAPIKeyAuth") -> bool:
        return user_api_key_dict.user_role in (
            LitellmUserRoles.PROXY_ADMIN.value,
            LitellmUserRoles.PROXY_ADMIN,
        )

    def check_user_access_to_response_id(
        self,
        response_id_user_id: Optional[str],
        response_id_team_id: Optional[str],
        user_api_key_dict: "UserAPIKeyAuth",
        response_id_key_hash: Optional[str] = None,
    ) -> bool:
        from litellm.proxy.proxy_server import general_settings

        if self._is_proxy_admin(user_api_key_dict):
            return True

        if general_settings.get("disable_responses_id_security", False):
            verbose_proxy_logger.debug(
                f"Responses ID Security is disabled. User {user_api_key_dict.user_id} is accessing response id owned "
                f"by user {response_id_user_id} / team {response_id_team_id}."
            )
            return True

        if response_id_key_hash and response_id_key_hash == user_api_key_dict.api_key:
            return True

        if response_id_user_id and response_id_user_id != user_api_key_dict.user_id:
            raise HTTPException(
                status_code=403,
                detail="Forbidden. The response id is not associated with the user, who this key belongs to. To disable this security feature, set general_settings::disable_responses_id_security to True in the config.yaml file.",
            )

        if response_id_team_id and response_id_team_id != user_api_key_dict.team_id:
            raise HTTPException(
                status_code=403,
                detail="Forbidden. The response id is not associated with the team, who this key belongs to. To disable this security feature, set general_settings::disable_responses_id_security to True in the config.yaml file.",
            )

        if not response_id_user_id and not response_id_team_id:
            raise HTTPException(
                status_code=403,
                detail="Forbidden. The response id carries no owner, so it cannot be verified as belonging to this "
                "key. To disable this security feature, set general_settings::disable_responses_id_security to True "
                "in the config.yaml file.",
            )

        return True

    def _is_encrypted_response_id(self, response_id: str) -> bool:
        return self._decrypt_response_id(response_id) is not None

    def _decrypt_response_id(self, response_id: str) -> Optional[DecryptedResponseID]:
        """
        Decrypt an id issued by this proxy, or return None when the id is not one (raw provider id, litellm base64
        managed id, forged id, or an id encrypted under a different signing key).
        """
        split_result = response_id.split("resp_")
        if len(split_result) < 2:
            return None

        decrypted_value = decrypt_value_helper(value=split_result[1], key="response_id", return_original_value=True)

        if decrypted_value is None or not decrypted_value.startswith(SpecialEnums.LITELM_MANAGED_FILE_ID_PREFIX.value):
            return None

        # Format: "litellm_proxy:responses_api:response_id:{};user_id:{};team_id:{};key_hash:{}"
        # key_hash is absent on ids issued before it was added.
        parts = decrypted_value.split(";")
        if len(parts) < 3:
            return None

        return DecryptedResponseID(
            response_id=parts[0].split("response_id:")[-1],
            user_id=parts[1].split("user_id:")[-1] or None,
            team_id=parts[2].split("team_id:")[-1] or None,
            key_hash=(parts[3].split("key_hash:")[-1] or None) if len(parts) > 3 else None,
        )

    def _get_signing_key(self) -> Optional[str]:
        """Get the signing key for encryption/decryption."""
        import os

        from litellm.proxy.proxy_server import master_key

        salt_key = os.getenv("LITELLM_SALT_KEY", None)
        if salt_key is None:
            salt_key = master_key
        return salt_key

    def _encrypt_response_id(
        self,
        response: BaseLiteLLMOpenAIResponseObject,
        user_api_key_dict: "UserAPIKeyAuth",
        request_cache: Optional[dict[str, str]] = None,
    ) -> BaseLiteLLMOpenAIResponseObject:
        # encrypt the response id using the symmetric key
        # encrypt the response id, and encode the user id and response id in base64

        # Check if signing key is available
        signing_key = self._get_signing_key()
        if signing_key is None:
            verbose_proxy_logger.warning(
                "Response ID encryption is enabled but no signing key is configured. "
                "Please set LITELLM_SALT_KEY environment variable or configure a master_key. "
                "Skipping response ID encryption. "
                "See: https://docs.litellm.ai/docs/proxy/prod#5-set-litellm-salt-key"
            )
            return response

        response_id = getattr(response, "id", None)
        response_obj = getattr(response, "response", None)

        if response_id and isinstance(response_id, str) and response_id.startswith("resp_"):
            # Check request-scoped cache first (for streaming consistency)
            if request_cache is not None and response_id in request_cache:
                setattr(response, "id", request_cache[response_id])
            else:
                encrypted_response_id = SpecialEnums.LITELLM_MANAGED_RESPONSE_API_RESPONSE_ID_COMPLETE_STR.value.format(
                    response_id,
                    user_api_key_dict.user_id or "",
                    user_api_key_dict.team_id or "",
                    user_api_key_dict.api_key or "",
                )

                encoded_user_id_and_response_id = encrypt_value_helper(value=encrypted_response_id)
                encrypted_id = f"resp_{encoded_user_id_and_response_id}"
                if request_cache is not None:
                    request_cache[response_id] = encrypted_id
                setattr(response, "id", encrypted_id)

        elif response_obj and isinstance(response_obj, ResponsesAPIResponse):
            # Check request-scoped cache first (for streaming consistency)
            if request_cache is not None and response_obj.id in request_cache:
                setattr(response_obj, "id", request_cache[response_obj.id])
            else:
                encrypted_response_id = SpecialEnums.LITELLM_MANAGED_RESPONSE_API_RESPONSE_ID_COMPLETE_STR.value.format(
                    response_obj.id,
                    user_api_key_dict.user_id or "",
                    user_api_key_dict.team_id or "",
                    user_api_key_dict.api_key or "",
                )
                encoded_user_id_and_response_id = encrypt_value_helper(value=encrypted_response_id)
                encrypted_id = f"resp_{encoded_user_id_and_response_id}"
                if request_cache is not None:
                    request_cache[response_obj.id] = encrypted_id
                setattr(response_obj, "id", encrypted_id)
            setattr(response, "response", response_obj)
        return response

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: "UserAPIKeyAuth",
        response: LLMResponseTypes,
    ) -> Any:
        """
        Queue response IDs for batch processing instead of writing directly to DB.

        This method adds response IDs to an in-memory queue, which are then
        batch-processed by the DBSpendUpdateWriter during regular database update cycles.
        """
        from litellm.proxy.proxy_server import general_settings

        if general_settings.get("disable_responses_id_security", False):
            return response
        if isinstance(response, ResponsesAPIResponse):
            response = cast(
                ResponsesAPIResponse,
                self._encrypt_response_id(response, user_api_key_dict, request_cache=None),
            )
        return response

    async def async_post_call_streaming_iterator_hook(  # type: ignore
        self, user_api_key_dict: "UserAPIKeyAuth", response: Any, request_data: dict
    ) -> AsyncGenerator[BaseLiteLLMOpenAIResponseObject, None]:
        from litellm.proxy.proxy_server import general_settings

        # Create a request-scoped cache for consistent encryption across streaming chunks.
        request_encryption_cache: dict[str, str] = {}

        async for chunk in response:
            if (
                isinstance(chunk, BaseLiteLLMOpenAIResponseObject)
                and user_api_key_dict.request_route
                == "/v1/responses"  # only encrypt the response id for the responses api
                and not general_settings.get("disable_responses_id_security", False)
            ):
                chunk = self._encrypt_response_id(chunk, user_api_key_dict, request_encryption_cache)
            yield chunk
