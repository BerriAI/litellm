"""
Security hook to prevent user B from seeing response from user A.

This hook uses the DBSpendUpdateWriter to batch-write response IDs to the database
instead of writing immediately on each request.
"""

from collections.abc import AsyncGenerator, Callable, Mapping
from typing import TYPE_CHECKING, Any, Final, cast

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


_RESPONSES_API_PROVIDER_PREFIX: Final = "/openai"
_RESPONSES_API_CREATE_ROUTES: Final = frozenset({"/v1/responses", "/responses"})

_ADDRESSED_RESPONSE_ID_KEY: Final = "_litellm_addressed_response_id"
_UNMANAGED_RESPONSE_ID_DETAIL: Final = (
    "Forbidden. This response id was not issued by this proxy, so the proxy cannot tell who owns it. "
    "To let keys address responses this proxy did not issue, set "
    "general_settings::allow_unmanaged_response_ids to True in the config.yaml file."
)
_PROXY_ADMIN_ROLES: Final = frozenset({LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN.value})


def _proxy_general_settings() -> Mapping[str, Any]:
    from litellm.proxy.proxy_server import general_settings

    return general_settings


def _proxy_signing_key() -> str | None:
    import os

    from litellm.proxy.proxy_server import master_key

    salt_key: Final = os.getenv("LITELLM_SALT_KEY", None)
    return master_key if salt_key is None else salt_key


def _is_responses_api_create_route(request_route: str | None) -> bool:
    if request_route is None:
        return False
    canonical: Final = (
        request_route[len(_RESPONSES_API_PROVIDER_PREFIX) :]
        if request_route.startswith(_RESPONSES_API_PROVIDER_PREFIX + "/")
        else request_route
    )
    return canonical in _RESPONSES_API_CREATE_ROUTES


class ResponsesIDSecurity(CustomLogger):
    def __init__(
        self,
        general_settings_reader: Callable[[], Mapping[str, Any]] = _proxy_general_settings,
        signing_key_reader: Callable[[], str | None] = _proxy_signing_key,
    ) -> None:
        self._general_settings_reader: Final = general_settings_reader
        self._signing_key_reader: Final = signing_key_reader

    async def async_pre_call_hook(
        self,
        user_api_key_dict: "UserAPIKeyAuth",
        cache: "DualCache",
        data: dict,
        call_type: CallTypesLiteral,
    ) -> Exception | str | dict | None:
        # MAP all the responses api response ids to the encrypted response ids
        responses_api_call_types: Final = {
            "aresponses",
            "aget_responses",
            "adelete_responses",
            "acancel_responses",
            "alist_input_items",
        }
        if call_type not in responses_api_call_types:
            return None
        addressed_id_field: Final = "previous_response_id" if call_type == "aresponses" else "response_id"
        retained_id: Final = data.get(_ADDRESSED_RESPONSE_ID_KEY)
        addressed_id: Final = (
            retained_id if isinstance(retained_id, str) and retained_id else data.get(addressed_id_field)
        )
        if not isinstance(addressed_id, str) or not addressed_id:
            return data
        authorized_id: Final = self._authorize_response_id(addressed_id, user_api_key_dict)
        data[addressed_id_field] = authorized_id
        data[_ADDRESSED_RESPONSE_ID_KEY] = addressed_id
        return data

    def _authorize_response_id(
        self,
        response_id: str,
        user_api_key_dict: "UserAPIKeyAuth",
    ) -> str:
        if self._is_encrypted_response_id(response_id):
            original_response_id, user_id, team_id = self._decrypt_response_id(response_id)
            self.check_user_access_to_response_id(user_id, team_id, user_api_key_dict)
            return original_response_id

        if self._unmanaged_response_ids_allowed(user_api_key_dict):
            return response_id

        raise HTTPException(status_code=403, detail=_UNMANAGED_RESPONSE_ID_DETAIL)

    def _unmanaged_response_ids_allowed(self, user_api_key_dict: "UserAPIKeyAuth") -> bool:
        general_settings: Final = self._general_settings_reader()

        if general_settings.get("disable_responses_id_security", False):
            return True
        if general_settings.get("allow_unmanaged_response_ids", False):
            return True
        if self._get_signing_key() is None:
            return True
        return user_api_key_dict.user_role in _PROXY_ADMIN_ROLES

    def check_user_access_to_response_id(
        self,
        response_id_user_id: str | None,
        response_id_team_id: str | None,
        user_api_key_dict: "UserAPIKeyAuth",
    ) -> bool:
        general_settings: Final = self._general_settings_reader()

        if (
            user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN.value
            or user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN
        ):
            return True

        if response_id_user_id and response_id_user_id != user_api_key_dict.user_id:
            if general_settings.get("disable_responses_id_security", False):
                verbose_proxy_logger.debug(
                    "Responses ID Security is disabled. User %s is accessing response id %s which is not associated with them.",
                    user_api_key_dict.user_id,
                    response_id_user_id,
                )
                return True
            raise HTTPException(
                status_code=403,
                detail="Forbidden. The response id is not associated with the user, who this key belongs to. To disable this security feature, set general_settings::disable_responses_id_security to True in the config.yaml file.",
            )

        if response_id_team_id and response_id_team_id != user_api_key_dict.team_id:
            if general_settings.get("disable_responses_id_security", False):
                verbose_proxy_logger.debug(
                    "Responses ID Security is disabled. Response belongs to team %s but user %s is accessing it with team id %s.",
                    response_id_team_id,
                    user_api_key_dict.user_id,
                    user_api_key_dict.team_id,
                )
                return True
            raise HTTPException(
                status_code=403,
                detail="Forbidden. The response id is not associated with the team, who this key belongs to. To disable this security feature, set general_settings::disable_responses_id_security to True in the config.yaml file.",
            )

        return True

    def _is_encrypted_response_id(self, response_id: str) -> bool:
        split_result: Final = response_id.split("resp_")
        if len(split_result) < 2:
            return False

        remaining_string: Final = split_result[1]
        decrypted_value = decrypt_value_helper(value=remaining_string, key="response_id", return_original_value=True)

        if decrypted_value is None:
            return False

        if decrypted_value.startswith(SpecialEnums.LITELM_MANAGED_FILE_ID_PREFIX.value):
            return True
        return False

    def _decrypt_response_id(self, response_id: str) -> tuple[str, str | None, str | None]:
        """
        Returns:
         - original_response_id: the original response id
         - user_id: the user id
         - team_id: the team id
        """
        split_result: Final = response_id.split("resp_")
        if len(split_result) < 2:
            return response_id, None, None

        remaining_string: Final = split_result[1]
        decrypted_value = decrypt_value_helper(value=remaining_string, key="response_id", return_original_value=True)

        if decrypted_value is None:
            return response_id, None, None

        if decrypted_value.startswith(SpecialEnums.LITELM_MANAGED_FILE_ID_PREFIX.value):
            # Expected format: "litellm_proxy:responses_api:response_id:{response_id};user_id:{user_id}"
            parts: Final = decrypted_value.split(";")

            if len(parts) >= 2:
                # Extract response_id from "litellm_proxy:responses_api:response_id:{response_id}"
                response_id_part: Final = parts[0]
                original_response_id: Final = response_id_part.split("response_id:")[-1]

                # Extract user_id from "user_id:{user_id}"
                user_id_part: Final = parts[1]
                user_id: Final = user_id_part.split("user_id:")[-1]

                # Extract team_id from "team_id:{team_id}"
                team_id_part: Final = parts[2]
                team_id: Final = team_id_part.split("team_id:")[-1]

                return original_response_id, user_id, team_id
            else:
                # Fallback if format is unexpected
                return response_id, None, None
        return response_id, None, None

    def _get_signing_key(self) -> str | None:
        return self._signing_key_reader()

    def _encrypt_response_id(
        self,
        response: BaseLiteLLMOpenAIResponseObject,
        user_api_key_dict: "UserAPIKeyAuth",
        request_cache: dict[str, str] | None = None,
    ) -> BaseLiteLLMOpenAIResponseObject:
        # encrypt the response id using the symmetric key
        # encrypt the response id, and encode the user id and response id in base64

        # Check if signing key is available
        signing_key: Final = self._get_signing_key()
        if signing_key is None:
            verbose_proxy_logger.debug(
                "Response ID encryption is enabled but no signing key is configured. "
                "Please set LITELLM_SALT_KEY environment variable or configure a master_key. "
                "Skipping response ID encryption. "
                "See: https://docs.litellm.ai/docs/proxy/prod#5-set-litellm-salt-key"
            )
            return response

        response_id: Final = getattr(response, "id", None)
        response_obj: Final = getattr(response, "response", None)

        if response_id and isinstance(response_id, str) and response_id.startswith("resp_"):
            # Check request-scoped cache first (for streaming consistency)
            if request_cache is not None and response_id in request_cache:
                setattr(response, "id", request_cache[response_id])
            else:
                encrypted_response_id = SpecialEnums.LITELLM_MANAGED_RESPONSE_API_RESPONSE_ID_COMPLETE_STR.value.format(
                    response_id,
                    user_api_key_dict.user_id or "",
                    user_api_key_dict.team_id or "",
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
        general_settings: Final = self._general_settings_reader()

        if general_settings.get("disable_responses_id_security", False):
            return response
        if isinstance(response, ResponsesAPIResponse):
            response = cast(
                ResponsesAPIResponse,
                self._encrypt_response_id(response, user_api_key_dict, request_cache=None),
            )
        return response

    async def async_post_call_streaming_iterator_hook(
        self, user_api_key_dict: "UserAPIKeyAuth", response: Any, request_data: dict
    ) -> AsyncGenerator[BaseLiteLLMOpenAIResponseObject, None]:
        general_settings: Final = self._general_settings_reader()

        # Create a request-scoped cache for consistent encryption across streaming chunks.
        request_encryption_cache: Final[dict[str, str]] = {}

        async for chunk in response:
            if (
                isinstance(chunk, BaseLiteLLMOpenAIResponseObject)
                and _is_responses_api_create_route(user_api_key_dict.request_route)
                and not general_settings.get("disable_responses_id_security", False)
            ):
                chunk = self._encrypt_response_id(chunk, user_api_key_dict, request_encryption_cache)
            yield chunk
