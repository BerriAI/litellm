"""
Security hook to prevent user B from seeing response from user A.

This hook uses the DBSpendUpdateWriter to batch-write response IDs to the database
instead of writing immediately on each request.
"""

from collections.abc import AsyncGenerator, Callable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, TypeVar

from fastapi import HTTPException
from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy._types import LitellmUserRoles
from litellm.proxy.common_utils.encrypt_decrypt_utils import (
    decrypt_value_helper,
    encrypt_value_helper,
)
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.llms.openai import (
    BaseLiteLLMOpenAIResponseObject,
    ResponsesAPIResponse,
)
from litellm.types.responses.main import DeleteResponseResult
from litellm.types.utils import ADDRESSED_RESPONSE_ID_FIELD, CallTypesLiteral, LLMResponseTypes, SpecialEnums

if TYPE_CHECKING:
    from litellm.caching.caching import DualCache
    from litellm.proxy._types import UserAPIKeyAuth


_RESPONSES_API_PROVIDER_PREFIX: Final = "/openai"
_RESPONSES_API_ROUTES: Final = frozenset({"/v1/responses", "/responses"})
_ADDRESSED_RESPONSE_ID_FIELD_BY_CALL_TYPE: Final[Mapping[str, str]] = MappingProxyType(
    {
        "aresponses": "previous_response_id",
        "acompact_responses": "previous_response_id",
        "aget_responses": "response_id",
        "adelete_responses": "response_id",
        "acancel_responses": "response_id",
        "alist_input_items": "response_id",
    }
)

_UNMANAGED_RESPONSE_ID_DETAIL: Final = (
    "Forbidden. This response id was not issued by this proxy, so the proxy cannot tell who owns it. "
    "To let keys address responses this proxy did not issue, set "
    "general_settings::allow_unmanaged_response_ids to True in the config.yaml file."
)
_PROXY_ADMIN_ROLES: Final = frozenset({LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN.value})


def _proxy_general_settings() -> Mapping[str, object]:
    from litellm.proxy.proxy_server import general_settings

    return general_settings


def _proxy_signing_key() -> str | None:
    import os

    from litellm.proxy.proxy_server import master_key

    salt_key: Final = os.getenv("LITELLM_SALT_KEY", None)
    return master_key if salt_key is None else salt_key


_RESPONSE_PAYLOAD_ADAPTER: Final = TypeAdapter(Mapping[str, object])
_ResponseObjectT = TypeVar("_ResponseObjectT", bound=BaseLiteLLMOpenAIResponseObject)
_AddressedResultT = TypeVar("_AddressedResultT", bound=ResponsesAPIResponse | DeleteResponseResult)


def _response_payload(response_obj: object) -> Mapping[str, object] | None:
    try:
        return _RESPONSE_PAYLOAD_ADAPTER.validate_python(response_obj)
    except ValidationError:
        return None


def _rewrite_advertised_id(
    event: _ResponseObjectT,
    rewrite: Callable[[str], str],
) -> _ResponseObjectT:
    event_id: Final = getattr(event, "id", None)
    if isinstance(event_id, str) and event_id.startswith("resp_"):
        setattr(event, "id", rewrite(event_id))
        return event

    nested: Final = getattr(event, "response", None)
    if isinstance(nested, ResponsesAPIResponse):
        setattr(nested, "id", rewrite(nested.id))
        setattr(event, "response", nested)
        return event

    payload: Final = _response_payload(nested)
    if payload is None:
        return event

    payload_id: Final = payload.get("id")
    if not isinstance(payload_id, str):
        return event

    rewritten: Final = {**payload, "id": rewrite(payload_id)}
    setattr(event, "response", rewritten)
    return event


def _streams_response_ids(request_route: str | None) -> bool:
    if request_route is None:
        return False
    canonical: Final = (
        request_route[len(_RESPONSES_API_PROVIDER_PREFIX) :]
        if request_route.startswith(_RESPONSES_API_PROVIDER_PREFIX + "/")
        else request_route
    )
    parent, _, response_id = canonical.rpartition("/")
    return canonical in _RESPONSES_API_ROUTES or (parent in _RESPONSES_API_ROUTES and response_id != "")


_NO_HELD_IDS: Final[Mapping[str, str]] = MappingProxyType({})


def _upstream_response_id(response_id: str) -> str:
    return ResponsesAPIRequestUtils.decode_previous_response_id_to_original_previous_response_id(response_id)


def _names_the_same_response(response_id: str | None, original_response_id: str) -> bool:
    return response_id is not None and _upstream_response_id(response_id) == _upstream_response_id(original_response_id)


class ResponsesIDSecurity(CustomLogger):
    def __init__(
        self,
        general_settings_reader: Callable[[], Mapping[str, object]] = _proxy_general_settings,
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
        addressed_id_field: Final = _ADDRESSED_RESPONSE_ID_FIELD_BY_CALL_TYPE.get(call_type)
        if addressed_id_field is None:
            return None
        retained_id: Final = data.get(ADDRESSED_RESPONSE_ID_FIELD)
        addressed_id: Final = (
            retained_id if isinstance(retained_id, str) and retained_id else data.get(addressed_id_field)
        )
        if not isinstance(addressed_id, str) or not addressed_id:
            return data
        authorized_id: Final = self.authorize_response_id(addressed_id, user_api_key_dict)
        data[addressed_id_field] = authorized_id
        data[ADDRESSED_RESPONSE_ID_FIELD] = addressed_id
        return data

    def authorize_response_id(
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
        response: _ResponseObjectT,
        user_api_key_dict: "UserAPIKeyAuth",
        request_cache: dict[str, str] | None = None,
    ) -> _ResponseObjectT:
        encrypt: Final = self._response_id_encrypter(user_api_key_dict, request_cache)
        return response if encrypt is None else _rewrite_advertised_id(response, encrypt)

    def _response_id_encrypter(
        self,
        user_api_key_dict: "UserAPIKeyAuth",
        request_cache: dict[str, str] | None,
    ) -> Callable[[str], str] | None:
        signing_key: Final = self._get_signing_key()
        if signing_key is None:
            verbose_proxy_logger.debug(
                "Response ID encryption is enabled but no signing key is configured. "
                "Please set LITELLM_SALT_KEY environment variable or configure a master_key. "
                "Skipping response ID encryption. "
                "See: https://docs.litellm.ai/docs/proxy/prod#5-set-litellm-salt-key"
            )
            return None

        def encrypt(original_id: str) -> str:
            cached: Final = request_cache.get(original_id) if request_cache is not None else None
            if cached is not None:
                return cached

            managed_id: Final = SpecialEnums.LITELLM_MANAGED_RESPONSE_API_RESPONSE_ID_COMPLETE_STR.value.format(
                original_id,
                user_api_key_dict.user_id or "",
                user_api_key_dict.team_id or "",
            )
            encrypted_id: Final = f"resp_{encrypt_value_helper(value=managed_id)}"
            if request_cache is not None:
                request_cache[original_id] = encrypted_id
            return encrypted_id

        return encrypt

    def _streamed_response_id_advertiser(
        self,
        user_api_key_dict: "UserAPIKeyAuth",
        request_data: Mapping[str, object],
    ) -> Callable[[str], str] | None:
        if not _streams_response_ids(user_api_key_dict.request_route):
            return None
        if self._general_settings_reader().get("disable_responses_id_security", False):
            return None
        encrypt: Final = self._response_id_encrypter(user_api_key_dict, request_cache={})
        if encrypt is None:
            return None
        logging_obj: Final = request_data.get("litellm_logging_obj")
        held_ids: Final = self._ids_the_caller_holds(request_data.get(ADDRESSED_RESPONSE_ID_FIELD))

        def advertise(original_id: str) -> str:
            logged_id: Final = encrypt(original_id)
            if isinstance(logging_obj, LiteLLMLoggingObj):
                logging_obj.client_facing_response_id = logged_id
            return held_ids.get(original_id, logged_id)

        return advertise

    def _ids_the_caller_holds(self, addressed_id: object) -> Mapping[str, str]:
        if not isinstance(addressed_id, str) or not self._is_encrypted_response_id(addressed_id):
            return _NO_HELD_IDS
        original_response_id, _, _ = self._decrypt_response_id(addressed_id)
        return MappingProxyType({original_response_id: addressed_id})

    def _id_the_caller_holds(self, response_id: str | None, addressed_id: object) -> str | None:
        return next(
            (
                held_id
                for original_response_id, held_id in self._ids_the_caller_holds(addressed_id).items()
                if _names_the_same_response(response_id, original_response_id)
            ),
            None,
        )

    def _advertised_response(
        self,
        response: _AddressedResultT,
        user_api_key_dict: "UserAPIKeyAuth",
        request_data: Mapping[str, object],
    ) -> _AddressedResultT:
        held_id: Final = self._id_the_caller_holds(response.id, request_data.get(ADDRESSED_RESPONSE_ID_FIELD))
        logged_response: Final = self._encrypt_response_id(response, user_api_key_dict, request_cache=None)
        return logged_response if held_id is None else logged_response.model_copy(update={"id": held_id})

    async def async_post_call_success_hook(
        self,
        data: dict[str, object],
        user_api_key_dict: "UserAPIKeyAuth",
        response: LLMResponseTypes,
    ) -> LLMResponseTypes:
        general_settings: Final = self._general_settings_reader()

        if general_settings.get("disable_responses_id_security", False):
            return response
        if isinstance(response, ResponsesAPIResponse | DeleteResponseResult):
            return self._advertised_response(response, user_api_key_dict, data)
        return response

    async def async_post_call_streaming_iterator_hook(
        self, user_api_key_dict: "UserAPIKeyAuth", response: Any, request_data: dict[str, object]
    ) -> AsyncGenerator[BaseLiteLLMOpenAIResponseObject, None]:
        advertise: Final = self._streamed_response_id_advertiser(user_api_key_dict, request_data)
        async for chunk in response:
            yield (
                _rewrite_advertised_id(chunk, advertise)
                if advertise is not None and isinstance(chunk, BaseLiteLLMOpenAIResponseObject)
                else chunk
            )


def decrypted_response_id(response_id: str) -> str:
    """The id a client-facing response id stands for, the same for every encryption of one response."""
    original_response_id, _, _ = ResponsesIDSecurity()._decrypt_response_id(response_id)
    return original_response_id
