"""
Security hook to prevent user B from seeing response from user A.

This hook uses the DBSpendUpdateWriter to batch-write response IDs to the database
instead of writing immediately on each request.
"""

from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
)

from litellm.proxy.hooks.base_id_security import BaseIDSecurity
from litellm.types.llms.openai import (
    BaseLiteLLMOpenAIResponseObject,
    ResponsesAPIResponse,
)
from litellm.types.utils import SpecialEnums

if TYPE_CHECKING:
    from litellm.caching.caching import DualCache
    from litellm.proxy._types import UserAPIKeyAuth


class ResponsesIDSecurity(BaseIDSecurity):
    """Security hook for Responses API to prevent unauthorized access to response IDs."""

    @property
    def resource_name(self) -> str:
        return "response"

    @property
    def id_prefix(self) -> str:
        return "resp_"

    @property
    def api_call_types(self) -> Set[str]:
        return {
            "aresponses",
            "aget_responses",
            "adelete_responses",
            "acancel_responses",
        }

    @property
    def special_enum_format(self) -> str:
        return SpecialEnums.LITELLM_MANAGED_RESPONSE_API_RESPONSE_ID_COMPLETE_STR.value

    @property
    def response_types(self) -> Tuple[type, ...]:
        return (ResponsesAPIResponse, BaseLiteLLMOpenAIResponseObject)

    @property
    def id_field_name(self) -> str:
        return "response_id"

    def get_resource_id_from_response(self, response: Any) -> Optional[str]:
        """Extract response ID from response object."""
        # First check direct id attribute
        response_id = getattr(response, "id", None)
        if response_id:
            return response_id
        
        # Check nested response object for ResponsesAPIResponse
        response_obj = getattr(response, "response", None)
        if response_obj and isinstance(response_obj, ResponsesAPIResponse):
            return getattr(response_obj, "id", None)
        
        return None

    def set_resource_id_in_response(self, response: Any, resource_id: str) -> Any:
        """Set response ID in response object."""
        # Handle direct id attribute
        if hasattr(response, "id") and getattr(response, "id", None):
            setattr(response, "id", resource_id)
        
        # Handle nested response object for ResponsesAPIResponse
        response_obj = getattr(response, "response", None)
        if response_obj and isinstance(response_obj, ResponsesAPIResponse):
            setattr(response_obj, "id", resource_id)
            setattr(response, "response", response_obj)
        
        return response

    # Backward compatibility aliases for existing tests and code
    def _is_encrypted_response_id(self, response_id: str) -> bool:
        """Alias for backward compatibility. Use _is_encrypted_id instead."""
        return self._is_encrypted_id(response_id)

    def _decrypt_response_id(self, response_id: str):
        """Alias for backward compatibility. Use _decrypt_id instead."""
        return self._decrypt_id(response_id)

    def _encrypt_response_id(self, response: Any, user_api_key_dict: "UserAPIKeyAuth") -> Any:
        """Alias for backward compatibility. Use _encrypt_id instead."""
        return self._encrypt_id(response, user_api_key_dict)

    def check_user_access_to_response_id(
        self,
        response_id_user_id: Optional[str],
        response_id_team_id: Optional[str],
        user_api_key_dict: "UserAPIKeyAuth",
    ) -> bool:
        """Alias for backward compatibility. Use check_user_access_to_resource_id instead."""
        return self.check_user_access_to_resource_id(
            response_id_user_id, response_id_team_id, user_api_key_dict
        )

    # Override async_pre_call_hook to ensure it's in this class's __dict__ for hook detection
    async def async_pre_call_hook(
        self,
        user_api_key_dict: "UserAPIKeyAuth",
        cache: "DualCache",
        data: dict,
        call_type: str,
    ) -> Optional[Union[Exception, str, dict]]:
        """Pre-call hook to verify user has access to the response they're trying to access."""
        return await super().async_pre_call_hook(user_api_key_dict, cache, data, call_type)

    async def async_post_call_streaming_iterator_hook(  # type: ignore
        self, user_api_key_dict: "UserAPIKeyAuth", response: Any, request_data: dict
    ) -> AsyncGenerator[BaseLiteLLMOpenAIResponseObject, None]:
        """Stream hook to encrypt response IDs in real-time streaming responses."""
        from litellm.proxy.proxy_server import general_settings

        async for chunk in response:
            if (
                isinstance(chunk, BaseLiteLLMOpenAIResponseObject)
                and user_api_key_dict.request_route
                == "/v1/responses"  # only encrypt the response id for the responses api
                and not general_settings.get(self.disable_security_setting_key, False)
            ):
                chunk = self._encrypt_id(chunk, user_api_key_dict)
            yield chunk
