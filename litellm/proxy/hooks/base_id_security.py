"""
Base class for ID security across different API endpoints.

This base class provides common functionality for encrypting and decrypting
resource IDs (video_id, response_id, batch_id, etc.) with embedded user_id and team_id
to prevent unauthorized access across users and teams.
"""

from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Literal,
    Optional,
    Set,
    Tuple,
    Union,
)

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import LitellmUserRoles
from litellm.proxy.common_utils.encrypt_decrypt_utils import (
    decrypt_value_helper,
    encrypt_value_helper,
)
from litellm.types.utils import SpecialEnums

if TYPE_CHECKING:
    from litellm.caching.caching import DualCache
    from litellm.proxy._types import UserAPIKeyAuth


class BaseIDSecurity(CustomLogger, ABC):
    """
    Abstract base class for implementing ID security across different endpoints.
    
    Subclasses must implement:
    - resource_name: Name of the resource (e.g., "video", "response")
    - id_prefix: ID prefix (e.g., "vid_", "resp_")
    - api_call_types: Set of call types to intercept
    - special_enum_format: SpecialEnums format string for encryption
    - response_types: Tuple of response types to encrypt
    - get_resource_id_from_response: Extract ID from response object
    - set_resource_id_in_response: Set ID in response object
    """

    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def resource_name(self) -> str:
        """Name of the resource (e.g., 'video', 'response')"""
        pass

    @property
    @abstractmethod
    def id_prefix(self) -> str:
        """ID prefix for this resource (e.g., 'vid_', 'resp_')"""
        pass

    @property
    @abstractmethod
    def api_call_types(self) -> Set[str]:
        """Set of API call types to intercept (e.g., {'avideo_status', 'avideo_content'})"""
        pass

    @property
    @abstractmethod
    def special_enum_format(self) -> str:
        """SpecialEnums format string for encryption"""
        pass

    @property
    @abstractmethod
    def response_types(self) -> Tuple[type, ...]:
        """Tuple of response types that should be encrypted"""
        pass

    @property
    @abstractmethod
    def id_field_name(self) -> str:
        """Name of the ID field in the data dict (e.g., 'video_id', 'response_id')"""
        pass

    @property
    def disable_security_setting_key(self) -> str:
        """Key for disabling security in general_settings"""
        return f"disable_{self.resource_name}_id_security"

    @property
    def decrypt_key(self) -> str:
        """Key used for decryption"""
        return f"{self.resource_name}_id"

    @abstractmethod
    def get_resource_id_from_response(self, response: Any) -> Optional[str]:
        """Extract the resource ID from a response object"""
        pass

    @abstractmethod
    def set_resource_id_in_response(self, response: Any, resource_id: str) -> Any:
        """Set the resource ID in a response object"""
        pass

    async def async_pre_call_hook(
        self,
        user_api_key_dict: "UserAPIKeyAuth",
        cache: "DualCache",
        data: dict,
        call_type: str,  # Changed from Literal to str to support all call types
    ) -> Optional[Union[Exception, str, dict]]:
        """
        Pre-call hook to verify user has access to the resource they're trying to access.
        """
        if call_type not in self.api_call_types:
            return None

        # Handle different field names for different call types
        resource_id = self._get_resource_id_from_data(data, call_type)

        if resource_id and self._is_encrypted_id(resource_id):
            original_id, user_id, team_id = self._decrypt_id(resource_id)
            
            self.check_user_access_to_resource_id(user_id, team_id, user_api_key_dict)
            
            # Update the data with the decrypted ID
            self._set_resource_id_in_data(data, call_type, original_id)

        return data

    def _get_resource_id_from_data(self, data: dict, call_type: str) -> Optional[str]:
        """
        Extract resource ID from data dict. Can be overridden for custom field names.
        Default checks both id_field_name and 'previous_{id_field_name}'.
        """
        # Check primary field
        resource_id = data.get(self.id_field_name)
        if resource_id:
            return resource_id
        
        # Check previous_ variant (for responses API)
        previous_field = f"previous_{self.id_field_name}"
        return data.get(previous_field)

    def _set_resource_id_in_data(self, data: dict, call_type: str, resource_id: str) -> None:
        """
        Set resource ID in data dict. Can be overridden for custom field names.
        """
        # Set in primary field if it exists
        if self.id_field_name in data:
            data[self.id_field_name] = resource_id
        
        # Set in previous_ variant if it exists
        previous_field = f"previous_{self.id_field_name}"
        if previous_field in data:
            data[previous_field] = resource_id

    def check_user_access_to_resource_id(
        self,
        resource_id_user_id: Optional[str],
        resource_id_team_id: Optional[str],
        user_api_key_dict: "UserAPIKeyAuth",
    ) -> bool:
        """
        Verify that the user making the request has permission to access the resource.
        
        Args:
            resource_id_user_id: The user_id encoded in the resource_id
            resource_id_team_id: The team_id encoded in the resource_id
            user_api_key_dict: The current user's API key information
            
        Returns:
            True if access is allowed
            
        Raises:
            HTTPException: If access is forbidden
        """
        from litellm.proxy.proxy_server import general_settings

        # Proxy admins can access any resource
        if (
            user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN.value
            or user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN
        ):
            return True

        # Check if user_id matches
        if resource_id_user_id and resource_id_user_id != user_api_key_dict.user_id:
            if general_settings.get(self.disable_security_setting_key, False):
                verbose_proxy_logger.debug(
                    f"{self.resource_name.title()} ID Security is disabled. User {user_api_key_dict.user_id} is accessing {self.resource_name} id {resource_id_user_id} which is not associated with them."
                )
                return True
            raise HTTPException(
                status_code=403,
                detail=f"Forbidden. The {self.resource_name} id is not associated with the user, who this key belongs to. To disable this security feature, set general_settings::{self.disable_security_setting_key} to True in the config.yaml file.",
            )

        # Check if team_id matches
        if resource_id_team_id and resource_id_team_id != user_api_key_dict.team_id:
            if general_settings.get(self.disable_security_setting_key, False):
                verbose_proxy_logger.debug(
                    f"{self.resource_name.title()} ID Security is disabled. {self.resource_name.title()} belongs to team {resource_id_team_id} but user {user_api_key_dict.user_id} is accessing it with team id {user_api_key_dict.team_id}."
                )
                return True
            raise HTTPException(
                status_code=403,
                detail=f"Forbidden. The {self.resource_name} id is not associated with the team, who this key belongs to. To disable this security feature, set general_settings::{self.disable_security_setting_key} to True in the config.yaml file.",
            )

        return True

    def _is_encrypted_id(self, resource_id: str) -> bool:
        """
        Check if a resource_id is encrypted (contains user/team information).
        
        Args:
            resource_id: The resource_id to check
            
        Returns:
            True if the resource_id is encrypted, False otherwise
        """
        split_result = resource_id.split(self.id_prefix)
        if len(split_result) < 2:
            return False

        remaining_string = split_result[1]
        decrypted_value = decrypt_value_helper(
            value=remaining_string, key=self.decrypt_key, return_original_value=True
        )

        if decrypted_value is None:
            return False

        if decrypted_value.startswith(SpecialEnums.LITELM_MANAGED_FILE_ID_PREFIX.value):
            return True
        return False

    def _decrypt_id(
        self, resource_id: str
    ) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Decrypt a resource_id to extract the original resource_id, user_id, and team_id.
        
        Args:
            resource_id: The encrypted resource_id
            
        Returns:
            Tuple of (original_resource_id, user_id, team_id)
        """
        split_result = resource_id.split(self.id_prefix)
        if len(split_result) < 2:
            return resource_id, None, None

        remaining_string = split_result[1]
        decrypted_value = decrypt_value_helper(
            value=remaining_string, key=self.decrypt_key, return_original_value=True
        )

        if decrypted_value is None:
            return resource_id, None, None

        if decrypted_value.startswith(SpecialEnums.LITELM_MANAGED_FILE_ID_PREFIX.value):
            # Expected format: "litellm_proxy:{api_name}:{id_field}:{id};user_id:{user_id};team_id:{team_id}"
            parts = decrypted_value.split(";")

            if len(parts) >= 3:
                # Extract resource_id from first part
                resource_id_part = parts[0]
                original_resource_id = resource_id_part.split(f"{self.id_field_name}:")[-1]

                # Extract user_id from "user_id:{user_id}"
                user_id_part = parts[1]
                user_id = user_id_part.split("user_id:")[-1]

                # Extract team_id from "team_id:{team_id}"
                team_id_part = parts[2]
                team_id = team_id_part.split("team_id:")[-1]

                return original_resource_id, user_id, team_id
            else:
                # Fallback if format is unexpected
                return resource_id, None, None
        return resource_id, None, None

    def _encrypt_id(
        self,
        response: Any,
        user_api_key_dict: "UserAPIKeyAuth",
    ) -> Any:
        """
        Encrypt the resource_id in the response by embedding user_id and team_id.
        
        Args:
            response: The API response object
            user_api_key_dict: The user's API key information
            
        Returns:
            The response with encrypted resource_id
        """
        resource_id = self.get_resource_id_from_response(response)

        if (
            resource_id
            and isinstance(resource_id, str)
            and resource_id.startswith(self.id_prefix)
        ):
            encrypted_resource_id = self.special_enum_format.format(
                resource_id,
                user_api_key_dict.user_id or "",
                user_api_key_dict.team_id or "",
            )

            encoded_id = encrypt_value_helper(value=encrypted_resource_id)
            new_id = f"{self.id_prefix}{encoded_id}"
            
            return self.set_resource_id_in_response(response, new_id)

        return response

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: "UserAPIKeyAuth",
        response: Any,
    ) -> Any:
        """
        Post-call hook to encrypt resource IDs in responses.
        This ensures that resource IDs returned to users contain embedded user/team information.
        """
        from litellm.proxy.proxy_server import general_settings

        if general_settings.get(self.disable_security_setting_key, False):
            return response

        # Check if response matches any of our response types
        for response_type in self.response_types:
            if isinstance(response, response_type):
                response = self._encrypt_id(response, user_api_key_dict)
                break

        return response

