"""
Security hook to prevent user B from seeing video from user A.

This hook uses encryption to embed user_id and team_id into video IDs,
ensuring that only the creating user/team can access the video.
"""

from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    Set,
    Tuple,
    Union,
)

from litellm.proxy.hooks.base_id_security import BaseIDSecurity
from litellm.types.utils import SpecialEnums
from litellm.types.videos.main import VideoObject

if TYPE_CHECKING:
    from litellm.caching.caching import DualCache
    from litellm.proxy._types import UserAPIKeyAuth


class VideoIDSecurity(BaseIDSecurity):
    """Security hook for Videos API to prevent unauthorized access to video IDs."""

    @property
    def resource_name(self) -> str:
        return "video"

    @property
    def id_prefix(self) -> str:
        return "video_"

    @property
    def api_call_types(self) -> Set[str]:
        return {
            "avideo_status",
            "avideo_content",
            "avideo_remix",
            "avideo_delete",
        }

    @property
    def special_enum_format(self) -> str:
        return SpecialEnums.LITELLM_MANAGED_VIDEO_API_VIDEO_ID_COMPLETE_STR.value

    @property
    def response_types(self) -> Tuple[type, ...]:
        return (VideoObject,)

    @property
    def id_field_name(self) -> str:
        return "video_id"

    def get_resource_id_from_response(self, response: Any) -> Optional[str]:
        """Extract video ID from response object."""
        return getattr(response, "id", None)

    def set_resource_id_in_response(self, response: Any, resource_id: str) -> Any:
        """Set video ID in response object."""
        setattr(response, "id", resource_id)
        return response

    # Override async_pre_call_hook to ensure it's in this class's __dict__ for hook detection
    async def async_pre_call_hook(
        self,
        user_api_key_dict: "UserAPIKeyAuth",
        cache: "DualCache",
        data: dict,
        call_type: str,
    ) -> Optional[Union[Exception, str, dict]]:
        """Pre-call hook to verify user has access to the video they're trying to access."""
        return await super().async_pre_call_hook(user_api_key_dict, cache, data, call_type)

    # Backward compatibility aliases for existing tests and code
    def _is_encrypted_video_id(self, video_id: str) -> bool:
        """Alias for backward compatibility. Use _is_encrypted_id instead."""
        return self._is_encrypted_id(video_id)

    def _decrypt_video_id(self, video_id: str):
        """Alias for backward compatibility. Use _decrypt_id instead."""
        return self._decrypt_id(video_id)

    def _encrypt_video_id(self, response: VideoObject, user_api_key_dict: "UserAPIKeyAuth") -> VideoObject:
        """Alias for backward compatibility. Use _encrypt_id instead."""
        return self._encrypt_id(response, user_api_key_dict)

    def check_user_access_to_video_id(
        self,
        video_id_user_id: Optional[str],
        video_id_team_id: Optional[str],
        user_api_key_dict: "UserAPIKeyAuth",
    ) -> bool:
        """Alias for backward compatibility. Use check_user_access_to_resource_id instead."""
        return self.check_user_access_to_resource_id(
            video_id_user_id, video_id_team_id, user_api_key_dict
        )

