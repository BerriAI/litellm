"""
Security hook to prevent user B from seeing video from user A.

This hook uses encryption to embed user_id and team_id into video IDs,
ensuring that only the creating user/team can access the video.
"""

from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Optional,
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
from litellm.types.videos.main import VideoObject

if TYPE_CHECKING:
    from litellm.caching.caching import DualCache
    from litellm.proxy._types import UserAPIKeyAuth


class VideoIDSecurity(CustomLogger):
    def __init__(self):
        pass

    async def async_pre_call_hook(
        self,
        user_api_key_dict: "UserAPIKeyAuth",
        cache: "DualCache",
        data: dict,
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "pass_through_endpoint",
            "rerank",
            "mcp_call",
            "anthropic_messages",
        ],
    ) -> Optional[Union[Exception, str, dict]]:
        """
        Pre-call hook to verify user has access to the video they're trying to access.
        Checks video_id in status, content, remix, and delete operations.
        """
        video_api_call_types = {
            "avideo_status",
            "avideo_content",
            "avideo_remix",
            "avideo_delete",
        }
        if call_type not in video_api_call_types:
            return None

        video_id = data.get("video_id")

        if video_id and self._is_encrypted_video_id(video_id):
            original_video_id, user_id, team_id = self._decrypt_video_id(video_id)

            self.check_user_access_to_video_id(user_id, team_id, user_api_key_dict)
            data["video_id"] = original_video_id

        return data

    def check_user_access_to_video_id(
        self,
        video_id_user_id: Optional[str],
        video_id_team_id: Optional[str],
        user_api_key_dict: "UserAPIKeyAuth",
    ) -> bool:
        """
        Verify that the user making the request has permission to access the video.
        
        Args:
            video_id_user_id: The user_id encoded in the video_id
            video_id_team_id: The team_id encoded in the video_id
            user_api_key_dict: The current user's API key information
            
        Returns:
            True if access is allowed
            
        Raises:
            HTTPException: If access is forbidden
        """
        from litellm.proxy.proxy_server import general_settings

        # Proxy admins can access any video
        if (
            user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN.value
            or user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN
        ):
            return True

        # Check if user_id matches
        if video_id_user_id and video_id_user_id != user_api_key_dict.user_id:
            if general_settings.get("disable_video_id_security", False):
                verbose_proxy_logger.debug(
                    f"Video ID Security is disabled. User {user_api_key_dict.user_id} is accessing video id {video_id_user_id} which is not associated with them."
                )
                return True
            raise HTTPException(
                status_code=403,
                detail="Forbidden. The video id is not associated with the user, who this key belongs to. To disable this security feature, set general_settings::disable_video_id_security to True in the config.yaml file.",
            )

        # Check if team_id matches
        if video_id_team_id and video_id_team_id != user_api_key_dict.team_id:
            if general_settings.get("disable_video_id_security", False):
                verbose_proxy_logger.debug(
                    f"Video ID Security is disabled. Video belongs to team {video_id_team_id} but user {user_api_key_dict.user_id} is accessing it with team id {user_api_key_dict.team_id}."
                )
                return True
            raise HTTPException(
                status_code=403,
                detail="Forbidden. The video id is not associated with the team, who this key belongs to. To disable this security feature, set general_settings::disable_video_id_security to True in the config.yaml file.",
            )

        return True

    def _is_encrypted_video_id(self, video_id: str) -> bool:
        """
        Check if a video_id is encrypted (contains user/team information).
        
        Args:
            video_id: The video_id to check
            
        Returns:
            True if the video_id is encrypted, False otherwise
        """
        split_result = video_id.split("vid_")
        if len(split_result) < 2:
            return False

        remaining_string = split_result[1]
        decrypted_value = decrypt_value_helper(
            value=remaining_string, key="video_id", return_original_value=True
        )

        if decrypted_value is None:
            return False

        if decrypted_value.startswith(SpecialEnums.LITELM_MANAGED_FILE_ID_PREFIX.value):
            return True
        return False

    def _decrypt_video_id(
        self, video_id: str
    ) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Decrypt a video_id to extract the original video_id, user_id, and team_id.
        
        Args:
            video_id: The encrypted video_id
            
        Returns:
            Tuple of (original_video_id, user_id, team_id)
        """
        split_result = video_id.split("vid_")
        if len(split_result) < 2:
            return video_id, None, None

        remaining_string = split_result[1]
        decrypted_value = decrypt_value_helper(
            value=remaining_string, key="video_id", return_original_value=True
        )

        if decrypted_value is None:
            return video_id, None, None

        if decrypted_value.startswith(SpecialEnums.LITELM_MANAGED_FILE_ID_PREFIX.value):
            # Expected format: "litellm_proxy:videos_api:video_id:{video_id};user_id:{user_id};team_id:{team_id}"
            parts = decrypted_value.split(";")

            if len(parts) >= 3:
                # Extract video_id from "litellm_proxy:videos_api:video_id:{video_id}"
                video_id_part = parts[0]
                original_video_id = video_id_part.split("video_id:")[-1]

                # Extract user_id from "user_id:{user_id}"
                user_id_part = parts[1]
                user_id = user_id_part.split("user_id:")[-1]

                # Extract team_id from "team_id:{team_id}"
                team_id_part = parts[2]
                team_id = team_id_part.split("team_id:")[-1]

                return original_video_id, user_id, team_id
            else:
                # Fallback if format is unexpected
                return video_id, None, None
        return video_id, None, None

    def _encrypt_video_id(
        self,
        response: VideoObject,
        user_api_key_dict: "UserAPIKeyAuth",
    ) -> VideoObject:
        """
        Encrypt the video_id in the response by embedding user_id and team_id.
        
        Args:
            response: The video generation response
            user_api_key_dict: The user's API key information
            
        Returns:
            The response with encrypted video_id
        """
        # encrypt the video id using the symmetric key
        # encrypt the video id, and encode the user id and video id
        video_id = getattr(response, "id", None)

        if video_id and isinstance(video_id, str) and video_id.startswith("vid_"):
            encrypted_video_id = SpecialEnums.LITELLM_MANAGED_VIDEO_API_VIDEO_ID_COMPLETE_STR.value.format(
                video_id,
                user_api_key_dict.user_id or "",
                user_api_key_dict.team_id or "",
            )

            encoded_user_id_and_video_id = encrypt_value_helper(
                value=encrypted_video_id
            )
            setattr(
                response, "id", f"vid_{encoded_user_id_and_video_id}"
            )  # maintain the 'vid_' prefix for the videos api video id

        return response

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: "UserAPIKeyAuth",
        response: Any,
    ) -> Any:
        """
        Post-call hook to encrypt video IDs in responses.
        This ensures that video IDs returned to users contain embedded user/team information.
        """
        from litellm.proxy.proxy_server import general_settings

        if general_settings.get("disable_video_id_security", False):
            return response

        if isinstance(response, VideoObject):
            response = self._encrypt_video_id(response, user_api_key_dict)

        return response

