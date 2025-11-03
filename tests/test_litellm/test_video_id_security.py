"""
Tests for Video ID Security hook to ensure users can only access their own videos.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi import HTTPException

from litellm.proxy.hooks.video_id_security import VideoIDSecurity
from litellm.proxy._types import UserAPIKeyAuth, LitellmUserRoles
from litellm.types.videos.main import VideoObject
from litellm.types.utils import SpecialEnums


@pytest.fixture
def video_id_security():
    """Fixture that creates a VideoIDSecurity instance."""
    return VideoIDSecurity()


@pytest.fixture
def mock_user_api_key_dict():
    """Fixture that creates a mock UserAPIKeyAuth object."""
    mock_auth = MagicMock()
    mock_auth.user_id = "test-user-123"
    mock_auth.team_id = "test-team-123"
    mock_auth.user_role = LitellmUserRoles.INTERNAL_USER.value
    return mock_auth


@pytest.fixture
def mock_cache():
    """Fixture that creates a mock DualCache object."""
    return MagicMock()


class TestVideoIDSecurity:
    """Test the VideoIDSecurity hook functionality."""

    def test_encrypt_video_id(self, video_id_security, mock_user_api_key_dict):
        """Test that video IDs are properly encrypted with user/team information."""
        response = VideoObject(
            id="vid_test123",
            object="video",
            status="completed",
            created_at=1234567890,
            model="openai/sora-2",
        )
        
        with patch(
            "litellm.proxy.hooks.video_id_security.encrypt_value_helper"
        ) as mock_encrypt:
            mock_encrypt.return_value = "encrypted_base64_value"
            
            encrypted_response = video_id_security._encrypt_video_id(
                response, mock_user_api_key_dict
            )
            
            assert encrypted_response.id.startswith("vid_")
            assert encrypted_response.id == "vid_encrypted_base64_value"
            mock_encrypt.assert_called_once()

    def test_is_encrypted_video_id_valid(self, video_id_security):
        """Test that a properly encrypted video ID is identified correctly."""
        with patch(
            "litellm.proxy.hooks.video_id_security.decrypt_value_helper"
        ) as mock_decrypt:
            mock_decrypt.return_value = (
                f"{SpecialEnums.LITELM_MANAGED_FILE_ID_PREFIX.value}:videos_api:video_id:vid_123;user_id:user-456;team_id:team-789"
            )
            
            result = video_id_security._is_encrypted_video_id("vid_encrypted_value")
            
            assert result is True

    def test_is_encrypted_video_id_invalid(self, video_id_security):
        """Test that an unencrypted video ID returns False."""
        with patch(
            "litellm.proxy.hooks.video_id_security.decrypt_value_helper"
        ) as mock_decrypt:
            mock_decrypt.return_value = None
            
            result = video_id_security._is_encrypted_video_id("vid_plain_value")
            
            assert result is False

    def test_decrypt_video_id_valid(self, video_id_security):
        """Test decrypting a valid encrypted video ID."""
        with patch(
            "litellm.proxy.hooks.video_id_security.decrypt_value_helper"
        ) as mock_decrypt:
            mock_decrypt.return_value = (
                f"{SpecialEnums.LITELM_MANAGED_FILE_ID_PREFIX.value}:videos_api:video_id:vid_original_123;user_id:user-456;team_id:team-789"
            )
            
            original_id, user_id, team_id = video_id_security._decrypt_video_id(
                "vid_encrypted_value"
            )
            
            assert original_id == "vid_original_123"
            assert user_id == "user-456"
            assert team_id == "team-789"

    def test_decrypt_video_id_no_encryption(self, video_id_security):
        """Test decrypting a non-encrypted video ID."""
        with patch(
            "litellm.proxy.hooks.video_id_security.decrypt_value_helper"
        ) as mock_decrypt:
            mock_decrypt.return_value = None
            
            original_id, user_id, team_id = video_id_security._decrypt_video_id(
                "vid_plain_value"
            )
            
            assert original_id == "vid_plain_value"
            assert user_id is None
            assert team_id is None

    def test_check_user_access_same_user(
        self, video_id_security, mock_user_api_key_dict
    ):
        """Test that a user can access their own video."""
        assert video_id_security.check_user_access_to_video_id(
            video_id_user_id="test-user-123",
            video_id_team_id="test-team-123",
            user_api_key_dict=mock_user_api_key_dict,
        )

    def test_check_user_access_different_user(
        self, video_id_security, mock_user_api_key_dict
    ):
        """Test that a user cannot access another user's video."""
        with patch("litellm.proxy.proxy_server.general_settings", {}):
            with pytest.raises(HTTPException) as exc_info:
                video_id_security.check_user_access_to_video_id(
                    video_id_user_id="different-user-999",
                    video_id_team_id="test-team-123",
                    user_api_key_dict=mock_user_api_key_dict,
                )
            
            assert exc_info.value.status_code == 403
            assert "Forbidden" in exc_info.value.detail

    def test_check_user_access_different_team(
        self, video_id_security, mock_user_api_key_dict
    ):
        """Test that a user from a different team cannot access the video."""
        with patch("litellm.proxy.proxy_server.general_settings", {}):
            with pytest.raises(HTTPException) as exc_info:
                video_id_security.check_user_access_to_video_id(
                    video_id_user_id="test-user-123",
                    video_id_team_id="different-team-999",
                    user_api_key_dict=mock_user_api_key_dict,
                )
            
            assert exc_info.value.status_code == 403
            assert "Forbidden" in exc_info.value.detail

    def test_check_user_access_proxy_admin(self, video_id_security):
        """Test that proxy admins can access any video."""
        mock_admin_auth = MagicMock()
        mock_admin_auth.user_id = "admin-user"
        mock_admin_auth.team_id = "admin-team"
        mock_admin_auth.user_role = LitellmUserRoles.PROXY_ADMIN.value
        
        assert video_id_security.check_user_access_to_video_id(
            video_id_user_id="some-other-user",
            video_id_team_id="some-other-team",
            user_api_key_dict=mock_admin_auth,
        )

    @pytest.mark.asyncio
    async def test_async_pre_call_hook_video_status(
        self, video_id_security, mock_user_api_key_dict, mock_cache
    ):
        """Test that pre-call hook properly decrypts and validates video IDs."""
        data = {"video_id": "vid_encrypted_value"}
        
        with patch.object(
            video_id_security, "_is_encrypted_video_id", return_value=True
        ):
            with patch.object(
                video_id_security,
                "_decrypt_video_id",
                return_value=("vid_test123", "test-user-123", "test-team-123"),
            ):
                result = await video_id_security.async_pre_call_hook(
                    user_api_key_dict=mock_user_api_key_dict,
                    cache=mock_cache,
                    data=data,
                    call_type="avideo_status",
                )
                
                assert result["video_id"] == "vid_test123"

    @pytest.mark.asyncio
    async def test_async_pre_call_hook_unauthorized_access(
        self, video_id_security, mock_cache
    ):
        """Test that pre-call hook blocks unauthorized access."""
        # User B trying to access User A's video
        mock_user_b = MagicMock()
        mock_user_b.user_id = "user-999"
        mock_user_b.team_id = "team-999"
        mock_user_b.user_role = LitellmUserRoles.INTERNAL_USER.value
        
        data = {"video_id": "vid_encrypted_value"}
        
        with patch.object(
            video_id_security, "_is_encrypted_video_id", return_value=True
        ):
            with patch.object(
                video_id_security,
                "_decrypt_video_id",
                return_value=("vid_test123", "user-123", "team-123"),
            ):
                with patch("litellm.proxy.proxy_server.general_settings", {}):
                    with pytest.raises(HTTPException) as exc_info:
                        await video_id_security.async_pre_call_hook(
                            user_api_key_dict=mock_user_b,
                            cache=mock_cache,
                            data=data,
                            call_type="avideo_status",
                        )
                    
                    assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_async_post_call_success_hook(
        self, video_id_security, mock_user_api_key_dict
    ):
        """Test that post-call hook encrypts video IDs in responses."""
        response = VideoObject(
            id="vid_test123",
            object="video",
            status="completed",
            created_at=1234567890,
            model="openai/sora-2",
        )
        
        with patch.object(
            video_id_security, "_encrypt_video_id", return_value=response
        ) as mock_encrypt:
            result = await video_id_security.async_post_call_success_hook(
                data={},
                user_api_key_dict=mock_user_api_key_dict,
                response=response,
            )
            
            mock_encrypt.assert_called_once_with(response, mock_user_api_key_dict)
            assert result == response
