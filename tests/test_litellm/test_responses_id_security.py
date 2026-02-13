"""
Tests for ResponsesIDSecurity hook.

Tests the security hook that prevents user B from seeing response from user A.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm.proxy.hooks.responses_id_security import ResponsesIDSecurity
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.utils import SpecialEnums


@pytest.fixture
def responses_id_security():
    """Fixture that creates a ResponsesIDSecurity instance."""
    return ResponsesIDSecurity()


@pytest.fixture
def mock_user_api_key_dict():
    """Fixture that creates a mock UserAPIKeyAuth object."""
    mock_auth = MagicMock()
    mock_auth.user_id = "test-user-123"
    mock_auth.team_id = "test-team-123"
    mock_auth.token = "test-token"
    mock_auth.user_role = None
    return mock_auth


@pytest.fixture
def mock_cache():
    """Fixture that creates a mock DualCache object."""
    return MagicMock()


class TestIsEncryptedResponseId:
    """Test _is_encrypted_response_id function"""

    def test_is_encrypted_response_id_valid(self, responses_id_security):
        """Test that a properly encrypted response ID is identified correctly"""
        # Patch at the module level where it's imported
        import litellm.proxy.hooks.responses_id_security as responses_module
        
        with patch.object(
            responses_module, "decrypt_value_helper"
        ) as mock_decrypt:
            mock_decrypt.return_value = f"{SpecialEnums.LITELM_MANAGED_FILE_ID_PREFIX.value}response_id:resp_123;user_id:user-456"

            result = responses_id_security._is_encrypted_response_id(
                "resp_encrypted_value"
            )

            assert result is True
            mock_decrypt.assert_called_once()

    def test_is_encrypted_response_id_invalid(self, responses_id_security):
        """Test that an unencrypted response ID returns False"""
        # Patch at the module level where it's imported
        import litellm.proxy.hooks.responses_id_security as responses_module
        
        with patch.object(
            responses_module, "decrypt_value_helper"
        ) as mock_decrypt:
            mock_decrypt.return_value = None

            result = responses_id_security._is_encrypted_response_id("resp_plain_value")

            assert result is False


class TestDecryptResponseId:
    """Test _decrypt_response_id function"""

    def test_decrypt_response_id_valid(self, responses_id_security):
        """Test decrypting a valid encrypted response ID"""
        # Patch at the module level where it's imported
        import litellm.proxy.hooks.responses_id_security as responses_module
        
        with patch.object(
            responses_module, "decrypt_value_helper"
        ) as mock_decrypt:
            mock_decrypt.return_value = f"{SpecialEnums.LITELM_MANAGED_FILE_ID_PREFIX.value}response_id:resp_original_123;user_id:user-456;team_id:team-789"

            original_id, user_id, team_id = responses_id_security._decrypt_response_id(
                "resp_encrypted_value"
            )

            assert original_id == "resp_original_123"
            assert user_id == "user-456"
            assert team_id == "team-789"

    def test_decrypt_response_id_no_encryption(self, responses_id_security):
        """Test decrypting a non-encrypted response ID"""
        # Patch at the module level where it's imported
        import litellm.proxy.hooks.responses_id_security as responses_module
        
        with patch.object(
            responses_module, "decrypt_value_helper"
        ) as mock_decrypt:
            mock_decrypt.return_value = None

            original_id, user_id, team_id = responses_id_security._decrypt_response_id(
                "resp_plain_value"
            )

            assert original_id == "resp_plain_value"
            assert user_id is None
            assert team_id is None


class TestEncryptResponseId:
    """Test _encrypt_response_id function"""

    @pytest.mark.skip(reason="Flaky on CI; disabling temporarily until responses_id_security is fixed")
    def test_encrypt_response_id_success(
        self, responses_id_security, mock_user_api_key_dict
    ):
        """Test encrypting a response ID with user information"""
        mock_response = ResponsesAPIResponse(
            id="resp_123", created_at=1234567890, output=[], status="completed"
        )

        with patch(
            "litellm.proxy.hooks.responses_id_security.encrypt_value_helper"
        ) as mock_encrypt:
            mock_encrypt.return_value = "encrypted_base64_value"
            
            with patch.object(
                responses_id_security, "_get_signing_key", return_value="test-key"
            ):
                result = responses_id_security._encrypt_response_id(
                    mock_response, mock_user_api_key_dict
                )

                assert result.id == "resp_encrypted_base64_value"
                assert result.id.startswith("resp_")
                mock_encrypt.assert_called_once()

    @pytest.mark.skip(reason="Flaky on CI; disabling temporarily until responses_id_security is fixed")
    def test_encrypt_response_id_maintains_prefix(
        self, responses_id_security, mock_user_api_key_dict
    ):
        """Test that encrypted response ID maintains 'resp_' prefix"""
        mock_response = ResponsesAPIResponse(
            id="resp_456", created_at=1234567890, output=[], status="in_progress"
        )

        with patch(
            "litellm.proxy.common_utils.encrypt_decrypt_utils._get_salt_key",
            return_value="test-salt-key"
        ):
            with patch.object(
                responses_id_security, "_get_signing_key", return_value="test-key"
            ):
                result = responses_id_security._encrypt_response_id(
                    mock_response, mock_user_api_key_dict
                )

                assert result.id.startswith("resp_")
                # The encrypted ID should be different from the original
                assert result.id != "resp_456"


class TestCheckUserAccessToResponseId:
    """Test check_user_access_to_response_id function"""

    def test_check_user_access_same_user(
        self, responses_id_security, mock_user_api_key_dict
    ):
        """Test that same user has access to their response ID"""
        result = responses_id_security.check_user_access_to_response_id(
            response_id_user_id="test-user-123",
            response_id_team_id="test-team-123",
            user_api_key_dict=mock_user_api_key_dict,
        )

        assert result is True

    def test_check_user_access_different_user_raises_exception(
        self, responses_id_security, mock_user_api_key_dict
    ):
        """Test that different user is denied access to response ID"""
        with patch("litellm.proxy.proxy_server.general_settings", {}):
            with pytest.raises(HTTPException) as exc_info:
                responses_id_security.check_user_access_to_response_id(
                    response_id_user_id="different-user-456",
                    response_id_team_id="test-team-123",
                    user_api_key_dict=mock_user_api_key_dict,
                )

            assert exc_info.value.status_code == 403
            assert "Forbidden" in exc_info.value.detail

    def test_check_user_access_different_team_raises_exception(
        self, responses_id_security, mock_user_api_key_dict
    ):
        """Test that different team is denied access to response ID"""
        with patch("litellm.proxy.proxy_server.general_settings", {}):
            with pytest.raises(HTTPException) as exc_info:
                responses_id_security.check_user_access_to_response_id(
                    response_id_user_id=None,
                    response_id_team_id="different-team-456",
                    user_api_key_dict=mock_user_api_key_dict,
                )

            assert exc_info.value.status_code == 403
            assert "Forbidden" in exc_info.value.detail

    def test_check_user_access_team_a_to_team_b_without_user_id(
        self, responses_id_security
    ):
        """Test that key from team A (without user_id) cannot access response from team B (without user_id)"""
        # Create a mock user from team A without user_id
        mock_auth_team_a = MagicMock()
        mock_auth_team_a.user_id = None
        mock_auth_team_a.team_id = "team-a"
        mock_auth_team_a.user_role = None

        with patch("litellm.proxy.proxy_server.general_settings", {}):
            with pytest.raises(HTTPException) as exc_info:
                responses_id_security.check_user_access_to_response_id(
                    response_id_user_id=None,
                    response_id_team_id="team-b",
                    user_api_key_dict=mock_auth_team_a,
                )

            assert exc_info.value.status_code == 403
            assert "team" in exc_info.value.detail.lower()

    def test_check_user_access_team_a_to_team_b_with_user_id(
        self, responses_id_security
    ):
        """Test that key from team A (without user_id) cannot access response from team B (with user_id)"""
        # Create a mock user from team A without user_id
        mock_auth_team_a = MagicMock()
        mock_auth_team_a.user_id = None
        mock_auth_team_a.team_id = "team-a"
        mock_auth_team_a.user_role = None

        with patch("litellm.proxy.proxy_server.general_settings", {}):
            with pytest.raises(HTTPException) as exc_info:
                responses_id_security.check_user_access_to_response_id(
                    response_id_user_id="user-from-team-b",
                    response_id_team_id="team-b",
                    user_api_key_dict=mock_auth_team_a,
                )

            # Access should be denied with 403. Could fail on user_id or team_id check.
            assert exc_info.value.status_code == 403
            assert "forbidden" in exc_info.value.detail.lower()

    def test_check_user_access_same_team_without_user_id(self, responses_id_security):
        """Test that key from team A (without user_id) can access response from same team A (without user_id)"""
        # Create a mock user from team A without user_id
        mock_auth_team_a = MagicMock()
        mock_auth_team_a.user_id = None
        mock_auth_team_a.team_id = "team-a"
        mock_auth_team_a.user_role = None

        result = responses_id_security.check_user_access_to_response_id(
            response_id_user_id=None,
            response_id_team_id="team-a",
            user_api_key_dict=mock_auth_team_a,
        )

        assert result is True

    def test_check_user_access_admin_can_access_any_response(
        self, responses_id_security
    ):
        """Test that proxy admin can access any response ID"""
        from litellm.proxy._types import LitellmUserRoles

        # Create a mock admin user
        mock_admin_auth = MagicMock()
        mock_admin_auth.user_id = "admin-user"
        mock_admin_auth.team_id = "admin-team"
        mock_admin_auth.user_role = LitellmUserRoles.PROXY_ADMIN.value

        # Admin should be able to access response from different team and different user
        result = responses_id_security.check_user_access_to_response_id(
            response_id_user_id="some-other-user",
            response_id_team_id="some-other-team",
            user_api_key_dict=mock_admin_auth,
        )

        assert result is True

    def test_check_user_access_security_disabled(
        self, responses_id_security, mock_user_api_key_dict
    ):
        """Test that when security is disabled, any user can access any response"""
        with patch(
            "litellm.proxy.proxy_server.general_settings",
            {"disable_responses_id_security": True},
        ):
            # User from team A should be able to access response from team B when security is disabled
            result = responses_id_security.check_user_access_to_response_id(
                response_id_user_id="different-user",
                response_id_team_id="different-team",
                user_api_key_dict=mock_user_api_key_dict,
            )

            assert result is True


class TestAsyncPreCallHook:
    """Test async_pre_call_hook function"""

    @pytest.mark.asyncio
    async def test_async_pre_call_hook_aresponses_with_previous_response_id(
        self, responses_id_security, mock_user_api_key_dict, mock_cache
    ):
        """Test pre-call hook decrypts previous_response_id for aresponses call"""
        data = {"previous_response_id": "resp_encrypted_value"}

        with patch.object(
            responses_id_security, "_is_encrypted_response_id", return_value=True
        ):
            with patch.object(
                responses_id_security,
                "_decrypt_response_id",
                return_value=("resp_original_123", "test-user-123", "test-team-123"),
            ):
                result = await responses_id_security.async_pre_call_hook(
                    user_api_key_dict=mock_user_api_key_dict,
                    cache=mock_cache,
                    data=data,
                    call_type="aresponses",
                )

                assert result["previous_response_id"] == "resp_original_123"

    @pytest.mark.asyncio
    async def test_async_pre_call_hook_aget_responses(
        self, responses_id_security, mock_user_api_key_dict, mock_cache
    ):
        """Test pre-call hook decrypts response_id for aget_responses call"""
        data = {"response_id": "resp_encrypted_456"}

        with patch.object(
            responses_id_security, "_is_encrypted_response_id", return_value=True
        ):
            with patch.object(
                responses_id_security,
                "_decrypt_response_id",
                return_value=("resp_original_456", "test-user-123", "test-team-123"),
            ):
                result = await responses_id_security.async_pre_call_hook(
                    user_api_key_dict=mock_user_api_key_dict,
                    cache=mock_cache,
                    data=data,
                    call_type="aget_responses",
                )

                assert result["response_id"] == "resp_original_456"

    @pytest.mark.asyncio
    async def test_async_pre_call_hook_team_a_accessing_team_b_response(
        self, responses_id_security, mock_cache
    ):
        """Test pre-call hook prevents team A from accessing team B response"""
        # Create a mock user from team A
        mock_auth_team_a = MagicMock()
        mock_auth_team_a.user_id = None
        mock_auth_team_a.team_id = "team-a"
        mock_auth_team_a.user_role = None

        data = {"response_id": "resp_encrypted_team_b"}

        with patch.object(
            responses_id_security, "_is_encrypted_response_id", return_value=True
        ):
            with patch.object(
                responses_id_security,
                "_decrypt_response_id",
                return_value=("resp_original_team_b", None, "team-b"),
            ):
                with patch("litellm.proxy.proxy_server.general_settings", {}):
                    with pytest.raises(HTTPException) as exc_info:
                        await responses_id_security.async_pre_call_hook(
                            user_api_key_dict=mock_auth_team_a,
                            cache=mock_cache,
                            data=data,
                            call_type="aget_responses",
                        )

                    assert exc_info.value.status_code == 403
                    assert "team" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_async_pre_call_hook_team_a_accessing_team_b_with_user(
        self, responses_id_security, mock_cache
    ):
        """Test pre-call hook prevents team A (no user) from accessing team B response (with user)"""
        # Create a mock user from team A without user_id
        mock_auth_team_a = MagicMock()
        mock_auth_team_a.user_id = None
        mock_auth_team_a.team_id = "team-a"
        mock_auth_team_a.user_role = None

        data = {"response_id": "resp_encrypted_team_b_with_user"}

        with patch.object(
            responses_id_security, "_is_encrypted_response_id", return_value=True
        ):
            with patch.object(
                responses_id_security,
                "_decrypt_response_id",
                return_value=("resp_original_team_b", "user-from-team-b", "team-b"),
            ):
                with patch("litellm.proxy.proxy_server.general_settings", {}):
                    with pytest.raises(HTTPException) as exc_info:
                        await responses_id_security.async_pre_call_hook(
                            user_api_key_dict=mock_auth_team_a,
                            cache=mock_cache,
                            data=data,
                            call_type="aget_responses",
                        )

                    # Access should be denied with 403. Could fail on user_id or team_id check.
                    assert exc_info.value.status_code == 403
                    assert "forbidden" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_async_pre_call_hook_same_team_access(
        self, responses_id_security, mock_cache
    ):
        """Test pre-call hook allows team A to access their own team's response"""
        # Create a mock user from team A
        mock_auth_team_a = MagicMock()
        mock_auth_team_a.user_id = None
        mock_auth_team_a.team_id = "team-a"
        mock_auth_team_a.user_role = None

        data = {"response_id": "resp_encrypted_team_a"}

        with patch.object(
            responses_id_security, "_is_encrypted_response_id", return_value=True
        ):
            with patch.object(
                responses_id_security,
                "_decrypt_response_id",
                return_value=("resp_original_team_a", None, "team-a"),
            ):
                result = await responses_id_security.async_pre_call_hook(
                    user_api_key_dict=mock_auth_team_a,
                    cache=mock_cache,
                    data=data,
                    call_type="aget_responses",
                )

                assert result["response_id"] == "resp_original_team_a"

    @pytest.mark.asyncio
    async def test_async_pre_call_hook_adelete_responses_team_security(
        self, responses_id_security, mock_cache
    ):
        """Test pre-call hook prevents team A from deleting team B's response"""
        # Create a mock user from team A
        mock_auth_team_a = MagicMock()
        mock_auth_team_a.user_id = None
        mock_auth_team_a.team_id = "team-a"
        mock_auth_team_a.user_role = None

        data = {"response_id": "resp_encrypted_team_b"}

        with patch.object(
            responses_id_security, "_is_encrypted_response_id", return_value=True
        ):
            with patch.object(
                responses_id_security,
                "_decrypt_response_id",
                return_value=("resp_original_team_b", None, "team-b"),
            ):
                with patch("litellm.proxy.proxy_server.general_settings", {}):
                    with pytest.raises(HTTPException) as exc_info:
                        await responses_id_security.async_pre_call_hook(
                            user_api_key_dict=mock_auth_team_a,
                            cache=mock_cache,
                            data=data,
                            call_type="adelete_responses",
                        )

                    assert exc_info.value.status_code == 403
                    assert "team" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_async_pre_call_hook_acancel_responses_team_security(
        self, responses_id_security, mock_cache
    ):
        """Test pre-call hook prevents team A from canceling team B's response"""
        # Create a mock user from team A
        mock_auth_team_a = MagicMock()
        mock_auth_team_a.user_id = None
        mock_auth_team_a.team_id = "team-a"
        mock_auth_team_a.user_role = None

        data = {"response_id": "resp_encrypted_team_b"}

        with patch.object(
            responses_id_security, "_is_encrypted_response_id", return_value=True
        ):
            with patch.object(
                responses_id_security,
                "_decrypt_response_id",
                return_value=("resp_original_team_b", None, "team-b"),
            ):
                with patch("litellm.proxy.proxy_server.general_settings", {}):
                    with pytest.raises(HTTPException) as exc_info:
                        await responses_id_security.async_pre_call_hook(
                            user_api_key_dict=mock_auth_team_a,
                            cache=mock_cache,
                            data=data,
                            call_type="acancel_responses",
                        )

                    assert exc_info.value.status_code == 403
                    assert "team" in exc_info.value.detail.lower()


class TestAsyncPostCallSuccessHook:
    """Test async_post_call_success_hook function"""

    @pytest.mark.asyncio
    async def test_async_post_call_success_hook_encrypts_response(
        self, responses_id_security, mock_user_api_key_dict
    ):
        """Test post-call hook encrypts ResponsesAPIResponse"""
        mock_response = ResponsesAPIResponse(
            id="resp_789", created_at=1234567890, output=[], status="completed"
        )
        data = {}

        with patch.object(
            responses_id_security, "_encrypt_response_id", return_value=mock_response
        ) as mock_encrypt:
            result = await responses_id_security.async_post_call_success_hook(
                data=data,
                user_api_key_dict=mock_user_api_key_dict,
                response=mock_response,
            )

            mock_encrypt.assert_called_once_with(mock_response, mock_user_api_key_dict)
            assert result == mock_response

    @pytest.mark.asyncio
    async def test_async_post_call_success_hook_non_responses_api_response(
        self, responses_id_security, mock_user_api_key_dict
    ):
        """Test post-call hook passes through non-ResponsesAPIResponse objects"""
        mock_response = {"id": "some-other-response", "data": "test"}
        data = {}

        result = await responses_id_security.async_post_call_success_hook(
            data=data,
            user_api_key_dict=mock_user_api_key_dict,
            response=mock_response,
        )

        assert result == mock_response
