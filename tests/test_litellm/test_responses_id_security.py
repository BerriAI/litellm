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
    mock_auth.token = "test-token"
    return mock_auth


@pytest.fixture
def mock_cache():
    """Fixture that creates a mock DualCache object."""
    return MagicMock()


class TestIsEncryptedResponseId:
    """Test _is_encrypted_response_id function"""

    def test_is_encrypted_response_id_valid(self, responses_id_security):
        """Test that a properly encrypted response ID is identified correctly"""
        with patch(
            "litellm.proxy.hooks.responses_id_security.decrypt_value_helper"
        ) as mock_decrypt:
            mock_decrypt.return_value = f"{SpecialEnums.LITELM_MANAGED_FILE_ID_PREFIX.value}response_id:resp_123;user_id:user-456"

            result = responses_id_security._is_encrypted_response_id(
                "resp_encrypted_value"
            )

            assert result is True
            mock_decrypt.assert_called_once()

    def test_is_encrypted_response_id_invalid(self, responses_id_security):
        """Test that an unencrypted response ID returns False"""
        with patch(
            "litellm.proxy.hooks.responses_id_security.decrypt_value_helper"
        ) as mock_decrypt:
            mock_decrypt.return_value = None

            result = responses_id_security._is_encrypted_response_id("resp_plain_value")

            assert result is False


class TestDecryptResponseId:
    """Test _decrypt_response_id function"""

    def test_decrypt_response_id_valid(self, responses_id_security):
        """Test decrypting a valid encrypted response ID"""
        with patch(
            "litellm.proxy.hooks.responses_id_security.decrypt_value_helper"
        ) as mock_decrypt:
            mock_decrypt.return_value = f"{SpecialEnums.LITELM_MANAGED_FILE_ID_PREFIX.value}response_id:resp_original_123;user_id:user-456"

            original_id, user_id = responses_id_security._decrypt_response_id(
                "resp_encrypted_value"
            )

            assert original_id == "resp_original_123"
            assert user_id == "user-456"

    def test_decrypt_response_id_no_encryption(self, responses_id_security):
        """Test decrypting a non-encrypted response ID"""
        with patch(
            "litellm.proxy.hooks.responses_id_security.decrypt_value_helper"
        ) as mock_decrypt:
            mock_decrypt.return_value = None

            original_id, user_id = responses_id_security._decrypt_response_id(
                "resp_plain_value"
            )

            assert original_id == "resp_plain_value"
            assert user_id is None


class TestEncryptResponseId:
    """Test _encrypt_response_id function"""

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

            result = responses_id_security._encrypt_response_id(
                mock_response, mock_user_api_key_dict
            )

            assert result.id == "resp_encrypted_base64_value"
            assert result.id.startswith("resp_")
            mock_encrypt.assert_called_once()

    def test_encrypt_response_id_maintains_prefix(
        self, responses_id_security, mock_user_api_key_dict
    ):
        """Test that encrypted response ID maintains 'resp_' prefix"""
        mock_response = ResponsesAPIResponse(
            id="resp_456", created_at=1234567890, output=[], status="in_progress"
        )

        with patch(
            "litellm.proxy.hooks.responses_id_security.encrypt_value_helper"
        ) as mock_encrypt:
            mock_encrypt.return_value = "encrypted_value_456"

            result = responses_id_security._encrypt_response_id(
                mock_response, mock_user_api_key_dict
            )

            assert result.id.startswith("resp_")


class TestCheckUserAccessToResponseId:
    """Test check_user_access_to_response_id function"""

    def test_check_user_access_same_user(
        self, responses_id_security, mock_user_api_key_dict
    ):
        """Test that same user has access to their response ID"""
        result = responses_id_security.check_user_access_to_response_id(
            "test-user-123", mock_user_api_key_dict
        )

        assert result is True

    def test_check_user_access_different_user_raises_exception(
        self, responses_id_security, mock_user_api_key_dict
    ):
        """Test that different user is denied access to response ID"""
        with patch("litellm.proxy.proxy_server.general_settings", {}):
            with pytest.raises(HTTPException) as exc_info:
                responses_id_security.check_user_access_to_response_id(
                    "different-user-456", mock_user_api_key_dict
                )

            assert exc_info.value.status_code == 403
            assert "Forbidden" in exc_info.value.detail


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
                return_value=("resp_original_123", "test-user-123"),
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
                return_value=("resp_original_456", "test-user-123"),
            ):
                result = await responses_id_security.async_pre_call_hook(
                    user_api_key_dict=mock_user_api_key_dict,
                    cache=mock_cache,
                    data=data,
                    call_type="aget_responses",
                )

                assert result["response_id"] == "resp_original_456"


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


class TestAsyncPostCallStreamingHook:
    """Test async_post_call_streaming_hook function"""

    @pytest.mark.asyncio
    async def test_streaming_hook_encrypts_response_created_event(
        self, responses_id_security, mock_user_api_key_dict
    ):
        """Test streaming hook encrypts response ID in ResponseCreatedEvent"""
        # Create a mock streaming event with a response object
        mock_event = MagicMock()
        mock_event.type = "response.created"
        
        mock_response_obj = ResponsesAPIResponse(
            id="resp_stream_123",
            created_at=1234567890,
            output=[],
            status="in_progress"
        )
        mock_event.response = mock_response_obj
        
        with patch.object(
            responses_id_security, "_encrypt_response_id", return_value=mock_response_obj
        ) as mock_encrypt:
            result = await responses_id_security.async_post_call_streaming_hook(
                user_api_key_dict=mock_user_api_key_dict,
                response=mock_event,
            )
            
            mock_encrypt.assert_called_once_with(mock_response_obj, mock_user_api_key_dict)
            assert result == mock_event

    @pytest.mark.asyncio
    async def test_streaming_hook_handles_non_responses_events(
        self, responses_id_security, mock_user_api_key_dict
    ):
        """Test streaming hook passes through non-Responses API events"""
        # Create a mock event without response object (like OutputTextDeltaEvent)
        mock_event = MagicMock()
        mock_event.type = "response.output_text.delta"
        mock_event.delta = "Hello "
        mock_event.response = None
        
        result = await responses_id_security.async_post_call_streaming_hook(
            user_api_key_dict=mock_user_api_key_dict,
            response=mock_event,
        )
        
        # Event should pass through unchanged
        assert result == mock_event
