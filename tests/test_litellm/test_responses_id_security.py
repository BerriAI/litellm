"""
Tests for ResponsesIDSecurity hook.

Tests the security hook that prevents user B from seeing response from user A.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm.proxy.hooks.responses_id_security import (
    ResponsesIDSecurity,
    _is_responses_api_create_route,
)
from litellm.types.llms.openai import (
    ResponseCompletedEvent,
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
)
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

        with patch.object(responses_module, "decrypt_value_helper") as mock_decrypt:
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

        with patch.object(responses_module, "decrypt_value_helper") as mock_decrypt:
            mock_decrypt.return_value = None

            result = responses_id_security._is_encrypted_response_id("resp_plain_value")

            assert result is False


class TestDecryptResponseId:
    """Test _decrypt_response_id function"""

    def test_decrypt_response_id_valid(self, responses_id_security):
        """Test decrypting a valid encrypted response ID"""
        # Patch at the module level where it's imported
        import litellm.proxy.hooks.responses_id_security as responses_module

        with patch.object(responses_module, "decrypt_value_helper") as mock_decrypt:
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

        with patch.object(responses_module, "decrypt_value_helper") as mock_decrypt:
            mock_decrypt.return_value = None

            original_id, user_id, team_id = responses_id_security._decrypt_response_id(
                "resp_plain_value"
            )

            assert original_id == "resp_plain_value"
            assert user_id is None
            assert team_id is None


class TestEncryptResponseId:
    """Test _encrypt_response_id function"""

    @pytest.mark.skip(
        reason="Flaky on CI; disabling temporarily until responses_id_security is fixed"
    )
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

    @pytest.mark.skip(
        reason="Flaky on CI; disabling temporarily until responses_id_security is fixed"
    )
    def test_encrypt_response_id_maintains_prefix(
        self, responses_id_security, mock_user_api_key_dict
    ):
        """Test that encrypted response ID maintains 'resp_' prefix"""
        mock_response = ResponsesAPIResponse(
            id="resp_456", created_at=1234567890, output=[], status="in_progress"
        )

        with patch(
            "litellm.proxy.common_utils.encrypt_decrypt_utils._get_salt_key",
            return_value="test-salt-key",
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


    @pytest.mark.asyncio
    async def test_async_pre_call_hook_alist_input_items_decrypts_response_id(
        self, responses_id_security, mock_user_api_key_dict, mock_cache
    ):
        data = {"response_id": "resp_encrypted_789"}

        with patch.object(
            responses_id_security, "_is_encrypted_response_id", return_value=True
        ):
            with patch.object(
                responses_id_security,
                "_decrypt_response_id",
                return_value=("resp_original_789", "test-user-123", "test-team-123"),
            ):
                result = await responses_id_security.async_pre_call_hook(
                    user_api_key_dict=mock_user_api_key_dict,
                    cache=mock_cache,
                    data=data,
                    call_type="alist_input_items",
                )

                assert result is not None
                assert result["response_id"] == "resp_original_789"

    @pytest.mark.asyncio
    async def test_async_pre_call_hook_alist_input_items_team_security(
        self, responses_id_security, mock_cache
    ):
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
                            call_type="alist_input_items",
                        )

                    assert exc_info.value.status_code == 403
                    assert "team" in exc_info.value.detail.lower()


class TestIsResponsesApiCreateRoute:
    """Test the route gate that decides whether a streamed response id is encrypted."""

    @pytest.mark.parametrize(
        "route",
        [
            "/v1/responses",
            "/responses",
            "/openai/v1/responses",
        ],
    )
    def test_create_routes_match(self, route):
        assert _is_responses_api_create_route(route) is True

    @pytest.mark.parametrize(
        "route",
        [
            None,
            "/chat/completions",
            "/openai/v1/chat/completions",
            "/v1/responses/{response_id}",
            "/openai/v1/responses/{response_id}",
            "/v1/responsesX",
            "/responsesX",
        ],
    )
    def test_non_create_routes_do_not_match(self, route):
        assert _is_responses_api_create_route(route) is False


class TestAsyncPostCallStreamingIteratorHook:
    """Regression test for LIT-6167: streamed responses on /openai/v1/responses and
    /responses must have their ids security-encrypted, not just on the exact
    /v1/responses path. A streamed create emits ResponseCompletedEvent, whose
    client-visible id lives on event.response.id, so the test drives that production
    event shape (not a top-level id) and uses real encryption, asserting the id
    round-trips back to the raw provider id plus the caller's user/team, which is the
    access-control wrapper the aliases were leaking without."""

    @staticmethod
    async def _agen(chunks):
        for chunk in chunks:
            yield chunk

    @staticmethod
    def _completed_event(response_id):
        return ResponseCompletedEvent(
            type=ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
            response=ResponsesAPIResponse(
                id=response_id,
                created_at=0,
                model="gpt-5.1",
                object="response",
                output=[],
                parallel_tool_calls=False,
                tool_choice="auto",
                tools=[],
            ),
        )

    async def _drain_streamed_id(self, responses_id_security, route, monkeypatch):
        monkeypatch.setenv("LITELLM_SALT_KEY", "sk-test-salt-key-abcdefghij")
        event = self._completed_event("resp_rawprovider123")

        mock_auth = MagicMock()
        mock_auth.user_id = "user-a"
        mock_auth.team_id = "team-a"
        mock_auth.request_route = route

        collected = [
            out
            async for out in responses_id_security.async_post_call_streaming_iterator_hook(
                user_api_key_dict=mock_auth,
                response=self._agen([event]),
                request_data={},
            )
        ]
        return collected[0].response.id

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "route",
        ["/v1/responses", "/responses", "/openai/v1/responses"],
    )
    async def test_streamed_id_encrypted_on_all_responses_routes(
        self, responses_id_security, route, monkeypatch
    ):
        streamed_id = await self._drain_streamed_id(responses_id_security, route, monkeypatch)

        assert streamed_id != "resp_rawprovider123"
        assert responses_id_security._is_encrypted_response_id(streamed_id)
        assert responses_id_security._decrypt_response_id(streamed_id) == (
            "resp_rawprovider123",
            "user-a",
            "team-a",
        )

    @pytest.mark.asyncio
    async def test_streamed_id_untouched_on_non_responses_route(
        self, responses_id_security, monkeypatch
    ):
        streamed_id = await self._drain_streamed_id(
            responses_id_security, "/chat/completions", monkeypatch
        )

        assert streamed_id == "resp_rawprovider123"
        assert not responses_id_security._is_encrypted_response_id(streamed_id)


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

            mock_encrypt.assert_called_once_with(
                mock_response, mock_user_api_key_dict, request_cache=None
            )
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



_FABRICATED_PROVIDER_RESPONSE_ID = "resp_fabricatedprovideridaaaaaaaaaaaaaaaa"
_FABRICATED_UNMANAGED_ID = "resp_fabricatedunmanagedidbbbbbbbbbbbbbbbb"
_UNIT_TEST_SALT_KEY = "lit6837-unit-test-salt-key"
_ADDRESSED_ID_FIELD_BY_CALL_TYPE = {
    "aresponses": "previous_response_id",
    "aget_responses": "response_id",
    "adelete_responses": "response_id",
    "acancel_responses": "response_id",
    "alist_input_items": "response_id",
}


@pytest.fixture
def salt_key_env(monkeypatch):
    """Give the encrypt/decrypt helpers a real salt key so ids round-trip for real."""
    monkeypatch.setenv("LITELLM_SALT_KEY", _UNIT_TEST_SALT_KEY)
    return _UNIT_TEST_SALT_KEY


def _hook(general_settings=None, signing_key=_UNIT_TEST_SALT_KEY):
    settings = general_settings if general_settings is not None else {}
    return ResponsesIDSecurity(
        general_settings_reader=lambda: settings,
        signing_key_reader=lambda: signing_key,
    )


def _auth(user_id="owner-user", team_id="owner-team", user_role=None):
    from litellm.proxy._types import UserAPIKeyAuth

    return UserAPIKeyAuth(user_id=user_id, team_id=team_id, user_role=user_role)


def _issue_managed_id(hook, owner, provider_response_id=_FABRICATED_PROVIDER_RESPONSE_ID):
    """Mint an id exactly the way the proxy hands one to a client on create."""
    issued = hook._encrypt_response_id(
        ResponsesAPIResponse(
            id=provider_response_id, created_at=1234567890, output=[], status="completed"
        ),
        owner,
    )
    return issued.id


class TestUnrecognizedResponseIdIsRejected:
    """An id this proxy never issued carries no owner, so it must not reach the provider."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("call_type", sorted(_ADDRESSED_ID_FIELD_BY_CALL_TYPE))
    async def test_unmanaged_id_is_rejected_and_not_forwarded(self, mock_cache, salt_key_env, call_type):
        field = _ADDRESSED_ID_FIELD_BY_CALL_TYPE[call_type]
        data = {field: _FABRICATED_UNMANAGED_ID}

        with pytest.raises(HTTPException) as exc_info:
            await _hook().async_pre_call_hook(
                user_api_key_dict=_auth(),
                cache=mock_cache,
                data=data,
                call_type=call_type,
            )

        assert exc_info.value.status_code == 403
        assert "allow_unmanaged_response_ids" in exc_info.value.detail
        assert data[field] == _FABRICATED_UNMANAGED_ID

    @pytest.mark.asyncio
    async def test_owner_can_still_address_the_id_the_proxy_issued_it(self, mock_cache, salt_key_env):
        hook = _hook()
        owner = _auth()
        data = {"response_id": _issue_managed_id(hook, owner)}

        result = await hook.async_pre_call_hook(
            user_api_key_dict=owner,
            cache=mock_cache,
            data=data,
            call_type="aget_responses",
        )

        assert result["response_id"] == _FABRICATED_PROVIDER_RESPONSE_ID

    @pytest.mark.asyncio
    async def test_stranger_cannot_address_an_id_issued_to_someone_else(self, mock_cache, salt_key_env):
        hook = _hook()
        issued_id = _issue_managed_id(hook, _auth())
        data = {"response_id": issued_id}

        with pytest.raises(HTTPException) as exc_info:
            await hook.async_pre_call_hook(
                user_api_key_dict=_auth(user_id="stranger-user", team_id="stranger-team"),
                cache=mock_cache,
                data=data,
                call_type="aget_responses",
            )

        assert exc_info.value.status_code == 403
        assert data["response_id"] == issued_id

    @pytest.mark.asyncio
    async def test_unmanaged_previous_response_id_cannot_seed_a_new_response(self, mock_cache, salt_key_env):
        data = {"model": "gpt-fake", "previous_response_id": _FABRICATED_UNMANAGED_ID}

        with pytest.raises(HTTPException) as exc_info:
            await _hook().async_pre_call_hook(
                user_api_key_dict=_auth(),
                cache=mock_cache,
                data=data,
                call_type="aresponses",
            )

        assert exc_info.value.status_code == 403
        assert data["previous_response_id"] == _FABRICATED_UNMANAGED_ID

    @pytest.mark.asyncio
    async def test_re_entering_the_hook_on_the_same_request_does_not_reject(self, mock_cache, salt_key_env):
        """The rate-limit fallback retry runs pre-call twice over one already-rewritten dict."""
        hook = _hook()
        owner = _auth()
        data = {"model": "gpt-fake", "previous_response_id": _issue_managed_id(hook, owner)}

        first = await hook.async_pre_call_hook(
            user_api_key_dict=owner, cache=mock_cache, data=data, call_type="aresponses"
        )
        second = await hook.async_pre_call_hook(
            user_api_key_dict=owner, cache=mock_cache, data=first, call_type="aresponses"
        )

        assert second["previous_response_id"] == _FABRICATED_PROVIDER_RESPONSE_ID


class TestUnmanagedResponseIdEscapeHatches:
    """Deployments that pass provider ids through on purpose must keep working."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "general_settings",
        [{"allow_unmanaged_response_ids": True}, {"disable_responses_id_security": True}],
    )
    async def test_opted_in_settings_forward_the_id_untouched(self, mock_cache, salt_key_env, general_settings):
        data = {"response_id": _FABRICATED_UNMANAGED_ID}

        result = await _hook(general_settings=general_settings).async_pre_call_hook(
            user_api_key_dict=_auth(),
            cache=mock_cache,
            data=data,
            call_type="aget_responses",
        )

        assert result["response_id"] == _FABRICATED_UNMANAGED_ID

    @pytest.mark.asyncio
    async def test_proxy_without_a_signing_key_forwards_the_id_untouched(self, mock_cache, monkeypatch):
        monkeypatch.delenv("LITELLM_SALT_KEY", raising=False)
        data = {"response_id": _FABRICATED_UNMANAGED_ID}

        result = await _hook(signing_key=None).async_pre_call_hook(
            user_api_key_dict=_auth(),
            cache=mock_cache,
            data=data,
            call_type="aget_responses",
        )

        assert result["response_id"] == _FABRICATED_UNMANAGED_ID

    @pytest.mark.asyncio
    async def test_proxy_admin_may_address_an_unmanaged_id(self, mock_cache, salt_key_env):
        from litellm.proxy._types import LitellmUserRoles

        data = {"response_id": _FABRICATED_UNMANAGED_ID}

        result = await _hook().async_pre_call_hook(
            user_api_key_dict=_auth(user_role=LitellmUserRoles.PROXY_ADMIN),
            cache=mock_cache,
            data=data,
            call_type="aget_responses",
        )

        assert result["response_id"] == _FABRICATED_UNMANAGED_ID


class TestClientSuppliedRetainedIdCannotBypassAuthorization:
    """The retained-id key travels in the request body, so it is re-authorized, never trusted."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("call_type", sorted(_ADDRESSED_ID_FIELD_BY_CALL_TYPE))
    async def test_forged_retained_id_is_still_authorized(self, mock_cache, salt_key_env, call_type):
        field = _ADDRESSED_ID_FIELD_BY_CALL_TYPE[call_type]
        data = {
            field: _FABRICATED_UNMANAGED_ID,
            "_litellm_addressed_response_id": _FABRICATED_UNMANAGED_ID,
        }

        with pytest.raises(HTTPException) as exc_info:
            await _hook().async_pre_call_hook(
                user_api_key_dict=_auth(),
                cache=mock_cache,
                data=data,
                call_type=call_type,
            )

        assert exc_info.value.status_code == 403
        assert data[field] == _FABRICATED_UNMANAGED_ID

    @pytest.mark.asyncio
    @pytest.mark.parametrize("forged", [{"nested": "value"}, ["list"], 42, "", None])
    async def test_non_string_retained_id_falls_back_to_the_addressed_field(self, mock_cache, salt_key_env, forged):
        data = {"response_id": _FABRICATED_UNMANAGED_ID, "_litellm_addressed_response_id": forged}

        with pytest.raises(HTTPException) as exc_info:
            await _hook().async_pre_call_hook(
                user_api_key_dict=_auth(),
                cache=mock_cache,
                data=data,
                call_type="aget_responses",
            )

        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_stranger_forging_their_own_id_never_reaches_someone_elses_response(
        self, mock_cache, salt_key_env
    ):
        hook = _hook()
        stranger = _auth(user_id="stranger-user", team_id="stranger-team")
        stranger_id = _issue_managed_id(hook, stranger, provider_response_id="resp_strangerownprovideridcccccccc")
        victim_provider_id = "resp_victimprovideriddddddddddddddddddddd"
        data = {"response_id": victim_provider_id, "_litellm_addressed_response_id": stranger_id}

        result = await hook.async_pre_call_hook(
            user_api_key_dict=stranger,
            cache=mock_cache,
            data=data,
            call_type="aget_responses",
        )

        assert result["response_id"] == "resp_strangerownprovideridcccccccc"
        assert result["response_id"] != victim_provider_id


class TestResponseIdOwnershipRefusalIsTheSameStep:
    """The WebSocket surface answers with an error frame, not a status code, so it reads
    the very same authorization step as a value. Its verdicts must not drift from the
    exception the HTTP routes raise."""

    def test_refusal_text_matches_the_exception_the_http_routes_raise(self, salt_key_env):
        hook = _hook()

        with pytest.raises(HTTPException) as exc_info:
            hook.authorize_response_id(_FABRICATED_UNMANAGED_ID, _auth())

        assert hook.response_id_ownership_refusal(_FABRICATED_UNMANAGED_ID, _auth()) == exc_info.value.detail

    def test_owner_addressing_an_id_the_proxy_issued_is_not_refused(self, salt_key_env):
        hook = _hook()
        owner = _auth()

        assert hook.response_id_ownership_refusal(_issue_managed_id(hook, owner), owner) is None

    def test_stranger_addressing_someone_elses_issued_id_is_refused(self, salt_key_env):
        hook = _hook()
        issued_id = _issue_managed_id(hook, _auth())
        stranger = _auth(user_id="stranger-user", team_id="stranger-team")

        assert hook.response_id_ownership_refusal(issued_id, stranger) is not None

    @pytest.mark.parametrize(
        "general_settings",
        [{"allow_unmanaged_response_ids": True}, {"disable_responses_id_security": True}],
    )
    def test_escape_hatches_refuse_nothing(self, salt_key_env, general_settings):
        hook = _hook(general_settings=general_settings)

        assert hook.response_id_ownership_refusal(_FABRICATED_UNMANAGED_ID, _auth()) is None
