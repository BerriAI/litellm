"""
Tests for ResponsesIDSecurity hook.

Tests the security hook that prevents user B from seeing response from user A.
"""

import datetime
from collections.abc import Callable, Mapping
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm.caching.dual_cache import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.responses_id_security import (
    ResponsesIDSecurity,
    _streams_response_ids,
)
from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.llms.openai import (
    GenericEvent,
    ResponseAPIUsage,
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
)
from litellm.types.responses.main import DeleteResponseResult
from litellm.types.utils import CallTypesLiteral, Choices, Message, ModelResponse, SpecialEnums, Usage


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
            mock_decrypt.return_value = (
                f"{SpecialEnums.LITELM_MANAGED_FILE_ID_PREFIX.value}response_id:resp_123;user_id:user-456"
            )

            result = responses_id_security._is_encrypted_response_id("resp_encrypted_value")

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

            original_id, user_id, team_id = responses_id_security._decrypt_response_id("resp_encrypted_value")

            assert original_id == "resp_original_123"
            assert user_id == "user-456"
            assert team_id == "team-789"

    def test_decrypt_response_id_no_encryption(self, responses_id_security):
        """Test decrypting a non-encrypted response ID"""
        # Patch at the module level where it's imported
        import litellm.proxy.hooks.responses_id_security as responses_module

        with patch.object(responses_module, "decrypt_value_helper") as mock_decrypt:
            mock_decrypt.return_value = None

            original_id, user_id, team_id = responses_id_security._decrypt_response_id("resp_plain_value")

            assert original_id == "resp_plain_value"
            assert user_id is None
            assert team_id is None


class TestCheckUserAccessToResponseId:
    """Test check_user_access_to_response_id function"""

    def test_check_user_access_same_user(self, responses_id_security, mock_user_api_key_dict):
        """Test that same user has access to their response ID"""
        result = responses_id_security.check_user_access_to_response_id(
            response_id_user_id="test-user-123",
            response_id_team_id="test-team-123",
            user_api_key_dict=mock_user_api_key_dict,
        )

        assert result is True

    def test_check_user_access_different_user_raises_exception(self, responses_id_security, mock_user_api_key_dict):
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

    def test_check_user_access_different_team_raises_exception(self, responses_id_security, mock_user_api_key_dict):
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

    def test_check_user_access_team_a_to_team_b_without_user_id(self, responses_id_security):
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

    def test_check_user_access_team_a_to_team_b_with_user_id(self, responses_id_security):
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

    def test_check_user_access_admin_can_access_any_response(self, responses_id_security):
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

    def test_check_user_access_security_disabled(self, responses_id_security, mock_user_api_key_dict):
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

        with patch.object(responses_id_security, "_is_encrypted_response_id", return_value=True):
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
    async def test_async_pre_call_hook_aget_responses(self, responses_id_security, mock_user_api_key_dict, mock_cache):
        """Test pre-call hook decrypts response_id for aget_responses call"""
        data = {"response_id": "resp_encrypted_456"}

        with patch.object(responses_id_security, "_is_encrypted_response_id", return_value=True):
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
    async def test_async_pre_call_hook_team_a_accessing_team_b_response(self, responses_id_security, mock_cache):
        """Test pre-call hook prevents team A from accessing team B response"""
        # Create a mock user from team A
        mock_auth_team_a = MagicMock()
        mock_auth_team_a.user_id = None
        mock_auth_team_a.team_id = "team-a"
        mock_auth_team_a.user_role = None

        data = {"response_id": "resp_encrypted_team_b"}

        with patch.object(responses_id_security, "_is_encrypted_response_id", return_value=True):
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
    async def test_async_pre_call_hook_team_a_accessing_team_b_with_user(self, responses_id_security, mock_cache):
        """Test pre-call hook prevents team A (no user) from accessing team B response (with user)"""
        # Create a mock user from team A without user_id
        mock_auth_team_a = MagicMock()
        mock_auth_team_a.user_id = None
        mock_auth_team_a.team_id = "team-a"
        mock_auth_team_a.user_role = None

        data = {"response_id": "resp_encrypted_team_b_with_user"}

        with patch.object(responses_id_security, "_is_encrypted_response_id", return_value=True):
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
    async def test_async_pre_call_hook_same_team_access(self, responses_id_security, mock_cache):
        """Test pre-call hook allows team A to access their own team's response"""
        # Create a mock user from team A
        mock_auth_team_a = MagicMock()
        mock_auth_team_a.user_id = None
        mock_auth_team_a.team_id = "team-a"
        mock_auth_team_a.user_role = None

        data = {"response_id": "resp_encrypted_team_a"}

        with patch.object(responses_id_security, "_is_encrypted_response_id", return_value=True):
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
    async def test_async_pre_call_hook_adelete_responses_team_security(self, responses_id_security, mock_cache):
        """Test pre-call hook prevents team A from deleting team B's response"""
        # Create a mock user from team A
        mock_auth_team_a = MagicMock()
        mock_auth_team_a.user_id = None
        mock_auth_team_a.team_id = "team-a"
        mock_auth_team_a.user_role = None

        data = {"response_id": "resp_encrypted_team_b"}

        with patch.object(responses_id_security, "_is_encrypted_response_id", return_value=True):
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
    async def test_async_pre_call_hook_acancel_responses_team_security(self, responses_id_security, mock_cache):
        """Test pre-call hook prevents team A from canceling team B's response"""
        # Create a mock user from team A
        mock_auth_team_a = MagicMock()
        mock_auth_team_a.user_id = None
        mock_auth_team_a.team_id = "team-a"
        mock_auth_team_a.user_role = None

        data = {"response_id": "resp_encrypted_team_b"}

        with patch.object(responses_id_security, "_is_encrypted_response_id", return_value=True):
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

        with patch.object(responses_id_security, "_is_encrypted_response_id", return_value=True):
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
    async def test_async_pre_call_hook_alist_input_items_team_security(self, responses_id_security, mock_cache):
        mock_auth_team_a = MagicMock()
        mock_auth_team_a.user_id = None
        mock_auth_team_a.team_id = "team-a"
        mock_auth_team_a.user_role = None

        data = {"response_id": "resp_encrypted_team_b"}

        with patch.object(responses_id_security, "_is_encrypted_response_id", return_value=True):
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


class TestStreamsResponseIds:
    """Test the route gate that decides whether a streamed response id is encrypted."""

    @pytest.mark.parametrize(
        "route",
        [
            "/v1/responses",
            "/responses",
            "/openai/v1/responses",
            "/v1/responses/resp_abc",
            "/responses/resp_abc",
            "/openai/v1/responses/resp_abc",
        ],
    )
    def test_create_and_retrieve_routes_match(self, route):
        assert _streams_response_ids(route) is True

    @pytest.mark.parametrize(
        "route",
        [
            None,
            "/chat/completions",
            "/openai/v1/chat/completions",
            "/v1/responses/",
            "/v1/responses/resp_abc/input_items",
            "/v1/responsesX",
            "/responsesX",
        ],
    )
    def test_other_routes_do_not_match(self, route):
        assert _streams_response_ids(route) is False


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
        [
            "/v1/responses",
            "/responses",
            "/openai/v1/responses",
            "/v1/responses/resp_rawprovider123",
            "/openai/v1/responses/resp_rawprovider123",
        ],
    )
    async def test_streamed_id_encrypted_on_all_responses_routes(self, responses_id_security, route, monkeypatch):
        streamed_id = await self._drain_streamed_id(responses_id_security, route, monkeypatch)

        assert streamed_id != "resp_rawprovider123"
        assert responses_id_security._is_encrypted_response_id(streamed_id)
        assert responses_id_security._decrypt_response_id(streamed_id) == (
            "resp_rawprovider123",
            "user-a",
            "team-a",
        )

    @pytest.mark.asyncio
    async def test_streamed_id_untouched_on_non_responses_route(self, responses_id_security, monkeypatch):
        streamed_id = await self._drain_streamed_id(responses_id_security, "/chat/completions", monkeypatch)

        assert streamed_id == "resp_rawprovider123"
        assert not responses_id_security._is_encrypted_response_id(streamed_id)


class TestStreamedGenericEventIdEncryption:
    """A background stream carries event types with no typed model, which arrive as
    GenericEvent holding a plain dict. Those used to skip encryption while their typed
    siblings were encrypted, so one stream advertised two ids and the unencrypted one
    skipped the ownership check. Asserts the property rather than one event type: every
    id a client can see is the same encrypted id, and the raw one appears in no frame."""

    RAW_ID = "resp_rawprovider123"

    @staticmethod
    async def _agen(chunks):
        for chunk in chunks:
            yield chunk

    @classmethod
    def _typed_event(cls, event_type):
        return {
            ResponsesAPIStreamEvents.RESPONSE_CREATED: ResponseCreatedEvent,
            ResponsesAPIStreamEvents.RESPONSE_COMPLETED: ResponseCompletedEvent,
        }[event_type](
            type=event_type,
            response=ResponsesAPIResponse(
                id=cls.RAW_ID,
                created_at=0,
                model="gpt-5.1",
                object="response",
                output=[],
                parallel_tool_calls=False,
                tool_choice="auto",
                tools=[],
            ),
        )

    @classmethod
    def _background_stream(cls):
        return [
            cls._typed_event(ResponsesAPIStreamEvents.RESPONSE_CREATED),
            GenericEvent(
                type="response.queued",
                response={"id": cls.RAW_ID, "status": "queued"},
            ),
            GenericEvent(type="keepalive"),
            GenericEvent(
                type="response.some_event_openai_adds_later",
                response={"id": cls.RAW_ID, "status": "in_progress"},
            ),
            cls._typed_event(ResponsesAPIStreamEvents.RESPONSE_COMPLETED),
        ]

    @staticmethod
    def _advertised_ids(events):
        nested = (getattr(event, "response", None) for event in events)
        return [
            payload["id"] if isinstance(payload, dict) else payload.id for payload in nested if payload is not None
        ] + [event.id for event in events if isinstance(getattr(event, "id", None), str)]

    async def _drain(self, responses_id_security, monkeypatch):
        monkeypatch.setenv("LITELLM_SALT_KEY", "sk-test-salt-key-abcdefghij")

        mock_auth = MagicMock()
        mock_auth.user_id = "user-a"
        mock_auth.team_id = "team-a"
        mock_auth.request_route = "/v1/responses"

        return [
            out
            async for out in responses_id_security.async_post_call_streaming_iterator_hook(
                user_api_key_dict=mock_auth,
                response=self._agen(self._background_stream()),
                request_data={},
            )
        ]

    @pytest.mark.asyncio
    async def test_every_event_advertises_the_same_encrypted_id(self, responses_id_security, monkeypatch):
        events = await self._drain(responses_id_security, monkeypatch)
        advertised = self._advertised_ids(events)

        assert len(advertised) == 4
        assert len(set(advertised)) == 1

        streamed_id = advertised[0]
        assert streamed_id != self.RAW_ID
        assert responses_id_security._is_encrypted_response_id(streamed_id)
        assert responses_id_security._decrypt_response_id(streamed_id) == (
            self.RAW_ID,
            "user-a",
            "team-a",
        )

    @pytest.mark.asyncio
    async def test_raw_provider_id_never_reaches_the_client(self, responses_id_security, monkeypatch):
        events = await self._drain(responses_id_security, monkeypatch)

        assert [self.RAW_ID in event.model_dump_json() for event in events] == [False] * len(events)

    @pytest.mark.asyncio
    async def test_sibling_fields_survive_the_rewrite(self, responses_id_security, monkeypatch):
        _, queued, keepalive, later, _ = await self._drain(responses_id_security, monkeypatch)

        assert queued.response["status"] == "queued"
        assert later.response["status"] == "in_progress"
        assert keepalive.type == "keepalive"
        assert getattr(keepalive, "response", None) is None


class TestAsyncPostCallSuccessHook:
    """Test async_post_call_success_hook function"""

    @pytest.mark.asyncio
    async def test_async_post_call_success_hook_encrypts_response(self, responses_id_security, mock_user_api_key_dict):
        """Test post-call hook encrypts ResponsesAPIResponse"""
        mock_response = ResponsesAPIResponse(id="resp_789", created_at=1234567890, output=[], status="completed")
        data = {}

        with patch.object(responses_id_security, "_encrypt_response_id", return_value=mock_response) as mock_encrypt:
            result = await responses_id_security.async_post_call_success_hook(
                data=data,
                user_api_key_dict=mock_user_api_key_dict,
                response=mock_response,
            )

            mock_encrypt.assert_called_once_with(mock_response, mock_user_api_key_dict, request_cache=None)
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
    "acompact_responses": "previous_response_id",
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
        ResponsesAPIResponse(id=provider_response_id, created_at=1234567890, output=[], status="completed"),
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
    @pytest.mark.parametrize("call_type", sorted(_ADDRESSED_ID_FIELD_BY_CALL_TYPE))
    async def test_owner_gets_the_provider_id_on_every_route(self, mock_cache, salt_key_env, call_type):
        field = _ADDRESSED_ID_FIELD_BY_CALL_TYPE[call_type]
        hook = _hook()
        owner = _auth()
        data = {field: _issue_managed_id(hook, owner)}

        result = await hook.async_pre_call_hook(
            user_api_key_dict=owner,
            cache=mock_cache,
            data=data,
            call_type=call_type,
        )

        assert result[field] == _FABRICATED_PROVIDER_RESPONSE_ID

    @pytest.mark.asyncio
    @pytest.mark.parametrize("call_type", sorted(_ADDRESSED_ID_FIELD_BY_CALL_TYPE))
    @pytest.mark.parametrize(
        ("caller_user_id", "caller_team_id", "detail"),
        [
            ("stranger-user", "owner-team", "not associated with the user"),
            ("owner-user", "stranger-team", "not associated with the team"),
        ],
    )
    async def test_other_user_or_team_is_rejected_on_every_route(
        self, mock_cache, salt_key_env, call_type, caller_user_id, caller_team_id, detail
    ):
        field = _ADDRESSED_ID_FIELD_BY_CALL_TYPE[call_type]
        hook = _hook()
        issued_id = _issue_managed_id(hook, _auth())
        data = {field: issued_id}

        with pytest.raises(HTTPException) as exc_info:
            await hook.async_pre_call_hook(
                user_api_key_dict=_auth(user_id=caller_user_id, team_id=caller_team_id),
                cache=mock_cache,
                data=data,
                call_type=call_type,
            )

        assert exc_info.value.status_code == 403
        assert detail in exc_info.value.detail
        assert data[field] == issued_id

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
    @pytest.mark.parametrize("call_type", sorted(_ADDRESSED_ID_FIELD_BY_CALL_TYPE))
    async def test_opted_in_settings_forward_the_id_untouched(
        self, mock_cache, salt_key_env, general_settings, call_type
    ):
        field = _ADDRESSED_ID_FIELD_BY_CALL_TYPE[call_type]
        data = {field: _FABRICATED_UNMANAGED_ID}

        result = await _hook(general_settings=general_settings).async_pre_call_hook(
            user_api_key_dict=_auth(),
            cache=mock_cache,
            data=data,
            call_type=call_type,
        )

        assert result[field] == _FABRICATED_UNMANAGED_ID

    @pytest.mark.asyncio
    @pytest.mark.parametrize("call_type", sorted(_ADDRESSED_ID_FIELD_BY_CALL_TYPE))
    async def test_proxy_without_a_signing_key_forwards_the_id_untouched(self, mock_cache, monkeypatch, call_type):
        monkeypatch.delenv("LITELLM_SALT_KEY", raising=False)
        field = _ADDRESSED_ID_FIELD_BY_CALL_TYPE[call_type]
        data = {field: _FABRICATED_UNMANAGED_ID}

        result = await _hook(signing_key=None).async_pre_call_hook(
            user_api_key_dict=_auth(),
            cache=mock_cache,
            data=data,
            call_type=call_type,
        )

        assert result[field] == _FABRICATED_UNMANAGED_ID

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
    async def test_stranger_forging_their_own_id_never_reaches_someone_elses_response(self, mock_cache, salt_key_env):
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


def _admin():
    from litellm.proxy._types import LitellmUserRoles

    return _auth(user_id="admin-user", team_id="admin-team", user_role=LitellmUserRoles.PROXY_ADMIN)


def _provider_answer(response_id: str) -> ResponsesAPIResponse:
    return ResponsesAPIResponse(id=response_id, created_at=1234567890, output=[], status="completed")


class TestReadsAnswerWithTheIdTheClientHolds:
    """A follow-up looks its history up by the id the client holds, so a read must not hand out a new one."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("call_type", ["aget_responses", "acancel_responses"])
    @pytest.mark.parametrize(
        "issuer, reader",
        [
            (_auth(), _auth()),
            (_auth(team_id=None), _auth(team_id="another-team")),
            (_auth(user_id=None), _auth(user_id=None)),
            (_auth(user_id=None), _auth(user_id="teammate-user")),
            (_auth(), _admin()),
            (_auth(user_id=None), _auth(user_id=None, team_id="admin-team", user_role=_admin().user_role)),
        ],
        ids=[
            "same-key",
            "same-user-new-team",
            "same-team-no-user",
            "teammate-reading-a-team-key-id",
            "admin-user",
            "admin-without-user",
        ],
    )
    async def test_a_caller_with_access_gets_back_the_exact_id_they_sent(
        self, mock_cache, salt_key_env, call_type, issuer, reader
    ):
        hook = _hook()
        issued_id = _issue_managed_id(hook, issuer)
        data = await hook.async_pre_call_hook(
            user_api_key_dict=reader, cache=mock_cache, data={"response_id": issued_id}, call_type=call_type
        )

        result = await hook.async_post_call_success_hook(
            data=data, user_api_key_dict=reader, response=_provider_answer(data["response_id"])
        )

        assert result.id == issued_id

    @pytest.mark.asyncio
    async def test_a_follow_up_gets_an_id_for_its_own_response_never_the_previous_one(self, mock_cache, salt_key_env):
        hook = _hook()
        owner = _auth()
        previous_id = _issue_managed_id(hook, owner)
        data = await hook.async_pre_call_hook(
            user_api_key_dict=owner,
            cache=mock_cache,
            data={"model": "gpt-fake", "previous_response_id": previous_id},
            call_type="aresponses",
        )

        result = await hook.async_post_call_success_hook(
            data=data, user_api_key_dict=owner, response=_provider_answer("resp_followupprovideridzzzzzzzzzzzzzzz")
        )

        assert result.id != previous_id
        assert hook._decrypt_response_id(result.id) == (
            "resp_followupprovideridzzzzzzzzzzzzzzz",
            "owner-user",
            "owner-team",
        )

    @pytest.mark.asyncio
    async def test_an_admin_read_by_a_raw_provider_id_gets_a_managed_id_back(self, mock_cache, salt_key_env):
        hook = _hook()
        admin = _admin()
        data = await hook.async_pre_call_hook(
            user_api_key_dict=admin,
            cache=mock_cache,
            data={"response_id": _FABRICATED_UNMANAGED_ID},
            call_type="aget_responses",
        )

        result = await hook.async_post_call_success_hook(
            data=data, user_api_key_dict=admin, response=_provider_answer(_FABRICATED_UNMANAGED_ID)
        )

        assert hook._is_encrypted_response_id(result.id)
        assert hook._decrypt_response_id(result.id) == (_FABRICATED_UNMANAGED_ID, "admin-user", "admin-team")


_LOGGED_MANAGED_ID = "resp_managedidlitellmbuiltbeforethehookran"


def _streamed_response(response_id):
    return ResponsesAPIResponse(
        id=response_id,
        created_at=0,
        model="gpt-5.1",
        object="response",
        output=[],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        usage=ResponseAPIUsage(input_tokens=3, output_tokens=2, total_tokens=5),
    )


def _upstream_frames():
    return (
        ResponseCreatedEvent(
            type=ResponsesAPIStreamEvents.RESPONSE_CREATED, response=_streamed_response(_LOGGED_MANAGED_ID)
        ),
        ResponseCompletedEvent(
            type=ResponsesAPIStreamEvents.RESPONSE_COMPLETED, response=_streamed_response(_LOGGED_MANAGED_ID)
        ),
    )


def _native_logged_result():
    return ResponseCompletedEvent(
        type=ResponsesAPIStreamEvents.RESPONSE_COMPLETED, response=_streamed_response(_LOGGED_MANAGED_ID)
    )


def _bridged_logged_result():
    return ModelResponse(
        id="chatcmpl-bridgedchatcompletionid",
        model="gpt-5.1",
        choices=[Choices(message=Message(role="assistant", content="hi"), finish_reason="stop", index=0)],
        usage=Usage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
    )


class _SpendRowRecorder(CustomLogger):
    def __init__(self):
        super().__init__()
        self.spend_row_ids = []
        self.standard_logging_ids = []
        self.callback_response_ids = []

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.spend_row_ids.append(get_logging_payload(kwargs, response_obj, start_time, end_time)["request_id"])
        self.callback_response_ids.append(response_obj.id)
        if "standard_logging_object" in kwargs:
            self.standard_logging_ids.append(kwargs["standard_logging_object"]["id"])


async def _stream_logged_mid_stream(
    hook: ResponsesIDSecurity,
    route: str,
    logged_result: ResponseCompletedEvent | ModelResponse,
    caller: UserAPIKeyAuth | None = None,
    request_data: Mapping[str, object] | None = None,
) -> tuple[str, _SpendRowRecorder]:
    streaming_caller = caller if caller is not None else _auth()
    recorder = _SpendRowRecorder()
    logging_obj = LiteLLMLoggingObj(
        model="gpt-5.1",
        messages=[],
        stream=True,
        call_type="aresponses",
        start_time=datetime.datetime.now(),
        litellm_call_id="streamed-responses-call",
        function_id="streamed-responses-fn",
        dynamic_async_success_callbacks=[recorder],
    )
    logging_obj.update_environment_variables(
        model="gpt-5.1", user="", optional_params={}, litellm_params={"api_base": ""}
    )
    upstream = _upstream_frames()

    async def frames():
        for frame in upstream:
            yield frame

    client_stream = hook.async_post_call_streaming_iterator_hook(
        user_api_key_dict=streaming_caller.model_copy(update={"request_route": route}),
        response=frames(),
        request_data={**(request_data or {}), "litellm_logging_obj": logging_obj},
    )
    first_frame = await anext(client_stream)
    now = datetime.datetime.now()
    await logging_obj.async_success_handler(result=logged_result, start_time=now, end_time=now, cache_hit=False)
    client_frames = [first_frame, *[frame async for frame in client_stream]]
    return client_frames[-1].response.id, recorder


class TestStreamedSpendRowCarriesTheIdTheClientSaw:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("logged_result", [_native_logged_result, _bridged_logged_result], ids=["native", "bridge"])
    @pytest.mark.parametrize("route", ["/v1/responses", "/responses", "/openai/v1/responses"])
    async def test_spend_row_and_callbacks_log_the_encrypted_id_of_the_terminal_frame(
        self, salt_key_env, route, logged_result
    ):
        hook = _hook()

        client_id, recorder = await _stream_logged_mid_stream(hook, route, logged_result())

        assert hook._decrypt_response_id(client_id) == (_LOGGED_MANAGED_ID, "owner-user", "owner-team")
        assert recorder.spend_row_ids == [client_id]
        assert recorder.standard_logging_ids == [client_id]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "hook, route",
        [
            (_hook(), "/chat/completions"),
            (_hook(general_settings={"disable_responses_id_security": True}), "/v1/responses"),
            (_hook(signing_key=None), "/v1/responses"),
        ],
        ids=["not-a-create-route", "security-disabled", "no-signing-key"],
    )
    async def test_stream_the_hook_leaves_alone_keeps_its_logged_id(self, salt_key_env, hook, route):
        client_id, recorder = await _stream_logged_mid_stream(hook, route, _native_logged_result())

        assert client_id == _LOGGED_MANAGED_ID
        assert recorder.spend_row_ids == [_LOGGED_MANAGED_ID]
        assert recorder.standard_logging_ids == [_LOGGED_MANAGED_ID]


class TestStreamedReadAnswersWithTheIdTheClientHolds:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("route_prefix", ["/v1/responses", "/responses", "/openai/v1/responses"])
    async def test_the_owner_sees_their_id_while_callbacks_log_one_of_the_reads_own(
        self, mock_cache, salt_key_env, route_prefix
    ):
        hook = _hook()
        owner = _auth()
        issued_id = _issue_managed_id(hook, owner, provider_response_id=_LOGGED_MANAGED_ID)
        data = await hook.async_pre_call_hook(
            user_api_key_dict=owner, cache=mock_cache, data={"response_id": issued_id}, call_type="aget_responses"
        )

        client_id, recorder = await _stream_logged_mid_stream(
            hook, f"{route_prefix}/{issued_id}", _native_logged_result(), caller=owner, request_data=data
        )

        assert client_id == issued_id
        assert recorder.spend_row_ids == recorder.standard_logging_ids
        assert len(recorder.standard_logging_ids) == 1
        assert recorder.standard_logging_ids[0] != issued_id
        assert hook._decrypt_response_id(recorder.standard_logging_ids[0]) == (
            _LOGGED_MANAGED_ID,
            "owner-user",
            "owner-team",
        )

    @pytest.mark.asyncio
    async def test_an_admin_streamed_read_of_someone_elses_response_sees_the_id_sent_and_logs_one_of_its_own(
        self, mock_cache, salt_key_env
    ):
        hook = _hook()
        admin = _admin()
        issued_id = _issue_managed_id(hook, _auth(), provider_response_id=_LOGGED_MANAGED_ID)
        data = await hook.async_pre_call_hook(
            user_api_key_dict=admin, cache=mock_cache, data={"response_id": issued_id}, call_type="aget_responses"
        )

        client_id, recorder = await _stream_logged_mid_stream(
            hook, f"/v1/responses/{issued_id}", _native_logged_result(), caller=admin, request_data=data
        )

        assert client_id == issued_id
        assert recorder.spend_row_ids == recorder.standard_logging_ids
        assert len(recorder.standard_logging_ids) == 1
        assert hook._decrypt_response_id(recorder.standard_logging_ids[0]) == (
            _LOGGED_MANAGED_ID,
            "admin-user",
            "admin-team",
        )


async def _read_logged_by_callbacks(
    hook: ResponsesIDSecurity,
    caller: UserAPIKeyAuth,
    issued_id: str,
    call_type: CallTypesLiteral,
    cache: DualCache,
    answer: Callable[[str], ResponsesAPIResponse | DeleteResponseResult] = _provider_answer,
) -> tuple[str, _SpendRowRecorder]:
    recorder = _SpendRowRecorder()
    now = datetime.datetime.now()
    logging_obj = LiteLLMLoggingObj(
        model="gpt-5.1",
        messages=[],
        stream=False,
        call_type=call_type,
        start_time=now,
        litellm_call_id=f"{call_type}-call",
        function_id=f"{call_type}-fn",
        dynamic_async_success_callbacks=[recorder],
    )
    logging_obj.update_environment_variables(
        model="gpt-5.1", user="", optional_params={}, litellm_params={"api_base": ""}
    )
    data = await hook.async_pre_call_hook(
        user_api_key_dict=caller, cache=cache, data={"response_id": issued_id}, call_type=call_type
    )
    provider_answer = answer(data["response_id"])

    client_response = await hook.async_post_call_success_hook(
        data=data, user_api_key_dict=caller, response=provider_answer
    )
    await logging_obj.async_success_handler(result=provider_answer, start_time=now, end_time=now, cache_hit=False)
    return client_response.id, recorder


class TestEveryReadLogsAnIdOfItsOwn:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("call_type", ["aget_responses", "acancel_responses"])
    @pytest.mark.parametrize("reader", [_auth(), _admin()], ids=["owner", "admin"])
    async def test_callbacks_and_the_spend_row_log_a_new_id_per_read_never_the_one_the_client_holds(
        self, mock_cache, salt_key_env, call_type, reader
    ):
        hook = _hook()
        issued_id = _issue_managed_id(hook, _auth())

        reads = [await _read_logged_by_callbacks(hook, reader, issued_id, call_type, mock_cache) for _ in range(2)]

        assert [client_id for client_id, _ in reads] == [issued_id, issued_id]
        logged_ids = [id_ for _, recorder in reads for id_ in recorder.standard_logging_ids]
        assert [recorder.spend_row_ids for _, recorder in reads] == [[logged_id] for logged_id in logged_ids]
        assert len(set(logged_ids)) == 2
        assert issued_id not in logged_ids
        assert {hook._decrypt_response_id(logged_id) for logged_id in logged_ids} == {
            (_FABRICATED_PROVIDER_RESPONSE_ID, reader.user_id, reader.team_id)
        }


_RAW_UPSTREAM_RESPONSE_ID = "resp_06de59f3upstreamiddddddddddddddddd"


def _routed_response_id(upstream_response_id: str) -> str:
    return ResponsesAPIRequestUtils.update_responses_api_response_id_with_model_id(
        _provider_answer(upstream_response_id), "openai", {"model_info": {"id": "deployment-1"}}
    ).id


def _deletion_of(response_id: str) -> DeleteResponseResult:
    return DeleteResponseResult(
        id=ResponsesAPIRequestUtils.decode_previous_response_id_to_original_previous_response_id(response_id),
        object="response.deleted",
        deleted=True,
    )


class TestDeleteAnswersWithTheIdTheClientHolds:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("reader", [_auth(), _admin()], ids=["owner", "admin"])
    @pytest.mark.parametrize(
        "stored_id",
        [_routed_response_id(_RAW_UPSTREAM_RESPONSE_ID), _RAW_UPSTREAM_RESPONSE_ID],
        ids=["routed-deployment", "plain-id"],
    )
    async def test_a_caller_with_access_gets_back_the_exact_id_they_sent(
        self, mock_cache, salt_key_env, reader, stored_id
    ):
        hook = _hook()
        issued_id = _issue_managed_id(hook, _auth(), provider_response_id=stored_id)
        data = await hook.async_pre_call_hook(
            user_api_key_dict=reader, cache=mock_cache, data={"response_id": issued_id}, call_type="adelete_responses"
        )

        result = await hook.async_post_call_success_hook(
            data=data, user_api_key_dict=reader, response=_deletion_of(data["response_id"])
        )

        assert (result.id, result.object, result.deleted) == (issued_id, "response.deleted", True)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("reader", [_auth(), _admin()], ids=["owner", "admin"])
    async def test_callbacks_and_the_spend_row_log_an_id_of_its_own_never_the_one_the_client_holds(
        self, mock_cache, salt_key_env, reader
    ):
        hook = _hook()
        issued_id = _issue_managed_id(
            hook, _auth(), provider_response_id=_routed_response_id(_RAW_UPSTREAM_RESPONSE_ID)
        )

        client_id, recorder = await _read_logged_by_callbacks(
            hook, reader, issued_id, "adelete_responses", mock_cache, answer=_deletion_of
        )

        assert client_id == issued_id
        assert recorder.spend_row_ids == recorder.callback_response_ids
        assert len(recorder.callback_response_ids) == 1
        assert recorder.callback_response_ids[0] != issued_id
        assert hook._decrypt_response_id(recorder.callback_response_ids[0]) == (
            _RAW_UPSTREAM_RESPONSE_ID,
            reader.user_id,
            reader.team_id,
        )
