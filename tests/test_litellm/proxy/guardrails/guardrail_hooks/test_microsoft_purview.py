"""Unit tests for the Microsoft Purview DLP guardrail."""

import time
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import HTTPException

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.microsoft_purview.base import (
    PurviewGuardrailBase,
)
from litellm.proxy.guardrails.guardrail_hooks.microsoft_purview.purview_dlp import (
    MicrosoftPurviewDLPGuardrail,
)


def _make_guardrail(**kwargs) -> MicrosoftPurviewDLPGuardrail:
    """Helper to construct a guardrail with test defaults."""
    defaults = {
        "guardrail_name": "test-purview",
        "tenant_id": "test-tenant-id",
        "client_id": "test-client-id",
        "client_secret": "test-client-secret",
    }
    defaults.update(kwargs)
    return MicrosoftPurviewDLPGuardrail(**defaults)


def _mock_token_response():
    """Mock a successful OAuth2 token response."""
    resp = Mock()
    resp.json.return_value = {
        "access_token": "mock-access-token",
        "expires_in": 3600,
    }
    return resp


def _mock_graph_response(policy_actions=None, protection_scope_state="unchanged"):
    """Mock a processContent Graph API response."""
    resp = Mock()
    body = {
        "protectionScopeState": protection_scope_state,
        "policyActions": policy_actions or [],
        "processingErrors": [],
    }
    resp.json.return_value = body
    resp.headers = {"ETag": "test-etag-123"}
    return resp


def _mock_scope_response():
    """Mock a protectionScopes/compute Graph API response."""
    resp = Mock()
    resp.json.return_value = {
        "value": [
            {
                "activities": "uploadText,downloadText",
                "executionMode": "evaluateInline",
                "policyActions": [],
            }
        ]
    }
    resp.headers = {"ETag": "scope-etag-123"}
    return resp


# ---------------------------------------------------------------
# _should_block
# ---------------------------------------------------------------


class TestShouldBlock:
    def test_empty_policy_actions(self):
        assert PurviewGuardrailBase._should_block({"policyActions": []}) is False

    def test_no_policy_actions_key(self):
        assert PurviewGuardrailBase._should_block({}) is False

    def test_restrict_access_block(self):
        response = {
            "policyActions": [
                {
                    "@odata.type": "#microsoft.graph.restrictAccessAction",
                    "action": "restrictAccess",
                    "restrictionAction": "block",
                }
            ]
        }
        assert PurviewGuardrailBase._should_block(response) is True

    def test_restrict_access_non_block(self):
        response = {
            "policyActions": [
                {
                    "@odata.type": "#microsoft.graph.restrictAccessAction",
                    "action": "restrictAccess",
                    "restrictionAction": "warn",
                }
            ]
        }
        assert PurviewGuardrailBase._should_block(response) is False

    def test_non_restrict_action(self):
        response = {
            "policyActions": [
                {
                    "@odata.type": "#microsoft.graph.auditAction",
                    "action": "audit",
                }
            ]
        }
        assert PurviewGuardrailBase._should_block(response) is False

    def test_multiple_actions_one_blocks(self):
        response = {
            "policyActions": [
                {"action": "audit"},
                {
                    "@odata.type": "#microsoft.graph.restrictAccessAction",
                    "action": "restrictAccess",
                    "restrictionAction": "block",
                },
            ]
        }
        assert PurviewGuardrailBase._should_block(response) is True


# ---------------------------------------------------------------
# User ID resolution
# ---------------------------------------------------------------


class TestResolveUserId:
    def test_from_metadata(self):
        guardrail = _make_guardrail()
        data = {"metadata": {"user_id": "entra-user-123"}}
        assert guardrail._resolve_user_id(data, Mock()) == "entra-user-123"

    def test_custom_field(self):
        guardrail = _make_guardrail(user_id_field="entra_id")
        data = {"metadata": {"entra_id": "custom-user-456"}}
        assert guardrail._resolve_user_id(data, Mock()) == "custom-user-456"

    def test_from_user_api_key_dict_user_id(self):
        guardrail = _make_guardrail()
        auth = UserAPIKeyAuth(api_key="test", user_id="key-user-789")
        assert guardrail._resolve_user_id({}, auth) == "key-user-789"

    def test_from_end_user_id(self):
        guardrail = _make_guardrail()
        auth = Mock()
        auth.user_id = None
        auth.end_user_id = "end-user-101"
        assert guardrail._resolve_user_id({}, auth) == "end-user-101"

    def test_none_when_missing(self):
        guardrail = _make_guardrail()
        auth = Mock()
        auth.user_id = None
        auth.end_user_id = None
        assert guardrail._resolve_user_id({}, auth) is None


# ---------------------------------------------------------------
# Pre-call hook
# ---------------------------------------------------------------


class TestPreCallHook:
    @pytest.mark.asyncio
    async def test_pre_call_allow(self):
        guardrail = _make_guardrail()

        with patch.object(
            guardrail, "_check_content", new_callable=AsyncMock
        ) as mock_check:
            mock_check.return_value = {"policyActions": []}

            await guardrail.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="test", user_id="user-123"),
                cache=None,
                data={"messages": [{"role": "user", "content": "Hello, how are you?"}]},
                call_type="completion",
            )

            mock_check.assert_called_once()
            assert mock_check.call_args.kwargs["activity"] == "uploadText"
            assert mock_check.call_args.kwargs["block_on_violation"] is True

    @pytest.mark.asyncio
    async def test_pre_call_block(self):
        guardrail = _make_guardrail()

        with patch.object(
            guardrail, "_check_content", new_callable=AsyncMock
        ) as mock_check:
            mock_check.side_effect = HTTPException(
                status_code=400,
                detail={"error": "Microsoft Purview DLP: Content blocked by policy"},
            )

            with pytest.raises(HTTPException) as exc_info:
                await guardrail.async_pre_call_hook(
                    user_api_key_dict=UserAPIKeyAuth(
                        api_key="test", user_id="user-123"
                    ),
                    cache=None,
                    data={
                        "messages": [
                            {
                                "role": "user",
                                "content": "SSN: 123-45-6789",
                            }
                        ]
                    },
                    call_type="completion",
                )

            assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_pre_call_no_user_id_skips(self):
        guardrail = _make_guardrail()

        with patch.object(
            guardrail, "_check_content", new_callable=AsyncMock
        ) as mock_check:
            result = await guardrail.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                cache=None,
                data={"messages": [{"role": "user", "content": "Hello"}]},
                call_type="completion",
            )

            mock_check.assert_not_called()
            # Returns data when skipping
            assert result is not None

    @pytest.mark.asyncio
    async def test_pre_call_no_messages_skips(self):
        guardrail = _make_guardrail()

        with patch.object(
            guardrail, "_check_content", new_callable=AsyncMock
        ) as mock_check:
            await guardrail.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="test", user_id="user-123"),
                cache=None,
                data={},
                call_type="completion",
            )

            mock_check.assert_not_called()


# ---------------------------------------------------------------
# Post-call hook
# ---------------------------------------------------------------


class TestPostCallHook:
    @pytest.mark.asyncio
    async def test_post_call_allow(self):
        from litellm.types.utils import Choices, Message, ModelResponse

        guardrail = _make_guardrail()
        response = ModelResponse(
            choices=[
                Choices(
                    index=0, message=Message(content="Safe response", role="assistant")
                )
            ],
        )

        with patch.object(
            guardrail, "_check_content", new_callable=AsyncMock
        ) as mock_check:
            mock_check.return_value = {"policyActions": []}

            result = await guardrail.async_post_call_success_hook(
                data={"metadata": {"user_id": "user-123"}},
                user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                response=response,
            )

            mock_check.assert_called_once()
            assert mock_check.call_args.kwargs["activity"] == "downloadText"
            assert result is response

    @pytest.mark.asyncio
    async def test_post_call_block(self):
        from litellm.types.utils import Choices, Message, ModelResponse

        guardrail = _make_guardrail()
        response = ModelResponse(
            choices=[
                Choices(
                    index=0,
                    message=Message(
                        content="Credit card: 4532-6677-8521-3500",
                        role="assistant",
                    ),
                )
            ],
        )

        with patch.object(
            guardrail, "_check_content", new_callable=AsyncMock
        ) as mock_check:
            mock_check.side_effect = HTTPException(
                status_code=400,
                detail={"error": "Microsoft Purview DLP: Content blocked by policy"},
            )

            with pytest.raises(HTTPException) as exc_info:
                await guardrail.async_post_call_success_hook(
                    data={"metadata": {"user_id": "user-123"}},
                    user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                    response=response,
                )

            assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_post_call_no_user_id_skips(self):
        from litellm.types.utils import Choices, Message, ModelResponse

        guardrail = _make_guardrail()
        response = ModelResponse(
            choices=[
                Choices(index=0, message=Message(content="Response", role="assistant"))
            ],
        )

        with patch.object(
            guardrail, "_check_content", new_callable=AsyncMock
        ) as mock_check:
            result = await guardrail.async_post_call_success_hook(
                data={},
                user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                response=response,
            )

            mock_check.assert_not_called()
            assert result is response


# ---------------------------------------------------------------
# _check_content — integration-level
# ---------------------------------------------------------------


class TestCheckContent:
    @pytest.mark.asyncio
    async def test_check_content_allow(self):
        guardrail = _make_guardrail()

        with (
            patch.object(
                guardrail,
                "_compute_protection_scopes",
                new_callable=AsyncMock,
                return_value=("etag-1", {}),
            ),
            patch.object(
                guardrail,
                "_process_content",
                new_callable=AsyncMock,
                return_value={
                    "protectionScopeState": "unchanged",
                    "policyActions": [],
                },
            ),
        ):
            result = await guardrail._check_content(
                user_id="user-1",
                text="Hello world",
                activity="uploadText",
                request_data={},
                block_on_violation=True,
            )

            assert result["policyActions"] == []

    @pytest.mark.asyncio
    async def test_check_content_block(self):
        guardrail = _make_guardrail()

        with (
            patch.object(
                guardrail,
                "_compute_protection_scopes",
                new_callable=AsyncMock,
                return_value=("etag-1", {}),
            ),
            patch.object(
                guardrail,
                "_process_content",
                new_callable=AsyncMock,
                return_value={
                    "protectionScopeState": "unchanged",
                    "policyActions": [
                        {
                            "@odata.type": "#microsoft.graph.restrictAccessAction",
                            "action": "restrictAccess",
                            "restrictionAction": "block",
                        }
                    ],
                },
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await guardrail._check_content(
                    user_id="user-1",
                    text="SSN: 123-45-6789",
                    activity="uploadText",
                    request_data={},
                    block_on_violation=True,
                )

            assert exc_info.value.status_code == 400
            assert "blocked by policy" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_check_content_logging_only_no_block(self):
        """In logging_only mode, violations should NOT raise."""
        guardrail = _make_guardrail(logging_only=True)

        with (
            patch.object(
                guardrail,
                "_compute_protection_scopes",
                new_callable=AsyncMock,
                return_value=("etag-1", {}),
            ),
            patch.object(
                guardrail,
                "_process_content",
                new_callable=AsyncMock,
                return_value={
                    "protectionScopeState": "unchanged",
                    "policyActions": [
                        {
                            "@odata.type": "#microsoft.graph.restrictAccessAction",
                            "action": "restrictAccess",
                            "restrictionAction": "block",
                        }
                    ],
                },
            ),
        ):
            # Should NOT raise even though violation detected
            result = await guardrail._check_content(
                user_id="user-1",
                text="SSN: 123-45-6789",
                activity="uploadText",
                request_data={},
                block_on_violation=False,
            )

            assert len(result["policyActions"]) == 1


# ---------------------------------------------------------------
# Token caching
# ---------------------------------------------------------------


class TestTokenCaching:
    @pytest.mark.asyncio
    async def test_token_cached(self):
        guardrail = _make_guardrail()

        with patch.object(
            guardrail.async_handler, "post", return_value=_mock_token_response()
        ) as mock_post:
            token1 = await guardrail._get_access_token()
            token2 = await guardrail._get_access_token()

            assert token1 == "mock-access-token"
            assert token2 == "mock-access-token"
            # Should only call the token endpoint once (cached)
            assert mock_post.call_count == 1

    @pytest.mark.asyncio
    async def test_token_refreshed_on_expiry(self):
        guardrail = _make_guardrail()

        with patch.object(
            guardrail.async_handler, "post", return_value=_mock_token_response()
        ) as mock_post:
            await guardrail._get_access_token()

            # Expire the token
            guardrail._token_cache = ("old-token", time.time() - 10)

            await guardrail._get_access_token()

            # Should have called token endpoint twice
            assert mock_post.call_count == 2


# ---------------------------------------------------------------
# Protection scope caching
# ---------------------------------------------------------------


class TestScopeCaching:
    @pytest.mark.asyncio
    async def test_scope_cached(self):
        guardrail = _make_guardrail()

        with patch.object(
            guardrail, "_graph_post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = (
                {
                    "value": [
                        {"activities": "uploadText", "executionMode": "evaluateInline"}
                    ]
                },
                {"ETag": "scope-etag"},
            )

            etag1, _ = await guardrail._compute_protection_scopes("user-1")
            etag2, _ = await guardrail._compute_protection_scopes("user-1")

            assert etag1 == "scope-etag"
            assert etag2 == "scope-etag"
            assert mock_post.call_count == 1

    @pytest.mark.asyncio
    async def test_scope_invalidated_on_modified(self):
        guardrail = _make_guardrail()

        with patch.object(
            guardrail, "_graph_post", new_callable=AsyncMock
        ) as mock_post:
            # First call: compute scopes
            mock_post.return_value = (
                {"value": []},
                {"ETag": "etag-1"},
            )
            await guardrail._compute_protection_scopes("user-1")

            # processContent returns modified
            mock_post.return_value = (
                {"protectionScopeState": "modified", "policyActions": []},
                {},
            )
            await guardrail._process_content("user-1", "text", "uploadText", "etag-1")

            # Scope cache should be invalidated
            assert "user-1" not in guardrail._scope_cache


# ---------------------------------------------------------------
# Initializer validation
# ---------------------------------------------------------------


class TestInitializerValidation:
    def test_missing_tenant_id(self):
        from litellm.proxy.guardrails.guardrail_hooks.microsoft_purview import (
            initialize_guardrail,
        )

        litellm_params = Mock()
        litellm_params.get.return_value = None
        litellm_params.api_key = "secret"
        litellm_params.mode = "pre_call"

        with pytest.raises(ValueError, match="tenant_id is required"):
            initialize_guardrail(litellm_params, {"guardrail_name": "test"})

    def test_missing_client_id(self):
        from litellm.proxy.guardrails.guardrail_hooks.microsoft_purview import (
            initialize_guardrail,
        )

        litellm_params = Mock()
        litellm_params.get.side_effect = lambda k, *a: (
            "test-tenant" if k == "tenant_id" else None
        )
        litellm_params.api_key = "secret"
        litellm_params.mode = "pre_call"

        with pytest.raises(ValueError, match="client_id is required"):
            initialize_guardrail(litellm_params, {"guardrail_name": "test"})

    def test_missing_client_secret(self):
        from litellm.proxy.guardrails.guardrail_hooks.microsoft_purview import (
            initialize_guardrail,
        )

        litellm_params = Mock()
        litellm_params.get.side_effect = lambda k, *a: {
            "tenant_id": "test-tenant",
            "client_id": "test-client",
        }.get(k)
        litellm_params.api_key = None
        litellm_params.mode = "pre_call"

        with pytest.raises(ValueError, match="client_secret"):
            initialize_guardrail(litellm_params, {"guardrail_name": "test"})


# ---------------------------------------------------------------
# Auto-discovery registration
# ---------------------------------------------------------------


class TestRegistration:
    def test_registry_contains_microsoft_purview(self):
        from litellm.proxy.guardrails.guardrail_hooks.microsoft_purview import (
            guardrail_class_registry,
            guardrail_initializer_registry,
        )

        assert "microsoft_purview" in guardrail_initializer_registry
        assert "microsoft_purview" in guardrail_class_registry
        assert (
            guardrail_class_registry["microsoft_purview"]
            is MicrosoftPurviewDLPGuardrail
        )
