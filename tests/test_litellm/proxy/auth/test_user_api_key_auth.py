import asyncio
import json
import os
import sys
from typing import Tuple
from unittest.mock import ANY, AsyncMock, MagicMock, patch

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import pytest

import litellm.proxy.proxy_server
from litellm.caching.dual_cache import DualCache
from litellm.proxy._types import (
    LiteLLM_JWTAuth,
    ProxyErrorTypes,
    ProxyException,
    UserAPIKeyAuth,
    JWTRoutingOverride,
)
from litellm.proxy.auth.handle_jwt import JWTHandler
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.auth.user_api_key_auth import (
    _run_post_custom_auth_checks,
    get_api_key,
    user_api_key_auth,
)


def test_get_api_key():
    bearer_token = "Bearer sk-12345678"
    api_key = "sk-12345678"
    passed_in_key = "Bearer sk-12345678"
    assert get_api_key(
        custom_litellm_key_header=None,
        api_key=bearer_token,
        azure_api_key_header=None,
        anthropic_api_key_header=None,
        google_ai_studio_api_key_header=None,
        azure_apim_header=None,
        pass_through_endpoints=None,
        route="",
        request=MagicMock(),
    ) == (api_key, passed_in_key)


@pytest.mark.asyncio
async def test_custom_auth_does_not_enforce_key_model_access_by_default():
    valid_token = UserAPIKeyAuth(token="test_token", models=["gpt-4o-mini"])
    request_data = {"model": "gpt-4o"}

    with patch(
        "litellm.proxy.auth.user_api_key_auth.can_key_call_model", new_callable=AsyncMock
    ) as mock_can_key, patch(
        "litellm.proxy.proxy_server.general_settings",
        {},
    ):
        await _run_post_custom_auth_checks(
            valid_token=valid_token,
            request=None,
            request_data=request_data,
            route="/v1/chat/completions",
            parent_otel_span=None,
        )
        mock_can_key.assert_not_awaited()


@pytest.mark.asyncio
async def test_custom_auth_honors_key_level_model_access_restriction_allowed_with_opt_in():
    valid_token = UserAPIKeyAuth(token="test_token", models=["gpt-4o-mini"])
    request_data = {"model": "gpt-4o-mini"}

    with patch(
        "litellm.proxy.auth.user_api_key_auth.can_key_call_model", new_callable=AsyncMock
    ) as mock_can_key, patch(
        "litellm.proxy.auth.user_api_key_auth.common_checks", new_callable=AsyncMock
    ), patch(
        "litellm.proxy.proxy_server.general_settings",
        {"custom_auth_run_common_checks": True},
    ):
        await _run_post_custom_auth_checks(
            valid_token=valid_token,
            request=None,
            request_data=request_data,
            route="/v1/chat/completions",
            parent_otel_span=None,
        )
        mock_can_key.assert_awaited_once_with(
            model="gpt-4o-mini",
            llm_model_list=ANY,
            valid_token=valid_token,
            llm_router=ANY,
        )


@pytest.mark.asyncio
async def test_custom_auth_honors_key_level_model_access_restriction_denied_with_opt_in():
    valid_token = UserAPIKeyAuth(token="test_token", models=["gpt-4o-mini"])
    request_data = {"model": "gpt-4o"}

    with patch(
        "litellm.proxy.auth.user_api_key_auth.can_key_call_model", new_callable=AsyncMock
    ) as mock_can_key, patch(
        "litellm.proxy.auth.user_api_key_auth.common_checks", new_callable=AsyncMock
    ), patch(
        "litellm.proxy.proxy_server.general_settings",
        {"custom_auth_run_common_checks": True},
    ):
        mock_can_key.side_effect = ProxyException(
            message="Key not allowed to access model",
            type=ProxyErrorTypes.key_model_access_denied,
            param="model",
            code=401,
        )
        with pytest.raises(ProxyException) as exc:
            await _run_post_custom_auth_checks(
                valid_token=valid_token,
                request=None,
                request_data=request_data,
                route="/v1/chat/completions",
                parent_otel_span=None,
            )

        assert exc.value.type == ProxyErrorTypes.key_model_access_denied


@pytest.mark.parametrize(
    "custom_litellm_key_header, api_key, passed_in_key",
    [
        ("Bearer sk-12345678", "sk-12345678", "Bearer sk-12345678"),
        ("Basic sk-12345678", "sk-12345678", "Basic sk-12345678"),
        ("bearer sk-12345678", "sk-12345678", "bearer sk-12345678"),
        ("sk-12345678", "sk-12345678", "sk-12345678"),
        # AWS Signature V4 format (LangChain AWS SDK)
        (
            "AWS4-HMAC-SHA256 Credential=Bearer sk-12345678/20260210/us-east-1/bedrock/aws4_request, SignedHeaders=host, Signature=abc123",
            "sk-12345678",
            "AWS4-HMAC-SHA256 Credential=Bearer sk-12345678/20260210/us-east-1/bedrock/aws4_request, SignedHeaders=host, Signature=abc123",
        ),
    ],
)
def test_get_api_key_with_custom_litellm_key_header(
    custom_litellm_key_header, api_key, passed_in_key
):
    assert get_api_key(
        custom_litellm_key_header=custom_litellm_key_header,
        api_key=None,
        azure_api_key_header=None,
        anthropic_api_key_header=None,
        google_ai_studio_api_key_header=None,
        azure_apim_header=None,
        pass_through_endpoints=None,
        route="",
        request=MagicMock(),
    ) == (api_key, passed_in_key)


def test_team_metadata_with_tags_flows_through_jwt_auth():
    """
    Test that team_metadata (specifically tags) flows through JWT authentication.

    This is a regression test for the issue where JWT auth was not populating
    team_metadata, causing team-level tags to be missing in litellm_pre_call_utils.py
    """
    from litellm.proxy._types import LiteLLM_TeamTable, UserAPIKeyAuth

    # Create a team object with metadata containing tags
    team_object = LiteLLM_TeamTable(
        team_id="test-team-id",
        team_alias="test-team-alias",
        metadata={"tags": ["production", "high-priority"], "department": "engineering"},
        tpm_limit=1000,
        rpm_limit=100,
        models=["gpt-4", "gpt-3.5-turbo"],
    )

    # Simulate constructing UserAPIKeyAuth like we do in JWT auth
    # This is the pattern from user_api_key_auth.py lines 552-587
    user_api_key_auth = UserAPIKeyAuth(
        api_key=None,
        team_id=team_object.team_id,
        team_tpm_limit=team_object.tpm_limit if team_object is not None else None,
        team_rpm_limit=team_object.rpm_limit if team_object is not None else None,
        team_models=team_object.models if team_object is not None else [],
        team_metadata=team_object.metadata if team_object is not None else None,
        user_role="internal_user",
        user_id="test-user",
    )

    # Verify team_metadata is set
    assert (
        user_api_key_auth.team_metadata is not None
    ), "team_metadata should be populated"
    assert user_api_key_auth.team_metadata == team_object.metadata, (
        f"team_metadata not correctly mapped. "
        f"Expected: {team_object.metadata}, Got: {user_api_key_auth.team_metadata}"
    )

    # Specifically verify tags are present
    assert "tags" in user_api_key_auth.team_metadata, "tags should be in team_metadata"
    assert user_api_key_auth.team_metadata["tags"] == ["production", "high-priority"], (
        f"tags not correctly mapped. "
        f"Expected: ['production', 'high-priority'], Got: {user_api_key_auth.team_metadata.get('tags')}"
    )


def test_route_checks_is_llm_api_route():
    """Test RouteChecks.is_llm_api_route() correctly identifies LLM API routes including passthrough endpoints"""

    # Test OpenAI routes
    openai_routes = [
        "/v1/chat/completions",
        "/chat/completions",
        "/v1/completions",
        "/completions",
        "/v1/embeddings",
        "/embeddings",
        "/v1/images/generations",
        "/images/generations",
        "/v1/audio/transcriptions",
        "/audio/transcriptions",
        "/v1/audio/speech",
        "/audio/speech",
        "/v1/moderations",
        "/moderations",
        "/v1/models",
        "/models",
        "/v1/rerank",
        "/rerank",
        "/v1/realtime",
        "/realtime",
    ]

    for route in openai_routes:
        assert RouteChecks.is_llm_api_route(
            route=route
        ), f"Route {route} should be identified as LLM API route"

    # Test Anthropic routes
    anthropic_routes = [
        "/v1/messages",
        "/v1/messages/count_tokens",
    ]

    for route in anthropic_routes:
        assert RouteChecks.is_llm_api_route(
            route=route
        ), f"Route {route} should be identified as LLM API route"

    # Test passthrough routes (this is the key improvement over the old route checking)
    passthrough_routes = [
        "/bedrock/v1/chat/completions",
        "/vertex-ai/v1/chat/completions",
        "/vertex_ai/v1/chat/completions",
        "/cohere/v1/chat/completions",
        "/gemini/v1/chat/completions",
        "/anthropic/v1/messages",
        "/langfuse/v1/chat/completions",
        "/azure/v1/chat/completions",
        "/openai/v1/chat/completions",
        "/assemblyai/v1/transcript",
        "/eu.assemblyai/v1/transcript",
        "/vllm/v1/chat/completions",
        "/mistral/v1/chat/completions",
    ]

    for route in passthrough_routes:
        assert RouteChecks.is_llm_api_route(
            route=route
        ), f"Route {route} should be identified as LLM API route"

    # Test MCP routes
    mcp_routes = [
        "/mcp",
        "/mcp/",
        "/mcp/test",
    ]

    for route in mcp_routes:
        assert RouteChecks.is_llm_api_route(
            route=route
        ), f"Route {route} should be identified as LLM API route"

    # Test LiteLLM native RAG routes
    rag_routes = [
        "/rag/ingest",
        "/v1/rag/ingest",
        "/rag/query",
        "/v1/rag/query",
    ]
    for route in rag_routes:
        assert RouteChecks.is_llm_api_route(
            route=route
        ), f"Route {route} should be identified as LLM API route"

    # Test routes with placeholders
    placeholder_routes = [
        "/v1/threads/thread_49EIN5QF32s4mH20M7GFKdlZ",
        "/threads/thread_49EIN5QF32s4mH20M7GFKdlZ",
        "/v1/assistants/assistant_123",
        "/assistants/assistant_123",
        "/v1/files/file_123",
        "/files/file_123",
        "/v1/batches/batch_123",
        "/batches/batch_123",
    ]

    for route in placeholder_routes:
        assert RouteChecks.is_llm_api_route(
            route=route
        ), f"Route {route} should be identified as LLM API route"

    # Test Azure OpenAI routes
    azure_routes = [
        "/openai/deployments/gpt-4/chat/completions",
        "/openai/deployments/gpt-3.5-turbo/completions",
        "/engines/gpt-4/chat/completions",
        "/engines/gpt-3.5-turbo/completions",
    ]

    for route in azure_routes:
        assert RouteChecks.is_llm_api_route(
            route=route
        ), f"Route {route} should be identified as LLM API route"

    # Test non-LLM routes (should return False)
    non_llm_routes = [
        "/health",
        "/metrics",
        "/key/list",
        "/team/list",
        "/user/list",
        "/config",
        "/routes",
        "/",
        "/admin/settings",
        "/logs",
        "/debug",
        "/test",
    ]

    for route in non_llm_routes:
        assert not RouteChecks.is_llm_api_route(
            route=route
        ), f"Route {route} should NOT be identified as LLM API route"

    # Test invalid inputs
    invalid_inputs = [
        None,
        123,
        [],
        {},
        "",
    ]

    for invalid_input in invalid_inputs:
        assert not RouteChecks.is_llm_api_route(
            route=invalid_input
        ), f"Invalid input {invalid_input} should return False"


@pytest.mark.asyncio
async def test_proxy_admin_expired_key_from_cache():
    """
    Test that PROXY_ADMIN keys retrieved from cache are checked for expiration
    before being returned. This prevents expired keys from bypassing expiration checks
    when retrieved from cache (which normally happens at lines 1014-1036).

    Regression test for issue where PROXY_ADMIN keys from cache skipped expiration check.
    """
    from datetime import datetime, timedelta, timezone

    from fastapi import Request
    from starlette.datastructures import URL

    from litellm.proxy._types import (
        LitellmUserRoles,
        ProxyErrorTypes,
        ProxyException,
        UserAPIKeyAuth,
    )
    from litellm.proxy.auth.user_api_key_auth import _user_api_key_auth_builder
    from litellm.proxy.proxy_server import hash_token

    # Create an expired PROXY_ADMIN key
    api_key = "sk-test-proxy-admin-key"
    hashed_key = hash_token(api_key)
    expired_time = datetime.now(timezone.utc) - timedelta(hours=1)  # Expired 1 hour ago

    expired_token = UserAPIKeyAuth(
        api_key=api_key,
        user_role=LitellmUserRoles.PROXY_ADMIN,
        expires=expired_time,
        token=hashed_key,
    )

    # Mock cache to return the expired token
    mock_cache = AsyncMock()
    mock_cache.async_get_cache = AsyncMock(return_value=expired_token)
    mock_cache.delete_cache = MagicMock()

    # Mock proxy_logging_obj
    mock_proxy_logging_obj = MagicMock()
    mock_proxy_logging_obj.internal_usage_cache = MagicMock()
    mock_proxy_logging_obj.internal_usage_cache.dual_cache = AsyncMock()
    mock_proxy_logging_obj.internal_usage_cache.dual_cache.async_delete_cache = (
        AsyncMock()
    )
    # Mock post_call_failure_hook as async function returning None (no transformation)
    mock_proxy_logging_obj.post_call_failure_hook = AsyncMock(return_value=None)

    # Mock prisma_client
    mock_prisma_client = MagicMock()

    # Mock get_key_object to return expired token from cache
    with patch(
        "litellm.proxy.auth.user_api_key_auth.get_key_object",
        new_callable=AsyncMock,
    ) as mock_get_key_object, patch(
        "litellm.proxy.auth.user_api_key_auth._delete_cache_key_object",
        new_callable=AsyncMock,
    ) as mock_delete_cache:
        mock_get_key_object.return_value = expired_token

        # Set attributes on proxy_server module (these are imported inside _user_api_key_auth_builder)
        import litellm.proxy.proxy_server as _proxy_server_mod

        _attrs_to_set = {
            "prisma_client": mock_prisma_client,
            "user_api_key_cache": mock_cache,
            "proxy_logging_obj": mock_proxy_logging_obj,
            "master_key": "sk-master-key",
            "general_settings": {},
            "llm_model_list": [],
            "llm_router": None,
            "open_telemetry_logger": None,
            "model_max_budget_limiter": MagicMock(),
            "user_custom_auth": None,
            "jwt_handler": None,
            "litellm_proxy_admin_name": "admin",
        }
        _original_values = {
            attr: getattr(_proxy_server_mod, attr, None) for attr in _attrs_to_set
        }
        try:
            for attr, val in _attrs_to_set.items():
                setattr(_proxy_server_mod, attr, val)

            # Create a mock request
            request = Request(scope={"type": "http"})
            request._url = URL(url="/chat/completions")
            request_data = {}

            # Call the auth builder - should raise ProxyException for expired key
            # Note: api_key needs "Bearer " prefix for get_api_key() to process it correctly
            with pytest.raises(ProxyException) as exc_info:
                await _user_api_key_auth_builder(
                    request=request,
                    api_key=f"Bearer {api_key}",  # Add Bearer prefix
                    azure_api_key_header="",
                    anthropic_api_key_header=None,
                    google_ai_studio_api_key_header=None,
                    azure_apim_header=None,
                    request_data=request_data,
                )

            # Verify that ProxyException was raised with expired_key type
            assert hasattr(
                exc_info.value, "type"
            ), "Exception should have 'type' attribute"
            assert (
                exc_info.value.type == ProxyErrorTypes.expired_key
            ), f"Expected expired_key error type, got {exc_info.value.type}"
            assert "Expired Key" in str(
                exc_info.value.message
            ), f"Exception message should mention 'Expired Key', got: {exc_info.value.message}"

            # Verify that the param field does NOT leak the full API key (Issue #18731)
            # The param should be abbreviated like "sk-...XXXX" not the full plaintext key
            assert (
                exc_info.value.param is not None
            ), "Exception should have 'param' attribute"
            assert exc_info.value.param != api_key, (
                f"SECURITY: Full API key should NOT be in param field! "
                f"Got: {exc_info.value.param}, Expected abbreviated format like 'sk-...XXXX'"
            )
            assert exc_info.value.param.startswith(
                "sk-..."
            ), f"Param should be abbreviated to 'sk-...XXXX' format. Got: {exc_info.value.param}"

            # Verify that cache deletion was called
            mock_delete_cache.assert_called_once()
            call_args = mock_delete_cache.call_args
            assert (
                call_args[1]["hashed_token"] == hashed_key
            ), "Cache deletion should be called with the hashed key"
        finally:
            # Restore all module-level attributes so subsequent tests are not affected
            for attr, val in _original_values.items():
                setattr(_proxy_server_mod, attr, val)


@pytest.mark.asyncio
async def test_return_user_api_key_auth_obj_user_spend_and_budget():
    """
    Test that _return_user_api_key_auth_obj correctly sets user_spend and user_max_budget
    from user_obj attributes.
    """
    from datetime import datetime

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.auth.user_api_key_auth import _return_user_api_key_auth_obj

    user_obj = type(
        "LiteLLM_UserTable",
        (),
        {
            "tpm_limit": 1000,
            "rpm_limit": 100,
            "user_email": "test@example.com",
            "spend": 250.0,
            "max_budget": 1000.0,
            "user_role": "internal_user",
        },
    )

    api_key = "sk-test-key"
    valid_token_dict = {
        "user_id": "test-user",
        "org_id": "test-org",
    }
    route = "/chat/completions"
    start_time = datetime.now()

    mock_service_logger = MagicMock()
    mock_service_logger.async_service_success_hook = AsyncMock()

    with patch(
        "litellm.proxy.auth.user_api_key_auth.user_api_key_service_logger_obj",
        new=mock_service_logger,
    ):
        result = await _return_user_api_key_auth_obj(
            user_obj=user_obj,
            api_key=api_key,
            parent_otel_span=None,
            valid_token_dict=valid_token_dict,
            route=route,
            start_time=start_time,
            user_role=None,
        )

    assert isinstance(result, UserAPIKeyAuth)
    assert result.user_spend == 250.0
    assert result.user_max_budget == 1000.0
    assert result.user_tpm_limit == 1000
    assert result.user_rpm_limit == 100
    assert result.user_email == "test@example.com"


def test_proxy_admin_jwt_auth_includes_identity_fields():
    """
    Test that the proxy admin early-return path in JWT auth populates
    user_id, team_id, team_alias, team_metadata, org_id, and end_user_id.

    Regression test: previously the is_proxy_admin branch only set user_role
    and parent_otel_span, discarding all identity fields resolved from the JWT.
    This caused blank Team Name and Internal User in Request Logs UI.
    """
    from litellm.proxy._types import LiteLLM_TeamTable, LitellmUserRoles, UserAPIKeyAuth

    team_object = LiteLLM_TeamTable(
        team_id="team-123",
        team_alias="my-team",
        metadata={"tags": ["prod"], "env": "production"},
    )

    # Simulate the proxy admin early-return path (user_api_key_auth.py ~line 586)
    result = UserAPIKeyAuth(
        api_key=None,
        user_role=LitellmUserRoles.PROXY_ADMIN,
        user_id="user-abc",
        team_id="team-123",
        team_alias=(team_object.team_alias if team_object is not None else None),
        team_metadata=team_object.metadata if team_object is not None else None,
        org_id="org-456",
        end_user_id="end-user-789",
        parent_otel_span=None,
    )

    assert result.user_role == LitellmUserRoles.PROXY_ADMIN
    assert result.user_id == "user-abc"
    assert result.team_id == "team-123"
    assert result.team_alias == "my-team"
    assert result.team_metadata == {"tags": ["prod"], "env": "production"}
    assert result.org_id == "org-456"
    assert result.end_user_id == "end-user-789"
    assert result.api_key is None


def test_proxy_admin_jwt_auth_handles_no_team_object():
    """
    Test that the proxy admin early-return path works correctly when
    team_object is None (user has admin role but no team association).
    """
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth

    team_object = None

    result = UserAPIKeyAuth(
        api_key=None,
        user_role=LitellmUserRoles.PROXY_ADMIN,
        user_id="admin-user",
        team_id=None,
        team_alias=(team_object.team_alias if team_object is not None else None),
        team_metadata=team_object.metadata if team_object is not None else None,
        org_id=None,
        end_user_id=None,
        parent_otel_span=None,
    )

    assert result.user_role == LitellmUserRoles.PROXY_ADMIN
    assert result.user_id == "admin-user"
    assert result.team_id is None
    assert result.team_alias is None
    assert result.team_metadata is None
    assert result.org_id is None
    assert result.end_user_id is None


class TestJWTOAuth2Coexistence:
    """
    Test that JWT and OAuth2 auth can coexist on the same instance.

    When both enable_jwt_auth and enable_oauth2_auth are True, the proxy should
    route tokens based on their format:
    - JWT tokens (3 dot-separated parts) -> JWT auth handler
    - Opaque tokens -> OAuth2 auth handler
    """

    def test_is_jwt_detects_jwt_tokens(self):
        """JWT tokens have 3 dot-separated parts."""
        assert JWTHandler.is_jwt("header.payload.signature") is True
        assert (
            JWTHandler.is_jwt("eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ1c2VyMSJ9.sig123")
            is True
        )

    def test_is_jwt_rejects_opaque_tokens(self):
        """Opaque OAuth2 tokens do not have 3 dot-separated parts."""
        assert JWTHandler.is_jwt("some-opaque-oauth2-token") is False
        assert JWTHandler.is_jwt("sk-12345678") is False
        assert JWTHandler.is_jwt("Bearer token") is False
        assert JWTHandler.is_jwt("two.parts") is False

    def test_is_jwt_returns_false_for_none(self):
        """None token (missing Authorization header) should not be treated as JWT."""
        assert JWTHandler.is_jwt(None) is False

    @pytest.mark.asyncio
    async def test_both_enabled_opaque_token_uses_oauth2(self):
        """
        When both enable_jwt_auth and enable_oauth2_auth are True,
        an opaque token should be handled by OAuth2 auth (not JWT).
        """
        opaque_token = "some-opaque-m2m-oauth2-token"

        general_settings = {
            "enable_oauth2_auth": True,
            "enable_jwt_auth": True,
        }

        mock_oauth2_response = UserAPIKeyAuth(
            api_key=opaque_token,
            user_id="machine-client-1",
            team_id="m2m-team",
        )

        mock_request = MagicMock()
        mock_request.url.path = "/v1/chat/completions"
        mock_request.headers = {"authorization": f"Bearer {opaque_token}"}
        mock_request.query_params = {}

        with patch(
            "litellm.proxy.proxy_server.general_settings", general_settings
        ), patch("litellm.proxy.proxy_server.premium_user", True), patch(
            "litellm.proxy.proxy_server.master_key", "sk-master"
        ), patch(
            "litellm.proxy.proxy_server.prisma_client", None
        ), patch(
            "litellm.proxy.auth.user_api_key_auth.Oauth2Handler.check_oauth2_token",
            new_callable=AsyncMock,
            return_value=mock_oauth2_response,
        ) as mock_oauth2, patch(
            "litellm.proxy.auth.user_api_key_auth.JWTAuthManager.auth_builder",
            new_callable=AsyncMock,
        ) as mock_jwt_auth:
            litellm.proxy.proxy_server.jwt_handler.update_environment(
                prisma_client=None,
                user_api_key_cache=DualCache(),
                litellm_jwtauth=LiteLLM_JWTAuth(),
            )

            result = await user_api_key_auth(
                request=mock_request,
                api_key=f"Bearer {opaque_token}",
            )

            # OAuth2 SHOULD be called for opaque tokens
            mock_oauth2.assert_called_once_with(token=opaque_token)
            # JWT auth should NOT be called
            mock_jwt_auth.assert_not_called()
            assert result.user_id == "machine-client-1"

    @pytest.mark.asyncio
    async def test_both_enabled_jwt_token_skips_oauth2(self):
        """
        When both enable_jwt_auth and enable_oauth2_auth are True,
        a JWT-formatted token should skip OAuth2 and reach the JWT handler.
        """
        jwt_token = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ1c2VyMSJ9.signature"

        general_settings = {
            "enable_oauth2_auth": True,
            "enable_jwt_auth": True,
        }

        mock_jwt_result = {
            "is_proxy_admin": True,
            "team_object": None,
            "user_object": None,
            "end_user_object": None,
            "org_object": None,
            "token": jwt_token,
            "team_id": "jwt-team",
            "user_id": "jwt-human-user",
            "end_user_id": None,
            "org_id": None,
            "team_membership": None,
            "jwt_claims": {"sub": "user1"},
        }

        mock_request = MagicMock()
        mock_request.url.path = "/v1/chat/completions"
        mock_request.headers = {"authorization": f"Bearer {jwt_token}"}
        mock_request.query_params = {}

        with patch(
            "litellm.proxy.proxy_server.general_settings", general_settings
        ), patch("litellm.proxy.proxy_server.premium_user", True), patch(
            "litellm.proxy.proxy_server.master_key", "sk-master"
        ), patch(
            "litellm.proxy.proxy_server.prisma_client", None
        ), patch(
            "litellm.proxy.auth.user_api_key_auth.Oauth2Handler.check_oauth2_token",
            new_callable=AsyncMock,
        ) as mock_oauth2, patch(
            "litellm.proxy.auth.user_api_key_auth.JWTAuthManager.auth_builder",
            new_callable=AsyncMock,
            return_value=mock_jwt_result,
        ) as mock_jwt_auth:
            litellm.proxy.proxy_server.jwt_handler.update_environment(
                prisma_client=None,
                user_api_key_cache=DualCache(),
                litellm_jwtauth=LiteLLM_JWTAuth(),
            )

            result = await user_api_key_auth(
                request=mock_request,
                api_key=f"Bearer {jwt_token}",
            )

            # OAuth2 should NOT be called for JWT tokens
            mock_oauth2.assert_not_called()
            # JWT auth SHOULD be called
            mock_jwt_auth.assert_called_once()
            assert result.user_id == "jwt-human-user"

    @pytest.mark.asyncio
    async def test_routing_override_routes_matching_jwt_to_oauth2(self):
        """
        When routing_overrides match JWT claims, route JWT-shaped token to OAuth2.
        """
        jwt_token = (
            "eyJhbGciOiJSUzI1NiJ9."
            "eyJpc3MiOiJtYWNoaW5lLWlzc3Vlci5leGFtcGxlLmNvbSIsImNsaWVudF9pZCI6Ik1JRF9MSVRFTExNIn0."
            "c2ln"
        )
        general_settings = {
            "enable_oauth2_auth": True,
            "enable_jwt_auth": True,
        }
        mock_oauth2_response = UserAPIKeyAuth(
            api_key=jwt_token,
            user_id="machine-client-override",
        )

        mock_request = MagicMock()
        mock_request.url.path = "/v1/chat/completions"
        mock_request.headers = {"authorization": f"Bearer {jwt_token}"}
        mock_request.query_params = {}

        with patch(
            "litellm.proxy.proxy_server.general_settings", general_settings
        ), patch("litellm.proxy.proxy_server.premium_user", True), patch(
            "litellm.proxy.proxy_server.master_key", "sk-master"
        ), patch(
            "litellm.proxy.proxy_server.prisma_client", None
        ), patch(
            "litellm.proxy.auth.user_api_key_auth.Oauth2Handler.check_oauth2_token",
            new_callable=AsyncMock,
            return_value=mock_oauth2_response,
        ) as mock_oauth2, patch(
            "litellm.proxy.auth.user_api_key_auth.JWTAuthManager.auth_builder",
            new_callable=AsyncMock,
        ) as mock_jwt_auth:
            litellm.proxy.proxy_server.jwt_handler.update_environment(
                prisma_client=None,
                user_api_key_cache=DualCache(),
                litellm_jwtauth=LiteLLM_JWTAuth(
                    routing_overrides=[
                        JWTRoutingOverride(
                            iss="machine-issuer.example.com",
                            client_id="MID_LITELLM",
                            path="oauth2",
                        )
                    ]
                ),
            )

            result = await user_api_key_auth(
                request=mock_request,
                api_key=f"Bearer {jwt_token}",
            )

            mock_oauth2.assert_called_once_with(token=jwt_token)
            mock_jwt_auth.assert_not_called()
            assert result.user_id == "machine-client-override"

    @pytest.mark.asyncio
    async def test_routing_override_does_not_match_client_id_falls_back_to_jwt(self):
        """
        If override ISS matches but client_id does not, continue default JWT flow.
        """
        jwt_token = (
            "eyJhbGciOiJSUzI1NiJ9."
            "eyJpc3MiOiJtYWNoaW5lLWlzc3Vlci5leGFtcGxlLmNvbSIsImNsaWVudF9pZCI6IlVTRVJfUE9SVEFMIn0."
            "c2ln"
        )
        general_settings = {
            "enable_oauth2_auth": True,
            "enable_jwt_auth": True,
        }
        mock_jwt_result = {
            "is_proxy_admin": True,
            "team_object": None,
            "user_object": None,
            "end_user_object": None,
            "org_object": None,
            "token": jwt_token,
            "team_id": "jwt-team",
            "user_id": "jwt-user-no-override",
            "end_user_id": None,
            "org_id": None,
            "team_membership": None,
            "jwt_claims": {"sub": "user1"},
        }

        mock_request = MagicMock()
        mock_request.url.path = "/v1/chat/completions"
        mock_request.headers = {"authorization": f"Bearer {jwt_token}"}
        mock_request.query_params = {}

        with patch(
            "litellm.proxy.proxy_server.general_settings", general_settings
        ), patch("litellm.proxy.proxy_server.premium_user", True), patch(
            "litellm.proxy.proxy_server.master_key", "sk-master"
        ), patch(
            "litellm.proxy.proxy_server.prisma_client", None
        ), patch(
            "litellm.proxy.auth.user_api_key_auth.Oauth2Handler.check_oauth2_token",
            new_callable=AsyncMock,
        ) as mock_oauth2, patch(
            "litellm.proxy.auth.user_api_key_auth.JWTAuthManager.auth_builder",
            new_callable=AsyncMock,
            return_value=mock_jwt_result,
        ) as mock_jwt_auth:
            litellm.proxy.proxy_server.jwt_handler.update_environment(
                prisma_client=None,
                user_api_key_cache=DualCache(),
                litellm_jwtauth=LiteLLM_JWTAuth(
                    routing_overrides=[
                        JWTRoutingOverride(
                            iss="machine-issuer.example.com",
                            client_id="MID_LITELLM",
                            path="oauth2",
                        )
                    ]
                ),
            )

            result = await user_api_key_auth(
                request=mock_request,
                api_key=f"Bearer {jwt_token}",
            )

            mock_oauth2.assert_not_called()
            mock_jwt_auth.assert_called_once()
            assert result.user_id == "jwt-user-no-override"

    @pytest.mark.asyncio
    async def test_routing_override_matches_aud_claim_list_and_list_selectors(self):
        """
        Match routing override when selectors are lists and token aud claim is a list.
        """
        jwt_token = (
            "eyJhbGciOiJSUzI1NiJ9."
            "eyJpc3MiOiJtYWNoaW5lLWlzc3Vlci5leGFtcGxlLmNvbSIsImNsaWVudF9pZCI6Ik1JRF9MSVRFTExNIiwiYXVkIjpbImFwaTovL2xpdGVsbG0iLCJhcGk6Ly9vdGhlciJdfQ."
            "c2ln"
        )
        general_settings = {
            "enable_oauth2_auth": True,
            "enable_jwt_auth": True,
        }
        mock_oauth2_response = UserAPIKeyAuth(
            api_key=jwt_token,
            user_id="machine-client-aud-list",
        )

        mock_request = MagicMock()
        mock_request.url.path = "/v1/chat/completions"
        mock_request.headers = {"authorization": f"Bearer {jwt_token}"}
        mock_request.query_params = {}

        with patch(
            "litellm.proxy.proxy_server.general_settings", general_settings
        ), patch("litellm.proxy.proxy_server.premium_user", True), patch(
            "litellm.proxy.proxy_server.master_key", "sk-master"
        ), patch(
            "litellm.proxy.proxy_server.prisma_client", None
        ), patch(
            "litellm.proxy.auth.user_api_key_auth.Oauth2Handler.check_oauth2_token",
            new_callable=AsyncMock,
            return_value=mock_oauth2_response,
        ) as mock_oauth2, patch(
            "litellm.proxy.auth.user_api_key_auth.JWTAuthManager.auth_builder",
            new_callable=AsyncMock,
        ) as mock_jwt_auth:
            litellm.proxy.proxy_server.jwt_handler.update_environment(
                prisma_client=None,
                user_api_key_cache=DualCache(),
                litellm_jwtauth=LiteLLM_JWTAuth(
                    routing_overrides=[
                        JWTRoutingOverride(
                            iss=[
                                "machine-issuer.example.com",
                                "other-issuer.example.com",
                            ],
                            client_id=["MID_LITELLM", "MID_BACKUP"],
                            aud=["api://litellm", "api://fallback"],
                            path="oauth2",
                        )
                    ]
                ),
            )

            result = await user_api_key_auth(
                request=mock_request,
                api_key=f"Bearer {jwt_token}",
            )

            mock_oauth2.assert_called_once_with(token=jwt_token)
            mock_jwt_auth.assert_not_called()
            assert result.user_id == "machine-client-aud-list"

    @pytest.mark.asyncio
    async def test_only_oauth2_enabled_handles_all_tokens(self):
        """
        When only enable_oauth2_auth is True (no JWT), all LLM API tokens
        should go through OAuth2 - backward compatible behavior.
        """
        jwt_like_token = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ1c2VyMSJ9.signature"

        general_settings = {
            "enable_oauth2_auth": True,
            "enable_jwt_auth": False,
        }

        mock_oauth2_response = UserAPIKeyAuth(
            api_key=jwt_like_token,
            user_id="oauth2-user",
        )

        mock_request = MagicMock()
        mock_request.url.path = "/v1/chat/completions"
        mock_request.headers = {"authorization": f"Bearer {jwt_like_token}"}
        mock_request.query_params = {}

        with patch(
            "litellm.proxy.proxy_server.general_settings", general_settings
        ), patch("litellm.proxy.proxy_server.premium_user", True), patch(
            "litellm.proxy.proxy_server.master_key", "sk-master"
        ), patch(
            "litellm.proxy.proxy_server.prisma_client", None
        ), patch(
            "litellm.proxy.auth.user_api_key_auth.Oauth2Handler.check_oauth2_token",
            new_callable=AsyncMock,
            return_value=mock_oauth2_response,
        ) as mock_oauth2:
            result = await user_api_key_auth(
                request=mock_request,
                api_key=f"Bearer {jwt_like_token}",
            )

            # OAuth2 should handle it since JWT auth is disabled
            mock_oauth2.assert_called_once_with(token=jwt_like_token)
            assert result.user_id == "oauth2-user"


@pytest.mark.asyncio
async def test_user_api_key_auth_builder_no_blocking_calls():
    """
    _user_api_key_auth_builder must never call any synchronous DualCache method
    (set_cache, get_cache, batch_get_cache, increment_cache, delete_cache) on
    the hot auth path — those methods call Redis synchronously and block the
    event loop. Only async_* variants are allowed.
    """
    from starlette.datastructures import URL
    from starlette.requests import Request

    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.auth.user_api_key_auth import _user_api_key_auth_builder

    _blocking_methods = [
        "set_cache",
        "get_cache",
        "batch_get_cache",
        "increment_cache",
        "delete_cache",
    ]

    api_key = "sk-test-no-blocking-cache"
    valid_token = UserAPIKeyAuth(
        api_key=api_key,
        token=api_key,
        user_role=LitellmUserRoles.INTERNAL_USER,
        team_id="team-abc",
    )

    mock_cache = AsyncMock()
    mock_cache.async_get_cache = AsyncMock(return_value=valid_token)
    mock_cache.async_set_cache = AsyncMock(return_value=None)
    # Wire sync methods on the instance as plain MagicMocks (no side_effect) so
    # calls are recorded but not raised — the function's broad except Exception
    # would swallow a raised error. We assert not_called() after the run instead.
    for _m in _blocking_methods:
        setattr(mock_cache, _m, MagicMock())

    mock_proxy_logging_obj = MagicMock()
    mock_proxy_logging_obj.internal_usage_cache = MagicMock()
    mock_proxy_logging_obj.internal_usage_cache.dual_cache = AsyncMock()
    mock_proxy_logging_obj.internal_usage_cache.dual_cache.async_delete_cache = (
        AsyncMock()
    )
    mock_proxy_logging_obj.post_call_failure_hook = AsyncMock(return_value=None)

    import litellm.proxy.proxy_server as _proxy_server_mod

    _attrs = {
        "prisma_client": MagicMock(),
        "user_api_key_cache": mock_cache,
        "proxy_logging_obj": mock_proxy_logging_obj,
        "master_key": "sk-master-key",
        "general_settings": {},
        "llm_model_list": [],
        "llm_router": None,
        "open_telemetry_logger": None,
        "model_max_budget_limiter": MagicMock(),
        "user_custom_auth": None,
        "jwt_handler": None,
        "litellm_proxy_admin_name": "admin",
    }
    _originals = {k: getattr(_proxy_server_mod, k, None) for k in _attrs}

    try:
        for k, v in _attrs.items():
            setattr(_proxy_server_mod, k, v)

        request = Request(scope={"type": "http"})
        request._url = URL(url="/chat/completions")

        import contextlib

        from litellm.caching.dual_cache import DualCache

        blocking_patches = [
            patch.object(
                DualCache,
                m,
                MagicMock(
                    side_effect=AssertionError(
                        f"Blocking DualCache.{m}() called on async hot path — use async_{m}() instead"
                    )
                ),
            )
            for m in _blocking_methods
        ]

        with contextlib.ExitStack() as stack:
            for p in blocking_patches:
                stack.enter_context(p)
            stack.enter_context(
                patch(
                    "litellm.proxy.auth.user_api_key_auth.get_key_object",
                    new_callable=AsyncMock,
                    return_value=valid_token,
                )
            )
            stack.enter_context(
                patch(
                    "litellm.proxy.auth.user_api_key_auth.get_team_object",
                    new_callable=AsyncMock,
                    return_value=None,
                )
            )
            await _user_api_key_auth_builder(
                request=request,
                api_key=f"Bearer {api_key}",
                azure_api_key_header="",
                anthropic_api_key_header=None,
                google_ai_studio_api_key_header=None,
                azure_apim_header=None,
                request_data={},
            )

        for _m in _blocking_methods:
            mock = getattr(mock_cache, _m)
            assert mock.call_count == 0, (
                f"Blocking DualCache.{_m}() was called {mock.call_count} time(s) "
                f"on the async hot path — use async_{_m}() instead"
            )

    finally:
        for k, v in _originals.items():
            setattr(_proxy_server_mod, k, v)
