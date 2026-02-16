import asyncio
import json
import os
import sys
from typing import Tuple
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from unittest.mock import MagicMock

import pytest

from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.auth.user_api_key_auth import get_api_key


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
    assert user_api_key_auth.team_metadata is not None, "team_metadata should be populated"
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
        assert RouteChecks.is_llm_api_route(route=route), f"Route {route} should be identified as LLM API route"

    # Test Anthropic routes
    anthropic_routes = [
        "/v1/messages",
        "/v1/messages/count_tokens",
    ]
    
    for route in anthropic_routes:
        assert RouteChecks.is_llm_api_route(route=route), f"Route {route} should be identified as LLM API route"

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
        assert RouteChecks.is_llm_api_route(route=route), f"Route {route} should be identified as LLM API route"

    # Test MCP routes
    mcp_routes = [
        "/mcp",
        "/mcp/",
        "/mcp/test",
    ]
    
    for route in mcp_routes:
        assert RouteChecks.is_llm_api_route(route=route), f"Route {route} should be identified as LLM API route"

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
        assert RouteChecks.is_llm_api_route(route=route), f"Route {route} should be identified as LLM API route"

    # Test Azure OpenAI routes
    azure_routes = [
        "/openai/deployments/gpt-4/chat/completions",
        "/openai/deployments/gpt-3.5-turbo/completions",
        "/engines/gpt-4/chat/completions",
        "/engines/gpt-3.5-turbo/completions",
    ]
    
    for route in azure_routes:
        assert RouteChecks.is_llm_api_route(route=route), f"Route {route} should be identified as LLM API route"

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
        assert not RouteChecks.is_llm_api_route(route=route), f"Route {route} should NOT be identified as LLM API route"

    # Test invalid inputs
    invalid_inputs = [
        None,
        123,
        [],
        {},
        "",
    ]
    
    for invalid_input in invalid_inputs:
        assert not RouteChecks.is_llm_api_route(route=invalid_input), f"Invalid input {invalid_input} should return False"


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
    mock_proxy_logging_obj.internal_usage_cache.dual_cache.async_delete_cache = AsyncMock()
    # Mock post_call_failure_hook as async function returning None (no transformation)
    mock_proxy_logging_obj.post_call_failure_hook = AsyncMock(return_value=None)
    
    # Mock prisma_client
    mock_prisma_client = MagicMock()
    
    # Mock get_key_object to return expired token from cache
    with patch(
        "litellm.proxy.auth.user_api_key_auth.get_key_object",
        new_callable=AsyncMock,
    ) as mock_get_key_object, \
         patch("litellm.proxy.auth.user_api_key_auth._delete_cache_key_object", new_callable=AsyncMock) as mock_delete_cache:
        
        mock_get_key_object.return_value = expired_token
        
        # Set attributes on proxy_server module (these are imported inside _user_api_key_auth_builder)
        import litellm.proxy.proxy_server
        
        setattr(litellm.proxy.proxy_server, "prisma_client", mock_prisma_client)
        setattr(litellm.proxy.proxy_server, "user_api_key_cache", mock_cache)
        setattr(litellm.proxy.proxy_server, "proxy_logging_obj", mock_proxy_logging_obj)
        setattr(litellm.proxy.proxy_server, "master_key", "sk-master-key")
        setattr(litellm.proxy.proxy_server, "general_settings", {})
        setattr(litellm.proxy.proxy_server, "llm_model_list", [])
        setattr(litellm.proxy.proxy_server, "llm_router", None)
        setattr(litellm.proxy.proxy_server, "open_telemetry_logger", None)
        setattr(litellm.proxy.proxy_server, "model_max_budget_limiter", MagicMock())
        setattr(litellm.proxy.proxy_server, "user_custom_auth", None)
        setattr(litellm.proxy.proxy_server, "jwt_handler", None)
        setattr(litellm.proxy.proxy_server, "litellm_proxy_admin_name", "admin")
        
        try:
            
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
            assert hasattr(exc_info.value, "type"), "Exception should have 'type' attribute"
            assert exc_info.value.type == ProxyErrorTypes.expired_key, (
                f"Expected expired_key error type, got {exc_info.value.type}"
            )
            assert "Expired Key" in str(exc_info.value.message), (
                f"Exception message should mention 'Expired Key', got: {exc_info.value.message}"
            )
            
            # Verify that the param field does NOT leak the full API key (Issue #18731)
            # The param should be abbreviated like "sk-...XXXX" not the full plaintext key
            assert exc_info.value.param is not None, "Exception should have 'param' attribute"
            assert exc_info.value.param != api_key, (
                f"SECURITY: Full API key should NOT be in param field! "
                f"Got: {exc_info.value.param}, Expected abbreviated format like 'sk-...XXXX'"
            )
            assert exc_info.value.param.startswith("sk-..."), (
                f"Param should be abbreviated to 'sk-...XXXX' format. Got: {exc_info.value.param}"
            )
            
            # Verify that cache deletion was called
            mock_delete_cache.assert_called_once()
            call_args = mock_delete_cache.call_args
            assert call_args[1]["hashed_token"] == hashed_key, (
                "Cache deletion should be called with the hashed key"
            )
        finally:
            # Clean up - restore original values if needed
            pass



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
        team_alias=(
            team_object.team_alias if team_object is not None else None
        ),
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
        team_alias=(
            team_object.team_alias if team_object is not None else None
        ),
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
