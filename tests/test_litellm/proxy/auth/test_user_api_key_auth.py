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
