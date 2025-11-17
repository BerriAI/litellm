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

from litellm.proxy.auth.user_api_key_auth import get_api_key
from litellm.proxy.auth.route_checks import RouteChecks


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
