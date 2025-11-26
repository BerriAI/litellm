"""
Test search API logging and cost tracking in proxy.

Tests that search API requests are properly logged to LiteLLM_SpendLogs
with correct fields populated (call_type, model, custom_llm_provider, 
model_group, spend, etc.)
"""
import asyncio
import os
import sys
import time
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))
import litellm
from litellm import Router
from litellm.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.proxy_track_cost_callback import _ProxyDBLogger
from litellm.proxy.spend_tracking.spend_management_endpoints import view_spend_logs
from litellm.proxy.utils import ProxyLogging, hash_token, update_spend
from litellm.llms.base_llm.search.transformation import SearchResponse, SearchResult


@pytest.fixture
def prisma_client():
    from litellm.proxy import proxy_server
    from litellm.proxy.proxy_cli import append_query_params
    from litellm.proxy.utils import PrismaClient

    params = {"connection_limit": 100, "pool_timeout": 60}
    database_url = os.getenv("DATABASE_URL")
    if database_url is None:
        pytest.skip("DATABASE_URL not set")
    
    modified_url = append_query_params(database_url, params)
    os.environ["DATABASE_URL"] = modified_url

    user_api_key_cache = DualCache()
    proxy_logging_obj = ProxyLogging(user_api_key_cache=user_api_key_cache)

    prisma_client = PrismaClient(
        database_url=os.environ["DATABASE_URL"], proxy_logging_obj=proxy_logging_obj
    )

    proxy_server.litellm_proxy_budget_name = (
        f"litellm-proxy-budget-{time.time()}"
    )
    proxy_server.user_custom_key_generate = None

    return prisma_client


@pytest.mark.asyncio
async def test_search_api_logging_and_cost_tracking(prisma_client):
    """
    Test that search API requests are logged with correct fields and cost tracking.
    
    Verifies:
    1. Search request creates a spend log entry
    2. call_type is set to "asearch"
    3. model is set to search_tool_name
    4. custom_llm_provider is set correctly
    5. model_group is set to search_tool_name
    6. spend is calculated and logged
    """
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    await litellm.proxy.proxy_server.prisma_client.connect()

    # Setup router with search tool
    search_tool_name = "tavily-search"
    search_provider = "tavily"
    
    router = Router(model_list=[])
    router.search_tools = [
        {
            "search_tool_name": search_tool_name,
            "litellm_params": {
                "search_provider": search_provider,
            },
        }
    ]
    
    setattr(litellm.proxy.proxy_server, "llm_router", router)

    # Generate a test API key
    from litellm.proxy.management_endpoints.key_management_endpoints import generate_key_fn
    from litellm.proxy._types import GenerateKeyRequest

    from litellm.proxy._types import LitellmUserRoles
    
    user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        api_key="sk-1234",
        user_id="test_user",
    )

    key_request = GenerateKeyRequest(models=[], duration=None)
    key_response = await generate_key_fn(
        data=key_request, user_api_key_dict=user_api_key_dict
    )
    generated_key = key_response.key
    user_id = key_response.user_id

    # Create mock search response
    mock_search_result = SearchResult(
        title="Test Result",
        url="https://example.com",
        snippet="Test snippet",
    )
    
    mock_search_response = SearchResponse(
        object="search",
        results=[mock_search_result],
    )

    # Mock the search function to return our mock response
    with patch("litellm.search.main.asearch", new_callable=AsyncMock) as mock_asearch:
        mock_asearch.return_value = mock_search_response

        # Setup proxy logging
        user_api_key_cache = DualCache()
        proxy_logging_obj = ProxyLogging(user_api_key_cache=user_api_key_cache)
        setattr(litellm.proxy.proxy_server, "proxy_logging_obj", proxy_logging_obj)

        # Call the track_cost_callback directly to simulate what happens after a search
        proxy_db_logger = _ProxyDBLogger()
        
        # Simulate the kwargs that would be passed from the search endpoint
        request_id = "search_test_123"
        kwargs = {
            "call_type": "asearch",
            "model": search_tool_name,
            "custom_llm_provider": search_provider,
            "litellm_call_id": request_id,  # Set request_id in kwargs
            "litellm_params": {
                "metadata": {
                    "user_api_key": hash_token(generated_key),
                    "user_api_key_user_id": user_id,
                    "model_group": search_tool_name,
                }
            },
            "metadata": {
                "user_api_key": hash_token(generated_key),
                "user_api_key_user_id": user_id,
                "model_group": search_tool_name,
            },
            "response_cost": 0.008,  # Mock cost for tavily search
        }
        
        # Set id on the response object
        mock_search_response.id = request_id

        await proxy_db_logger._PROXY_track_cost_callback(
            kwargs=kwargs,
            completion_response=mock_search_response,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        # Wait for async operations
        await asyncio.sleep(2)
        await update_spend(
            prisma_client=prisma_client,
            db_writer_client=None,
            proxy_logging_obj=proxy_logging_obj,
        )

        # Query spend logs
        spend_logs = await view_spend_logs(
            request_id=request_id,
            user_api_key_dict=UserAPIKeyAuth(api_key=generated_key),
        )

        # Verify spend log was created
        assert len(spend_logs) == 1, f"Expected 1 spend log, got {len(spend_logs)}"

        spend_log = spend_logs[0]

        # Verify all fields are populated correctly
        assert spend_log.request_id == request_id
        assert spend_log.call_type == "asearch"
        assert spend_log.model == search_tool_name
        assert spend_log.custom_llm_provider == search_provider
        assert spend_log.model_group == search_tool_name
        assert spend_log.spend == 0.008
        # API key should be hashed (either the generated key or the one from metadata)
        assert spend_log.api_key != ""  # Should be populated
        # Note: user field may be empty if not set in the request, but user_id should be in metadata
        assert spend_log.metadata.get("user_api_key_user_id") == user_id or spend_log.user == user_id

        print(f"✅ Search API logging test passed!")
        print(f"   - call_type: {spend_log.call_type}")
        print(f"   - model: {spend_log.model}")
        print(f"   - custom_llm_provider: {spend_log.custom_llm_provider}")
        print(f"   - model_group: {spend_log.model_group}")
        print(f"   - spend: {spend_log.spend}")

