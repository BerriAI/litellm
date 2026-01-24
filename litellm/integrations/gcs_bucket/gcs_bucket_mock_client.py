"""
Mock client for GCS Bucket integration testing.

This module intercepts GCS API calls and Vertex AI auth calls, returning successful
mock responses, allowing full code execution without making actual network calls.

Usage:
    Set GCS_MOCK=true in environment variables or config to enable mock mode.
"""

import asyncio

from litellm._logging import verbose_logger
from litellm.integrations.mock_client_factory import MockClientConfig, create_mock_client_factory, MockResponse

# Use factory for POST handler
_config = MockClientConfig(
    name="GCS",
    env_var="GCS_MOCK",
    default_latency_ms=150,
    default_status_code=200,
    default_json_data={"kind": "storage#object", "name": "mock-object"},
    url_matchers=["storage.googleapis.com"],
    patch_async_handler=True,
    patch_sync_client=False,
)

_create_mock_gcs_post, should_use_gcs_mock = create_mock_client_factory(_config)

# Store original methods for GET/DELETE (GCS-specific)
_original_async_handler_get = None
_original_async_handler_delete = None
_mocks_initialized = False

# Default mock latency in seconds (simulates network round-trip)
# Typical GCS API calls take 100-300ms for uploads, 50-150ms for GET/DELETE
_MOCK_LATENCY_SECONDS = float(__import__("os").getenv("GCS_MOCK_LATENCY_MS", "150")) / 1000.0


async def _mock_async_handler_get(self, url, params=None, headers=None, follow_redirects=None):
    """Monkey-patched AsyncHTTPHandler.get that intercepts GCS calls."""
    # Only mock GCS API calls
    if isinstance(url, str) and "storage.googleapis.com" in url:
        verbose_logger.info(f"[GCS MOCK] GET to {url}")
        await asyncio.sleep(_MOCK_LATENCY_SECONDS)
        # Return a minimal but valid StandardLoggingPayload JSON string as bytes
        # This matches what GCS returns when downloading with ?alt=media
        mock_payload = {
            "id": "mock-request-id",
            "trace_id": "mock-trace-id",
            "call_type": "completion",
            "stream": False,
            "response_cost": 0.0,
            "status": "success",
            "status_fields": {"llm_api_status": "success"},
            "custom_llm_provider": "mock",
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "startTime": 0.0,
            "endTime": 0.0,
            "completionStartTime": 0.0,
            "response_time": 0.0,
            "model_map_information": {"model": "mock-model"},
            "model": "mock-model",
            "model_id": None,
            "model_group": None,
            "api_base": "https://api.mock.com",
            "metadata": {},
            "cache_hit": None,
            "cache_key": None,
            "saved_cache_cost": 0.0,
            "request_tags": [],
            "end_user": None,
            "requester_ip_address": None,
            "messages": None,
            "response": None,
            "error_str": None,
            "error_information": None,
            "model_parameters": {},
            "hidden_params": {},
            "guardrail_information": None,
            "standard_built_in_tools_params": None,
        }
        return MockResponse(
            status_code=200,
            json_data=mock_payload,
            url=url,
            elapsed_seconds=_MOCK_LATENCY_SECONDS
        )
    if _original_async_handler_get is not None:
        return await _original_async_handler_get(self, url=url, params=params, headers=headers, follow_redirects=follow_redirects)
    raise RuntimeError("Original AsyncHTTPHandler.get not available")


async def _mock_async_handler_delete(self, url, data=None, json=None, params=None, headers=None, timeout=None, stream=False, content=None):
    """Monkey-patched AsyncHTTPHandler.delete that intercepts GCS calls."""
    # Only mock GCS API calls
    if isinstance(url, str) and "storage.googleapis.com" in url:
        verbose_logger.info(f"[GCS MOCK] DELETE to {url}")
        await asyncio.sleep(_MOCK_LATENCY_SECONDS)
        # DELETE returns 204 No Content with empty body (not JSON)
        return MockResponse(
            status_code=204,
            json_data=None,  # Empty body for DELETE
            url=url,
            elapsed_seconds=_MOCK_LATENCY_SECONDS
        )
    if _original_async_handler_delete is not None:
        return await _original_async_handler_delete(self, url=url, data=data, json=json, params=params, headers=headers, timeout=timeout, stream=stream, content=content)
    raise RuntimeError("Original AsyncHTTPHandler.delete not available")


def create_mock_gcs_client():
    """
    Monkey-patch AsyncHTTPHandler methods to intercept GCS calls.
    
    AsyncHTTPHandler is used by LiteLLM's get_async_httpx_client() which is what
    GCSBucketBase uses for making API calls.
    
    This function is idempotent - it only initializes mocks once, even if called multiple times.
    """
    global _original_async_handler_get, _original_async_handler_delete, _mocks_initialized
    
    # Use factory for POST handler
    _create_mock_gcs_post()
    
    # If already initialized, skip GET/DELETE patching
    if _mocks_initialized:
        return
    
    verbose_logger.debug("[GCS MOCK] Initializing GCS GET/DELETE handlers...")
    
    # Patch GET and DELETE handlers (GCS-specific)
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
    
    if _original_async_handler_get is None:
        _original_async_handler_get = AsyncHTTPHandler.get
        AsyncHTTPHandler.get = _mock_async_handler_get  # type: ignore
        verbose_logger.debug("[GCS MOCK] Patched AsyncHTTPHandler.get")
    
    if _original_async_handler_delete is None:
        _original_async_handler_delete = AsyncHTTPHandler.delete
        AsyncHTTPHandler.delete = _mock_async_handler_delete  # type: ignore
        verbose_logger.debug("[GCS MOCK] Patched AsyncHTTPHandler.delete")
    
    verbose_logger.debug(f"[GCS MOCK] Mock latency set to {_MOCK_LATENCY_SECONDS*1000:.0f}ms")
    verbose_logger.debug("[GCS MOCK] GCS mock client initialization complete")
    
    _mocks_initialized = True


def mock_vertex_auth_methods():
    """
    Monkey-patch Vertex AI auth methods to return fake tokens.
    This prevents auth failures when GCS_MOCK is enabled.
    
    This function is idempotent - it only patches once, even if called multiple times.
    """
    from litellm.llms.vertex_ai.vertex_llm_base import VertexBase
    
    # Store original methods if not already stored
    if not hasattr(VertexBase, '_original_ensure_access_token_async'):
        setattr(VertexBase, '_original_ensure_access_token_async', VertexBase._ensure_access_token_async)
        setattr(VertexBase, '_original_ensure_access_token', VertexBase._ensure_access_token)
        setattr(VertexBase, '_original_get_token_and_url', VertexBase._get_token_and_url)
        
        async def _mock_ensure_access_token_async(self, credentials, project_id, custom_llm_provider):
            """Mock async auth method - returns fake token."""
            verbose_logger.debug("[GCS MOCK] Vertex AI auth: _ensure_access_token_async called")
            return ("mock-gcs-token", "mock-project-id")
        
        def _mock_ensure_access_token(self, credentials, project_id, custom_llm_provider):
            """Mock sync auth method - returns fake token."""
            verbose_logger.debug("[GCS MOCK] Vertex AI auth: _ensure_access_token called")
            return ("mock-gcs-token", "mock-project-id")
        
        def _mock_get_token_and_url(self, model, auth_header, vertex_credentials, vertex_project, 
                                    vertex_location, gemini_api_key, stream, custom_llm_provider, api_base):
            """Mock get_token_and_url - returns fake token."""
            verbose_logger.debug("[GCS MOCK] Vertex AI auth: _get_token_and_url called")
            return ("mock-gcs-token", "https://storage.googleapis.com")
        
        # Patch the methods
        VertexBase._ensure_access_token_async = _mock_ensure_access_token_async  # type: ignore
        VertexBase._ensure_access_token = _mock_ensure_access_token  # type: ignore
        VertexBase._get_token_and_url = _mock_get_token_and_url  # type: ignore
        
        verbose_logger.debug("[GCS MOCK] Patched Vertex AI auth methods")


# should_use_gcs_mock is already created by the factory
