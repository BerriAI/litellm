"""
Mock client for GCS Bucket integration testing.

This module intercepts GCS API calls and Vertex AI auth calls, returning successful
mock responses, allowing full code execution without making actual network calls.

Usage:
    Set GCS_MOCK=true in environment variables or config to enable mock mode.
"""

import httpx
import json
import asyncio
from datetime import timedelta
from typing import Dict, Optional

from litellm._logging import verbose_logger

# Store original methods for restoration
_original_async_handler_post = None
_original_async_handler_get = None
_original_async_handler_delete = None

# Track if mocks have been initialized to avoid duplicate initialization
_mocks_initialized = False

# Default mock latency in seconds (simulates network round-trip)
# Typical GCS API calls take 100-300ms for uploads, 50-150ms for GET/DELETE
_MOCK_LATENCY_SECONDS = float(__import__("os").getenv("GCS_MOCK_LATENCY_MS", "150")) / 1000.0


class MockGCSResponse:
    """Mock httpx.Response that satisfies GCS API requirements."""
    
    def __init__(self, status_code: int = 200, json_data: Optional[Dict] = None, url: Optional[str] = None, elapsed_seconds: float = 0.0):
        self.status_code = status_code
        self._json_data = json_data or {"kind": "storage#object", "name": "mock-object"}
        self.headers = httpx.Headers({})
        self.is_success = status_code < 400
        self.is_error = status_code >= 400
        self.is_redirect = 300 <= status_code < 400
        self.url = httpx.URL(url) if url else httpx.URL("")
        # Set realistic elapsed time based on mock latency
        elapsed_time = elapsed_seconds if elapsed_seconds > 0 else _MOCK_LATENCY_SECONDS
        self.elapsed = timedelta(seconds=elapsed_time)
        self._text = json.dumps(self._json_data)
        self._content = self._text.encode("utf-8")
    
    @property
    def text(self) -> str:
        """Return response text."""
        return self._text
    
    @property
    def content(self) -> bytes:
        """Return response content."""
        return self._content
    
    def json(self) -> Dict:
        """Return JSON response data."""
        return self._json_data
    
    def read(self) -> bytes:
        """Read response content."""
        return self._content
    
    def raise_for_status(self):
        """Raise exception for error status codes."""
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


async def _mock_async_handler_post(self, url, data=None, json=None, params=None, headers=None, timeout=None, stream=False, logging_obj=None, files=None, content=None):
    """Monkey-patched AsyncHTTPHandler.post that intercepts GCS calls."""
    # Only mock GCS API calls
    if isinstance(url, str) and "storage.googleapis.com" in url:
        verbose_logger.info(f"[GCS MOCK] POST to {url}")
        # Simulate network latency
        await asyncio.sleep(_MOCK_LATENCY_SECONDS)
        return MockGCSResponse(
            status_code=200, 
            json_data={"kind": "storage#object", "name": "mock-object"}, 
            url=url,
            elapsed_seconds=_MOCK_LATENCY_SECONDS
        )
    # For non-GCS calls, use original method
    if _original_async_handler_post is not None:
        return await _original_async_handler_post(self, url=url, data=data, json=json, params=params, headers=headers, timeout=timeout, stream=stream, logging_obj=logging_obj, files=files, content=content)
    # Fallback: if original not set, raise error
    raise RuntimeError("Original AsyncHTTPHandler.post not available")


async def _mock_async_handler_get(self, url, params=None, headers=None, follow_redirects=None):
    """Monkey-patched AsyncHTTPHandler.get that intercepts GCS calls."""
    # Only mock GCS API calls
    if isinstance(url, str) and "storage.googleapis.com" in url:
        verbose_logger.info(f"[GCS MOCK] GET to {url}")
        # Simulate network latency
        await asyncio.sleep(_MOCK_LATENCY_SECONDS)
        return MockGCSResponse(
            status_code=200, 
            json_data={"data": "mock-log-data"}, 
            url=url,
            elapsed_seconds=_MOCK_LATENCY_SECONDS
        )
    # For non-GCS calls, use original method
    if _original_async_handler_get is not None:
        return await _original_async_handler_get(self, url=url, params=params, headers=headers, follow_redirects=follow_redirects)
    # Fallback: if original not set, raise error
    raise RuntimeError("Original AsyncHTTPHandler.get not available")


async def _mock_async_handler_delete(self, url, data=None, json=None, params=None, headers=None, timeout=None, stream=False, content=None):
    """Monkey-patched AsyncHTTPHandler.delete that intercepts GCS calls."""
    # Only mock GCS API calls
    if isinstance(url, str) and "storage.googleapis.com" in url:
        verbose_logger.info(f"[GCS MOCK] DELETE to {url}")
        # Simulate network latency
        await asyncio.sleep(_MOCK_LATENCY_SECONDS)
        return MockGCSResponse(
            status_code=204, 
            json_data={}, 
            url=url,
            elapsed_seconds=_MOCK_LATENCY_SECONDS
        )
    # For non-GCS calls, use original method
    if _original_async_handler_delete is not None:
        return await _original_async_handler_delete(self, url=url, data=data, json=json, params=params, headers=headers, timeout=timeout, stream=stream, content=content)
    # Fallback: if original not set, raise error
    raise RuntimeError("Original AsyncHTTPHandler.delete not available")


def create_mock_gcs_client():
    """
    Monkey-patch AsyncHTTPHandler methods to intercept GCS calls.
    
    AsyncHTTPHandler is used by LiteLLM's get_async_httpx_client() which is what
    GCSBucketBase uses for making API calls.
    
    This function is idempotent - it only initializes mocks once, even if called multiple times.
    """
    global _original_async_handler_post, _original_async_handler_get, _original_async_handler_delete
    global _mocks_initialized
    
    # If already initialized, skip
    if _mocks_initialized:
        return
    
    verbose_logger.debug("[GCS MOCK] Initializing GCS mock client...")
    
    # Patch AsyncHTTPHandler methods (used by LiteLLM's custom httpx handler)
    if _original_async_handler_post is None:
        from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
        _original_async_handler_post = AsyncHTTPHandler.post
        AsyncHTTPHandler.post = _mock_async_handler_post  # type: ignore
        verbose_logger.debug("[GCS MOCK] Patched AsyncHTTPHandler.post")
    
    if _original_async_handler_get is None:
        from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
        _original_async_handler_get = AsyncHTTPHandler.get
        AsyncHTTPHandler.get = _mock_async_handler_get  # type: ignore
        verbose_logger.debug("[GCS MOCK] Patched AsyncHTTPHandler.get")
    
    if _original_async_handler_delete is None:
        from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
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


def should_use_gcs_mock() -> bool:
    """
    Determine if GCS should run in mock mode.
    
    Checks the GCS_MOCK environment variable.
    
    Returns:
        bool: True if mock mode should be enabled
    """
    import os
    from litellm.secret_managers.main import str_to_bool
    
    mock_mode = os.getenv("GCS_MOCK", "false")
    result = str_to_bool(mock_mode)
    
    # Ensure we return a bool, not None
    result = bool(result) if result is not None else False
    
    if result:
        verbose_logger.info("GCS Mock Mode: ENABLED - API calls will be mocked")
    
    return result
