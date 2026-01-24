"""
Mock client for LangSmith integration testing.

This module intercepts LangSmith API calls and returns successful mock responses,
allowing full code execution without making actual network calls.

Usage:
    Set LANGSMITH_MOCK=true in environment variables or config to enable mock mode.
"""

import httpx
import json
import asyncio
from datetime import timedelta
from typing import Dict, Optional

from litellm._logging import verbose_logger

# Store original methods for restoration
_original_async_handler_post = None

# Track if mocks have been initialized to avoid duplicate initialization
_mocks_initialized = False

# Default mock latency in seconds (simulates network round-trip)
# Typical LangSmith API calls take 50-150ms
_MOCK_LATENCY_SECONDS = float(__import__("os").getenv("LANGSMITH_MOCK_LATENCY_MS", "100")) / 1000.0


class MockLangsmithResponse:
    """Mock httpx.Response that satisfies LangSmith API requirements."""
    
    def __init__(self, status_code: int = 200, json_data: Optional[Dict] = None, url: Optional[str] = None, elapsed_seconds: float = 0.0):
        self.status_code = status_code
        self._json_data = json_data or {"status": "success"}
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


def _is_langsmith_url(url) -> bool:
    """Check if URL is a LangSmith domain."""
    try:
        parsed_url = httpx.URL(url) if isinstance(url, str) else url
        hostname = parsed_url.host or ""
        
        return (
            hostname.endswith(".smith.langchain.com") or
            hostname == "api.smith.langchain.com" or
            "smith.langchain.com" in hostname or
            (hostname in ("localhost", "127.0.0.1") and "langsmith" in str(parsed_url).lower())
        )
    except Exception:
        return False


async def _mock_async_handler_post(self, url, data=None, json=None, params=None, headers=None, timeout=None, stream=False, logging_obj=None, files=None, content=None):
    """Monkey-patched AsyncHTTPHandler.post that intercepts LangSmith calls."""
    # Only mock LangSmith API calls
    if isinstance(url, str) and _is_langsmith_url(url):
        verbose_logger.info(f"[LANGSMITH MOCK] POST to {url}")
        # Simulate network latency
        await asyncio.sleep(_MOCK_LATENCY_SECONDS)
        return MockLangsmithResponse(
            status_code=200, 
            json_data={"status": "success", "ids": ["mock-run-id"]}, 
            url=url,
            elapsed_seconds=_MOCK_LATENCY_SECONDS
        )
    # For non-LangSmith calls, use original method
    if _original_async_handler_post is not None:
        return await _original_async_handler_post(self, url=url, data=data, json=json, params=params, headers=headers, timeout=timeout, stream=stream, logging_obj=logging_obj, files=files, content=content)
    # Fallback: if original not set, raise error
    raise RuntimeError("Original AsyncHTTPHandler.post not available")


def create_mock_langsmith_client():
    """
    Monkey-patch AsyncHTTPHandler.post to intercept LangSmith calls.
    
    AsyncHTTPHandler is used by LiteLLM's get_async_httpx_client() which is what
    LangsmithLogger uses for making API calls.
    
    This function is idempotent - it only initializes mocks once, even if called multiple times.
    """
    global _original_async_handler_post
    global _mocks_initialized
    
    # If already initialized, skip
    if _mocks_initialized:
        return
    
    verbose_logger.debug("[LANGSMITH MOCK] Initializing LangSmith mock client...")
    
    # Patch AsyncHTTPHandler.post (used by LiteLLM's custom httpx handler)
    if _original_async_handler_post is None:
        from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
        _original_async_handler_post = AsyncHTTPHandler.post
        AsyncHTTPHandler.post = _mock_async_handler_post  # type: ignore
        verbose_logger.debug("[LANGSMITH MOCK] Patched AsyncHTTPHandler.post")
    
    verbose_logger.debug(f"[LANGSMITH MOCK] Mock latency set to {_MOCK_LATENCY_SECONDS*1000:.0f}ms")
    verbose_logger.debug("[LANGSMITH MOCK] LangSmith mock client initialization complete")
    
    _mocks_initialized = True


def should_use_langsmith_mock() -> bool:
    """
    Determine if LangSmith should run in mock mode.
    
    Checks the LANGSMITH_MOCK environment variable.
    
    Returns:
        bool: True if mock mode should be enabled
    """
    import os
    from litellm.secret_managers.main import str_to_bool
    
    mock_mode = os.getenv("LANGSMITH_MOCK", "false")
    result = str_to_bool(mock_mode)
    result = bool(result) if result is not None else False
    
    if result:
        verbose_logger.info("LangSmith Mock Mode: ENABLED - API calls will be mocked")
    
    return result
