"""
Mock httpx client for Langfuse integration testing.

This module intercepts Langfuse API calls and returns successful mock responses,
allowing full code execution without making actual network calls.

Usage:
    Set LANGFUSE_MOCK=true in environment variables or config to enable mock mode.
"""

import httpx
import json
from typing import Dict, Optional

from litellm._logging import verbose_logger

# Store original post method for restoration
_original_httpx_post = None


class MockLangfuseResponse:
    """Mock httpx.Response that satisfies Langfuse SDK requirements."""
    
    def __init__(self, status_code: int = 200, json_data: Optional[Dict] = None, url: Optional[str] = None):
        self.status_code = status_code
        self._json_data = json_data or {"status": "success"}
        self.headers = httpx.Headers({})
        self.is_success = status_code < 400
        self.is_error = status_code >= 400
        self.is_redirect = 300 <= status_code < 400
        self.url = httpx.URL(url) if url else httpx.URL("")
        self.elapsed = httpx.Timeout(0.0)
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


def _mock_httpx_post(self, url, **kwargs):
    """Monkey-patched httpx.Client.post that intercepts Langfuse calls."""
    # Only mock Langfuse API calls
    if isinstance(url, str) and ("langfuse.com" in url or "langfuse" in url.lower()):
        print(f"[LANGFUSE MOCK] POST to {url}")
        return MockLangfuseResponse(status_code=200, json_data={"status": "success"}, url=url)
    # For non-Langfuse calls, use original method
    if _original_httpx_post is not None:
        return _original_httpx_post(self, url, **kwargs)
    # Fallback: if original not set, create a temporary client for this call
    import httpx
    with httpx.Client() as client:
        return client.post(url, **kwargs)


def create_mock_langfuse_client():
    """
    Monkey-patch httpx.Client.post to intercept Langfuse calls.
    
    Returns a real httpx.Client instance - the monkey-patch intercepts all calls.
    """
    global _original_httpx_post
    
    if _original_httpx_post is None:
        _original_httpx_post = httpx.Client.post
        httpx.Client.post = _mock_httpx_post  # type: ignore
        print("[LANGFUSE MOCK] Patched httpx.Client.post")
    
    # Return real client - monkey-patch handles interception
    return httpx.Client()


def should_use_langfuse_mock() -> bool:
    """
    Determine if Langfuse should run in mock mode.
    
    Checks the LANGFUSE_MOCK environment variable.
    
    Returns:
        bool: True if mock mode should be enabled
    """
    import os
    from litellm.secret_managers.main import str_to_bool
    
    mock_mode = os.getenv("LANGFUSE_MOCK", "false")
    result = str_to_bool(mock_mode)
    
    # Ensure we return a bool, not None
    result = bool(result) if result is not None else False
    
    if result:
        verbose_logger.info("Langfuse Mock Mode: ENABLED - API calls will be mocked")
    
    return result
