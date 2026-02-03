"""
Mock httpx client for Langfuse integration testing.

This module intercepts Langfuse API calls and returns successful mock responses,
allowing full code execution without making actual network calls.

Usage:
    Set LANGFUSE_MOCK=true in environment variables or config to enable mock mode.
"""

import httpx
import json
from datetime import timedelta
from typing import Dict, Optional

from litellm._logging import verbose_logger

_original_httpx_post = None

# Default mock latency in seconds (simulates network round-trip)
# Typical Langfuse API calls take 50-150ms
_MOCK_LATENCY_SECONDS = float(__import__("os").getenv("LANGFUSE_MOCK_LATENCY_MS", "100")) / 1000.0


class MockLangfuseResponse:
    """Mock httpx.Response that satisfies Langfuse SDK requirements."""
    
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
        return self._text
    
    @property
    def content(self) -> bytes:
        return self._content
    
    def json(self) -> Dict:
        return self._json_data
    
    def read(self) -> bytes:
        return self._content
    
    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


def _is_langfuse_url(url) -> bool:
    """Check if URL is a Langfuse domain."""
    try:
        parsed_url = httpx.URL(url) if isinstance(url, str) else url
        hostname = parsed_url.host or ""
        
        return (
            hostname.endswith(".langfuse.com") or
            hostname == "langfuse.com" or
            (hostname in ("localhost", "127.0.0.1") and "langfuse" in str(parsed_url).lower())
        )
    except Exception:
        return False


def _mock_httpx_post(self, url, **kwargs):
    """Monkey-patched httpx.Client.post that intercepts Langfuse calls."""
    if _is_langfuse_url(url):
        verbose_logger.info(f"[LANGFUSE MOCK] POST to {url}")
        return MockLangfuseResponse(status_code=200, json_data={"status": "success"}, url=url, elapsed_seconds=_MOCK_LATENCY_SECONDS)
    
    if _original_httpx_post is not None:
        return _original_httpx_post(self, url, **kwargs)


def create_mock_langfuse_client():
    """
    Monkey-patch httpx.Client.post to intercept Langfuse calls.
    
    Returns a real httpx.Client instance - the monkey-patch intercepts all calls.
    """
    global _original_httpx_post
    
    if _original_httpx_post is None:
        _original_httpx_post = httpx.Client.post
        httpx.Client.post = _mock_httpx_post  # type: ignore
        verbose_logger.debug("[LANGFUSE MOCK] Patched httpx.Client.post")
    
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
    result = bool(result) if result is not None else False
    
    if result:
        verbose_logger.info("Langfuse Mock Mode: ENABLED - API calls will be mocked")
    
    return result
