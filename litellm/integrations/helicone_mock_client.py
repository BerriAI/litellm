"""
Mock HTTP client for Helicone integration testing.

This module intercepts Helicone API calls and returns successful mock responses,
allowing full code execution without making actual network calls.

Usage:
    Set HELICONE_MOCK=true in environment variables or config to enable mock mode.
"""

import os
import time

from litellm._logging import verbose_logger
from litellm.integrations.mock_client_factory import MockClientConfig, MockResponse, create_mock_client_factory

# Use factory for should_use_mock and MockResponse
# HTTPHandler uses self.client.send(), not self.client.post(), so we need custom patching
_config = MockClientConfig(
    name="HELICONE",
    env_var="HELICONE_MOCK",
    default_latency_ms=100,
    default_status_code=200,
    default_json_data={"status": "success"},
    url_matchers=[
        ".hconeai.com",
        "hconeai.com",
        ".helicone.ai",
        "helicone.ai",
    ],
    patch_async_handler=False,
    patch_sync_client=False,  # HTTPHandler uses self.client.send(), not self.client.post()
)

# Get should_use_mock from factory (but don't use its patching since HTTPHandler is different)
_, should_use_helicone_mock = create_mock_client_factory(_config)

# Store original HTTPHandler.post method (Helicone-specific)
_original_http_handler_post = None
_mocks_initialized = False

# Default mock latency in seconds
_MOCK_LATENCY_SECONDS = float(os.getenv("HELICONE_MOCK_LATENCY_MS", "100")) / 1000.0


def _is_helicone_url(url: str) -> bool:
    """Check if URL is a Helicone API URL."""
    url_lower = url.lower()
    return "hconeai.com" in url_lower or "helicone.ai" in url_lower


def _mock_http_handler_post(self, url, data=None, json=None, params=None, headers=None, timeout=None, stream=False, files=None, content=None, logging_obj=None):
    """Monkey-patched HTTPHandler.post that intercepts Helicone calls."""
    # Only mock Helicone API calls
    if isinstance(url, str) and _is_helicone_url(url):
        verbose_logger.info(f"[HELICONE MOCK] POST to {url}")
        time.sleep(_MOCK_LATENCY_SECONDS)
        return MockResponse(
            status_code=_config.default_status_code,
            json_data=_config.default_json_data,
            url=url,
            elapsed_seconds=_MOCK_LATENCY_SECONDS
        )
    if _original_http_handler_post is not None:
        return _original_http_handler_post(self, url=url, data=data, json=json, params=params, headers=headers, timeout=timeout, stream=stream, files=files, content=content, logging_obj=logging_obj)
    raise RuntimeError("Original HTTPHandler.post not available")


def create_mock_helicone_client():
    """
    Monkey-patch HTTPHandler.post to intercept Helicone calls.
    
    Helicone uses litellm.module_level_client which is an HTTPHandler instance.
    HTTPHandler.post uses self.client.send(), not self.client.post(), so we need
    custom patching (similar to how GCS has custom GET/DELETE handlers).
    
    This function is idempotent - it only initializes mocks once, even if called multiple times.
    """
    global _original_http_handler_post, _mocks_initialized
    
    if _mocks_initialized:
        return
    
    verbose_logger.debug("[HELICONE MOCK] Initializing Helicone mock client...")
    
    from litellm.llms.custom_httpx.http_handler import HTTPHandler
    
    if _original_http_handler_post is None:
        _original_http_handler_post = HTTPHandler.post
        HTTPHandler.post = _mock_http_handler_post  # type: ignore
        verbose_logger.debug("[HELICONE MOCK] Patched HTTPHandler.post")
    
    verbose_logger.debug(f"[HELICONE MOCK] Mock latency set to {_MOCK_LATENCY_SECONDS*1000:.0f}ms")
    verbose_logger.debug("[HELICONE MOCK] Helicone mock client initialization complete")
    
    _mocks_initialized = True
