"""
Factory for creating mock HTTP clients for integration testing.

This module provides a simple factory pattern to create mock clients that intercept
API calls and return successful mock responses, allowing full code execution without
making actual network calls.
"""

import httpx
import json
import asyncio
from datetime import timedelta
from typing import Dict, Optional, List, cast
from dataclasses import dataclass

from litellm._logging import verbose_logger


@dataclass
class MockClientConfig:
    """Configuration for creating a mock client."""
    name: str  # e.g., "GCS", "LANGFUSE", "LANGSMITH", "DATADOG"
    env_var: str  # e.g., "GCS_MOCK", "LANGFUSE_MOCK"
    default_latency_ms: int = 100  # Default mock latency in milliseconds
    default_status_code: int = 200  # Default HTTP status code
    default_json_data: Optional[Dict] = None  # Default JSON response data
    url_matchers: Optional[List[str]] = None  # List of strings to match in URLs (e.g., ["storage.googleapis.com"])
    patch_async_handler: bool = True  # Whether to patch AsyncHTTPHandler.post
    patch_sync_client: bool = False  # Whether to patch httpx.Client.post
    patch_http_handler: bool = False  # Whether to patch HTTPHandler.post (for sync calls that use HTTPHandler)
    
    def __post_init__(self):
        """Ensure url_matchers is a list."""
        if self.url_matchers is None:
            self.url_matchers = []


class MockResponse:
    """Generic mock httpx.Response that satisfies API requirements."""
    
    def __init__(self, status_code: int = 200, json_data: Optional[Dict] = None, url: Optional[str] = None, elapsed_seconds: float = 0.0):
        self.status_code = status_code
        self._json_data = json_data or {"status": "success"}
        self.headers = httpx.Headers({})
        self.is_success = status_code < 400
        self.is_error = status_code >= 400
        self.is_redirect = 300 <= status_code < 400
        self.url = httpx.URL(url) if url else httpx.URL("")
        self.elapsed = timedelta(seconds=elapsed_seconds)
        self._text = json.dumps(self._json_data) if json_data else ""
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


def _is_url_match(url, matchers: List[str]) -> bool:
    """Check if URL matches any of the provided matchers."""
    try:
        parsed_url = httpx.URL(url) if isinstance(url, str) else url
        url_str = str(parsed_url).lower()
        hostname = parsed_url.host or ""
        
        for matcher in matchers:
            if matcher.lower() in url_str or matcher.lower() in hostname.lower():
                return True
        
        # Also check for localhost with matcher in path
        if hostname in ("localhost", "127.0.0.1"):
            for matcher in matchers:
                if matcher.lower() in url_str:
                    return True
        
        return False
    except Exception:
        return False


def create_mock_client_factory(config: MockClientConfig):  # noqa: PLR0915
    """
    Factory function that creates mock client functions based on configuration.
    
    Returns:
        tuple: (create_mock_client_func, should_use_mock_func)
    """
    # Store original methods for restoration
    _original_async_handler_post = None
    _original_sync_client_post = None
    _original_http_handler_post = None
    _mocks_initialized = False
    
    # Calculate mock latency
    import os
    latency_env = f"{config.name.upper()}_MOCK_LATENCY_MS"
    _MOCK_LATENCY_SECONDS = float(os.getenv(latency_env, str(config.default_latency_ms))) / 1000.0
    
    # Create URL matcher function
    def _is_mock_url(url) -> bool:
        # url_matchers is guaranteed to be a list after __post_init__
        return _is_url_match(url, cast(List[str], config.url_matchers))
    
    # Create async handler mock
    async def _mock_async_handler_post(self, url, data=None, json=None, params=None, headers=None, timeout=None, stream=False, logging_obj=None, files=None, content=None):
        """Monkey-patched AsyncHTTPHandler.post that intercepts API calls."""
        if isinstance(url, str) and _is_mock_url(url):
            verbose_logger.info(f"[{config.name} MOCK] POST to {url}")
            await asyncio.sleep(_MOCK_LATENCY_SECONDS)
            return MockResponse(
                status_code=config.default_status_code,
                json_data=config.default_json_data,
                url=url,
                elapsed_seconds=_MOCK_LATENCY_SECONDS
            )
        if _original_async_handler_post is not None:
            return await _original_async_handler_post(self, url=url, data=data, json=json, params=params, headers=headers, timeout=timeout, stream=stream, logging_obj=logging_obj, files=files, content=content)
        raise RuntimeError("Original AsyncHTTPHandler.post not available")
    
    # Create sync client mock
    def _mock_sync_client_post(self, url, **kwargs):
        """Monkey-patched httpx.Client.post that intercepts API calls."""
        if _is_mock_url(url):
            verbose_logger.info(f"[{config.name} MOCK] POST to {url} (sync)")
            return MockResponse(
                status_code=config.default_status_code,
                json_data=config.default_json_data,
                url=url,
                elapsed_seconds=_MOCK_LATENCY_SECONDS
            )
        if _original_sync_client_post is not None:
            return _original_sync_client_post(self, url, **kwargs)
    
    # Create HTTPHandler mock (for sync calls that use HTTPHandler.post)
    def _mock_http_handler_post(self, url, data=None, json=None, params=None, headers=None, timeout=None, stream=False, files=None, content=None, logging_obj=None):
        """Monkey-patched HTTPHandler.post that intercepts API calls."""
        if isinstance(url, str) and _is_mock_url(url):
            verbose_logger.info(f"[{config.name} MOCK] POST to {url}")
            import time
            time.sleep(_MOCK_LATENCY_SECONDS)
            return MockResponse(
                status_code=config.default_status_code,
                json_data=config.default_json_data,
                url=url,
                elapsed_seconds=_MOCK_LATENCY_SECONDS
            )
        if _original_http_handler_post is not None:
            return _original_http_handler_post(self, url=url, data=data, json=json, params=params, headers=headers, timeout=timeout, stream=stream, files=files, content=content, logging_obj=logging_obj)
        raise RuntimeError("Original HTTPHandler.post not available")
    
    # Create mock client initialization function
    def create_mock_client():
        """Initialize the mock client by patching HTTP handlers."""
        nonlocal _original_async_handler_post, _original_sync_client_post, _original_http_handler_post, _mocks_initialized
        
        if _mocks_initialized:
            return
        
        verbose_logger.debug(f"[{config.name} MOCK] Initializing {config.name} mock client...")
        
        if config.patch_async_handler and _original_async_handler_post is None:
            from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
            _original_async_handler_post = AsyncHTTPHandler.post
            AsyncHTTPHandler.post = _mock_async_handler_post  # type: ignore
            verbose_logger.debug(f"[{config.name} MOCK] Patched AsyncHTTPHandler.post")
        
        if config.patch_sync_client and _original_sync_client_post is None:
            _original_sync_client_post = httpx.Client.post
            httpx.Client.post = _mock_sync_client_post  # type: ignore
            verbose_logger.debug(f"[{config.name} MOCK] Patched httpx.Client.post")
        
        if config.patch_http_handler and _original_http_handler_post is None:
            from litellm.llms.custom_httpx.http_handler import HTTPHandler
            _original_http_handler_post = HTTPHandler.post
            HTTPHandler.post = _mock_http_handler_post  # type: ignore
            verbose_logger.debug(f"[{config.name} MOCK] Patched HTTPHandler.post")
        
        verbose_logger.debug(f"[{config.name} MOCK] Mock latency set to {_MOCK_LATENCY_SECONDS*1000:.0f}ms")
        verbose_logger.debug(f"[{config.name} MOCK] {config.name} mock client initialization complete")
        
        _mocks_initialized = True
    
    # Create should_use_mock function
    def should_use_mock() -> bool:
        """Determine if mock mode should be enabled."""
        import os
        from litellm.secret_managers.main import str_to_bool
        
        mock_mode = os.getenv(config.env_var, "false")
        result = str_to_bool(mock_mode)
        result = bool(result) if result is not None else False
        
        if result:
            verbose_logger.info(f"{config.name} Mock Mode: ENABLED - API calls will be mocked")
        
        return result
    
    return create_mock_client, should_use_mock
