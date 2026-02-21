"""
Mock HTTP client for Braintrust integration testing.

This module intercepts Braintrust API calls and returns successful mock responses,
allowing full code execution without making actual network calls.

Usage:
    Set BRAINTRUST_MOCK=true in environment variables or config to enable mock mode.
"""

import os
import time
from urllib.parse import urlparse

from litellm._logging import verbose_logger
from litellm.integrations.mock_client_factory import MockClientConfig, MockResponse, create_mock_client_factory

# Use factory for should_use_mock and MockResponse
# Braintrust uses both HTTPHandler (sync) and AsyncHTTPHandler (async)
# Braintrust needs endpoint-specific responses, so we use custom HTTPHandler.post patching
_config = MockClientConfig(
    "BRAINTRUST",
    "BRAINTRUST_MOCK",
    default_latency_ms=100,
    default_status_code=200,
    default_json_data={"id": "mock-project-id", "status": "success"},
    url_matchers=[
        ".braintrustdata.com",
        "braintrustdata.com",
        ".braintrust.dev",
        "braintrust.dev",
    ],
    patch_async_handler=True,  # Patch AsyncHTTPHandler.post for async calls
    patch_sync_client=False,  # HTTPHandler uses self.client.send(), not self.client.post()
    patch_http_handler=False,  # We use custom patching for endpoint-specific responses
)

# Get should_use_mock and create_mock_client from factory
# We need to call the factory's create_mock_client to patch AsyncHTTPHandler.post
create_mock_braintrust_factory_client, should_use_braintrust_mock = create_mock_client_factory(_config)

# Store original HTTPHandler.post method (Braintrust-specific for sync calls with custom logic)
_original_http_handler_post = None
_mocks_initialized = False

# Default mock latency in seconds
_MOCK_LATENCY_SECONDS = float(os.getenv("BRAINTRUST_MOCK_LATENCY_MS", "100")) / 1000.0


def _is_braintrust_url(url: str) -> bool:
    """Check if URL is a Braintrust API URL."""
    if not isinstance(url, str):
        return False

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()

    if not host:
        return False

    return (
        host == "braintrustdata.com"
        or host.endswith(".braintrustdata.com")
        or host == "braintrust.dev"
        or host.endswith(".braintrust.dev")
    )


def _mock_http_handler_post(self, url, data=None, json=None, params=None, headers=None, timeout=None, stream=False, files=None, content=None, logging_obj=None):
    """Monkey-patched HTTPHandler.post that intercepts Braintrust calls with endpoint-specific responses."""
    # Only mock Braintrust API calls
    if isinstance(url, str) and _is_braintrust_url(url):
        verbose_logger.info(f"[BRAINTRUST MOCK] POST to {url}")
        time.sleep(_MOCK_LATENCY_SECONDS)
        # Return appropriate mock response based on endpoint
        if "/project" in url:
            # Project creation/retrieval/register endpoint
            project_name = json.get("name", "litellm") if json else "litellm"
            mock_data = {"id": f"mock-project-id-{project_name}", "name": project_name}
        elif "/project_logs" in url:
            # Log insertion endpoint
            mock_data = {"status": "success"}
        else:
            mock_data = _config.default_json_data
        return MockResponse(
            status_code=_config.default_status_code,
            json_data=mock_data,
            url=url,
            elapsed_seconds=_MOCK_LATENCY_SECONDS
        )
    if _original_http_handler_post is not None:
        return _original_http_handler_post(self, url=url, data=data, json=json, params=params, headers=headers, timeout=timeout, stream=stream, files=files, content=content, logging_obj=logging_obj)
    raise RuntimeError("Original HTTPHandler.post not available")


def create_mock_braintrust_client():
    """
    Monkey-patch HTTPHandler.post to intercept Braintrust sync calls.
    
    Braintrust uses HTTPHandler for sync calls and AsyncHTTPHandler for async calls.
    HTTPHandler.post uses self.client.send(), not self.client.post(), so we need
    custom patching for sync (similar to Helicone).
    AsyncHTTPHandler.post is patched by the factory.
    
    We use custom patching instead of factory's patch_http_handler because we need
    endpoint-specific responses (different for /project vs /project_logs).
    
    This function is idempotent - it only initializes mocks once, even if called multiple times.
    """
    global _original_http_handler_post, _mocks_initialized
    
    if _mocks_initialized:
        return
    
    verbose_logger.debug("[BRAINTRUST MOCK] Initializing Braintrust mock client...")
    
    from litellm.llms.custom_httpx.http_handler import HTTPHandler
    
    if _original_http_handler_post is None:
        _original_http_handler_post = HTTPHandler.post
        HTTPHandler.post = _mock_http_handler_post  # type: ignore
        verbose_logger.debug("[BRAINTRUST MOCK] Patched HTTPHandler.post")
    
    # CRITICAL: Call the factory's initialization function to patch AsyncHTTPHandler.post
    # This is required for async calls to be mocked
    create_mock_braintrust_factory_client()
    
    verbose_logger.debug(f"[BRAINTRUST MOCK] Mock latency set to {_MOCK_LATENCY_SECONDS*1000:.0f}ms")
    verbose_logger.debug("[BRAINTRUST MOCK] Braintrust mock client initialization complete")
    
    _mocks_initialized = True
