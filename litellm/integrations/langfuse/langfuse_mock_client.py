"""
Mock httpx client for Langfuse integration testing.

This module intercepts Langfuse API calls and returns successful mock responses,
allowing full code execution without making actual network calls.

Usage:
    Set LANGFUSE_MOCK=true in environment variables or config to enable mock mode.
"""

import httpx
from litellm.integrations.mock_client_factory import MockClientConfig, create_mock_client_factory

# Create mock client using factory
_config = MockClientConfig(
    name="LANGFUSE",
    env_var="LANGFUSE_MOCK",
    default_latency_ms=100,
    default_status_code=200,
    default_json_data={"status": "success"},
    url_matchers=[
        ".langfuse.com",
        "langfuse.com",
    ],
    patch_async_handler=False,
    patch_sync_client=True,
)

_create_mock_langfuse_client_internal, should_use_langfuse_mock = create_mock_client_factory(_config)

# Langfuse needs to return an httpx.Client instance
def create_mock_langfuse_client():
    """Create and return an httpx.Client instance - the monkey-patch intercepts all calls."""
    _create_mock_langfuse_client_internal()
    return httpx.Client()
