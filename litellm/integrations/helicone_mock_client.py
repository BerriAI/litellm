"""
Mock HTTP client for Helicone integration testing.

This module intercepts Helicone API calls and returns successful mock responses,
allowing full code execution without making actual network calls.

Usage:
    Set HELICONE_MOCK=true in environment variables or config to enable mock mode.
"""

from litellm.integrations.mock_client_factory import MockClientConfig, create_mock_client_factory

# Create mock client using factory
# Helicone uses HTTPHandler which internally uses httpx.Client.send(), not httpx.Client.post()
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
    patch_http_handler=True,  # Patch HTTPHandler.post directly
)

create_mock_helicone_client, should_use_helicone_mock = create_mock_client_factory(_config)
