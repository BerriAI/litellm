"""
Mock httpx client for PostHog integration testing.

This module intercepts PostHog API calls and returns successful mock responses,
allowing full code execution without making actual network calls.

Usage:
    Set POSTHOG_MOCK=true in environment variables or config to enable mock mode.
"""

from litellm.integrations.mock_client_factory import MockClientConfig, create_mock_client_factory

# Create mock client using factory
_config = MockClientConfig(
    name="POSTHOG",
    env_var="POSTHOG_MOCK",
    default_latency_ms=100,
    default_status_code=200,
    default_json_data={"status": "success"},
    url_matchers=[
        ".posthog.com",
        "posthog.com",
        "us.i.posthog.com",
        "app.posthog.com",
    ],
    patch_async_handler=True,
    patch_sync_client=True,
)

create_mock_posthog_client, should_use_posthog_mock = create_mock_client_factory(_config)
