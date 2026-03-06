"""
Mock client for Datadog integration testing.

This module intercepts Datadog API calls and returns successful mock responses,
allowing full code execution without making actual network calls.

Usage:
    Set DATADOG_MOCK=true in environment variables or config to enable mock mode.
"""

from litellm.integrations.mock_client_factory import MockClientConfig, create_mock_client_factory

# Create mock client using factory
_config = MockClientConfig(
    name="DATADOG",
    env_var="DATADOG_MOCK",
    default_latency_ms=100,
    default_status_code=202,
    default_json_data={"status": "ok"},
    url_matchers=[
        ".datadoghq.com",
        "datadoghq.com",
    ],
    patch_async_handler=True,
    patch_sync_client=True,
)

create_mock_datadog_client, should_use_datadog_mock = create_mock_client_factory(_config)
