"""
Mock client for LangSmith integration testing.

This module intercepts LangSmith API calls and returns successful mock responses,
allowing full code execution without making actual network calls.

Usage:
    Set LANGSMITH_MOCK=true in environment variables or config to enable mock mode.
"""

from litellm.integrations.mock_client_factory import MockClientConfig, create_mock_client_factory

# Create mock client using factory
_config = MockClientConfig(
    name="LANGSMITH",
    env_var="LANGSMITH_MOCK",
    default_latency_ms=100,
    default_status_code=200,
    default_json_data={"status": "success", "ids": ["mock-run-id"]},
    url_matchers=[
        ".smith.langchain.com",
        "api.smith.langchain.com",
        "smith.langchain.com",
    ],
    patch_async_handler=True,
    patch_sync_client=False,
)

create_mock_langsmith_client, should_use_langsmith_mock = create_mock_client_factory(_config)
