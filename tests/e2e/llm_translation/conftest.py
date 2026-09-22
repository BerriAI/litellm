"""LLM-translation suite's `client` fixture.

The shared lifecycle (resources/scoped_key), proxy liveness gate, and e2e marker
live in the parent tests/e2e/conftest.py. PassthroughClient holds the shared
ProxyClient, so the `resources` fixture cleans up keys this suite creates. The
`sdk` fixture hands tests real provider SDK clients (OpenAI, Anthropic) pointed
at the proxy, the way customers actually call it.
"""

import pytest

from passthrough_client import PassthroughClient, build_client
from proxy_client import ProxyClient
from sdk_clients import SdkClients, build_sdk_clients


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "covers: registry cell a test covers, e.g. llm.chat_completions.provider.basic.nonstream.works",
    )


@pytest.fixture(scope="session")
def client(proxy: ProxyClient) -> PassthroughClient:
    return build_client(proxy)


@pytest.fixture(scope="session")
def sdk() -> SdkClients:
    return build_sdk_clients()
