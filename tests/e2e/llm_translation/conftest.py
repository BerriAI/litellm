"""LLM-translation suite's `client` fixture.

The shared lifecycle (resources/scoped_key), proxy liveness gate, and e2e marker
live in the parent tests/e2e/conftest.py. PassthroughClient holds the shared
Gateway, so the `resources` fixture cleans up keys this suite creates.
"""

import pytest

from endpoints_client import EndpointsClient, build_endpoints_client
from passthrough_client import PassthroughClient, build_client


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "covers: registry cell a test covers, e.g. llm.chat_completions.provider.basic.nonstream.works",
    )


@pytest.fixture(scope="session")
def client() -> PassthroughClient:
    return build_client()


@pytest.fixture(scope="session")
def endpoints_client() -> EndpointsClient:
    return build_endpoints_client()
