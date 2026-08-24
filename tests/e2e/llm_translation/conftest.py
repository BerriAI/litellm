"""LLM-translation suite's `client` fixture.

The shared lifecycle (resources/scoped_key), proxy liveness gate, and e2e marker
live in the parent tests/e2e/conftest.py. PassthroughClient holds the shared
ProxyClient, so the `resources` fixture cleans up keys this suite creates.
"""

import os

import pytest

from e2e_config import BEDROCK_OIDC_OPT_IN_ENV
from endpoints_client import EndpointsClient, build_endpoints_client
from passthrough_client import PassthroughClient, build_client
from proxy_client import ProxyClient


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "covers: registry cell a test covers, e.g. llm.chat_completions.provider.basic.nonstream.works",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Drop the Bedrock web-identity tests unless the operator opted in.

    Deselection, not skipping: the suite hard-fails rather than skips, so a
    test whose IAM prerequisites are absent must never be collected in the
    first place. Mirrors load/conftest.py's weekly gate.
    """
    if os.environ.get(BEDROCK_OIDC_OPT_IN_ENV):
        return
    deselected = [item for item in items if item.get_closest_marker("bedrock_oidc") is not None]
    if not deselected:
        return
    config.hook.pytest_deselected(items=deselected)
    items[:] = [item for item in items if item.get_closest_marker("bedrock_oidc") is None]


@pytest.fixture(scope="session")
def client(proxy: ProxyClient) -> PassthroughClient:
    return build_client(proxy)


@pytest.fixture(scope="session")
def endpoints_client(proxy: ProxyClient) -> EndpointsClient:
    return build_endpoints_client(proxy)
