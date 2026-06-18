"""LLM-translation suite's `client` fixture.

The shared lifecycle (resources/scoped_key), proxy liveness skip, and e2e marker
live in the parent tests/e2e_tests/conftest.py. PassthroughClient subclasses
ProxyClient, so it satisfies lifecycle.ResourceClient and the shared `resources`
fixture cleans up keys this suite creates.
"""

import pytest

from passthrough_client import PassthroughClient, build_client


@pytest.fixture(scope="session")
def client() -> PassthroughClient:
    return build_client()
