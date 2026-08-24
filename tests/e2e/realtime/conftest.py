"""Realtime suite's `client` fixture.

The shared lifecycle (resources/scoped_key), proxy liveness skip, and e2e marker
live in the parent tests/e2e/conftest.py. RealtimeClient holds the shared
Gateway, so the `resources` fixture cleans up keys this suite creates.
"""

import pytest

from realtime_client import RealtimeClient, build_client


@pytest.fixture(scope="session")
def client() -> RealtimeClient:
    return build_client()


@pytest.fixture(scope="session")
def configured_models(client: RealtimeClient) -> frozenset[str]:
    return client.configured_models()
