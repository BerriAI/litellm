"""Router suite's `client` fixture.

The shared lifecycle (resources/scoped_key), proxy liveness skip, and e2e marker
live in the parent tests/e2e/conftest.py. ComplexityRouterClient holds the shared
Gateway, so the `resources` fixture cleans up keys this suite creates.
"""

import pytest

from complexity_router_client import ComplexityRouterClient, build_client


@pytest.fixture(scope="session")
def client() -> ComplexityRouterClient:
    return build_client()
