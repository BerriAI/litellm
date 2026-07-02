"""Batches suite's `client` fixture.

The shared lifecycle (resources/scoped_key), proxy liveness skip, and e2e marker
live in the parent tests/e2e/conftest.py. BatchClient holds the shared Gateway, so
the `resources` fixture cleans up keys through it; tests register file deletes and
batch cancels via `resources.defer(...)`.
"""

import pytest

from batch_client import BatchClient, build_client


@pytest.fixture(scope="session")
def client() -> BatchClient:
    return build_client()
