"""Batch-files suite's `client` fixture."""

import pytest

from batch_client import BatchFilesClient, build_client


@pytest.fixture(scope="session")
def client() -> BatchFilesClient:
    return build_client()
