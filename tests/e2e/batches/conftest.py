"""Batches suite's `client` fixture.

The shared lifecycle (resources/scoped_key), proxy liveness skip, and e2e marker
live in the parent tests/e2e/conftest.py. BatchClient holds the shared Gateway, so
the `resources` fixture cleans up keys through it; tests register file deletes and
batch cancels via `resources.defer(...)`.

Batch deployments (openai-batch, azure-batch, vertex-batch, ...) are registered
once per session via /model/new and deleted on teardown so they need not live in
the proxy config.
"""

from __future__ import annotations

from typing import Iterator

import pytest

from batch_client import BatchClient, build_client
from capabilities import PROVIDERS
from e2e_http import NoBody


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "covers: registry cell a test covers, e.g. llm.batches.openai.basic.nonstream.works",
    )


@pytest.fixture(scope="session")
def client() -> BatchClient:
    return build_client()


@pytest.fixture(scope="session")
def batch_deployments(client: BatchClient) -> Iterator[None]:
    probe = client.gateway.probe("/health/liveliness", params=NoBody())
    if not probe.healthy:
        yield
        return

    registered: list[str] = []
    try:
        for provider in PROVIDERS:
            registered.append(
                client.create_model(provider.model, provider.litellm_params())
            )
        yield
    finally:
        for model_id in registered:
            client.delete_model(model_id)
