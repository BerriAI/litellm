"""Realtime suite's `client` and `realtime_models` fixtures.

The shared lifecycle (resources/scoped_key), proxy liveness gate, and e2e marker
live in the parent tests/e2e/conftest.py. RealtimeClient holds the shared Gateway,
so the `resources` fixture cleans up keys this suite creates.

`realtime_models` registers every provider's realtime deployment through /model/new
at session start and deletes them at teardown, so the suite provisions the models it
uses through the management endpoints instead of depending on a static (or
misconfigured) gateway model_list.
"""

from collections.abc import Iterator

import pytest

from realtime_client import PROVIDERS, RealtimeClient, build_client


@pytest.fixture(scope="session")
def client() -> RealtimeClient:
    return build_client()


@pytest.fixture(scope="session")
def realtime_models(client: RealtimeClient) -> Iterator[dict[str, str]]:
    """Provision each provider's realtime deployment via /model/new and yield a
    provider-id -> model-name map the tests connect with; delete them on teardown.
    Every provider is provisioned (never skipped): a provider whose credentials or
    upstream model are missing on the gateway hard-fails its test, per the suite's
    fail-on-behavior contract in tests/e2e/CLAUDE.md."""
    records = tuple((provider.id, *client.provision(provider)) for provider in PROVIDERS)
    try:
        yield {provider_id: model_name for provider_id, model_name, _ in records}
    finally:
        for _, _, model_id in records:
            client.gateway.delete_model(model_id)
