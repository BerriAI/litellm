"""Guardrails suite's `client` fixture.

Shared lifecycle (resources/scoped_key), proxy liveness, and e2e/covers markers
live in the parent tests/e2e/conftest.py. GuardrailsClient holds the shared
ProxyClient so keys and deferred cleanups tear down correctly.
"""

from __future__ import annotations

import pytest

from guardrails_client import GuardrailsClient, build_client
from proxy_client import ProxyClient


@pytest.fixture(scope="session")
def client(proxy: ProxyClient) -> GuardrailsClient:
    return build_client(proxy)
