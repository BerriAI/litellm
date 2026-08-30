"""`other` suite's `client` fixture.

Lifecycle (resources/scoped_key), proxy liveness gate, and the e2e/covers
markers all live in the parent tests/e2e/conftest.py. OtherClient holds the
shared ProxyClient so anything these tests create tears down through it.
"""

from __future__ import annotations

import pytest

from other_client import OtherClient, build_client
from proxy_client import ProxyClient


@pytest.fixture(scope="session")
def client(proxy: ProxyClient) -> OtherClient:
    return build_client(proxy)
