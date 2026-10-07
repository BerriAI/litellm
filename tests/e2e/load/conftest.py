from __future__ import annotations

import pytest

from load_client import LoadClient, build_client
from proxy_client import ProxyClient


@pytest.fixture(scope="session")
def client(proxy: ProxyClient) -> LoadClient:
    return build_client(proxy)
