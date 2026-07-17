"""Access-control suite client fixture; lifecycle/skip/marker live in the parent conftest."""

import pytest

from access_control_client import AccessControlClient, build_client
from proxy_client import ProxyClient


@pytest.fixture(scope="session")
def client(proxy: ProxyClient) -> AccessControlClient:
    return build_client(proxy)
