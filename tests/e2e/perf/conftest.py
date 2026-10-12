from __future__ import annotations

import pytest

from lifecycle import ResourceManager
from load_client import LoadClient, build_client
from perf_rig import PerfRig
from proxy_client import ProxyClient


@pytest.fixture(scope="session")
def client(proxy: ProxyClient) -> LoadClient:
    return build_client(proxy)


@pytest.fixture
def perf_rig(proxy: ProxyClient, resources: ResourceManager) -> PerfRig:
    return PerfRig(proxy=proxy, resources=resources)
