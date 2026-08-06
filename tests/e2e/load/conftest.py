from __future__ import annotations

import os

import pytest

from e2e_config import WEEKLY_ANOMALY_OPT_IN_ENV
from load_client import LoadClient, build_client
from proxy_client import ProxyClient


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if os.environ.get(WEEKLY_ANOMALY_OPT_IN_ENV):
        return
    deselected = [
        item for item in items if item.get_closest_marker("weekly") is not None
    ]
    if not deselected:
        return
    config.hook.pytest_deselected(items=deselected)
    items[:] = [
        item for item in items if item.get_closest_marker("weekly") is None
    ]


@pytest.fixture(scope="session")
def client(proxy: ProxyClient) -> LoadClient:
    return build_client(proxy)
