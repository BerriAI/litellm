from __future__ import annotations

import os

import pytest
from e2e_config import REDIS_CHAOS_OPT_IN_ENV, WEEKLY_ANOMALY_OPT_IN_ENV
from load_client import LoadClient, build_client
from proxy_client import ProxyClient

_OPT_IN_MARKERS = (
    ("weekly", WEEKLY_ANOMALY_OPT_IN_ENV),
    ("redis_chaos", REDIS_CHAOS_OPT_IN_ENV),
)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    opted_out = {marker for marker, opt_in_env in _OPT_IN_MARKERS if not os.environ.get(opt_in_env)}
    deselected = [item for item in items if any(item.get_closest_marker(marker) is not None for marker in opted_out)]
    if not deselected:
        return
    config.hook.pytest_deselected(items=deselected)
    items[:] = [item for item in items if item not in deselected]


@pytest.fixture(scope="session")
def client(proxy: ProxyClient) -> LoadClient:
    return build_client(proxy)
