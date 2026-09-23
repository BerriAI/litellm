from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final

import pytest

from e2e_config import SECRET_MANAGER_OPT_IN_ENV
from proxy_client import ProxyClient
from secret_backends import BACKENDS, selected_backend
from secret_store import SecretBackend, SecretStore

REQUIRES_CAPABILITY: Final = "requires_capability"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        f"{REQUIRES_CAPABILITY}(capability): secret_manager test deselected when the backend "
        f"{SECRET_MANAGER_OPT_IN_ENV} names lacks the capability (secret_store.Capability)",
    )


def _lacks_capability(item: pytest.Item, backend: SecretBackend) -> bool:
    marker: Final = item.get_closest_marker(REQUIRES_CAPABILITY)
    return marker is not None and marker.args[0] not in backend.capabilities


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    backend: Final = BACKENDS.get(os.environ.get(SECRET_MANAGER_OPT_IN_ENV, "").strip())
    if backend is None:
        return
    deselected: Final = [item for item in items if _lacks_capability(item, backend)]
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = [item for item in items if not _lacks_capability(item, backend)]


@dataclass(frozen=True, slots=True)
class SecretManagerClient:
    proxy: ProxyClient


@pytest.fixture(scope="session")
def client(proxy: ProxyClient) -> SecretManagerClient:
    return SecretManagerClient(proxy)


@pytest.fixture(scope="session")
def backend() -> SecretBackend:
    return selected_backend()


@pytest.fixture(scope="session")
def store(backend: SecretBackend) -> SecretStore:
    return backend.from_env()
