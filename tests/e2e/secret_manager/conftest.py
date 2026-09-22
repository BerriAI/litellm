"""`secret_manager` suite fixtures.

Lifecycle (resources/scoped_key), proxy liveness gate, the e2e/covers markers and
the E2E_SECRET_MANAGER opt-in all live in the parent tests/e2e/conftest.py. The
suite has no routes of its own, so its `client` is the shared ProxyClient;
`backend` is the secret manager E2E_SECRET_MANAGER names (secret_backends.py) and
`store` reaches the same one the proxy under test is configured against.

A test that needs something not every backend does carries
`requires_capability("<capability>")` (secret_store.Capability) and is deselected
on lanes whose backend lacks it: deselected rather than skipped, so its coverage
cell counts only where it actually runs.
"""

from __future__ import annotations

import os
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
    # An unset or unknown backend is left alone here: unset, the parent conftest has
    # already deselected the suite; unknown, the `backend` fixture fails naming the
    # valid ones.
    backend: Final = BACKENDS.get(os.environ.get(SECRET_MANAGER_OPT_IN_ENV, "").strip())
    if backend is None:
        return
    deselected: Final = [item for item in items if _lacks_capability(item, backend)]
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = [item for item in items if not _lacks_capability(item, backend)]


@pytest.fixture(scope="session")
def client(proxy: ProxyClient) -> ProxyClient:
    return proxy


@pytest.fixture(scope="session")
def backend() -> SecretBackend:
    return selected_backend()


@pytest.fixture(scope="session")
def store(backend: SecretBackend) -> SecretStore:
    return backend.from_env()
