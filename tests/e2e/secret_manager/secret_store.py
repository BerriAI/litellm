"""What the secret_manager suite needs from a backend, independent of which one.

The proxy under test runs with one `key_management_system` (it is global), and the
suite talks to that same backend directly through a SecretStore: it seeds the
secrets a deployment points at, and reads back what the proxy wrote. Each backend
contributes a store module (secret_store_<system>.py) exporting a SecretBackend,
registered in secret_backends.BACKENDS; the tests never name a backend.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal, Protocol

SECRET_MANAGER_CONFIG_DIR: Final = "gateway"


class SecretStore(Protocol):
    def write(self, name: str, value: str) -> None:
        """Store `value` under `name`, failing the test when the backend refuses."""
        ...

    def read(self, name: str) -> str | None:
        """The current value under `name`, or None when the backend holds none."""
        ...

    def destroy(self, name: str) -> None:
        """Teardown for a secret a test seeded or the proxy wrote. Idempotent."""
        ...


# What a backend may lack. "deletes_stored_keys": the proxy's async_delete_secret
# really removes the secret; CyberArk Conjur, for one, answers not_supported and
# keeps it (cyberark_secret_manager.py).
Capability = Literal["deletes_stored_keys"]


@dataclass(frozen=True, slots=True)
class SecretBackend:
    # A litellm KeyManagementSystem value (litellm/types/secret_managers/main.py),
    # kept as a literal so the suite does not import litellm.
    system: str
    from_env: Callable[[], SecretStore]
    capabilities: frozenset[Capability]

    @property
    def proxy_config(self) -> str:
        """The proxy config for this backend's lane, relative to tests/e2e."""
        return f"{SECRET_MANAGER_CONFIG_DIR}/secret_manager_{self.system}_ci_config.yml"
