from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal, Protocol

SECRET_MANAGER_CONFIG_DIR: Final = "gateway"


class SecretStore(Protocol):
    def write(self, name: str, value: str) -> None: ...

    def read(self, name: str) -> str | None: ...

    def destroy(self, name: str) -> None: ...


Capability = Literal["deletes_stored_keys"]


@dataclass(frozen=True, slots=True)
class SecretBackend:
    system: str
    from_env: Callable[[], SecretStore]
    capabilities: frozenset[Capability]

    @property
    def proxy_config(self) -> str:
        return f"{SECRET_MANAGER_CONFIG_DIR}/secret_manager_{self.system}_ci_config.yml"
