from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Final

import pytest
from e2e_http import (
    Headers,
    NetworkError,
    Success,
    UnknownApiError,
    delete_external,
    get_external,
    post_json_external,
)
from pydantic import BaseModel, Field

from secret_store import SecretBackend

VAULT_ADDR_ENV: Final = "E2E_VAULT_ADDR"
VAULT_TOKEN_ENV: Final = "E2E_VAULT_TOKEN"
VAULT_MOUNT_ENV: Final = "E2E_VAULT_MOUNT_NAME"

DEFAULT_VAULT_ADDR: Final = "http://127.0.0.1:8200"
DEFAULT_MOUNT: Final = "secret"

SYSTEM: Final = "hashicorp_vault"

_START_HINT: Final = (
    f"Start one with `bash tests/e2e/secret_manager/backend.sh up {SYSTEM}`, which writes the env for "
    f"the proxy (booted from gateway/secret_manager_{SYSTEM}_ci_config.yml) and for the tests"
)


class VaultHeaders(Headers):
    x_vault_token: str = Field(serialization_alias="X-Vault-Token", repr=False)


class KvData(BaseModel):
    key: str = Field(repr=False)


class KvWriteBody(BaseModel):
    data: KvData


class KvReadData(BaseModel):
    data: KvData


class KvReadResponse(BaseModel):
    data: KvReadData


@dataclass(frozen=True, slots=True)
class Vault:
    base_url: str
    token: str = field(repr=False)
    mount: str = DEFAULT_MOUNT

    def _headers(self) -> VaultHeaders:
        return VaultHeaders(x_vault_token=self.token)

    def _data_url(self, name: str) -> str:
        return f"{self.base_url}/v1/{self.mount}/data/{name}"

    def _metadata_url(self, name: str) -> str:
        return f"{self.base_url}/v1/{self.mount}/metadata/{name}"

    def write(self, name: str, value: str) -> None:
        write: Final = post_json_external(
            self._data_url(name), headers=self._headers(), json=KvWriteBody(data=KvData(key=value))
        )
        if write.status_code == -1:
            pytest.fail(f"No live Vault at {self.base_url}: {write.body}. {_START_HINT}")
        if not write.ok:
            pytest.fail(f"Vault refused to write {name}: HTTP {write.status_code} {write.body[:300]}")

    def read(self, name: str) -> str | None:
        result: Final = get_external(self._data_url(name), headers=self._headers(), response_type=KvReadResponse)
        match result:
            case Success(data=body):
                return body.data.data.key
            case UnknownApiError(status_code=404):
                return None
            case NetworkError(message=message):
                return pytest.fail(f"No live Vault at {self.base_url}: {message}. {_START_HINT}")
            case _:
                return pytest.fail(f"Vault refused to read {name}: {result}")

    def destroy(self, name: str) -> None:
        write: Final = delete_external(self._metadata_url(name), headers=self._headers())
        if not write.ok and write.status_code != 404:
            pytest.fail(f"Vault refused to destroy {name}: HTTP {write.status_code} {write.body[:300]}")


def vault_from_env() -> Vault:
    token: Final = os.environ.get(VAULT_TOKEN_ENV, "").strip()
    if not token:
        pytest.fail(f"The hashicorp_vault lane needs {VAULT_TOKEN_ENV} to reach its Vault. {_START_HINT}")
    return Vault(
        base_url=os.environ.get(VAULT_ADDR_ENV, DEFAULT_VAULT_ADDR).rstrip("/"),
        token=token,
        mount=os.environ.get(VAULT_MOUNT_ENV, "").strip() or DEFAULT_MOUNT,
    )


HASHICORP_VAULT: Final = SecretBackend(
    system=SYSTEM,
    from_env=vault_from_env,
    capabilities=frozenset({"deletes_stored_keys"}),
)
