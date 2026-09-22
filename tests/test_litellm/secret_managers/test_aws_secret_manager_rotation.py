from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Final, TypeAlias

import pytest

from litellm.secret_managers.aws_secret_manager_v2 import AWSSecretsManagerV2


OptionalParams: TypeAlias = Mapping[str, object] | None
Timeout: TypeAlias = object
WriteCall: TypeAlias = tuple[str, str, str | None, OptionalParams, Timeout]
PutCall: TypeAlias = tuple[str, str, OptionalParams, Timeout]
DeleteCall: TypeAlias = tuple[str, int | None, OptionalParams, Timeout]


@dataclass(frozen=True, slots=True)
class StatefulSecretStorage:
    values: Mapping[str, str]
    events: tuple[str, ...] = ()
    reads: tuple[str, ...] = ()
    writes: tuple[WriteCall, ...] = ()
    puts: tuple[PutCall, ...] = ()
    deletions: tuple[DeleteCall, ...] = ()

    def read(self, secret_name: str) -> tuple["StatefulSecretStorage", str | None]:
        return (
            replace(self, events=(*self.events, f"read:{secret_name}"), reads=(*self.reads, secret_name)),
            self.values.get(secret_name),
        )

    def write(
        self,
        secret_name: str,
        secret_value: str,
        description: str | None,
        optional_params: OptionalParams,
        timeout: Timeout,
    ) -> tuple["StatefulSecretStorage", dict[str, str]]:
        values: Final = MappingProxyType({**self.values, secret_name: secret_value})
        return (
            replace(
                self,
                values=values,
                events=(*self.events, f"write:{secret_name}"),
                writes=(*self.writes, (secret_name, secret_value, description, optional_params, timeout)),
            ),
            {"ARN": f"arn:synthetic:{secret_name}"},
        )

    def put(
        self,
        secret_name: str,
        secret_value: str,
        optional_params: OptionalParams,
        timeout: Timeout,
    ) -> tuple["StatefulSecretStorage", dict[str, str]]:
        values: Final = MappingProxyType({**self.values, secret_name: secret_value})
        return (
            replace(
                self,
                values=values,
                events=(*self.events, f"put:{secret_name}"),
                puts=(*self.puts, (secret_name, secret_value, optional_params, timeout)),
            ),
            {"ARN": f"arn:synthetic:{secret_name}"},
        )

    def delete(
        self,
        secret_name: str,
        recovery_window_in_days: int | None,
        optional_params: OptionalParams,
        timeout: Timeout,
    ) -> tuple["StatefulSecretStorage", dict[str, object]]:
        values: Final = MappingProxyType({name: value for name, value in self.values.items() if name != secret_name})
        return (
            replace(
                self,
                values=values,
                events=(*self.events, f"delete:{secret_name}"),
                deletions=(*self.deletions, (secret_name, recovery_window_in_days, optional_params, timeout)),
            ),
            {},
        )


class StatefulAWSSecretsManager(AWSSecretsManagerV2):
    def __init__(self, storage: StatefulSecretStorage) -> None:
        super().__init__()
        self.storage = storage

    async def async_read_secret(
        self,
        secret_name: str,
        optional_params: OptionalParams = None,
        timeout: Timeout = None,
        primary_secret_name: str | None = None,
    ) -> str | None:
        storage, secret_value = self.storage.read(secret_name)
        self.storage = storage
        return secret_value

    async def async_write_secret(
        self,
        secret_name: str,
        secret_value: str,
        description: str | None = None,
        optional_params: OptionalParams = None,
        timeout: Timeout = None,
        tags: object = None,
    ) -> dict[str, str]:
        storage, response = self.storage.write(secret_name, secret_value, description, optional_params, timeout)
        self.storage = storage
        return response

    async def async_put_secret_value(
        self,
        secret_name: str,
        secret_value: str,
        optional_params: OptionalParams = None,
        timeout: Timeout = None,
    ) -> dict[str, str]:
        storage, response = self.storage.put(secret_name, secret_value, optional_params, timeout)
        self.storage = storage
        return response

    async def async_delete_secret(
        self,
        secret_name: str,
        recovery_window_in_days: int | None = 7,
        optional_params: OptionalParams = None,
        timeout: Timeout = None,
    ) -> dict[str, object]:
        storage, response = self.storage.delete(secret_name, recovery_window_in_days, optional_params, timeout)
        self.storage = storage
        return response


@pytest.mark.asyncio
async def test_rotate_secret_same_name_writes_requested_value_in_place() -> None:
    secret_name: Final = "synthetic/current-alias"
    new_value: Final = "synthetic-new-value"
    unrelated_secret_name: Final = "synthetic/unrelated"
    unrelated_value: Final = "synthetic-unrelated-value"
    storage: Final = StatefulSecretStorage(
        MappingProxyType(
            {
                secret_name: "synthetic-old-value",
                unrelated_secret_name: unrelated_value,
            }
        )
    )
    manager: Final = StatefulAWSSecretsManager(storage)

    assert await manager.async_rotate_secret(
        current_secret_name=secret_name,
        new_secret_name=secret_name,
        new_secret_value=new_value,
    ) == {"ARN": f"arn:synthetic:{secret_name}"}

    assert manager.storage.events == (f"put:{secret_name}",)
    assert manager.storage.puts == ((secret_name, new_value, None, None),)
    assert manager.storage.writes == ()
    assert manager.storage.deletions == ()
    assert manager.storage.values[secret_name] == new_value
    assert manager.storage.values[unrelated_secret_name] == unrelated_value


@pytest.mark.asyncio
async def test_rotate_secret_different_names_persists_requested_value_and_deletes_old_alias() -> None:
    current_name: Final = "synthetic/old-alias"
    new_name: Final = "synthetic/new-alias"
    new_value: Final = "synthetic-new-value"
    unrelated_secret_name: Final = "synthetic/unrelated"
    unrelated_value: Final = "synthetic-unrelated-value"
    storage: Final = StatefulSecretStorage(
        MappingProxyType(
            {
                current_name: "synthetic-old-value",
                unrelated_secret_name: unrelated_value,
            }
        )
    )
    manager: Final = StatefulAWSSecretsManager(storage)

    await manager.async_rotate_secret(
        current_secret_name=current_name,
        new_secret_name=new_name,
        new_secret_value=new_value,
    )

    assert manager.storage.events == (
        f"read:{current_name}",
        f"write:{new_name}",
        f"read:{new_name}",
        f"delete:{current_name}",
    )
    assert manager.storage.reads == (current_name, new_name)
    assert manager.storage.writes == ((new_name, new_value, f"Rotated from {current_name}", None, None),)
    assert manager.storage.puts == ()
    assert manager.storage.deletions == ((current_name, 7, None, None),)
    assert manager.storage.values[new_name] == new_value
    assert current_name not in manager.storage.values
    assert manager.storage.values[unrelated_secret_name] == unrelated_value
