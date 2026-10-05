import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Final, TypeAlias

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.secret_managers.aws_secret_manager_v2 import AWSSecretsManagerV2
from litellm.types.llms.custom_http import httpxSpecialProvider


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


@dataclass(frozen=True, slots=True)
class FakeSecretsManagerState:
    live: Mapping[str, str]
    scheduled_for_deletion: frozenset[str] = frozenset()
    actions: tuple[str, ...] = ()
    descriptions: Mapping[str, str] = MappingProxyType({})
    failing_actions: frozenset[str] = frozenset()


class FakeSecretsManagerService:
    def __init__(self, state: FakeSecretsManagerState) -> None:
        self.state = state

    def handle(self, request: httpx.Request) -> httpx.Response:
        action: Final = request.headers["X-Amz-Target"].removeprefix("secretsmanager.")
        body: Final = json.loads(request.content)
        name: Final = str(body.get("Name") or body.get("SecretId"))
        self.state = replace(self.state, actions=(*self.state.actions, f"{action}:{name}"))
        if action in self.state.failing_actions:
            return self._error("InternalServiceError", f"injected failure for {action}")
        match action:
            case "CreateSecret":
                if name in self.state.live:
                    return self._error("ResourceExistsException", f"The secret {name} already exists")
                self.state = replace(
                    self.state,
                    live=MappingProxyType({**self.state.live, name: str(body["SecretString"])}),
                    descriptions=MappingProxyType({**self.state.descriptions, name: str(body.get("Description", ""))}),
                )
                return httpx.Response(200, json={"ARN": f"arn:fake:{name}", "Name": name})
            case "UpdateSecret":
                if name in self.state.scheduled_for_deletion:
                    return self._error(
                        "InvalidRequestException",
                        "You can't perform this operation on the secret because it was marked for deletion.",
                    )
                self.state = replace(
                    self.state,
                    live=MappingProxyType({**self.state.live, name: str(body["SecretString"])}),
                    descriptions=MappingProxyType({**self.state.descriptions, name: str(body.get("Description", ""))}),
                )
                return httpx.Response(200, json={"ARN": f"arn:fake:{name}", "Name": name})
            case "DescribeSecret":
                if name not in self.state.live:
                    return self._error("ResourceNotFoundException", "Secrets Manager can't find the specified secret.")
                deleted: Final = "2026-01-01T00:00:00Z" if name in self.state.scheduled_for_deletion else None
                return httpx.Response(200, json={"ARN": f"arn:fake:{name}", "Name": name, "DeletedDate": deleted})
            case "RestoreSecret":
                self.state = replace(self.state, scheduled_for_deletion=self.state.scheduled_for_deletion - {name})
                return httpx.Response(200, json={"ARN": f"arn:fake:{name}", "Name": name})
            case "PutSecretValue":
                if name in self.state.scheduled_for_deletion:
                    return self._error(
                        "InvalidRequestException",
                        "You can't perform this operation on the secret because it was marked for deletion.",
                    )
                self.state = replace(
                    self.state, live=MappingProxyType({**self.state.live, name: str(body["SecretString"])})
                )
                return httpx.Response(200, json={"ARN": f"arn:fake:{name}", "Name": name})
            case "GetSecretValue":
                if name not in self.state.live or name in self.state.scheduled_for_deletion:
                    return self._error("ResourceNotFoundException", "Secrets Manager can't find the specified secret.")
                return httpx.Response(200, json={"SecretString": self.state.live[name]})
            case "DeleteSecret":
                self.state = replace(self.state, scheduled_for_deletion=self.state.scheduled_for_deletion | {name})
                return httpx.Response(200, json={"ARN": f"arn:fake:{name}", "Name": name})
        return self._error("UnsupportedAction", action)

    @staticmethod
    def _error(error_type: str, message: str) -> httpx.Response:
        return httpx.Response(400, json={"__type": error_type, "message": message})


@contextmanager
def fake_secrets_manager(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeSecretsManagerService]:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "synthetic-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "synthetic-secret-key")
    monkeypatch.setenv("AWS_REGION_NAME", "us-east-1")
    service: Final = FakeSecretsManagerService(FakeSecretsManagerState(live=MappingProxyType({})))
    cache_key: Final = "async_httpx_clienttimeout_None" + httpxSpecialProvider.SecretManager
    litellm.in_memory_llm_clients_cache.set_cache(
        key=cache_key,
        value=AsyncHTTPHandler(transport=httpx.MockTransport(service.handle)),
    )
    try:
        yield service
    finally:
        litellm.in_memory_llm_clients_cache.delete_cache(
            litellm.in_memory_llm_clients_cache.update_cache_key_with_event_loop(cache_key)
        )


@pytest.mark.asyncio
async def test_rotate_secret_back_to_name_inside_recovery_window_restores_and_stores_new_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alias_a: Final = "synthetic/alias-a"
    alias_b: Final = "synthetic/alias-b"
    with fake_secrets_manager(monkeypatch) as fake:
        manager: Final = AWSSecretsManagerV2(aws_region_name="us-east-1")
        await manager.async_write_secret(secret_name=alias_a, secret_value="value-1")
        await manager.async_rotate_secret(
            current_secret_name=alias_a, new_secret_name=alias_b, new_secret_value="value-2"
        )
        assert alias_a in fake.state.scheduled_for_deletion

        await manager.async_rotate_secret(
            current_secret_name=alias_b, new_secret_name=alias_a, new_secret_value="value-3"
        )

        assert await manager.async_read_secret(secret_name=alias_a) == "value-3"
        assert await manager.async_read_secret(secret_name=alias_b) is None
        assert fake.state.scheduled_for_deletion == frozenset({alias_b})
        assert fake.state.descriptions[alias_a] == f"Rotated from {alias_b}"


@pytest.mark.asyncio
async def test_write_secret_to_name_inside_recovery_window_reschedules_deletion_when_update_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alias: Final = "synthetic/deleted-alias"
    with fake_secrets_manager(monkeypatch) as fake:
        manager: Final = AWSSecretsManagerV2(aws_region_name="us-east-1")
        await manager.async_write_secret(secret_name=alias, secret_value="value-1")
        await manager.async_delete_secret(secret_name=alias, recovery_window_in_days=7)
        fake.state = replace(fake.state, failing_actions=frozenset({"UpdateSecret"}))

        with pytest.raises(ValueError, match="injected failure for UpdateSecret"):
            await manager.async_write_secret(secret_name=alias, secret_value="value-2")

        assert fake.state.scheduled_for_deletion == frozenset({alias})
        assert fake.state.live[alias] == "value-1"


@pytest.mark.asyncio
async def test_write_secret_to_name_inside_recovery_window_restores_and_stores_new_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alias: Final = "synthetic/deleted-alias"
    with fake_secrets_manager(monkeypatch) as fake:
        manager: Final = AWSSecretsManagerV2(aws_region_name="us-east-1")
        await manager.async_write_secret(secret_name=alias, secret_value="value-1")
        await manager.async_delete_secret(secret_name=alias, recovery_window_in_days=7)

        assert await manager.async_write_secret(secret_name=alias, secret_value="value-2") == {
            "ARN": f"arn:fake:{alias}",
            "Name": alias,
        }

        assert await manager.async_read_secret(secret_name=alias) == "value-2"
        assert fake.state.scheduled_for_deletion == frozenset()


@pytest.mark.asyncio
async def test_write_secret_to_live_existing_name_still_fails_without_overwriting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alias: Final = "synthetic/live-alias"
    with fake_secrets_manager(monkeypatch) as fake:
        manager: Final = AWSSecretsManagerV2(aws_region_name="us-east-1")
        await manager.async_write_secret(secret_name=alias, secret_value="value-1")

        with pytest.raises(ValueError, match="ResourceExistsException"):
            await manager.async_write_secret(secret_name=alias, secret_value="value-2")

        assert await manager.async_read_secret(secret_name=alias) == "value-1"
        assert f"RestoreSecret:{alias}" not in fake.state.actions
