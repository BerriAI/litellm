from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final, Protocol, runtime_checkable

from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.catalog import Rules, SecretManagerContext, decision
from litellm.rust_bridge.configuration import Decision
from litellm.types.secret_managers.main import KeyManagementSettings, KeyManagementSystem


@dataclass(frozen=True, slots=True)
class NativeSecretManagerConfig:
    system: str
    environment: tuple[tuple[str, str], ...] = field(repr=False)
    settings_json: str = field(repr=False)
    enterprise_enabled: bool
    owner_type: type[object]
    environment_attributes: tuple[tuple[str, str], ...]
    settings_attributes: tuple[str, ...]
    methods: tuple[tuple[str, object], ...] = field(repr=False)


def register_native_secret_manager(
    client: object,
    system: KeyManagementSystem,
    builtin_type: type[object],
    *,
    settings: KeyManagementSettings | None = None,
    environment: Mapping[str, str] | None = None,
    enterprise_enabled: bool = False,
    environment_attributes: Mapping[str, str] | None = None,
    settings_attributes: tuple[str, ...] = (),
    methods: tuple[str, ...] = (),
) -> None:
    if type(client) is not builtin_type:
        return
    prefixes: Final = ("AWS_", "AZURE_", "GOOGLE_", "VERTEX_", "GCS_", "HCP_VAULT_", "CYBERARK_")
    captured: Final = tuple(
        (name, value)
        for name, value in os.environ.items()
        if name.startswith(prefixes) or name == "SECRET_MANAGER_REFRESH_INTERVAL"
    )
    config: Final = NativeSecretManagerConfig(
        system=system.value,
        environment=(*captured, *(environment.items() if environment is not None else ())),
        settings_json=(settings or KeyManagementSettings()).model_dump_json(),
        enterprise_enabled=enterprise_enabled,
        owner_type=builtin_type,
        environment_attributes=tuple(environment_attributes.items()) if environment_attributes is not None else (),
        settings_attributes=settings_attributes,
        methods=tuple((name, getattr(builtin_type, name)) for name in methods),
    )
    setattr(client, "_litellm_native_secret_config", config)


class NativeSecretManagerRuntime(Protocol):
    @property
    def system(self) -> str: ...

    def read_secret(self, name: str, settings_json: str | None = None) -> str | None: ...


@runtime_checkable
class NativeSecretManagerFactory(Protocol):
    @staticmethod
    def from_client(client: object) -> NativeSecretManagerRuntime | None: ...


def _factory(value: object) -> NativeSecretManagerFactory | None:
    return value if isinstance(value, NativeSecretManagerFactory) and callable(value.from_client) else None


_RUNTIME: Final = NativeBinding("_SecretManagerRuntime", validate=_factory)


def resolve_native_secret_manager(
    client: object,
    system: str,
    rules: Rules | None = None,
) -> NativeSecretManagerRuntime | None:
    if system in ("custom", "local"):
        return None
    selected: Final = decision(SecretManagerContext(system=system), rules)
    if selected is Decision.PYTHON:
        return None
    factory: Final = _RUNTIME.load()
    if factory is None:
        if selected is Decision.RUST_REQUIRED:
            raise RuntimeError("Rust secret manager runtime is unavailable")
        return None
    runtime: Final = factory.from_client(client)
    if runtime is not None and runtime.system != system:
        raise ValueError("Native secret manager system does not match configuration")
    return runtime
