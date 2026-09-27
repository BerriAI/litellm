from __future__ import annotations

import os
from collections.abc import Awaitable, Mapping
from dataclasses import dataclass, field
from importlib import import_module
from typing import Final, Protocol, runtime_checkable

import httpx
from pydantic import JsonValue

from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.catalog import Rules, SecretManagerContext, decision
from litellm.rust_bridge.configuration import Decision
from litellm.types.secret_managers.main import KeyManagementSettings, KeyManagementSystem


@dataclass(frozen=True, slots=True)
class NativeSecretManagerConfig:
    system: str
    environment: tuple[tuple[str, str], ...] = field(repr=False)
    settings: Mapping[str, object] = field(repr=False)
    enterprise_enabled: bool
    owner_type: type[object]
    environment_attributes: tuple[tuple[str, str], ...]
    settings_attributes: tuple[str, ...]
    methods: tuple[tuple[str, object], ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class _ClientAdapter:
    system: KeyManagementSystem
    module: str
    name: str
    methods: tuple[str, ...]
    environment_attributes: tuple[tuple[str, str], ...] = ()
    settings_attributes: tuple[str, ...] = ()
    enterprise_enabled: bool = False


_ADAPTERS: Final = (
    _ClientAdapter(
        KeyManagementSystem.AWS_SECRET_MANAGER,
        "litellm.secret_managers.aws_secret_manager_v2",
        "AWSSecretsManagerV2",
        ("sync_read_secret", "async_read_secret"),
        settings_attributes=(
            "aws_region_name",
            "aws_role_name",
            "aws_session_name",
            "aws_external_id",
            "aws_profile_name",
            "aws_web_identity_token",
            "aws_sts_endpoint",
            "replica_regions",
            "kms_key_id",
        ),
    ),
    _ClientAdapter(
        KeyManagementSystem.HASHICORP_VAULT,
        "litellm.secret_managers.hashicorp_secret_manager",
        "HashicorpSecretManager",
        ("sync_read_secret", "async_read_secret", "async_write_secret", "async_delete_secret", "async_rotate_secret"),
        environment_attributes=(
            ("HCP_VAULT_ADDR", "vault_addr"),
            ("HCP_VAULT_TOKEN", "vault_token"),
            ("HCP_VAULT_NAMESPACE", "vault_namespace"),
            ("HCP_VAULT_LOGIN_NAMESPACE", "login_namespace_override"),
            ("HCP_VAULT_SECRET_NAMESPACE", "secret_namespace_override"),
            ("HCP_VAULT_MOUNT_NAME", "vault_mount_name"),
            ("HCP_VAULT_PATH_PREFIX", "vault_path_prefix"),
            ("HCP_VAULT_CLIENT_CERT", "tls_cert_path"),
            ("HCP_VAULT_CLIENT_KEY", "tls_key_path"),
            ("HCP_VAULT_CERT_ROLE", "vault_cert_role"),
            ("HCP_VAULT_APPROLE_ROLE_ID", "approle_role_id"),
            ("HCP_VAULT_APPROLE_SECRET_ID", "approle_secret_id"),
            ("HCP_VAULT_APPROLE_MOUNT_PATH", "approle_mount_path"),
            ("HCP_VAULT_REFRESH_INTERVAL", "cache.default_ttl"),
        ),
        enterprise_enabled=True,
    ),
    _ClientAdapter(
        KeyManagementSystem.CYBERARK,
        "litellm.secret_managers.cyberark_secret_manager",
        "CyberArkSecretManager",
        ("sync_read_secret", "async_read_secret", "async_write_secret", "async_delete_secret", "async_rotate_secret"),
        environment_attributes=(
            ("CYBERARK_API_BASE", "conjur_addr"),
            ("CYBERARK_ACCOUNT", "conjur_account"),
            ("CYBERARK_USERNAME", "conjur_username"),
            ("CYBERARK_API_KEY", "conjur_api_key"),
            ("CYBERARK_CLIENT_CERT", "tls_cert_path"),
            ("CYBERARK_CLIENT_KEY", "tls_key_path"),
            ("CYBERARK_SSL_VERIFY", "ssl_verify"),
            ("CYBERARK_REFRESH_INTERVAL", "cache.default_ttl"),
        ),
        enterprise_enabled=True,
    ),
    _ClientAdapter(
        KeyManagementSystem.GOOGLE_SECRET_MANAGER,
        "litellm.secret_managers.google_secret_manager",
        "GoogleSecretManager",
        ("get_secret_from_google_secret_manager",),
        environment_attributes=(
            ("GOOGLE_SECRET_MANAGER_PROJECT_ID", "PROJECT_ID"),
            ("GOOGLE_SECRET_MANAGER_REFRESH_INTERVAL", "cache.default_ttl"),
            ("GOOGLE_SECRET_MANAGER_ALWAYS_READ_SECRET_MANAGER", "always_read_secret_manager"),
        ),
        enterprise_enabled=True,
    ),
)

_SDK_ADAPTERS: Final = (
    _ClientAdapter(KeyManagementSystem.AZURE_KEY_VAULT, "azure.keyvault.secrets", "SecretClient", ("get_secret",)),
    _ClientAdapter(KeyManagementSystem.GOOGLE_KMS, "google.cloud.kms_v1", "KeyManagementServiceClient", ("decrypt",)),
)


def _capture(client: object, adapter: _ClientAdapter) -> NativeSecretManagerConfig:
    prefixes: Final = ("AWS_", "AZURE_", "GOOGLE_", "VERTEX_", "GCS_", "HCP_VAULT_", "CYBERARK_")
    config: Final = NativeSecretManagerConfig(
        system=adapter.system.value,
        environment=tuple(
            (name, value)
            for name, value in os.environ.items()
            if name.startswith(prefixes) or name == "SECRET_MANAGER_REFRESH_INTERVAL"
        ),
        settings=KeyManagementSettings().model_dump(mode="json"),
        enterprise_enabled=adapter.enterprise_enabled,
        owner_type=type(client),
        environment_attributes=adapter.environment_attributes,
        settings_attributes=adapter.settings_attributes,
        methods=tuple((name, getattr(type(client), name)) for name in adapter.methods),
    )
    vars(client)["_litellm_native_secret_config"] = config
    return config


def capture_secret_manager(client: object, system: str) -> None:
    for adapter in (*_ADAPTERS, *_SDK_ADAPTERS):
        if (
            adapter.system.value == system
            and type(client).__name__ == adapter.name
            and type(client) is getattr(import_module(adapter.module), adapter.name)
        ):
            _capture(client, adapter)
            return
    if (
        system == KeyManagementSystem.AWS_KMS.value
        and type(client).__module__ == "botocore.client"
        and type(client).__name__ == "KMS"
    ):
        _capture(client, _ClientAdapter(KeyManagementSystem.AWS_KMS, "botocore.client", "KMS", ("decrypt",)))


def native_secret_manager_config(client: object) -> NativeSecretManagerConfig | None:
    captured: Final = getattr(client, "_litellm_native_secret_config", None)
    if isinstance(captured, NativeSecretManagerConfig):
        return captured
    for adapter in _ADAPTERS:
        if type(client).__module__ == adapter.module and type(client) is getattr(
            import_module(adapter.module), adapter.name
        ):
            return _capture(client, adapter)
    return None


class NativeSecretManagerRuntime(Protocol):
    @property
    def system(self) -> str: ...

    def read_secret(self, name: str, settings: Mapping[str, object] | None = None) -> JsonValue: ...


@runtime_checkable
class NativeSecretManagerFactory(Protocol):
    @staticmethod
    def from_client(client: object) -> NativeSecretManagerRuntime | None: ...


def _factory(value: object) -> NativeSecretManagerFactory | None:
    return value if isinstance(value, NativeSecretManagerFactory) and callable(value.from_client) else None


NATIVE_SECRET_MANAGER: Final = NativeBinding("_SecretManagerRuntime", validate=_factory)


def resolve_native_secret_manager(
    client: object,
    system: str,
    rules: Rules | None = None,
    *,
    binding: NativeBinding[NativeSecretManagerFactory] = NATIVE_SECRET_MANAGER,
) -> NativeSecretManagerRuntime | None:
    if system in ("custom", "local"):
        return None
    selected: Final = decision(SecretManagerContext(system=system), rules)
    if selected is Decision.PYTHON:
        return None
    factory: Final = binding.load()
    if factory is None:
        if selected is Decision.RUST_REQUIRED:
            raise RuntimeError("Rust secret manager runtime is unavailable")
        return None
    runtime: Final = factory.from_client(client)
    if runtime is not None and runtime.system != system:
        raise ValueError("Native secret manager system does not match configuration")
    return runtime


@runtime_checkable
class NativeProviderReader(Protocol):
    def sync_read_secret(
        self,
        secret_name: str,
        optional_params: Mapping[str, object] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> str | None: ...

    def async_read_secret(
        self,
        secret_name: str,
        optional_params: Mapping[str, object] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> Awaitable[str | None]: ...


def resolve_native_provider_reader(
    client: object,
    system: str,
    rules: Rules | None = None,
    *,
    binding: NativeBinding[NativeSecretManagerFactory] = NATIVE_SECRET_MANAGER,
) -> NativeProviderReader | None:
    runtime: Final = resolve_native_secret_manager(client, system, rules, binding=binding)
    if runtime is None:
        return None
    if not isinstance(runtime, NativeProviderReader):
        raise TypeError("Rust secret manager provider reads are unavailable")
    return runtime


@runtime_checkable
class NativeProviderWriter(Protocol):
    def async_write_secret(
        self,
        secret_name: str,
        secret_value: str,
        description: str | None = None,
        optional_params: Mapping[str, object] | None = None,
        timeout: float | httpx.Timeout | None = None,
        tags: object = None,
    ) -> Awaitable[dict[str, JsonValue]]: ...

    def async_delete_secret(
        self,
        secret_name: str,
        recovery_window_in_days: int | None = 7,
        optional_params: Mapping[str, object] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> Awaitable[dict[str, JsonValue]]: ...

    def async_rotate_secret(
        self,
        current_secret_name: str,
        new_secret_name: str,
        new_secret_value: str,
        optional_params: Mapping[str, object] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> Awaitable[dict[str, JsonValue]]: ...


def resolve_native_provider_writer(
    client: object,
    system: str,
    rules: Rules | None = None,
    *,
    binding: NativeBinding[NativeSecretManagerFactory] = NATIVE_SECRET_MANAGER,
) -> NativeProviderWriter | None:
    runtime: Final = resolve_native_secret_manager(client, system, rules, binding=binding)
    if runtime is None:
        return None
    if not isinstance(runtime, NativeProviderWriter):
        raise TypeError("Rust secret manager provider writes are unavailable")
    return runtime
