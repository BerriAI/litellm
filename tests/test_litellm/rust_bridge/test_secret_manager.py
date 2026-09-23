from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Final

import pytest

import litellm
from litellm.rust_bridge import bindings
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.catalog import Rules, SecretManagerRule
from litellm.rust_bridge.configuration import Rollout
from litellm.rust_bridge.secret_manager import (
    NativeSecretManagerFactory,
    NativeSecretManagerRuntime,
    capture_secret_manager,
    native_secret_manager_config,
    resolve_native_secret_manager,
)
from litellm.secret_managers.aws_secret_manager_v2 import AWSSecretsManagerV2
from litellm.secret_managers.dispatch import get_secret_from_manager
from litellm.secret_managers.hashicorp_secret_manager import HashicorpSecretManager
from litellm.types.secret_managers.main import KeyManagementSettings, KeyManagementSystem
from tests.test_litellm_rust.support.recording_server import ResponseSpec, recording_service

@pytest.fixture(autouse=True)
def preserve_manager_globals(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "secret_manager_client", litellm.secret_manager_client)
    monkeypatch.setattr(litellm, "_key_management_system", litellm._key_management_system)
    monkeypatch.setattr(litellm, "_key_management_settings", litellm._key_management_settings)


def _vault(monkeypatch: pytest.MonkeyPatch, address: str) -> HashicorpSecretManager:
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "premium_user", True)
    monkeypatch.setenv("HCP_VAULT_ADDR", address)
    monkeypatch.setenv("HCP_VAULT_TOKEN", "token")
    return HashicorpSecretManager()


def _vault_body(value: str) -> dict[str, object]:
    return {
        "data": {
            "data": {"key": value},
            "metadata": {
                "created_time": "",
                "deletion_time": "",
                "custom_metadata": None,
                "destroyed": False,
                "version": 1,
            },
        },
        "lease_id": "",
        "lease_duration": 0,
        "renewable": False,
        "request_id": "",
        "warnings": None,
        "wrap_info": None,
    }


@pytest.mark.parametrize("system", ("aws_secret_manager", "hashicorp_vault", "cyberark"))
async def test_python_native_handle_reuses_backend_across_sync_and_async_reads(system: str) -> None:
    native: Final = pytest.importorskip("litellm.rust_bridge._native")
    with recording_service() as server:
        environment: Final = {
            "AWS_REGION_NAME": "us-east-1",
            "AWS_ACCESS_KEY_ID": "captured-access",
            "AWS_SECRET_ACCESS_KEY": "captured-secret",
            "AWS_BEDROCK_RUNTIME_ENDPOINT": server.base_url,
            "AZURE_KEY_VAULT_URI": server.base_url,
            "AZURE_AD_TOKEN": "azure-token",
            "HCP_VAULT_ADDR": server.base_url,
            "HCP_VAULT_TOKEN": "vault-token",
            "CYBERARK_API_BASE": server.base_url,
            "CYBERARK_API_KEY": "cyberark-key",
            "CYBERARK_ACCOUNT": "account",
            "CYBERARK_USERNAME": "reader",
        }
        responses: Final = {
            "aws_secret_manager": {"SecretString": "native-value"},
            "azure_key_vault": {"value": "native-value"},
            "hashicorp_vault": _vault_body("native-value"),
            "cyberark": "native-value",
        }
        server.default_response = ResponseSpec(body=responses[system])
        server.expected_requests = 2 if system == "cyberark" else 1 if system == "hashicorp_vault" else 3
        if system == "cyberark":
            server.enqueue(ResponseSpec(body="authentication-token"))
        handle: Final = native._SecretManagerRuntime.from_config(system, environment, enterprise_enabled=True)
        expected: Final = '"native-value"' if system == "cyberark" else "native-value"
        assert handle.read_secret("KEY") == expected
        assert await handle.async_read_secret("KEY") == expected
        assert handle.read_secret("KEY") == expected


def test_shared_initializer_captures_credentials_and_tracks_instance_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    native: Final = pytest.importorskip("litellm.rust_bridge._native")
    with recording_service() as server:
        server.default_response = ResponseSpec(body={"SecretString": "native-value"})
        server.expected_requests = 2
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "captured-access")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "captured-secret")
        monkeypatch.setenv("AWS_BEDROCK_RUNTIME_ENDPOINT", server.base_url)
        from litellm.proxy.proxy_server import ProxyConfig

        monkeypatch.setattr(litellm, "_key_management_settings", KeyManagementSettings(aws_region_name="us-east-1"))
        ProxyConfig().initialize_secret_manager(KeyManagementSystem.AWS_SECRET_MANAGER.value)
        manager: Final = litellm.secret_manager_client
        assert isinstance(manager, AWSSecretsManagerV2)
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "changed-access")
        monkeypatch.setenv("AWS_BEDROCK_RUNTIME_ENDPOINT", "http://127.0.0.1:1")
        first: Final = native._SecretManagerRuntime.from_client(manager)
        assert first is not None
        assert native._SecretManagerRuntime.from_client(manager) is first
        assert first.read_secret("KEY") == "native-value"
        manager.aws_region_name = "us-west-2"
        second: Final = native._SecretManagerRuntime.from_client(manager)
        assert second is not None
        assert second is not first
        assert second.read_secret("KEY") == "native-value"
        assert "Credential=captured-access/" in server.requests[0].headers["authorization"]
        assert "/us-east-1/" in server.requests[0].headers["authorization"]
        assert "/us-west-2/" in server.requests[1].headers["authorization"]


def test_configuration_replacement_rebuilds_without_invalidating_existing_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native: Final = pytest.importorskip("litellm.rust_bridge._native")
    with recording_service() as first_server, recording_service() as second_server:
        first_server.default_response = ResponseSpec(body=_vault_body("first"))
        second_server.default_response = ResponseSpec(body=_vault_body("second"))
        manager: Final = _vault(monkeypatch, first_server.base_url)
        first: Final = native._SecretManagerRuntime.from_client(manager)
        assert first is not None
        assert first.read_secret("KEY") == "first"
        manager.vault_addr = second_server.base_url
        second: Final = native._SecretManagerRuntime.from_client(manager)
        assert second is not None
        assert second is not first
        assert second.read_secret("KEY") == "second"
        assert first.read_secret("KEY") == "first"


def test_custom_subclass_keeps_its_python_reader_under_native_selection() -> None:
    class CustomManager(AWSSecretsManagerV2):
        def sync_read_secret(self, secret_name: str, primary_secret_name: str | None = None) -> str:
            return f"custom:{secret_name}"

    class DecliningFactory:
        @staticmethod
        def from_client(client: object) -> NativeSecretManagerRuntime | None:
            assert native_secret_manager_config(client) is None
            return None

    binding: Final[NativeBinding[NativeSecretManagerFactory]] = NativeBinding("unused", validate=lambda value: None)
    binding.override(DecliningFactory)
    rules: Final[Rules] = (SecretManagerRule(Rollout.RUST_REQUIRED, systems=frozenset({"aws_secret_manager"})),)
    assert get_secret_from_manager(
        CustomManager(aws_region_name="us-east-1"), "aws_secret_manager", "KEY", rules=rules, binding=binding
    ) == "custom:KEY"


def test_read_dispatch_uses_native_backend_with_explicit_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    native: Final = pytest.importorskip("litellm.rust_bridge._native")
    with recording_service() as server:
        server.default_response = ResponseSpec(body=_vault_body("native-value"))
        manager: Final = _vault(monkeypatch, server.base_url)
        rules: Final[Rules] = (SecretManagerRule(Rollout.RUST_REQUIRED, systems=frozenset({"hashicorp_vault"})),)
        assert get_secret_from_manager(manager, "hashicorp_vault", "KEY", rules=rules) == "native-value"
        runtime: Final = native._SecretManagerRuntime.from_client(manager)
        assert runtime is not None
        assert runtime.read_secret("KEY") == "native-value"
        assert len(server.requests) == 1


@pytest.mark.parametrize("system", ("google_secret_manager", "hashicorp_vault", "cyberark"))
def test_enterprise_backends_cannot_initialize_without_entitlement(system: str) -> None:
    native: Final = pytest.importorskip("litellm.rust_bridge._native")
    with pytest.raises(ValueError, match=r"[Ee]nterprise|[Pp]remium"):
        native._SecretManagerRuntime.from_config(system, {"CYBERARK_API_KEY": "key"})


def test_azure_factory_rejects_unencrypted_vault_endpoints() -> None:
    native: Final = pytest.importorskip("litellm.rust_bridge._native")
    with pytest.raises(ValueError, match="https"):
        native._SecretManagerRuntime.from_config("azure_key_vault", {"AZURE_KEY_VAULT_URI": "http://127.0.0.1:1"})


def test_direct_builtin_constructor_can_use_native_without_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    native: Final = pytest.importorskip("litellm.rust_bridge._native")
    with recording_service() as server:
        server.default_response = ResponseSpec(body=_vault_body("native-value"))
        manager: Final = _vault(monkeypatch, server.base_url)
        handle: Final = native._SecretManagerRuntime.from_client(manager)
        assert handle is not None
        assert handle.read_secret("KEY") == "native-value"


def test_binding_rejects_a_different_backend_before_reading() -> None:
    native: Final = pytest.importorskip("litellm.rust_bridge._native")
    with recording_service() as server:
        server.expected_requests = 0
        handle: Final = native._SecretManagerRuntime.from_config(
            "hashicorp_vault",
            {"HCP_VAULT_ADDR": server.base_url, "HCP_VAULT_TOKEN": "token"},
            enterprise_enabled=True,
        )
        rules: Final[Rules] = (SecretManagerRule(Rollout.RUST_REQUIRED, systems=frozenset({"aws_secret_manager"})),)
        with pytest.raises(ValueError, match="system does not match"):
            resolve_native_secret_manager(handle, "aws_secret_manager", rules)


@pytest.mark.parametrize("rollout", (Rollout.PYTHON_ONLY, Rollout.RUST_OPT_OUT))
def test_python_selection_and_missing_extension_preserve_python_reader(
    monkeypatch: pytest.MonkeyPatch, rollout: Rollout
) -> None:
    binding: Final[NativeBinding[NativeSecretManagerFactory]] = NativeBinding("unused", validate=lambda value: None)
    binding.override(None)
    with recording_service() as server:
        server.default_response = ResponseSpec(body=_vault_body("python-value"))
        manager: Final = _vault(monkeypatch, server.base_url)
        rules: Final[Rules] = (SecretManagerRule(rollout, systems=frozenset({"hashicorp_vault"})),)
        assert (
            get_secret_from_manager(manager, "hashicorp_vault", "KEY", rules=rules, binding=binding) == "python-value"
        )
        assert server.requests[0].headers["x-vault-token"] == "token"


def test_required_native_missing_extension_does_not_read_python(monkeypatch: pytest.MonkeyPatch) -> None:
    binding: Final[NativeBinding[NativeSecretManagerFactory]] = NativeBinding("unused", validate=lambda value: None)
    binding.override(None)
    with recording_service() as server:
        server.expected_requests = 0
        manager: Final = _vault(monkeypatch, server.base_url)
        rules: Final[Rules] = (SecretManagerRule(Rollout.RUST_REQUIRED, systems=frozenset({"hashicorp_vault"})),)
        with pytest.raises(RuntimeError, match="unavailable"):
            get_secret_from_manager(manager, "hashicorp_vault", "KEY", rules=rules, binding=binding)


def test_native_failure_is_not_replayed_in_python(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with recording_service() as server:
        server.default_response = ResponseSpec(status=403, body={"errors": ["denied"]})
        manager: Final = _vault(monkeypatch, server.base_url)
        rules: Final[Rules] = (SecretManagerRule(Rollout.RUST_OPT_OUT, systems=frozenset({"hashicorp_vault"})),)
        with pytest.raises(ValueError, match="HashiCorp Vault"):
            get_secret_from_manager(manager, "hashicorp_vault", "KEY", rules=rules)
        assert len(server.requests) == 1


@pytest.mark.parametrize("explicit_capture", (False, True))
def test_config_capture_preserves_credentials_and_excludes_unrelated_environment(
    monkeypatch: pytest.MonkeyPatch, explicit_capture: bool
) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "initial-access")
    monkeypatch.setenv("SECRET_MANAGER_REFRESH_INTERVAL", "45")
    monkeypatch.setenv("UNRELATED_PRIVATE_TOKEN", "unrelated-secret")
    manager: Final = AWSSecretsManagerV2(aws_region_name="us-east-1")
    if explicit_capture:
        capture_secret_manager(manager, "aws_secret_manager")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "replacement-access")
    captured: Final = native_secret_manager_config(manager)
    assert captured is not None
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "replacement-access")

    retained: Final = native_secret_manager_config(manager)

    assert retained is captured
    assert retained.system == "aws_secret_manager"
    assert dict(retained.environment)["AWS_ACCESS_KEY_ID"] == "initial-access"
    assert dict(retained.environment)["SECRET_MANAGER_REFRESH_INTERVAL"] == "45"
    assert "UNRELATED_PRIVATE_TOKEN" not in dict(retained.environment)
    assert "initial-access" not in repr(retained)
    assert retained.settings == KeyManagementSettings().model_dump(mode="json")


def test_kms_sdk_client_capture_preserves_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    import boto3

    monkeypatch.setenv("AWS_REGION_NAME", "captured-region")
    client: Final = boto3.client(
        "kms", region_name="us-east-1", aws_access_key_id="access", aws_secret_access_key="secret"
    )
    try:
        capture_secret_manager(client, "aws_kms")
        monkeypatch.setenv("AWS_REGION_NAME", "replacement-region")

        captured: Final = native_secret_manager_config(client)

        assert captured is not None
        assert captured.system == "aws_kms"
        assert dict(captured.environment)["AWS_REGION_NAME"] == "captured-region"
    finally:
        client.close()


def test_same_named_custom_client_is_not_captured() -> None:
    class AWSSecretsManagerV2:
        pass

    client: Final = AWSSecretsManagerV2()
    capture_secret_manager(client, "aws_secret_manager")

    assert native_secret_manager_config(client) is None


@dataclass(slots=True)
class _RecordingRuntime:
    system: str
    result: str | None
    calls: tuple[tuple[str, Mapping[str, object] | None], ...] = ()

    def read_secret(self, name: str, settings: Mapping[str, object] | None = None) -> str | None:
        self.calls = (*self.calls, (name, settings))
        return self.result


@pytest.mark.parametrize("value", (None, " value\n"))
@pytest.mark.parametrize("settings", (None, KeyManagementSettings(primary_secret_name="primary")))
def test_native_dispatch_forwards_settings_and_preserves_missing_or_unmodified_values(
    value: str | None, settings: KeyManagementSettings | None
) -> None:
    client: Final = object()
    runtime: Final = _RecordingRuntime("aws_secret_manager", value)

    class Factory:
        @staticmethod
        def from_client(candidate: object) -> NativeSecretManagerRuntime:
            assert candidate is client
            return runtime

    binding: Final[NativeBinding[NativeSecretManagerFactory]] = NativeBinding("unused", validate=lambda value: None)
    binding.override(Factory)
    rules: Final[Rules] = (SecretManagerRule(Rollout.RUST_REQUIRED, systems=frozenset({runtime.system})),)

    result: Final = get_secret_from_manager(client, runtime.system, "KEY", settings, rules=rules, binding=binding)

    assert result == value
    assert runtime.calls == (("KEY", settings.model_dump(mode="json") if settings is not None else None),)


def test_native_system_mismatch_is_rejected_before_reading() -> None:
    runtime: Final = _RecordingRuntime("hashicorp_vault", "wrong-provider")

    class Factory:
        @staticmethod
        def from_client(client: object) -> NativeSecretManagerRuntime:
            return runtime

    binding: Final[NativeBinding[NativeSecretManagerFactory]] = NativeBinding("unused", validate=lambda value: None)
    binding.override(Factory)
    rules: Final[Rules] = (SecretManagerRule(Rollout.RUST_REQUIRED, systems=frozenset({"aws_secret_manager"})),)

    with pytest.raises(ValueError, match="system does not match"):
        get_secret_from_manager(object(), "aws_secret_manager", "KEY", rules=rules, binding=binding)

    assert runtime.calls == ()


@pytest.mark.parametrize("system", ("custom", "local"))
def test_python_only_manager_types_never_construct_a_native_backend(system: str) -> None:
    class ForbiddenFactory:
        @staticmethod
        def from_client(client: object) -> NativeSecretManagerRuntime:
            raise AssertionError("custom and local clients cannot use native backends")

    binding: Final[NativeBinding[NativeSecretManagerFactory]] = NativeBinding("unused", validate=lambda value: None)
    binding.override(ForbiddenFactory)
    rules: Final[Rules] = (SecretManagerRule(Rollout.RUST_REQUIRED, systems=frozenset({system})),)

    assert resolve_native_secret_manager(object(), system, rules, binding=binding) is None


@pytest.mark.parametrize("factory", (None, SimpleNamespace(from_client="not callable")))
def test_invalid_native_factories_are_reported_as_unavailable(
    monkeypatch: pytest.MonkeyPatch, factory: object
) -> None:
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: SimpleNamespace(_SecretManagerRuntime=factory))
    rules: Final[Rules] = (SecretManagerRule(Rollout.RUST_REQUIRED, systems=frozenset({"aws_secret_manager"})),)

    with pytest.raises(RuntimeError, match="runtime is unavailable"):
        resolve_native_secret_manager(object(), "aws_secret_manager", rules)


def test_native_binding_accepts_a_callable_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime: Final = _RecordingRuntime("aws_secret_manager", "value")

    class Factory:
        @staticmethod
        def from_client(client: object) -> NativeSecretManagerRuntime:
            return runtime

    monkeypatch.setattr(bindings, "get_native_bridge", lambda: SimpleNamespace(_SecretManagerRuntime=Factory))
    rules: Final[Rules] = (SecretManagerRule(Rollout.RUST_REQUIRED, systems=frozenset({runtime.system})),)

    assert resolve_native_secret_manager(object(), runtime.system, rules) is runtime
