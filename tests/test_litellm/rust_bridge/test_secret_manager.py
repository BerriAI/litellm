from __future__ import annotations

from typing import Final

import pytest

from litellm.rust_bridge.catalog import Rules, SecretManagerRule
from litellm.rust_bridge.configuration import Rollout
from litellm.rust_bridge.secret_manager import register_native_secret_manager, resolve_native_secret_manager
from litellm.secret_managers.aws_secret_manager_v2 import AWSSecretsManagerV2
from litellm.types.secret_managers.main import KeyManagementSystem
from tests.test_litellm_rust.support.recording_server import ResponseSpec, recording_service

native: Final = pytest.importorskip("litellm.rust_bridge._native")


class _Facade:
    def sync_read_secret(self, secret_name: str) -> str:
        raise AssertionError("a native backend must not call the Python reader")


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


def test_registered_aws_uses_instance_settings_and_captured_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    with recording_service() as server:
        server.default_response = ResponseSpec(body={"SecretString": "native-value"})
        server.expected_requests = 2
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "captured-access")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "captured-secret")
        monkeypatch.setenv("AWS_BEDROCK_RUNTIME_ENDPOINT", server.base_url)
        manager: Final = AWSSecretsManagerV2(aws_region_name="us-east-1")
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


def test_configuration_replacement_rebuilds_without_invalidating_existing_handles() -> None:
    with recording_service() as first_server, recording_service() as second_server:
        first_server.default_response = ResponseSpec(body=_vault_body("first"))
        second_server.default_response = ResponseSpec(body=_vault_body("second"))
        facade: Final = _Facade()
        register_native_secret_manager(
            facade,
            KeyManagementSystem.HASHICORP_VAULT,
            _Facade,
            environment={"HCP_VAULT_ADDR": first_server.base_url, "HCP_VAULT_TOKEN": "token"},
            enterprise_enabled=True,
            methods=("sync_read_secret",),
        )
        first: Final = native._SecretManagerRuntime.from_client(facade)
        assert first is not None
        assert first.read_secret("KEY") == "first"
        register_native_secret_manager(
            facade,
            KeyManagementSystem.HASHICORP_VAULT,
            _Facade,
            environment={"HCP_VAULT_ADDR": second_server.base_url, "HCP_VAULT_TOKEN": "token"},
            enterprise_enabled=True,
            methods=("sync_read_secret",),
        )
        second: Final = native._SecretManagerRuntime.from_client(facade)
        assert second is not None
        assert second.read_secret("KEY") == "second"
        assert first.read_secret("KEY") == "first"


def test_custom_subclasses_are_not_registered_as_builtin_managers() -> None:
    class CustomManager(AWSSecretsManagerV2):
        pass

    assert native._SecretManagerRuntime.from_client(CustomManager(aws_region_name="us-east-1")) is None


def test_python_lookup_resolves_registered_backend_with_an_explicit_decision(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.rust_bridge import secret_manager
    from litellm.secret_managers.secret_manager_handler import get_secret_from_manager

    with recording_service() as server:
        server.default_response = ResponseSpec(body=_vault_body("native-value"))
        facade: Final = _Facade()
        register_native_secret_manager(
            facade,
            KeyManagementSystem.HASHICORP_VAULT,
            _Facade,
            environment={"HCP_VAULT_ADDR": server.base_url, "HCP_VAULT_TOKEN": "token"},
            enterprise_enabled=True,
        )
        rules: Final[Rules] = (SecretManagerRule(Rollout.RUST_REQUIRED, systems=frozenset({"hashicorp_vault"})),)
        runtime: Final = resolve_native_secret_manager(facade, "hashicorp_vault", rules)
        assert runtime is not None
        assert runtime.read_secret("KEY") == "native-value"
        monkeypatch.setattr(
            secret_manager,
            "resolve_native_secret_manager",
            lambda client, system: resolve_native_secret_manager(client, system, rules),
        )
        assert get_secret_from_manager(facade, "hashicorp_vault", "KEY") == "native-value"


@pytest.mark.parametrize("system", ("google_secret_manager", "hashicorp_vault", "cyberark"))
def test_enterprise_backends_cannot_initialize_without_entitlement(system: str) -> None:
    with pytest.raises(ValueError, match=r"[Ee]nterprise|[Pp]remium"):
        native._SecretManagerRuntime.from_config(system, {"CYBERARK_API_KEY": "key"})


def test_azure_factory_rejects_unencrypted_vault_endpoints() -> None:
    with pytest.raises(ValueError, match="https"):
        native._SecretManagerRuntime.from_config("azure_key_vault", {"AZURE_KEY_VAULT_URI": "http://127.0.0.1:1"})


def test_unregistered_builtin_requires_explicit_configuration() -> None:
    manager: Final = AWSSecretsManagerV2(aws_region_name="us-east-1")
    delattr(manager, "_litellm_native_secret_config")
    with pytest.raises(native.RustBridgeDeclined, match="native configuration"):
        native._SecretManagerRuntime.from_client(manager)


def test_binding_rejects_a_different_backend_before_reading() -> None:
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
