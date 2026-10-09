from __future__ import annotations

from types import SimpleNamespace
from typing import Final

import pytest

import litellm
from litellm.rust_bridge.secret_manager import capture_secret_manager, native_secret_manager_config
from litellm.secret_managers.aws_secret_manager_v2 import AWSSecretsManagerV2
from litellm.types.secret_managers.main import KeyManagementSettings
from tests._support.rust_secret_manager import preserve_manager_globals as preserve_manager_globals


def test_custom_subclass_is_not_captured_for_a_native_handle() -> None:
    native: Final = pytest.importorskip("litellm.rust_bridge._native")

    class CustomManager(AWSSecretsManagerV2):
        def sync_read_secret(self, secret_name: str, primary_secret_name: str | None = None) -> str:
            return f"custom:{secret_name}"

    manager: Final = CustomManager(aws_region_name="us-east-1")
    assert native._SecretManagerRuntime.from_client(manager) is None
    assert manager.sync_read_secret("KEY") == "custom:KEY"


@pytest.mark.parametrize("system", ("google_secret_manager", "hashicorp_vault", "cyberark"))
def test_enterprise_backends_cannot_initialize_without_entitlement(system: str) -> None:
    native: Final = pytest.importorskip("litellm.rust_bridge._native")
    with pytest.raises(ValueError, match=r"[Ee]nterprise|[Pp]remium"):
        native._SecretManagerRuntime.from_config(system, {"CYBERARK_API_KEY": "key"})


def test_azure_factory_rejects_unencrypted_vault_endpoints() -> None:
    native: Final = pytest.importorskip("litellm.rust_bridge._native")
    with pytest.raises(ValueError, match="https"):
        native._SecretManagerRuntime.from_config("azure_key_vault", {"AZURE_KEY_VAULT_URI": "http://127.0.0.1:1"})


def test_uncaptured_clients_have_no_native_handle() -> None:
    native: Final = pytest.importorskip("litellm.rust_bridge._native")
    assert native._SecretManagerRuntime.from_client(SimpleNamespace()) is None


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


def test_public_aws_bootstrap_read_does_not_hit_the_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "bootstrap-value")
    assert AWSSecretsManagerV2().sync_read_secret("AWS_ACCESS_KEY_ID") == "bootstrap-value"
