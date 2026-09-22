import dataclasses
import logging
from pathlib import Path
from typing import Final

import httpx
import pytest
from pydantic import TypeAdapter
from typing_extensions import ReadOnly, TypedDict

import litellm
from litellm.integrations.custom_secret_manager import CustomSecretManager
from litellm.llms.custom_httpx.http_handler import default_user_agent
from litellm.rust_bridge import settings
from litellm.types.secret_managers.main import KeyManagementSettings, KeyManagementSystem

CONTRACT_PATH: Final = Path(__file__).parents[3] / "litellm-rust/crates/python-bridge/python_settings.json"


class SettingSpec(TypedDict):
    adapter: ReadOnly[str]
    required: ReadOnly[bool]
    precedence: ReadOnly[str]
    sensitive: ReadOnly[bool]
    shapes: ReadOnly[list[str]]
    unsupported_live: ReadOnly[str | None]


class SettingsGroup(TypedDict):
    version: ReadOnly[int]
    fields: ReadOnly[dict[str, SettingSpec]]


def test_the_rust_contract_matches_the_returned_fields() -> None:
    contract: Final = TypeAdapter(dict[str, SettingsGroup]).validate_json(CONTRACT_PATH.read_text())

    assert {name: tuple(group["fields"]) for name, group in contract.items()} == {
        "http_settings": tuple(field.name for field in dataclasses.fields(settings.http_settings())),
        "url_policy": tuple(field.name for field in dataclasses.fields(settings.url_policy())),
        "provider_defaults": tuple(field.name for field in dataclasses.fields(settings.provider_defaults())),
        "secret_manager": tuple(field.name for field in dataclasses.fields(settings.secret_manager())),
    }


def test_url_policy_reads_the_litellm_globals(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "user_url_validation", False)
    monkeypatch.setattr(litellm, "user_url_allowed_hosts", ["docs.internal:8443"])

    assert settings.url_policy() == settings.UrlPolicy(
        user_url_validation=False,
        user_url_allowed_hosts=["docs.internal:8443"],
    )


def test_http_settings_reads_the_litellm_globals(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "ssl_verify", "/etc/ssl/corp.pem")
    monkeypatch.setattr(litellm, "ssl_certificate", "/etc/ssl/client.pem")
    monkeypatch.setattr(litellm, "ssl_security_level", "DEFAULT@SECLEVEL=1")
    monkeypatch.setattr(litellm, "ssl_ecdh_curve", "X25519")
    monkeypatch.setattr(litellm, "force_ipv4", True)
    monkeypatch.setattr(litellm, "http2", True)
    monkeypatch.setattr(litellm, "aiohttp_trust_env", True)
    monkeypatch.setattr(litellm, "disable_aiohttp_trust_env", True)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

    assert settings.http_settings() == settings.HttpSettings(
        ssl_verify="/etc/ssl/corp.pem",
        ssl_certificate="/etc/ssl/client.pem",
        ssl_security_level="DEFAULT@SECLEVEL=1",
        ssl_ecdh_curve="X25519",
        force_ipv4=True,
        http2=True,
        aiohttp_trust_env=True,
        disable_aiohttp_trust_env=True,
        disable_aiohttp_transport=True,
        user_agent=default_user_agent(),
    )


def test_http_settings_ignores_environment_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_USER_AGENT", "operator/1")
    monkeypatch.setenv("SSL_VERIFY", "false")
    monkeypatch.setattr(litellm, "ssl_verify", True)

    result: Final = settings.http_settings()

    assert result.user_agent == default_user_agent()
    assert result.ssl_verify is True


def test_warn_reaches_the_litellm_logger(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        settings.warn("ssl_ecdh_curve 'secp521r1' is not supported")

    assert [record.getMessage() for record in caplog.records] == ["ssl_ecdh_curve 'secp521r1' is not supported"]


class _VaultSecrets(CustomSecretManager):
    def __init__(self, secrets: dict[str, str]) -> None:
        super().__init__(secret_manager_name="rust_bridge_settings_test")
        self.secrets = secrets

    async def async_read_secret(
        self,
        secret_name: str,
        optional_params: dict[str, object] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> str | None:
        return self.secrets.get(secret_name)

    def sync_read_secret(
        self,
        secret_name: str,
        optional_params: dict[str, object] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> str | None:
        return self.secrets.get(secret_name)


def test_secret_manager_projects_custom_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    manager_settings: Final = KeyManagementSettings(
        access_mode="read_and_write",
        hosted_keys=["MISTRAL_API_KEY"],
        primary_secret_name="primary",
        aws_region_name="us-east-1",
    )
    client: Final = _VaultSecrets({"MISTRAL_API_KEY": "vault-key"})
    monkeypatch.setattr(litellm, "secret_manager_client", client)
    monkeypatch.setattr(litellm, "_key_management_system", KeyManagementSystem.CUSTOM)
    monkeypatch.setattr(litellm, "_key_management_settings", manager_settings)

    assert settings.secret_manager() == settings.SecretManager(
        system="custom",
        access_mode="read_and_write",
        hosted_keys=["MISTRAL_API_KEY"],
        primary_secret_name="primary",
        store_virtual_keys=manager_settings.store_virtual_keys,
        prefix_for_stored_virtual_keys=manager_settings.prefix_for_stored_virtual_keys,
        kms_key_id=manager_settings.kms_key_id,
        custom_secret_manager=manager_settings.custom_secret_manager,
        aws_region_name="us-east-1",
        aws_role_name=manager_settings.aws_role_name,
        aws_session_name=manager_settings.aws_session_name,
        aws_external_id=manager_settings.aws_external_id,
        aws_profile_name=manager_settings.aws_profile_name,
        aws_web_identity_token=manager_settings.aws_web_identity_token,
        aws_sts_endpoint=manager_settings.aws_sts_endpoint,
        replica_regions=manager_settings.replica_regions,
        client=client,
        settings_object=manager_settings,
    )


def test_secret_manager_without_a_client_has_no_system(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "secret_manager_client", None)

    assert settings.secret_manager().system is None


def test_secret_manager_uses_key_management_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "secret_manager_client", None)
    monkeypatch.setattr(litellm, "_key_management_settings", None)

    defaults: Final = KeyManagementSettings()
    result: Final = settings.secret_manager()

    assert result == settings.SecretManager(
        system=None,
        access_mode=defaults.access_mode,
        hosted_keys=defaults.hosted_keys,
        primary_secret_name=defaults.primary_secret_name,
        store_virtual_keys=defaults.store_virtual_keys,
        prefix_for_stored_virtual_keys=defaults.prefix_for_stored_virtual_keys,
        kms_key_id=defaults.kms_key_id,
        custom_secret_manager=defaults.custom_secret_manager,
        aws_region_name=defaults.aws_region_name,
        aws_role_name=defaults.aws_role_name,
        aws_session_name=defaults.aws_session_name,
        aws_external_id=defaults.aws_external_id,
        aws_profile_name=defaults.aws_profile_name,
        aws_web_identity_token=defaults.aws_web_identity_token,
        aws_sts_endpoint=defaults.aws_sts_endpoint,
        replica_regions=defaults.replica_regions,
        client=None,
        settings_object=None,
    )


def test_provider_defaults_read_the_litellm_globals(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "vertex_project", "configured-project")
    monkeypatch.setattr(litellm, "vertex_location", "europe-west4")
    monkeypatch.setattr(litellm, "enable_azure_ad_token_refresh", True)

    assert settings.provider_defaults() == settings.ProviderDefaults(
        vertex_project="configured-project",
        vertex_location="europe-west4",
        enable_azure_ad_token_refresh=True,
    )
