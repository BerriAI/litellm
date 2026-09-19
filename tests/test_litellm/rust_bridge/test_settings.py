import dataclasses
import logging
from pathlib import Path
from typing import Final

import httpx
import pytest
from pydantic import TypeAdapter

import litellm
from litellm.integrations.custom_secret_manager import CustomSecretManager
from litellm.llms.custom_httpx.http_handler import default_user_agent
from litellm.rust_bridge import settings
from litellm.types.secret_managers.main import KeyManagementSettings, KeyManagementSystem

CONTRACT_PATH: Final = Path(__file__).parents[3] / "litellm-rust/crates/python-bridge/python_settings.json"


def test_the_rust_contract_matches_the_returned_fields() -> None:
    contract: Final = TypeAdapter(dict[str, list[str]]).validate_json(CONTRACT_PATH.read_text())

    assert contract == {
        "http_settings": [field.name for field in dataclasses.fields(settings.http_settings())],
        "url_policy": [field.name for field in dataclasses.fields(settings.url_policy())],
        "provider_defaults": [field.name for field in dataclasses.fields(settings.provider_defaults())],
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


def test_secret_prefers_the_secret_manager_and_falls_back_to_the_environment_on_a_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "stale-env-key")
    monkeypatch.setenv("REDUCTO_API_KEY", "env-only-key")
    monkeypatch.setattr(litellm, "secret_manager_client", _VaultSecrets({"MISTRAL_API_KEY": "vault-key"}))
    monkeypatch.setattr(litellm, "_key_management_system", KeyManagementSystem.CUSTOM)
    monkeypatch.setattr(litellm, "_key_management_settings", KeyManagementSettings(access_mode="read_only"))

    assert settings.secret("MISTRAL_API_KEY") == "vault-key"
    assert settings.secret("REDUCTO_API_KEY") == "env-only-key"
    assert settings.secret("ABSENT_KEY") is None


def test_secret_reads_the_environment_without_a_secret_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "env-key")
    monkeypatch.setattr(litellm, "secret_manager_client", None)

    assert settings.secret("MISTRAL_API_KEY") == "env-key"


def test_provider_defaults_read_the_litellm_globals(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "vertex_project", "configured-project")
    monkeypatch.setattr(litellm, "vertex_location", "europe-west4")
    monkeypatch.setattr(litellm, "enable_azure_ad_token_refresh", True)

    assert settings.provider_defaults() == settings.ProviderDefaults(
        vertex_project="configured-project",
        vertex_location="europe-west4",
        enable_azure_ad_token_refresh=True,
    )
