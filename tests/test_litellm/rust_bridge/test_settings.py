import dataclasses
from pathlib import Path
from typing import Final

import pytest
from pydantic import TypeAdapter

import litellm
from litellm.llms.custom_httpx.http_handler import default_user_agent
from litellm.rust_bridge import settings

CONTRACT_PATH: Final = Path(__file__).parents[3] / "litellm-rust/crates/python-bridge/python_settings.json"


def test_the_rust_contract_matches_the_returned_fields() -> None:
    contract: Final = TypeAdapter(dict[str, list[str]]).validate_json(CONTRACT_PATH.read_text())

    assert contract == {
        "http_settings": [field.name for field in dataclasses.fields(settings.http_settings())],
        "url_policy": [field.name for field in dataclasses.fields(settings.url_policy())],
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
