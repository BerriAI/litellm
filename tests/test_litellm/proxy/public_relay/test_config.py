import base64

from litellm.proxy.public_relay.config import PublicRelaySettings


def _secret() -> str:
    return base64.urlsafe_b64encode(b"x" * 32).decode()


def test_operational_configuration_requires_base_url(monkeypatch) -> None:
    values = {
        "PUBLIC_RELAY_ENABLED": "true",
        "PUBLIC_RELAY_SESSION_SECRET": _secret(),
        "PUBLIC_RELAY_CONTENT_ENCRYPTION_KEY": _secret(),
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("PUBLIC_RELAY_BASE_URL", raising=False)

    missing = PublicRelaySettings.from_env().missing_runtime_configuration()

    assert missing == ("PUBLIC_RELAY_BASE_URL",)


def test_operational_configuration_accepts_single_server_settings(monkeypatch) -> None:
    values = {
        "PUBLIC_RELAY_ENABLED": "true",
        "PUBLIC_RELAY_SESSION_SECRET": _secret(),
        "PUBLIC_RELAY_CONTENT_ENCRYPTION_KEY": _secret(),
        "PUBLIC_RELAY_BASE_URL": "https://relay.example.test",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)

    assert PublicRelaySettings.from_env().missing_runtime_configuration() == ()
