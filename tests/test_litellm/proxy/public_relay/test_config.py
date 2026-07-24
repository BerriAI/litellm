import base64

from litellm.proxy.public_relay.config import PublicRelaySettings


def _secret() -> str:
    return base64.urlsafe_b64encode(b"x" * 32).decode()


def test_operational_configuration_requires_email_delivery(monkeypatch) -> None:
    values = {
        "PUBLIC_RELAY_ENABLED": "true",
        "PUBLIC_RELAY_SESSION_SECRET": _secret(),
        "PUBLIC_RELAY_CONTENT_ENCRYPTION_KEY": _secret(),
        "PUBLIC_RELAY_TURNSTILE_VERIFY_URL": "https://turnstile.example.test/verify",
        "STRIPE_SECRET_KEY": "sk_test_value",
        "STRIPE_WEBHOOK_SECRET": "whsec_value",
        "PUBLIC_RELAY_CHECKOUT_SUCCESS_URL": "https://relay.example.test/portal/billing",
        "PUBLIC_RELAY_CHECKOUT_CANCEL_URL": "https://relay.example.test/portal/billing",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    for key in ("RESEND_API_KEY", "RESEND_FROM_EMAIL", "SMTP_HOST", "SMTP_SENDER_EMAIL"):
        monkeypatch.delenv(key, raising=False)

    missing = PublicRelaySettings.from_env().missing_runtime_configuration()

    assert missing == ("RESEND_API_KEY/SMTP_HOST and sender email",)


def test_operational_configuration_accepts_smtp(monkeypatch) -> None:
    values = {
        "PUBLIC_RELAY_ENABLED": "true",
        "PUBLIC_RELAY_SESSION_SECRET": _secret(),
        "PUBLIC_RELAY_CONTENT_ENCRYPTION_KEY": _secret(),
        "PUBLIC_RELAY_TURNSTILE_VERIFY_URL": "https://turnstile.example.test/verify",
        "STRIPE_SECRET_KEY": "sk_test_value",
        "STRIPE_WEBHOOK_SECRET": "whsec_value",
        "PUBLIC_RELAY_CHECKOUT_SUCCESS_URL": "https://relay.example.test/portal/billing",
        "PUBLIC_RELAY_CHECKOUT_CANCEL_URL": "https://relay.example.test/portal/billing",
        "SMTP_HOST": "smtp.example.test",
        "SMTP_SENDER_EMAIL": "relay@example.test",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)

    assert PublicRelaySettings.from_env().missing_runtime_configuration() == ()
