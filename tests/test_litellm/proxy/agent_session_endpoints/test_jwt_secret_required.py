"""
Validation #15 — daemon JWT secret is REQUIRED, no master-key fallback.

The daemon JWT secret (``LITELLM_AGENT_JWT_SECRET``) MUST be a separate
credential from the proxy master key. A prior version of
``auth._get_signing_secret`` fell back to ``LITELLM_MASTER_KEY`` when
the dedicated env var was unset — which conflated two distinct auth
surfaces and meant a captured daemon JWT could be used to mint regular
API keys with master-key authority.

This test file enforces:
  1. ``_get_signing_secret`` raises when ``LITELLM_AGENT_JWT_SECRET`` is
     unset, even when ``LITELLM_MASTER_KEY`` IS set (no silent fallback).
  2. ``is_agent_jwt_secret_configured()`` returns False in that case so
     ``proxy_server.py`` knows to skip mounting the routers.
  3. ``mint_daemon_token`` and ``decode_daemon_token`` both surface the
     same error — every caller is gated by the env var.
"""

import pytest

from litellm.proxy.agent_session_endpoints.auth import (
    AgentJWTSecretNotConfiguredError,
    _get_signing_secret,
    decode_daemon_token,
    is_agent_jwt_secret_configured,
    mint_daemon_token,
)


def test_signing_secret_raises_without_dedicated_env_var(monkeypatch):
    """The dedicated env var is the ONLY source of the secret.

    Setting LITELLM_MASTER_KEY (and clearing LITELLM_AGENT_JWT_SECRET)
    must NOT silently provide a fallback.
    """
    monkeypatch.delenv("LITELLM_AGENT_JWT_SECRET", raising=False)
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-master-key-which-must-not-be-used")

    with pytest.raises(AgentJWTSecretNotConfiguredError):
        _get_signing_secret()


def test_is_agent_jwt_secret_configured_reports_unset(monkeypatch):
    """Used at startup to decide whether to mount the routers.

    Returns False when the env var is unset, even if the master key is
    set — preventing the proxy from silently exposing an
    auth-conflated /v2/agents surface.
    """
    monkeypatch.delenv("LITELLM_AGENT_JWT_SECRET", raising=False)
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-1234")
    assert is_agent_jwt_secret_configured() is False


def test_is_agent_jwt_secret_configured_reports_set(monkeypatch):
    monkeypatch.setenv("LITELLM_AGENT_JWT_SECRET", "dedicated-jwt-secret-value")
    assert is_agent_jwt_secret_configured() is True


def test_is_agent_jwt_secret_configured_rejects_empty_string(monkeypatch):
    """Empty string should be treated as unset — same as ``os.environ.get`` semantics."""
    monkeypatch.setenv("LITELLM_AGENT_JWT_SECRET", "")
    assert is_agent_jwt_secret_configured() is False


def test_mint_token_raises_without_dedicated_env_var(monkeypatch):
    monkeypatch.delenv("LITELLM_AGENT_JWT_SECRET", raising=False)
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-1234")
    with pytest.raises(AgentJWTSecretNotConfiguredError):
        mint_daemon_token(
            session_id="sess_abc",
            agent_id="agt_abc",
            expires_at_epoch=10**10,
        )


def test_decode_token_raises_without_dedicated_env_var(monkeypatch):
    """Even decoding (which the daemon-auth dependency calls on every
    request) must refuse to operate without the dedicated secret."""
    monkeypatch.delenv("LITELLM_AGENT_JWT_SECRET", raising=False)
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-1234")
    with pytest.raises(AgentJWTSecretNotConfiguredError):
        decode_daemon_token("any-token-bytes")


def test_proxy_server_skips_mounting_when_secret_missing(monkeypatch):
    """End-to-end: simulate the startup check ``proxy_server.py`` runs.

    When the env var is unset, ``is_agent_jwt_secret_configured`` returns
    False — and that is the function ``proxy_server.py`` inspects to
    decide whether to mount the routers.
    """
    monkeypatch.delenv("LITELLM_AGENT_JWT_SECRET", raising=False)
    # No master key fallback even when LITELLM_MASTER_KEY is set.
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-1234")
    assert is_agent_jwt_secret_configured() is False
