"""
Tests for `_resolve_proxy_url` — the helper that picks the URL embedded in
the install one-liner returned to the operator.

This is security-sensitive: if an attacker can influence the URL, they can
redirect the worker (and the freshly-issued pair token) to a host they
control. The contract we test:

* `LITELLM_CLOUD_AGENT_PROXY_BASE_URL` (when set) wins, regardless of what
  the request claims. Operator-configured value is the only fully trusted
  source.
* `X-Forwarded-Host` / `X-Forwarded-Proto` are IGNORED unless the operator
  explicitly opts in via `LITELLM_TRUST_PROXY_HEADERS=1`. This is the fix
  for the header-injection path Greptile flagged on PR #27332.
* When neither override is present, we fall back to the request's direct
  `Host` header — that reflects the actual TCP destination of the request,
  not an attacker-controlled hop hint.
"""

from types import SimpleNamespace

import pytest

from litellm.proxy.agent_settings_endpoints.worker_endpoints import (
    _resolve_proxy_url,
)


def _fake_request(
    *,
    host: str = "proxy.example.com",
    scheme: str = "https",
    forwarded_host: str = "",
    forwarded_proto: str = "",
):
    """Build a minimal stand-in for fastapi.Request with just the bits
    `_resolve_proxy_url` actually reads. Avoids spinning up the full
    Starlette request object."""
    headers = {"host": host}
    if forwarded_host:
        headers["x-forwarded-host"] = forwarded_host
    if forwarded_proto:
        headers["x-forwarded-proto"] = forwarded_proto
    return SimpleNamespace(
        headers=headers,
        url=SimpleNamespace(scheme=scheme),
        client=SimpleNamespace(host="127.0.0.1"),
    )


class TestEnvOverrideWins:
    def test_explicit_proxy_base_url_used_verbatim(self, monkeypatch):
        monkeypatch.setenv(
            "LITELLM_CLOUD_AGENT_PROXY_BASE_URL", "https://configured.example"
        )
        url = _resolve_proxy_url(
            _fake_request(forwarded_host="attacker.example", forwarded_proto="http")
        )
        assert url == "https://configured.example"

    def test_proxy_base_url_strips_trailing_slash(self, monkeypatch):
        monkeypatch.setenv(
            "LITELLM_CLOUD_AGENT_PROXY_BASE_URL", "https://configured.example/"
        )
        url = _resolve_proxy_url(_fake_request())
        assert url == "https://configured.example"


class TestForwardedHeadersIgnoredByDefault:
    def test_xff_host_ignored_without_trust_flag(self, monkeypatch):
        monkeypatch.delenv("LITELLM_CLOUD_AGENT_PROXY_BASE_URL", raising=False)
        monkeypatch.delenv("LITELLM_TRUST_PROXY_HEADERS", raising=False)
        url = _resolve_proxy_url(
            _fake_request(
                host="trusted.example.com",
                forwarded_host="attacker.example",
                forwarded_proto="http",
            )
        )
        # The forged X-Forwarded-Host MUST NOT show up.
        assert "attacker.example" not in url
        assert url == "https://trusted.example.com"

    def test_xff_host_explicit_off(self, monkeypatch):
        monkeypatch.delenv("LITELLM_CLOUD_AGENT_PROXY_BASE_URL", raising=False)
        monkeypatch.setenv("LITELLM_TRUST_PROXY_HEADERS", "0")
        url = _resolve_proxy_url(
            _fake_request(
                host="trusted.example.com",
                forwarded_host="attacker.example",
            )
        )
        assert "attacker.example" not in url


class TestForwardedHeadersOptIn:
    def test_xff_honored_when_trust_flag_set(self, monkeypatch):
        monkeypatch.delenv("LITELLM_CLOUD_AGENT_PROXY_BASE_URL", raising=False)
        monkeypatch.setenv("LITELLM_TRUST_PROXY_HEADERS", "1")
        url = _resolve_proxy_url(
            _fake_request(
                host="origin.internal",
                forwarded_host="public.example.com",
                forwarded_proto="https",
            )
        )
        assert url == "https://public.example.com"

    def test_xff_falls_back_to_host_when_only_proto_forwarded(self, monkeypatch):
        # If only x-forwarded-proto is present (no host), we still use the
        # direct Host header — we never silently mix forwarded scheme with
        # request-direct host or vice versa.
        monkeypatch.delenv("LITELLM_CLOUD_AGENT_PROXY_BASE_URL", raising=False)
        monkeypatch.setenv("LITELLM_TRUST_PROXY_HEADERS", "1")
        url = _resolve_proxy_url(
            _fake_request(
                host="proxy.example.com",
                scheme="https",
                forwarded_proto="http",
            )
        )
        assert url == "https://proxy.example.com"


class TestFallbacks:
    def test_fallback_to_localhost_when_no_host(self, monkeypatch):
        monkeypatch.delenv("LITELLM_CLOUD_AGENT_PROXY_BASE_URL", raising=False)
        monkeypatch.delenv("LITELLM_TRUST_PROXY_HEADERS", raising=False)
        request = SimpleNamespace(
            headers={},
            url=SimpleNamespace(scheme="https"),
            client=SimpleNamespace(host="127.0.0.1"),
        )
        url = _resolve_proxy_url(request)
        assert url == "https://localhost:4000"


@pytest.fixture(autouse=True)
def _no_op_fixture():
    yield
