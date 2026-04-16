import socket

import pytest

import litellm
from litellm.litellm_core_utils import url_utils
from litellm.litellm_core_utils.url_utils import SSRFError, _is_blocked_ip, validate_url


@pytest.fixture
def mock_dns_public(monkeypatch):
    """Resolve any hostname to 93.184.216.34 (public)."""

    def fake_getaddrinfo(host, port, *args, **kwargs):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port or 80))
        ]

    monkeypatch.setattr(url_utils.socket, "getaddrinfo", fake_getaddrinfo)


@pytest.fixture
def mock_dns_failure(monkeypatch):
    """Make every DNS lookup raise gaierror."""

    def fake_getaddrinfo(host, port, *args, **kwargs):
        raise socket.gaierror("Name or service not known")

    monkeypatch.setattr(url_utils.socket, "getaddrinfo", fake_getaddrinfo)


class TestIsBlockedIp:
    def test_blocks_private(self):
        assert _is_blocked_ip("10.0.0.1") is True

    def test_allows_public(self):
        assert _is_blocked_ip("8.8.8.8") is False

    def test_unparseable_is_blocked(self):
        assert _is_blocked_ip("not-an-ip") is True


class TestValidateUrl:
    def test_blocks_loopback(self):
        with pytest.raises(SSRFError):
            validate_url("http://127.0.0.1/test")

    def test_blocks_imds(self):
        with pytest.raises(SSRFError):
            validate_url("http://169.254.169.254/latest/meta-data/")

    def test_blocks_rfc1918_class_a(self):
        with pytest.raises(SSRFError):
            validate_url("http://10.0.1.5:8080/v1/completions")

    def test_blocks_rfc1918_class_b(self):
        with pytest.raises(SSRFError):
            validate_url("http://172.16.0.1/")

    def test_blocks_rfc1918_class_c(self):
        with pytest.raises(SSRFError):
            validate_url("http://192.168.1.1/")

    def test_blocks_file_scheme(self):
        with pytest.raises(SSRFError):
            validate_url("file:///etc/passwd")

    def test_blocks_ftp_scheme(self):
        with pytest.raises(SSRFError):
            validate_url("ftp://internal.host/data")

    def test_blocks_no_hostname(self):
        with pytest.raises(SSRFError):
            validate_url("http:///path")

    def test_allows_public_https(self, mock_dns_public):
        rewritten, host = validate_url("https://example.com/image.png")
        assert host == "example.com"
        assert rewritten == "https://example.com/image.png"

    def test_rewrites_public_http_to_ip(self, mock_dns_public):
        rewritten, host = validate_url("http://example.com/image.png")
        assert host == "example.com"
        assert "example.com" not in rewritten

    def test_preserves_path_and_query(self, mock_dns_public):
        rewritten, host = validate_url("http://example.com/path?key=value")
        assert "/path" in rewritten
        assert "key=value" in rewritten

    def test_dns_failure_raises(self, mock_dns_failure):
        with pytest.raises(SSRFError, match="DNS resolution failed"):
            validate_url("http://this-domain-does-not-exist-xyz123.invalid/test")

    def test_blocks_localhost_hostname(self):
        with pytest.raises(SSRFError):
            validate_url("http://localhost/")

    def test_blocks_ipv6_loopback(self):
        with pytest.raises(SSRFError):
            validate_url("http://[::1]/")

    def test_https_rewrites_when_ssl_verify_disabled(
        self, monkeypatch, mock_dns_public
    ):
        monkeypatch.setattr(litellm, "ssl_verify", False)
        rewritten, host = validate_url("https://example.com/image.png")
        assert host == "example.com"
        assert "example.com" not in rewritten  # rewritten to IP

    def test_https_not_rewritten_when_ssl_verify_enabled(
        self, monkeypatch, mock_dns_public
    ):
        monkeypatch.setattr(litellm, "ssl_verify", True)
        rewritten, host = validate_url("https://example.com/image.png")
        assert rewritten == "https://example.com/image.png"


class TestHostHeaderFormatting:
    """RFC 7230 §5.4: IPv6 literals must be bracketed in the Host header."""

    def test_ipv4_no_port(self, monkeypatch):
        def fake(host, port, *a, **kw):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.2.3.4", port))]

        monkeypatch.setattr(url_utils.socket, "getaddrinfo", fake)
        _, host = validate_url("http://example.com/")
        assert host == "example.com"

    def test_ipv4_with_explicit_nondefault_port(self, monkeypatch):
        def fake(host, port, *a, **kw):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.2.3.4", port))]

        monkeypatch.setattr(url_utils.socket, "getaddrinfo", fake)
        _, host = validate_url("http://example.com:8080/")
        assert host == "example.com:8080"

    def test_ipv4_with_explicit_default_port_strips_port(self, monkeypatch):
        def fake(host, port, *a, **kw):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.2.3.4", port))]

        monkeypatch.setattr(url_utils.socket, "getaddrinfo", fake)
        _, host = validate_url("http://example.com:80/")
        assert host == "example.com"

    def test_ipv6_literal_is_bracketed_with_port(self, monkeypatch):
        """Regression: IPv6 + port produced ambiguous `Host: 2001:db8::1:8080`."""
        monkeypatch.setattr(litellm, "user_url_allowed_hosts", ["[2001:db8::1]"])

        def fake(host, port, *a, **kw):
            return [
                (
                    socket.AF_INET6,
                    socket.SOCK_STREAM,
                    6,
                    "",
                    ("2001:db8::1", port, 0, 0),
                )
            ]

        monkeypatch.setattr(url_utils.socket, "getaddrinfo", fake)
        _, host = validate_url("http://[2001:db8::1]:8080/")
        assert host == "[2001:db8::1]:8080"

    def test_ipv6_literal_is_bracketed_without_port(self, monkeypatch):
        monkeypatch.setattr(litellm, "user_url_allowed_hosts", ["[2001:db8::1]"])

        def fake(host, port, *a, **kw):
            return [
                (
                    socket.AF_INET6,
                    socket.SOCK_STREAM,
                    6,
                    "",
                    ("2001:db8::1", port, 0, 0),
                )
            ]

        monkeypatch.setattr(url_utils.socket, "getaddrinfo", fake)
        _, host = validate_url("http://[2001:db8::1]/")
        assert host == "[2001:db8::1]"


class TestValidationMasterSwitch:
    def test_disabled_bypasses_fetch_in_safe_get(self, monkeypatch):
        """When user_url_validation is False, safe_get delegates to client.get without validation."""
        monkeypatch.setattr(litellm, "user_url_validation", False)

        calls = []

        class FakeClient:
            def get(self, url, **kwargs):
                calls.append((url, kwargs))

                class R:
                    is_redirect = False

                return R()

        url_utils.safe_get(FakeClient(), "http://127.0.0.1/internal")
        assert calls and calls[0][0] == "http://127.0.0.1/internal"
        assert calls[0][1].get("follow_redirects") is True

    def test_enabled_still_blocks(self, monkeypatch):
        monkeypatch.setattr(litellm, "user_url_validation", True)
        with pytest.raises(SSRFError):
            validate_url("http://127.0.0.1/")


class TestHostAllowlist:
    def test_allowlisted_hostname_permits_private_ip(self, monkeypatch):
        monkeypatch.setattr(litellm, "user_url_allowed_hosts", ["internal.corp"])

        def fake(host, port, *a, **kw):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.1.5", port))]

        monkeypatch.setattr(url_utils.socket, "getaddrinfo", fake)
        rewritten, host = validate_url("http://internal.corp/path")
        assert host == "internal.corp"
        assert "10.0.1.5" in rewritten

    def test_non_allowlisted_hostname_still_blocked(self, monkeypatch):
        monkeypatch.setattr(litellm, "user_url_allowed_hosts", ["internal.corp"])

        def fake(host, port, *a, **kw):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.1.5", port))]

        monkeypatch.setattr(url_utils.socket, "getaddrinfo", fake)
        with pytest.raises(SSRFError):
            validate_url("http://other.corp/")

    def test_allowlist_case_insensitive(self, monkeypatch):
        monkeypatch.setattr(litellm, "user_url_allowed_hosts", ["Internal.Corp"])

        def fake(host, port, *a, **kw):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.1.5", port))]

        monkeypatch.setattr(url_utils.socket, "getaddrinfo", fake)
        rewritten, _ = validate_url("http://internal.corp/")
        assert "10.0.1.5" in rewritten

    def test_allowlist_with_port_matches_explicit_port(self, monkeypatch):
        monkeypatch.setattr(litellm, "user_url_allowed_hosts", ["internal.corp:8080"])

        def fake(host, port, *a, **kw):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.1.5", port))]

        monkeypatch.setattr(url_utils.socket, "getaddrinfo", fake)
        rewritten, host = validate_url("http://internal.corp:8080/")
        assert host == "internal.corp:8080"
        assert "10.0.1.5" in rewritten

    def test_allowlist_with_port_matches_default_port(self, monkeypatch):
        """Admin entry `host:443` matches `https://host/` (port=None, default 443)."""
        monkeypatch.setattr(litellm, "user_url_allowed_hosts", ["internal.corp:443"])

        def fake(host, port, *a, **kw):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.1.5", port))]

        monkeypatch.setattr(url_utils.socket, "getaddrinfo", fake)
        # Should succeed — no SSRFError raised
        validate_url("https://internal.corp/")

    def test_allowlist_port_specific_does_not_match_other_port(self, monkeypatch):
        monkeypatch.setattr(litellm, "user_url_allowed_hosts", ["internal.corp:8080"])

        def fake(host, port, *a, **kw):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.1.5", port))]

        monkeypatch.setattr(url_utils.socket, "getaddrinfo", fake)
        with pytest.raises(SSRFError):
            validate_url("http://internal.corp:9090/")

    def test_allowlist_host_entry_matches_any_port(self, monkeypatch):
        monkeypatch.setattr(litellm, "user_url_allowed_hosts", ["internal.corp"])

        def fake(host, port, *a, **kw):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.1.5", port))]

        monkeypatch.setattr(url_utils.socket, "getaddrinfo", fake)
        validate_url("http://internal.corp:9090/")
        validate_url("https://internal.corp:8443/")

    def test_allowlist_permits_loopback(self, monkeypatch):
        """Admin may opt into loopback if they explicitly configure it."""
        monkeypatch.setattr(litellm, "user_url_allowed_hosts", ["localhost"])
        # localhost resolves locally without needing mocks
        rewritten, host = validate_url("http://localhost:8080/")
        assert host == "localhost:8080"

    def test_empty_allowlist_retains_default_deny(self, monkeypatch):
        monkeypatch.setattr(litellm, "user_url_allowed_hosts", [])
        with pytest.raises(SSRFError):
            validate_url("http://127.0.0.1/")

    def test_allowlist_strips_trailing_dot(self, monkeypatch):
        monkeypatch.setattr(litellm, "user_url_allowed_hosts", ["internal.corp."])

        def fake(host, port, *a, **kw):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.1.5", port))]

        monkeypatch.setattr(url_utils.socket, "getaddrinfo", fake)
        validate_url("http://internal.corp/")
