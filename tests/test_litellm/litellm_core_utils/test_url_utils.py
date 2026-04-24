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

    # Coverage delta picked up by switching to `not ip.is_global` (RFC 6890)
    # over the old hand-maintained CIDR list.
    def test_blocks_cgnat_alibaba_metadata(self):
        """100.100.100.200 is Alibaba Cloud metadata; lives in CGNAT."""
        assert _is_blocked_ip("100.100.100.200") is True

    def test_blocks_ietf_protocol_assignments_old_oracle_metadata(self):
        """192.0.0.192 was the legacy Oracle Cloud metadata IP."""
        assert _is_blocked_ip("192.0.0.192") is True

    def test_blocks_documentation_ranges(self):
        assert _is_blocked_ip("192.0.2.1") is True
        assert _is_blocked_ip("198.51.100.1") is True
        assert _is_blocked_ip("203.0.113.1") is True

    def test_blocks_multicast(self):
        assert _is_blocked_ip("224.0.0.1") is True

    def test_blocks_reserved_future_use(self):
        assert _is_blocked_ip("240.0.0.1") is True

    def test_blocks_broadcast(self):
        assert _is_blocked_ip("255.255.255.255") is True

    def test_blocks_azure_wire_server(self):
        """168.63.129.16 is globally routable but cloud-internal — explicit exception."""
        assert _is_blocked_ip("168.63.129.16") is True

    def test_blocks_aws_ipv6_imds(self):
        """fd00:ec2::254 is AWS's IPv6 IMDS, in IPv6 ULA (fc00::/7)."""
        assert _is_blocked_ip("fd00:ec2::254") is True

    def test_blocks_ipv4_mapped_private(self):
        """::ffff:10.0.0.1 must be unwrapped and blocked as 10.0.0.1."""
        assert _is_blocked_ip("::ffff:10.0.0.1") is True

    def test_blocks_ipv4_mapped_azure_wire_server(self):
        """::ffff:168.63.129.16 must be unwrapped and blocked via the exception list."""
        assert _is_blocked_ip("::ffff:168.63.129.16") is True


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

    def test_blocks_localhost_hostname(self, monkeypatch):
        def fake(host, port, *a, **kw):
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port or 80))
            ]

        monkeypatch.setattr(url_utils.socket, "getaddrinfo", fake)
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


class TestRedirectHostnamePreservation:
    """Relative-location redirects must keep the original hostname, not the
    rewritten IP, so the next hop's Host header still identifies the site."""

    def test_relative_redirect_preserves_hostname_for_next_hop(self, monkeypatch):
        def fake(host, port, *a, **kw):
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))
            ]

        monkeypatch.setattr(url_utils.socket, "getaddrinfo", fake)

        class FakeResponse:
            def __init__(self, status, location=None):
                self.status_code = status
                self.headers = {"location": location} if location else {}
                self.is_redirect = 300 <= status < 400

        hops = []

        class FakeClient:
            def __init__(self):
                self._n = 0

            def get(self, url, headers=None, follow_redirects=False, **kw):
                hops.append({"url": url, "host": (headers or {}).get("Host")})
                self._n += 1
                if self._n == 1:
                    return FakeResponse(302, "/redirected")
                return FakeResponse(200)

        url_utils.safe_get(FakeClient(), "http://example.com/initial")
        assert len(hops) == 2
        # Both hops must carry the ORIGINAL hostname in the Host header.
        assert hops[0]["host"] == "example.com"
        assert hops[1]["host"] == "example.com"
        # Both outbound URLs go to the resolved IP (rewritten), not the hostname.
        assert "93.184.216.34" in hops[0]["url"]
        assert "93.184.216.34" in hops[1]["url"]
        # The second hop resolved /redirected relative to the original, not the IP.
        assert hops[1]["url"].endswith("/redirected")


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

        def fake(host, port, *a, **kw):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port))]

        monkeypatch.setattr(url_utils.socket, "getaddrinfo", fake)
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
