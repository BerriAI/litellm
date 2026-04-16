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
