"""
Tests for SSRF protection in custom code guardrail HTTP primitives.
"""

import ipaddress
from unittest.mock import patch

import pytest

from litellm.proxy.guardrails.guardrail_hooks.custom_code.primitives import (
    _is_private_ip,
    _validate_url_for_ssrf,
    http_request,
)


# ---------------------------------------------------------------------------
# _is_private_ip
# ---------------------------------------------------------------------------


class TestIsPrivateIp:
    """Verify that private/reserved addresses are correctly identified."""

    @pytest.mark.parametrize(
        "ip",
        [
            "127.0.0.1",
            "10.0.0.1",
            "172.16.0.1",
            "192.168.1.1",
            "169.254.169.254",  # AWS/GCP metadata
            "0.0.0.0",
            "100.64.0.1",  # Carrier-grade NAT
            "::1",  # IPv6 loopback
            "fc00::1",  # IPv6 unique-local
            "fe80::1",  # IPv6 link-local
        ],
    )
    def test_private_ips_blocked(self, ip):
        addr = ipaddress.ip_address(ip)
        assert _is_private_ip(addr) is True

    @pytest.mark.parametrize(
        "ip",
        [
            "8.8.8.8",
            "1.1.1.1",
            "151.101.1.140",
            "2607:f8b0:4004:800::200e",  # Google public IPv6
        ],
    )
    def test_public_ips_allowed(self, ip):
        addr = ipaddress.ip_address(ip)
        assert _is_private_ip(addr) is False


# ---------------------------------------------------------------------------
# _validate_url_for_ssrf
# ---------------------------------------------------------------------------


class TestValidateUrlForSsrf:
    """URL-level SSRF validation."""

    def test_blocks_raw_private_ipv4(self):
        err = _validate_url_for_ssrf("http://127.0.0.1/admin")
        assert err is not None
        assert "private" in err.lower() or "reserved" in err.lower()

    def test_blocks_metadata_endpoint(self):
        err = _validate_url_for_ssrf(
            "http://169.254.169.254/latest/meta-data/iam/security-credentials/"
        )
        assert err is not None

    def test_blocks_raw_private_ipv6(self):
        err = _validate_url_for_ssrf("http://[::1]/secret")
        assert err is not None

    def test_allows_public_ip(self):
        err = _validate_url_for_ssrf("https://8.8.8.8/dns-query")
        assert err is None

    def test_blocks_no_hostname(self):
        err = _validate_url_for_ssrf("file:///etc/passwd")
        assert err is not None

    @patch("socket.getaddrinfo")
    def test_blocks_dns_rebinding_to_private(self, mock_getaddrinfo):
        """Hostname resolves to a private IP — must be blocked."""
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("127.0.0.1", 80)),
        ]
        err = _validate_url_for_ssrf("http://evil.example.com/steal")
        assert err is not None
        assert "private" in err.lower() or "reserved" in err.lower()

    @patch("socket.getaddrinfo")
    def test_allows_dns_to_public(self, mock_getaddrinfo):
        """Hostname resolves to a public IP — should be allowed."""
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("151.101.1.140", 443)),
        ]
        err = _validate_url_for_ssrf("https://api.example.com/check")
        assert err is None

    @patch("socket.getaddrinfo", side_effect=OSError("DNS failure"))
    def test_blocks_unresolvable_host(self, mock_getaddrinfo):
        err = _validate_url_for_ssrf("http://doesnotexist.invalid/path")
        assert err is not None
        assert "resolve" in err.lower()


# ---------------------------------------------------------------------------
# http_request integration
# ---------------------------------------------------------------------------


class TestHttpRequestSsrf:
    """End-to-end: http_request must reject SSRF attempts."""

    @pytest.mark.asyncio
    async def test_http_request_blocks_localhost(self):
        result = await http_request("http://127.0.0.1:8080/admin")
        assert result["success"] is False
        assert result["error"] is not None
        assert "private" in result["error"].lower() or "reserved" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_http_request_blocks_metadata(self):
        result = await http_request(
            "http://169.254.169.254/latest/meta-data/"
        )
        assert result["success"] is False
        assert result["error"] is not None

    @pytest.mark.asyncio
    async def test_http_request_blocks_internal_network(self):
        result = await http_request("http://10.0.0.1/internal-api")
        assert result["success"] is False
        assert result["error"] is not None

    @pytest.mark.asyncio
    async def test_http_request_blocks_ipv6_loopback(self):
        result = await http_request("http://[::1]/secret")
        assert result["success"] is False
        assert result["error"] is not None
