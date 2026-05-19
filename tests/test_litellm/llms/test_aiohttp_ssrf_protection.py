import ipaddress
import pytest
from unittest.mock import patch

from litellm.llms.custom_httpx.aiohttp_handler import (
    _assert_not_private_url,
    _is_blocked_address,
)


class TestBlockedAddress:
    def test_ipv4_mapped_ipv6_private_blocked(self):
        # ::ffff:10.0.0.1 is IPv4-mapped IPv6 for 10.0.0.1 — must be blocked
        addr = ipaddress.ip_address("::ffff:10.0.0.1")
        assert _is_blocked_address(addr)

    def test_ipv4_mapped_ipv6_public_allowed(self):
        addr = ipaddress.ip_address("::ffff:104.18.7.8")
        assert not _is_blocked_address(addr)

    def test_ipv6_link_local_blocked(self):
        addr = ipaddress.ip_address("fe80::1")
        assert _is_blocked_address(addr)

    def test_ipv6_ula_blocked(self):
        addr = ipaddress.ip_address("fc00::1")
        assert _is_blocked_address(addr)

    def test_0_0_0_0_blocked(self):
        addr = ipaddress.ip_address("0.0.0.0")
        assert _is_blocked_address(addr)


class TestAiohttpSSRFProtection:
    def test_aws_metadata_endpoint_blocked(self):
        with pytest.raises(ValueError, match="private/reserved"):
            _assert_not_private_url("http://169.254.169.254/latest/meta-data/")

    def test_localhost_blocked(self):
        with pytest.raises(ValueError, match="private/reserved"):
            _assert_not_private_url("http://127.0.0.1/admin")

    def test_private_10_network_blocked(self):
        with pytest.raises(ValueError, match="private/reserved"):
            _assert_not_private_url("http://10.0.0.1/internal")

    def test_private_172_16_network_blocked(self):
        with pytest.raises(ValueError, match="private/reserved"):
            _assert_not_private_url("http://172.16.0.1/internal")

    def test_private_192_168_network_blocked(self):
        with pytest.raises(ValueError, match="private/reserved"):
            _assert_not_private_url("http://192.168.1.1/internal")

    def test_cgnat_blocked(self):
        with pytest.raises(ValueError, match="private/reserved"):
            _assert_not_private_url("http://100.64.0.1/internal")

    def test_all_dns_answers_checked(self):
        # Domain returns public IP first, private IP second — must still block
        with patch(
            "socket.getaddrinfo",
            return_value=[
                (None, None, None, None, ("104.18.7.8", None)),
                (None, None, None, None, ("10.0.0.1", None)),
            ],
        ):
            with pytest.raises(ValueError, match="private/reserved"):
                _assert_not_private_url("https://evil-rebinding.example.com/")

    def test_public_ip_allowed(self):
        with patch(
            "socket.getaddrinfo",
            return_value=[(None, None, None, None, ("104.18.7.8", None))],
        ):
            _assert_not_private_url("https://api.openai.com/v1/chat/completions")

    def test_empty_hostname_allowed(self):
        _assert_not_private_url("not-a-url")

    def test_dns_failure_does_not_block(self):
        import socket as _socket

        with patch("socket.getaddrinfo", side_effect=_socket.gaierror("DNS fail")):
            _assert_not_private_url("https://nonexistent.invalid/path")
