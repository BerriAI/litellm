import asyncio
import ipaddress
import pytest
from unittest.mock import patch

from litellm.llms.custom_httpx.aiohttp_handler import (
    _SSRFGuardResolver,
    _assert_not_private_url,
    _is_blocked_address,
)


class TestBlockedAddress:
    def test_ipv4_mapped_ipv6_private_blocked(self):
        addr = ipaddress.ip_address("::ffff:10.0.0.1")
        assert _is_blocked_address(addr)

    def test_ipv4_mapped_ipv6_public_allowed(self):
        addr = ipaddress.ip_address("::ffff:104.18.7.8")
        assert not _is_blocked_address(addr)

    def test_ipv6_link_local_blocked(self):
        assert _is_blocked_address(ipaddress.ip_address("fe80::1"))

    def test_ipv6_ula_blocked(self):
        assert _is_blocked_address(ipaddress.ip_address("fc00::1"))

    def test_0_0_0_0_blocked(self):
        assert _is_blocked_address(ipaddress.ip_address("0.0.0.0"))


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


class TestSSRFGuardResolver:
    """Tests for the async resolver that eliminates TOCTOU DNS rebinding."""

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_private_ip_blocked_at_connection_time(self):
        resolver = _SSRFGuardResolver()
        mock_infos = [
            (2, 1, 6, "", ("10.0.0.1", 443)),
        ]
        with patch("asyncio.AbstractEventLoop.getaddrinfo", return_value=mock_infos):

            async def run():
                loop = asyncio.get_event_loop()
                with patch.object(loop, "getaddrinfo", return_value=mock_infos):
                    with pytest.raises(ValueError, match="private/reserved"):
                        await resolver.resolve("evil.internal", 443)

            self._run(run())

    def test_public_ip_passes_resolver(self):
        resolver = _SSRFGuardResolver()
        mock_infos = [
            (2, 1, 6, "", ("104.18.7.8", 443)),
        ]

        async def run():
            loop = asyncio.get_event_loop()
            with patch.object(loop, "getaddrinfo", return_value=mock_infos):
                result = await resolver.resolve("api.openai.com", 443)
            assert result[0]["host"] == "104.18.7.8"

        self._run(run())

    def test_all_answers_checked_by_resolver(self):
        resolver = _SSRFGuardResolver()
        mock_infos = [
            (2, 1, 6, "", ("104.18.7.8", 443)),
            (2, 1, 6, "", ("169.254.169.254", 443)),
        ]

        async def run():
            loop = asyncio.get_event_loop()
            with patch.object(loop, "getaddrinfo", return_value=mock_infos):
                with pytest.raises(ValueError, match="private/reserved"):
                    await resolver.resolve("rebinding.example.com", 443)

        self._run(run())
