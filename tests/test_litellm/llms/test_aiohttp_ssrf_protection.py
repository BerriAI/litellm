import asyncio
import ipaddress
import pytest
from unittest.mock import patch

import litellm
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

    def test_unparseable_ip_in_dns_answer_skipped(self):
        # If getaddrinfo returns a non-IP string (edge case), it should be skipped
        with patch(
            "socket.getaddrinfo",
            return_value=[(None, None, None, None, ("not-an-ip", None))],
        ):
            _assert_not_private_url("https://example.com/")  # should not raise


class TestAllowInternalIpsOptOut:
    """allow_requests_to_internal_ips=True disables SSRF protection for on-prem use."""

    def setup_method(self):
        self._original = litellm.allow_requests_to_internal_ips

    def teardown_method(self):
        litellm.allow_requests_to_internal_ips = self._original

    def test_private_ip_allowed_when_flag_set(self):
        litellm.allow_requests_to_internal_ips = True
        _assert_not_private_url("http://10.0.0.1/internal")  # must not raise

    def test_localhost_allowed_when_flag_set(self):
        litellm.allow_requests_to_internal_ips = True
        _assert_not_private_url("http://127.0.0.1:11434/api/chat")  # Ollama local

    def test_aws_metadata_allowed_when_flag_set(self):
        litellm.allow_requests_to_internal_ips = True
        _assert_not_private_url("http://169.254.169.254/latest/meta-data/")

    def test_flag_false_still_blocks(self):
        litellm.allow_requests_to_internal_ips = False
        with pytest.raises(ValueError, match="private/reserved"):
            _assert_not_private_url("http://10.0.0.1/internal")

    @pytest.mark.asyncio
    async def test_resolver_allows_private_when_flag_set(self):
        litellm.allow_requests_to_internal_ips = True
        resolver = _SSRFGuardResolver()
        mock_infos = [(2, 1, 6, "", ("10.0.0.1", 443))]

        async def run():
            loop = asyncio.get_running_loop()
            with patch.object(loop, "getaddrinfo", return_value=mock_infos):
                result = await resolver.resolve("internal.corp", 443)
            assert result[0]["host"] == "10.0.0.1"

        await run()

    @pytest.mark.asyncio
    async def test_resolver_blocks_private_when_flag_false(self):
        litellm.allow_requests_to_internal_ips = False
        resolver = _SSRFGuardResolver()
        mock_infos = [(2, 1, 6, "", ("10.0.0.1", 443))]

        async def run():
            loop = asyncio.get_running_loop()
            with patch.object(loop, "getaddrinfo", return_value=mock_infos):
                with pytest.raises(ValueError, match="private/reserved"):
                    await resolver.resolve("evil.internal", 443)

        await run()


class TestSSRFGuardOnRequestMethods:
    """Verify _assert_not_private_url is actually called in the request paths."""

    @pytest.mark.asyncio
    async def test_make_common_async_call_blocks_private_ip(self):
        from unittest.mock import AsyncMock, Mock

        from litellm.llms.custom_httpx.aiohttp_handler import BaseLLMAIOHTTPHandler

        handler = BaseLLMAIOHTTPHandler()
        mock_config = Mock()
        mock_config.max_retry_on_unprocessable_entity_error = 1
        mock_session = AsyncMock()

        with pytest.raises(ValueError, match="private/reserved"):
            await handler._make_common_async_call(
                async_client_session=mock_session,
                provider_config=mock_config,
                api_base="http://169.254.169.254/latest/meta-data/",
                headers={},
                data={},
                timeout=30,
                litellm_params={},
            )

    def test_make_common_sync_call_blocks_private_ip(self):
        from unittest.mock import Mock

        from litellm.llms.custom_httpx.aiohttp_handler import BaseLLMAIOHTTPHandler

        handler = BaseLLMAIOHTTPHandler()
        mock_config = Mock()
        mock_config.max_retry_on_unprocessable_entity_error = 1
        mock_sync_client = Mock()

        with pytest.raises(ValueError, match="private/reserved"):
            handler._make_common_sync_call(
                sync_httpx_client=mock_sync_client,
                provider_config=mock_config,
                api_base="http://10.0.0.1/internal",
                headers={},
                data={},
                timeout=30,
                litellm_params={},
            )


class TestSSRFGuardResolver:
    """Tests for the async resolver that eliminates TOCTOU DNS rebinding."""

    def _run(self, coro):
        return asyncio.run(coro)

    def test_private_ip_blocked_at_connection_time(self):
        resolver = _SSRFGuardResolver()
        mock_infos = [
            (2, 1, 6, "", ("10.0.0.1", 443)),
        ]

        async def run():
            loop = asyncio.get_running_loop()
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
            loop = asyncio.get_running_loop()
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
            loop = asyncio.get_running_loop()
            with patch.object(loop, "getaddrinfo", return_value=mock_infos):
                with pytest.raises(ValueError, match="private/reserved"):
                    await resolver.resolve("rebinding.example.com", 443)

        self._run(run())

    def test_resolver_dns_failure_propagates(self):
        import socket as _socket

        resolver = _SSRFGuardResolver()

        async def run():
            loop = asyncio.get_running_loop()
            with patch.object(
                loop, "getaddrinfo", side_effect=_socket.gaierror("DNS fail")
            ):
                with pytest.raises(_socket.gaierror):
                    await resolver.resolve("nonexistent.invalid", 443)

        self._run(run())

    def test_resolver_unparseable_ip_skipped(self):
        resolver = _SSRFGuardResolver()
        mock_infos = [
            (2, 1, 6, "", ("not-an-ip", 443)),
            (2, 1, 6, "", ("104.18.7.8", 443)),
        ]

        async def run():
            loop = asyncio.get_running_loop()
            with patch.object(loop, "getaddrinfo", return_value=mock_infos):
                result = await resolver.resolve("example.com", 443)
            assert any(r["host"] == "104.18.7.8" for r in result)

        self._run(run())

    @pytest.mark.asyncio
    async def test_resolver_close_is_noop(self):
        resolver = _SSRFGuardResolver()
        await resolver.close()  # Should not raise
