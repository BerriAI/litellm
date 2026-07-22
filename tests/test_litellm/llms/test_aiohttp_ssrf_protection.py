import asyncio
import ipaddress
import unittest.mock
import pytest
from unittest.mock import Mock, patch

import httpx

import litellm
from litellm.llms.custom_httpx.aiohttp_handler import (
    _SSRFGuardResolver,
    _SSRFGuardTransport,
    _assert_not_private_ip_literal,
    _assert_not_private_url,
    _get_ssrf_safe_sync_client,
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

    def test_ipv6_unspecified_blocked(self):
        assert _is_blocked_address(ipaddress.ip_address("::"))

    def test_0_0_0_0_blocked(self):
        assert _is_blocked_address(ipaddress.ip_address("0.0.0.0"))

    def test_benchmarking_range_blocked(self):
        assert _is_blocked_address(ipaddress.ip_address("198.18.0.1"))

    def test_class_e_reserved_blocked(self):
        assert _is_blocked_address(ipaddress.ip_address("240.0.0.1"))

    def test_documentation_range_blocked(self):
        assert _is_blocked_address(ipaddress.ip_address("192.0.2.1"))


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


class TestAssertNotPrivateIpLiteral:
    """Tests for the IP-literal bypass guard on the async path."""

    def test_aws_metadata_ip_literal_blocked(self):
        with pytest.raises(ValueError, match="private/reserved"):
            _assert_not_private_ip_literal("http://169.254.169.254/latest/meta-data/")

    def test_localhost_ip_literal_blocked(self):
        with pytest.raises(ValueError, match="private/reserved"):
            _assert_not_private_ip_literal("http://127.0.0.1/admin")

    def test_private_10_network_ip_literal_blocked(self):
        with pytest.raises(ValueError, match="private/reserved"):
            _assert_not_private_ip_literal("http://10.0.0.1/internal")

    def test_hostname_not_blocked(self):
        # Hostnames are not IP literals — resolver handles them
        _assert_not_private_ip_literal("https://api.openai.com/v1/chat/completions")

    def test_public_ip_literal_allowed(self):
        _assert_not_private_ip_literal("https://104.18.7.8/v1/chat/completions")

    def test_ipv6_loopback_literal_blocked(self):
        with pytest.raises(ValueError, match="private/reserved"):
            _assert_not_private_ip_literal("http://[::1]/admin")

    def test_ipv4_mapped_ipv6_private_literal_blocked(self):
        with pytest.raises(ValueError, match="private/reserved"):
            _assert_not_private_ip_literal("http://[::ffff:10.0.0.1]/internal")

    def test_no_hostname_is_allowed(self):
        # URLs with no parseable hostname (e.g. opaque URIs) pass through
        _assert_not_private_ip_literal("not-a-url")

    def test_flag_disables_check(self):
        litellm.allow_requests_to_internal_ips = True
        try:
            _assert_not_private_ip_literal("http://169.254.169.254/latest/meta-data/")
        finally:
            litellm.allow_requests_to_internal_ips = False


class TestSSRFGuardOnRequestMethods:
    """Verify SSRF protection is enforced on both sync and async request paths."""

    @pytest.mark.asyncio
    async def test_make_common_async_call_blocks_ip_literal_without_dns(self):
        """Async path blocks IP-literal api_base at preflight (no DNS I/O needed).
        This closes the TCPConnector bypass where aiohttp skips the custom resolver
        when the host is already an IP address."""
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

    @pytest.mark.asyncio
    async def test_make_common_async_call_hostname_defers_to_resolver(self):
        """Async path does NOT do a blocking DNS preflight for hostname-based URLs.
        SSRF protection for those is handled by _SSRFGuardResolver at TCP-connect time.
        """
        from unittest.mock import AsyncMock, Mock

        from litellm.llms.custom_httpx.aiohttp_handler import BaseLLMAIOHTTPHandler

        handler = BaseLLMAIOHTTPHandler()
        mock_config = Mock()
        mock_config.max_retry_on_unprocessable_entity_error = 1
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.ok = True
        mock_session.post.return_value.__aenter__ = AsyncMock(
            return_value=mock_response
        )
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=False)

        # Hostname-based private URL must NOT raise at preflight — resolver handles it
        try:
            await handler._make_common_async_call(
                async_client_session=mock_session,
                provider_config=mock_config,
                api_base="http://internal.corp/api",
                headers={},
                data={},
                timeout=30,
                litellm_params={},
            )
        except ValueError as e:
            pytest.fail(f"Async path must not do a blocking DNS preflight: {e}")
        except Exception:
            pass  # Other errors (e.g. from mock) are fine

    def test_make_common_sync_call_blocks_private_ip(self):
        """Sync path _assert_not_private_url preflight still blocks private IPs
        for externally-supplied clients (defence-in-depth).  Internally created
        clients use _SSRFGuardTransport for connect-time validation."""
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

    def test_make_common_sync_call_succeeds_for_public_url(self):
        """Sync code path still works for legitimate (public-IP) API bases after
        introducing _SSRFGuardTransport.  Uses _get_ssrf_safe_sync_client() as the
        internally-created client — the same path taken by BaseLLMAIOHTTPHandler
        when no explicit client is supplied.

        Verifies both that:
        - the transport's validation passes (public IP, no block), and
        - the request is forwarded to the pinned IP (no second DNS lookup).
        """
        from litellm.llms.custom_httpx.aiohttp_handler import (
            BaseLLMAIOHTTPHandler,
            _get_ssrf_safe_sync_client,
        )

        handler = BaseLLMAIOHTTPHandler()
        mock_config = Mock()
        mock_config.max_retry_on_unprocessable_entity_error = 1
        mock_config.should_retry_llm_api_inside_llm_translation_on_http_error = Mock(
            return_value=False
        )

        # Simulate DNS resolving to a public IP (legitimate provider)
        public_dns = [(2, 1, 6, "", ("104.18.7.8", 443))]

        # httpx.Client._send_single_request asserts isinstance(response.stream,
        # SyncByteStream) after calling transport.handle_request, so we must
        # return a real httpx.Response (not a plain Mock).
        fake_response = httpx.Response(
            status_code=200,
            content=b'{"id": "chatcmpl-test", "choices": []}',
            headers={"content-type": "application/json"},
        )

        ssrf_safe_client = _get_ssrf_safe_sync_client()
        forwarded_hosts: list = []

        def spy_handle(self_transport, request):
            forwarded_hosts.append(request.url.host)
            return fake_response

        with patch("socket.getaddrinfo", return_value=public_dns):
            with patch.object(httpx.HTTPTransport, "handle_request", spy_handle):
                result = handler._make_common_sync_call(
                    sync_httpx_client=ssrf_safe_client,
                    provider_config=mock_config,
                    api_base="https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": "Bearer sk-test"},
                    data={"model": "gpt-4", "messages": []},
                    timeout=30,
                    litellm_params={},
                )

        assert result is fake_response
        # Transport received the pinned IP, proving no second DNS resolution occurs
        assert forwarded_hosts == ["104.18.7.8"]


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


class TestSSRFGuardTransport:
    """_SSRFGuardTransport eliminates DNS-rebinding TOCTOU on the sync httpx path.

    It resolves the hostname, validates every returned IP, then rewrites the
    URL to an IP literal so httpcore connects directly — no second DNS lookup.
    """

    def test_transport_blocks_hostname_resolving_to_private_ip(self):
        """Raises when DNS returns a private IP for a hostname-based URL."""
        transport = _SSRFGuardTransport()
        mock_infos = [(2, 1, 6, "", ("10.0.0.1", 443))]
        request = httpx.Request("POST", "https://evil.internal/v1/chat")

        with patch("socket.getaddrinfo", return_value=mock_infos):
            with pytest.raises(ValueError, match="private/reserved"):
                transport.handle_request(request)

    def test_transport_blocks_aws_metadata_endpoint(self):
        transport = _SSRFGuardTransport()
        mock_infos = [(2, 1, 6, "", ("169.254.169.254", 80))]
        request = httpx.Request("GET", "http://metadata.example.com/latest/meta-data/")

        with patch("socket.getaddrinfo", return_value=mock_infos):
            with pytest.raises(ValueError, match="private/reserved"):
                transport.handle_request(request)

    def test_transport_blocks_all_dns_answers(self):
        """Raises when ANY answer resolves to a private IP (A-record rotation)."""
        transport = _SSRFGuardTransport()
        mock_infos = [
            (2, 1, 6, "", ("104.18.7.8", 443)),  # public — passes
            (2, 1, 6, "", ("169.254.169.254", 443)),  # private — must still block
        ]
        request = httpx.Request("POST", "https://rebinding.example.com/v1")

        with patch("socket.getaddrinfo", return_value=mock_infos):
            with pytest.raises(ValueError, match="private/reserved"):
                transport.handle_request(request)

    def test_transport_pins_ip_and_preserves_hostname(self):
        """Forwards request with URL rewritten to pinned IP; original hostname
        in Host header and sni_hostname extension for TLS."""
        transport = _SSRFGuardTransport()
        mock_infos = [(2, 1, 6, "", ("104.18.7.8", 443))]
        request = httpx.Request(
            "POST",
            "https://api.openai.com/v1/chat/completions",
            content=b'{"model":"gpt-4"}',
        )
        mock_response = Mock(spec=httpx.Response)

        with patch("socket.getaddrinfo", return_value=mock_infos):
            with patch.object(
                httpx.HTTPTransport, "handle_request", return_value=mock_response
            ) as mock_super:
                result = transport.handle_request(request)

        assert result is mock_response
        forwarded: httpx.Request = mock_super.call_args[0][0]
        # URL host is the pinned IP — httpcore skips DNS, TOCTOU window closed
        assert forwarded.url.host == "104.18.7.8"
        # Original hostname preserved for virtual hosting
        assert forwarded.headers.get("host") == "api.openai.com"
        # Original hostname preserved for TLS SNI so cert validation passes
        assert forwarded.extensions.get("sni_hostname") == "api.openai.com"

    def test_transport_dns_failure_raises(self):
        """DNS failure raises gaierror (fail-closed) to prevent bypass via transient failure."""
        import socket as _socket

        transport = _SSRFGuardTransport()
        request = httpx.Request("POST", "https://nonexistent.invalid/v1")

        with patch("socket.getaddrinfo", side_effect=_socket.gaierror("DNS fail")):
            with pytest.raises(_socket.gaierror):
                transport.handle_request(request)

    def test_transport_public_ip_literal_allowed(self):
        """Public IP-literal URLs are allowed and bypass DNS (no resolution needed)."""
        transport = _SSRFGuardTransport()
        request = httpx.Request("POST", "https://104.18.7.8/v1/chat")
        mock_response = Mock(spec=httpx.Response)

        # getaddrinfo must NOT be called for an IP-literal URL
        with patch("socket.getaddrinfo") as mock_dns:
            with patch.object(
                httpx.HTTPTransport, "handle_request", return_value=mock_response
            ):
                result = transport.handle_request(request)

        mock_dns.assert_not_called()
        assert result is mock_response

    def test_transport_blocks_private_ip_literal(self):
        """Private IP-literal in URL is blocked directly — no DNS lookup required."""
        transport = _SSRFGuardTransport()
        request = httpx.Request("POST", "http://10.0.0.1/api")

        with patch("socket.getaddrinfo") as mock_dns:
            with pytest.raises(ValueError, match="private/reserved"):
                transport.handle_request(request)

        mock_dns.assert_not_called()

    def test_transport_blocks_aws_metadata_ip_literal(self):
        """169.254.169.254 as an IP literal is blocked without DNS lookup."""
        transport = _SSRFGuardTransport()
        request = httpx.Request("GET", "http://169.254.169.254/latest/meta-data/")

        with patch("socket.getaddrinfo") as mock_dns:
            with pytest.raises(ValueError, match="private/reserved"):
                transport.handle_request(request)

        mock_dns.assert_not_called()

    def test_transport_blocks_ipv6_loopback_literal(self):
        """IPv6 loopback ::1 as an IP literal is blocked without DNS lookup."""
        transport = _SSRFGuardTransport()
        request = httpx.Request("GET", "http://[::1]/admin")

        with patch("socket.getaddrinfo") as mock_dns:
            with pytest.raises(ValueError, match="private/reserved"):
                transport.handle_request(request)

        mock_dns.assert_not_called()

    def test_transport_ip_literal_bypassed_when_flag_set(self):
        """allow_requests_to_internal_ips=True skips IP-literal validation too."""
        transport = _SSRFGuardTransport()
        request = httpx.Request("POST", "http://10.0.0.1/internal")
        mock_response = Mock(spec=httpx.Response)

        litellm.allow_requests_to_internal_ips = True
        try:
            with patch.object(
                httpx.HTTPTransport, "handle_request", return_value=mock_response
            ):
                result = transport.handle_request(request)
            assert result is mock_response
        finally:
            litellm.allow_requests_to_internal_ips = False

    def test_transport_bypassed_when_flag_set(self):
        """allow_requests_to_internal_ips=True disables all transport-level checks."""
        transport = _SSRFGuardTransport()
        mock_infos = [(2, 1, 6, "", ("10.0.0.1", 443))]
        request = httpx.Request("POST", "https://internal.corp/v1/chat")
        mock_response = Mock(spec=httpx.Response)

        litellm.allow_requests_to_internal_ips = True
        try:
            with patch("socket.getaddrinfo", return_value=mock_infos):
                with patch.object(
                    httpx.HTTPTransport, "handle_request", return_value=mock_response
                ):
                    result = transport.handle_request(request)
            assert result is mock_response
        finally:
            litellm.allow_requests_to_internal_ips = False

    def test_get_ssrf_safe_sync_client_uses_guard_transport(self):
        """_get_ssrf_safe_sync_client() produces an HTTPHandler backed by
        _SSRFGuardTransport so the sync code path gets connect-time validation."""
        from litellm.llms.custom_httpx.http_handler import HTTPHandler

        handler = _get_ssrf_safe_sync_client()
        assert isinstance(handler, HTTPHandler)
        # The underlying httpx.Client must use _SSRFGuardTransport
        assert isinstance(handler.client._transport, _SSRFGuardTransport)

    def test_ssrf_safe_client_blocks_private_ip_at_connect_time(self):
        """End-to-end: _get_ssrf_safe_sync_client raises on a private-IP hostname
        even when called via HTTPHandler.post(), proving the sync code path works."""
        handler = _get_ssrf_safe_sync_client()
        mock_infos = [(2, 1, 6, "", ("10.0.0.1", 443))]

        with patch("socket.getaddrinfo", return_value=mock_infos):
            with pytest.raises(ValueError, match="private/reserved"):
                handler.post("https://evil.internal/v1/chat", json={"test": True})

    def test_ssrf_safe_client_blocks_private_ip_literal(self):
        """End-to-end: _get_ssrf_safe_sync_client raises on a private IP-literal URL
        even without DNS, proving the sync path catches direct IP addresses."""
        handler = _get_ssrf_safe_sync_client()

        # No DNS mock needed — the transport must block before any resolution
        with patch("socket.getaddrinfo") as mock_dns:
            with pytest.raises(ValueError, match="private/reserved"):
                handler.post("http://10.0.0.1/api", json={"test": True})

        mock_dns.assert_not_called()

    def test_transport_skips_unparseable_ip_in_dns_answers(self):
        """Unparseable entries in getaddrinfo answers are skipped; valid entries
        still checked — mirrors _SSRFGuardResolver behaviour on the sync path."""
        transport = _SSRFGuardTransport()
        # One entry is unparseable; the other is a safe public IP.
        mock_infos = [
            (2, 1, 6, "", ("not-an-ip", 443)),
            (2, 1, 6, "", ("104.18.7.8", 443)),
        ]
        request = httpx.Request("POST", "https://api.example.com/v1/chat")
        mock_response = Mock(spec=httpx.Response)

        with patch("socket.getaddrinfo", return_value=mock_infos):
            with patch.object(
                httpx.HTTPTransport, "handle_request", return_value=mock_response
            ):
                result = transport.handle_request(request)

        assert result is mock_response

    def test_transport_blocks_redirect_to_private_ip_literal(self):
        """Redirect targets with private IP literals are blocked by the transport.
        httpx calls handle_request for each redirect hop, so the guard applies."""
        transport = _SSRFGuardTransport()
        # Simulate httpx presenting the redirect target URL to the transport.
        redirect_request = httpx.Request(
            "GET", "http://169.254.169.254/latest/meta-data/"
        )

        with patch("socket.getaddrinfo") as mock_dns:
            with pytest.raises(ValueError, match="private/reserved"):
                transport.handle_request(redirect_request)

        mock_dns.assert_not_called()

    def test_transport_blocks_redirect_to_private_hostname(self):
        """Redirect targets resolving to private IPs are blocked by the transport."""
        transport = _SSRFGuardTransport()
        mock_infos = [(2, 1, 6, "", ("10.0.0.1", 80))]
        redirect_request = httpx.Request("GET", "http://internal.corp/sensitive")

        with patch("socket.getaddrinfo", return_value=mock_infos):
            with pytest.raises(ValueError, match="private/reserved"):
                transport.handle_request(redirect_request)


class TestSSRFSafeClientCachingAndSSL:
    """_get_ssrf_safe_sync_client must reuse a cached client and forward SSL config."""

    def setup_method(self):
        # Clear the cache before each test so tests are independent
        cache = getattr(litellm, "in_memory_llm_clients_cache", None)
        if cache is not None:
            try:
                cache.delete_cache("ssrf_safe_httpx_client")
            except Exception:
                pass

    def test_creates_cache_when_absent(self):
        """If in_memory_llm_clients_cache is None, a new LLMClientCache is created."""
        original = getattr(litellm, "in_memory_llm_clients_cache", None)
        try:
            litellm.in_memory_llm_clients_cache = None
            client = _get_ssrf_safe_sync_client()
            assert client is not None
            assert litellm.in_memory_llm_clients_cache is not None
        finally:
            litellm.in_memory_llm_clients_cache = original

    def test_client_is_cached_across_calls(self):
        """Two consecutive calls return the same HTTPHandler instance."""
        client1 = _get_ssrf_safe_sync_client()
        client2 = _get_ssrf_safe_sync_client()
        assert client1 is client2

    def test_ssl_config_is_forwarded_to_transport(self):
        """get_ssl_configuration() return value is passed as verify= to
        _SSRFGuardTransport so custom CA bundles and ssl_context objects are
        honoured by the SSRF-guarded sync client."""
        from litellm.llms.custom_httpx.aiohttp_handler import (
            _SSRF_SAFE_CLIENT_CACHE_KEY,
        )

        cache = getattr(litellm, "in_memory_llm_clients_cache", None)
        if cache:
            try:
                cache.delete_cache(_SSRF_SAFE_CLIENT_CACHE_KEY)
            except Exception:
                pass

        sentinel_ssl = object()  # unique sentinel — proves it was forwarded
        with patch(
            "litellm.llms.custom_httpx.aiohttp_handler.get_ssl_configuration",
            return_value=sentinel_ssl,
        ):
            with patch.object(
                _SSRFGuardTransport, "__init__", wraps=_SSRFGuardTransport.__init__
            ) as spy:
                try:
                    _get_ssrf_safe_sync_client()
                except Exception:
                    pass  # construction may fail with a non-SSL sentinel
                if spy.call_args:
                    _, kwargs = spy.call_args
                    assert kwargs.get("verify") is sentinel_ssl

    def test_ssl_certificate_is_forwarded_to_transport(self):
        """SSL_CERTIFICATE env-var (used for mTLS / custom CA) is forwarded to
        _SSRFGuardTransport as cert= so it is not silently dropped."""
        from litellm.llms.custom_httpx.aiohttp_handler import (
            _SSRF_SAFE_CLIENT_CACHE_KEY,
        )

        cache = getattr(litellm, "in_memory_llm_clients_cache", None)
        if cache:
            try:
                cache.delete_cache(_SSRF_SAFE_CLIENT_CACHE_KEY)
            except Exception:
                pass

        with patch(
            "litellm.llms.custom_httpx.aiohttp_handler.os.getenv",
            side_effect=lambda k, d=None: (
                "/path/to/client.pem" if k == "SSL_CERTIFICATE" else d
            ),
        ):
            with patch.object(
                _SSRFGuardTransport, "__init__", wraps=_SSRFGuardTransport.__init__
            ) as spy:
                try:
                    _get_ssrf_safe_sync_client()
                except Exception:
                    pass
                if spy.call_args:
                    _, kwargs = spy.call_args
                    assert kwargs.get("cert") == "/path/to/client.pem"


class TestSSRFTraceConfig:
    """Tests for the on_request_start trace hook that blocks IP-literal aiohttp
    redirect targets — the gap _SSRFGuardResolver cannot cover because aiohttp
    skips DNS resolution when the Location header is already an IP literal."""

    def _run(self, coro):
        return asyncio.run(coro)

    def _mock_params(self, url: str):
        params = unittest.mock.Mock()
        params.url = url
        return params

    def test_trace_callback_blocks_aws_metadata_ip_literal(self):
        from litellm.llms.custom_httpx.aiohttp_handler import _on_ssrf_request_start

        async def run():
            with pytest.raises(ValueError, match="private/reserved"):
                await _on_ssrf_request_start(
                    unittest.mock.Mock(),
                    unittest.mock.Mock(),
                    self._mock_params("http://169.254.169.254/latest/meta-data/"),
                )

        self._run(run())

    def test_trace_callback_blocks_private_rfc1918_ip_literal(self):
        from litellm.llms.custom_httpx.aiohttp_handler import _on_ssrf_request_start

        async def run():
            with pytest.raises(ValueError, match="private/reserved"):
                await _on_ssrf_request_start(
                    unittest.mock.Mock(),
                    unittest.mock.Mock(),
                    self._mock_params("http://10.0.0.1/internal"),
                )

        self._run(run())

    def test_trace_callback_allows_public_ip_literal(self):
        from litellm.llms.custom_httpx.aiohttp_handler import _on_ssrf_request_start

        async def run():
            # Public IP — must not raise
            await _on_ssrf_request_start(
                unittest.mock.Mock(),
                unittest.mock.Mock(),
                self._mock_params("https://104.18.7.8/v1/chat/completions"),
            )

        self._run(run())

    def test_trace_callback_passes_hostname_urls_through(self):
        """Hostname URLs are not IP literals — resolver handles them at DNS time."""
        from litellm.llms.custom_httpx.aiohttp_handler import _on_ssrf_request_start

        async def run():
            await _on_ssrf_request_start(
                unittest.mock.Mock(),
                unittest.mock.Mock(),
                self._mock_params("https://api.openai.com/v1/chat/completions"),
            )

        self._run(run())

    def test_trace_callback_bypassed_when_flag_set(self):
        from litellm.llms.custom_httpx.aiohttp_handler import _on_ssrf_request_start

        litellm.allow_requests_to_internal_ips = True
        try:

            async def run():
                # Private IP literal must NOT raise when flag is set
                await _on_ssrf_request_start(
                    unittest.mock.Mock(),
                    unittest.mock.Mock(),
                    self._mock_params("http://169.254.169.254/latest/meta-data/"),
                )

            self._run(run())
        finally:
            litellm.allow_requests_to_internal_ips = False
