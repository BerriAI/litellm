"""
Unit tests for MCP IP-based access control.

Tests that internal callers see all MCP servers while
external callers only see servers with available_on_public_internet=True.
"""

from unittest.mock import MagicMock, patch

from fastapi import Request

from litellm.proxy.auth.ip_address_utils import IPAddressUtils
from litellm.types.mcp_server.mcp_server_manager import MCPServer


def _make_server(server_id, available_on_public_internet=False):
    return MCPServer(
        server_id=server_id,
        name=server_id,
        server_name=server_id,
        transport="http",
        available_on_public_internet=available_on_public_internet,
    )


def _make_manager(servers):
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        MCPServerManager,
    )

    manager = MCPServerManager()
    for s in servers:
        manager.config_mcp_servers[s.server_id] = s
    return manager


class TestIsInternalIp:
    """Tests that IP classification works for private, public, and edge cases."""

    def test_private_ranges_are_internal(self):
        assert IPAddressUtils.is_internal_ip("127.0.0.1") is True
        assert IPAddressUtils.is_internal_ip("10.0.0.1") is True
        assert IPAddressUtils.is_internal_ip("172.16.0.1") is True
        assert IPAddressUtils.is_internal_ip("192.168.1.1") is True
        assert IPAddressUtils.is_internal_ip("::1") is True

    def test_public_ips_are_external(self):
        assert IPAddressUtils.is_internal_ip("8.8.8.8") is False
        assert IPAddressUtils.is_internal_ip("1.1.1.1") is False
        assert IPAddressUtils.is_internal_ip("172.32.0.1") is False

    def test_xff_chain_uses_leftmost_ip(self):
        assert IPAddressUtils.is_internal_ip("8.8.8.8, 10.0.0.1") is False
        assert IPAddressUtils.is_internal_ip("10.0.0.1, 8.8.8.8") is True

    def test_fails_closed_on_bad_input(self):
        assert IPAddressUtils.is_internal_ip("") is False
        assert IPAddressUtils.is_internal_ip(None) is False
        assert IPAddressUtils.is_internal_ip("not-an-ip") is False


class TestMCPClientIPExtraction:
    def test_fails_closed_when_xff_enabled_without_trusted_proxy_ranges(self):
        request = MagicMock(spec=Request)
        request.client = MagicMock()
        request.client.host = "203.0.113.5"
        request.headers = {"x-forwarded-for": "10.0.0.1"}

        result = IPAddressUtils.get_mcp_client_ip(
            request,
            general_settings={"use_x_forwarded_for": True},
        )

        # XFF is untrusted (no mcp_trusted_proxy_ranges) so it must be ignored,
        # and we must not trust the direct peer either: fail closed so the caller
        # is classified as external and is_internal_ip("") is False.
        assert result == ""
        assert IPAddressUtils.is_internal_ip(result) is False

    def test_private_proxy_peer_does_not_grant_internal_access(self):
        # Regression: behind an internal reverse proxy with use_x_forwarded_for
        # enabled but mcp_trusted_proxy_ranges unset, the direct peer is the
        # proxy's private IP. Returning it would mis-classify an external caller
        # as internal and expose available_on_public_internet=false servers.
        request = MagicMock(spec=Request)
        request.client = MagicMock()
        request.client.host = "10.0.0.7"
        request.headers = {"x-forwarded-for": "8.8.8.8"}

        result = IPAddressUtils.get_mcp_client_ip(
            request,
            general_settings={"use_x_forwarded_for": True},
        )

        assert result == ""
        assert IPAddressUtils.is_internal_ip(result) is False

    def test_honours_xff_from_trusted_proxy(self):
        request = MagicMock(spec=Request)
        request.client = MagicMock()
        request.client.host = "10.0.0.5"
        request.headers = {"x-forwarded-for": "192.168.1.10"}

        result = IPAddressUtils.get_mcp_client_ip(
            request,
            general_settings={
                "use_x_forwarded_for": True,
                "mcp_trusted_proxy_ranges": ["10.0.0.0/8"],
            },
        )

        assert result == "192.168.1.10"

    def test_ignores_xff_from_untrusted_direct_caller(self):
        request = MagicMock(spec=Request)
        request.client = MagicMock()
        request.client.host = "203.0.113.5"
        request.headers = {"x-forwarded-for": "10.0.0.1"}

        result = IPAddressUtils.get_mcp_client_ip(
            request,
            general_settings={
                "use_x_forwarded_for": True,
                "mcp_trusted_proxy_ranges": ["10.0.0.0/8"],
            },
        )

        assert result == "203.0.113.5"


class TestMCPServerIPFiltering:
    """Tests that external callers only see public MCP servers."""

    @patch("litellm.public_mcp_servers", [])
    @patch("litellm.proxy.proxy_server.general_settings", {})
    def test_external_ip_only_sees_public_servers(self):
        pub = _make_server("pub", available_on_public_internet=True)
        priv = _make_server("priv", available_on_public_internet=False)
        manager = _make_manager([pub, priv])

        result = manager.filter_server_ids_by_ip(["pub", "priv"], client_ip="8.8.8.8")
        assert result == ["pub"]

    @patch("litellm.public_mcp_servers", [])
    @patch("litellm.proxy.proxy_server.general_settings", {})
    def test_internal_ip_sees_all_servers(self):
        pub = _make_server("pub", available_on_public_internet=True)
        priv = _make_server("priv", available_on_public_internet=False)
        manager = _make_manager([pub, priv])

        result = manager.filter_server_ids_by_ip(
            ["pub", "priv"], client_ip="192.168.1.1"
        )
        assert result == ["pub", "priv"]

    @patch("litellm.public_mcp_servers", [])
    @patch("litellm.proxy.proxy_server.general_settings", {})
    def test_no_ip_means_no_filtering(self):
        priv = _make_server("priv", available_on_public_internet=False)
        manager = _make_manager([priv])

        result = manager.filter_server_ids_by_ip(["priv"], client_ip=None)
        assert result == ["priv"]


class TestFilterServerIdsByIpWithInfo:
    """Tests that filter_server_ids_by_ip_with_info returns accurate block counts."""

    @patch("litellm.public_mcp_servers", [])
    @patch("litellm.proxy.proxy_server.general_settings", {})
    def test_external_ip_reports_blocked_count(self):
        pub = _make_server("pub", available_on_public_internet=True)
        priv = _make_server("priv", available_on_public_internet=False)
        manager = _make_manager([pub, priv])

        allowed, blocked = manager.filter_server_ids_by_ip_with_info(
            ["pub", "priv"], client_ip="8.8.8.8"
        )
        assert allowed == ["pub"]
        assert blocked == 1

    @patch("litellm.public_mcp_servers", [])
    @patch("litellm.proxy.proxy_server.general_settings", {})
    def test_internal_ip_reports_zero_blocked(self):
        pub = _make_server("pub", available_on_public_internet=True)
        priv = _make_server("priv", available_on_public_internet=False)
        manager = _make_manager([pub, priv])

        allowed, blocked = manager.filter_server_ids_by_ip_with_info(
            ["pub", "priv"], client_ip="192.168.1.1"
        )
        assert allowed == ["pub", "priv"]
        assert blocked == 0

    @patch("litellm.public_mcp_servers", [])
    @patch("litellm.proxy.proxy_server.general_settings", {})
    def test_no_ip_returns_all_with_zero_blocked(self):
        priv = _make_server("priv", available_on_public_internet=False)
        manager = _make_manager([priv])

        allowed, blocked = manager.filter_server_ids_by_ip_with_info(
            ["priv"], client_ip=None
        )
        assert allowed == ["priv"]
        assert blocked == 0

    @patch("litellm.public_mcp_servers", [])
    @patch("litellm.proxy.proxy_server.general_settings", {})
    def test_all_private_external_ip_reports_all_blocked(self):
        priv1 = _make_server("priv1", available_on_public_internet=False)
        priv2 = _make_server("priv2", available_on_public_internet=False)
        manager = _make_manager([priv1, priv2])

        allowed, blocked = manager.filter_server_ids_by_ip_with_info(
            ["priv1", "priv2"], client_ip="1.2.3.4"
        )
        assert allowed == []
        assert blocked == 2
