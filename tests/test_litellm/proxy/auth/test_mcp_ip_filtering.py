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
    def test_public_direct_peer_spoofing_xff_stays_external(self):
        # A caller connecting directly to the gateway (public direct peer) with
        # use_x_forwarded_for on but no mcp_trusted_proxy_ranges cannot spoof an
        # internal X-Forwarded-For to reach available_on_public_internet=false
        # servers: the forged XFF is ignored and the real public peer is used.
        request = MagicMock(spec=Request)
        request.client = MagicMock()
        request.client.host = "203.0.113.5"
        request.headers = {"x-forwarded-for": "10.0.0.1"}

        result = IPAddressUtils.get_mcp_client_ip(
            request,
            general_settings={"use_x_forwarded_for": True},
        )

        assert result == "203.0.113.5"
        assert IPAddressUtils.is_internal_ip(result) is False

    def test_external_client_behind_private_proxy_stays_external(self):
        # Behind an internal reverse proxy (private direct peer) with XFF on but
        # no mcp_trusted_proxy_ranges, an external client (public XFF) is still
        # classified external by its forwarded address, so it cannot reach
        # internal-only servers.
        request = MagicMock(spec=Request)
        request.client = MagicMock()
        request.client.host = "10.0.0.7"
        request.headers = {"x-forwarded-for": "8.8.8.8"}

        result = IPAddressUtils.get_mcp_client_ip(
            request,
            general_settings={"use_x_forwarded_for": True},
        )

        assert result == "8.8.8.8"
        assert IPAddressUtils.is_internal_ip(result) is False

    def test_internal_client_behind_private_proxy_without_trusted_ranges_is_internal(
        self,
    ):
        # LIT-3964 regression: an internal client reaching the gateway through a
        # private reverse proxy (private direct peer, internal XFF) with
        # use_x_forwarded_for on but no mcp_trusted_proxy_ranges must be
        # classified internal. Returning "" here (the regressed behaviour) made
        # is_internal_ip("") False, so every internal-only MCP server 404'd.
        request = MagicMock(spec=Request)
        request.client = MagicMock()
        request.client.host = "10.0.0.1"
        request.headers = {"x-forwarded-for": "10.0.0.9"}

        result = IPAddressUtils.get_mcp_client_ip(
            request,
            general_settings={"use_x_forwarded_for": True},
        )

        assert result == "10.0.0.9"
        assert IPAddressUtils.is_internal_ip(result) is True

    def test_unknown_direct_peer_spoofing_xff_stays_external(self):
        # When the ASGI transport does not expose request.client (Unix-socket
        # transports, some test clients) the direct peer is unknown. With XFF on
        # and no mcp_trusted_proxy_ranges, a forged internal X-Forwarded-For must
        # not be trusted. The result must be a non-None external value: returning
        # None would mean "no filtering" downstream and leak internal access.
        request = MagicMock(spec=Request)
        request.client = None
        request.headers = {"x-forwarded-for": "10.0.0.9"}

        result = IPAddressUtils.get_mcp_client_ip(
            request,
            general_settings={"use_x_forwarded_for": True},
        )

        assert result is not None
        assert IPAddressUtils.is_internal_ip(result) is False

    def test_xff_honoured_when_peer_internal_under_custom_ranges(self):
        # The private-proxy carve-out must use the same internal-range
        # definition as the access-control gate (mcp_internal_ip_ranges), not
        # only the RFC1918 defaults. A reverse proxy internal under a custom
        # CIDR (100.64.0.0/10, outside the defaults) is a private proxy, so its
        # forwarded client IP must be honoured. Classifying it by defaults would
        # drop the XFF and return the proxy's own custom-internal IP, letting a
        # public forwarded client inherit internal access.
        request = MagicMock(spec=Request)
        request.client = MagicMock()
        request.client.host = "100.64.0.5"
        request.headers = {"x-forwarded-for": "8.8.8.8"}

        result = IPAddressUtils.get_mcp_client_ip(
            request,
            general_settings={
                "use_x_forwarded_for": True,
                "mcp_internal_ip_ranges": ["100.64.0.0/10"],
            },
        )

        assert result == "8.8.8.8"

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

    @patch(
        "litellm.proxy.proxy_server.general_settings",
        {"use_x_forwarded_for": True},
    )
    @patch("litellm.public_mcp_servers", [])
    def test_internal_client_behind_proxy_can_reach_internal_only_server(self):
        # LIT-3964 end-to-end: the rest/oauth paths extract the client IP via
        # get_mcp_client_ip and feed it straight into filter_server_ids_by_ip.
        # With use_x_forwarded_for on, no mcp_trusted_proxy_ranges, a private
        # proxy peer and an internal forwarded client, the internal-only server
        # must survive the filter instead of 404ing.
        priv = _make_server("priv", available_on_public_internet=False)
        manager = _make_manager([priv])

        request = MagicMock(spec=Request)
        request.client = MagicMock()
        request.client.host = "10.0.0.1"
        request.headers = {"x-forwarded-for": "10.0.0.9"}
        client_ip = IPAddressUtils.get_mcp_client_ip(
            request,
            general_settings={"use_x_forwarded_for": True},
        )

        result = manager.filter_server_ids_by_ip(["priv"], client_ip=client_ip)
        assert result == ["priv"]

    @patch(
        "litellm.proxy.proxy_server.general_settings",
        {"use_x_forwarded_for": True},
    )
    @patch("litellm.public_mcp_servers", [])
    def test_unknown_peer_spoofing_xff_cannot_reach_internal_only_server(self):
        # Security regression: if the transport doesn't expose request.client, a
        # forged internal X-Forwarded-For must not unlock internal-only servers.
        # get_mcp_client_ip classifies external and the filter drops the private
        # server. A None result (the fail-open bug) would leave it accessible.
        priv = _make_server("priv", available_on_public_internet=False)
        manager = _make_manager([priv])

        request = MagicMock(spec=Request)
        request.client = None
        request.headers = {"x-forwarded-for": "10.0.0.9"}
        client_ip = IPAddressUtils.get_mcp_client_ip(
            request,
            general_settings={"use_x_forwarded_for": True},
        )

        result = manager.filter_server_ids_by_ip(["priv"], client_ip=client_ip)
        assert result == []

    @patch(
        "litellm.proxy.proxy_server.general_settings",
        {
            "use_x_forwarded_for": True,
            "mcp_internal_ip_ranges": ["100.64.0.0/10"],
        },
    )
    @patch("litellm.public_mcp_servers", [])
    def test_public_client_via_custom_internal_proxy_cannot_reach_internal_only_server(
        self,
    ):
        # Security: with custom mcp_internal_ip_ranges, a public client forwarded
        # by a proxy that is internal only under those custom ranges must still
        # be blocked from internal-only servers. get_mcp_client_ip recognises the
        # proxy as private and honours the public XFF, and the filter then drops
        # the internal-only server. Classifying the peer by RFC1918 defaults
        # would return the proxy's own IP and leak the server to the caller.
        priv = _make_server("priv", available_on_public_internet=False)
        manager = _make_manager([priv])

        request = MagicMock(spec=Request)
        request.client = MagicMock()
        request.client.host = "100.64.0.5"
        request.headers = {"x-forwarded-for": "8.8.8.8"}
        client_ip = IPAddressUtils.get_mcp_client_ip(
            request,
            general_settings={
                "use_x_forwarded_for": True,
                "mcp_internal_ip_ranges": ["100.64.0.0/10"],
            },
        )

        result = manager.filter_server_ids_by_ip(["priv"], client_ip=client_ip)
        assert result == []


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
