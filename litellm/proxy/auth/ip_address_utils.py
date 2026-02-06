"""
IP address utilities for MCP public/private access control.

Internal callers (private IPs) see all MCP servers.
External callers (public IPs) only see servers with available_on_public_internet=True.
"""

import ipaddress
from typing import Any, Dict, List, Optional, Union

from fastapi import Request

from litellm._logging import verbose_proxy_logger
from litellm.proxy.auth.auth_utils import _get_request_ip_address


class IPAddressUtils:
    """Static utilities for IP-based MCP access control."""

    _DEFAULT_INTERNAL_NETWORKS = [
        ipaddress.ip_network("10.0.0.0/8"),
        ipaddress.ip_network("172.16.0.0/12"),
        ipaddress.ip_network("192.168.0.0/16"),
        ipaddress.ip_network("127.0.0.0/8"),
        ipaddress.ip_network("::1/128"),
        ipaddress.ip_network("fc00::/7"),
    ]

    @staticmethod
    def parse_internal_networks(
        configured_ranges: Optional[List[str]],
    ) -> List[Union[ipaddress.IPv4Network, ipaddress.IPv6Network]]:
        """Parse configured CIDR ranges into network objects, falling back to defaults."""
        if not configured_ranges:
            return IPAddressUtils._DEFAULT_INTERNAL_NETWORKS
        networks: List[Union[ipaddress.IPv4Network, ipaddress.IPv6Network]] = []
        for cidr in configured_ranges:
            try:
                networks.append(ipaddress.ip_network(cidr, strict=False))
            except ValueError:
                verbose_proxy_logger.warning(
                    "Invalid CIDR in mcp_internal_ip_ranges: %s, skipping", cidr
                )
        return networks if networks else IPAddressUtils._DEFAULT_INTERNAL_NETWORKS

    @staticmethod
    def is_internal_ip(
        client_ip: Optional[str],
        internal_networks: Optional[
            List[Union[ipaddress.IPv4Network, ipaddress.IPv6Network]]
        ] = None,
    ) -> bool:
        """
        Check if a client IP is from an internal/private network.

        Handles X-Forwarded-For comma chains (takes leftmost = original client).
        Fails closed: empty/invalid IPs are treated as external.
        """
        if not client_ip:
            return False

        # X-Forwarded-For may contain comma-separated chain; leftmost is original client
        if "," in client_ip:
            client_ip = client_ip.split(",")[0].strip()

        networks = internal_networks or IPAddressUtils._DEFAULT_INTERNAL_NETWORKS

        try:
            addr = ipaddress.ip_address(client_ip.strip())
        except ValueError:
            return False

        return any(addr in network for network in networks)

    @staticmethod
    def get_mcp_client_ip(
        request: Request,
        general_settings: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Extract client IP from a FastAPI request for MCP access control.

        Uses the proxy's use_x_forwarded_for setting to determine whether
        to trust X-Forwarded-For headers.

        Args:
            request: FastAPI request object
            general_settings: Optional settings dict. If not provided, imports from proxy_server.
        """
        if general_settings is None:
            from litellm.proxy.proxy_server import (
                general_settings as proxy_general_settings,
            )
            general_settings = proxy_general_settings

        use_xff = general_settings.get("use_x_forwarded_for", False)
        return _get_request_ip_address(request, use_x_forwarded_for=use_xff)
