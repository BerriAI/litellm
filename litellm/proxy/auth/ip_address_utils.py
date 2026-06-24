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

# One-shot warnings so operators upgrading from the prior "always trust X-Forwarded-*"
# behaviour see an actionable message in their logs the first time each path triggers.
_warned_xff_without_trusted_ranges = False
_warned_xff_access_control_without_trusted_ranges = False


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
    def parse_trusted_proxy_networks(
        configured_ranges: Optional[List[str]],
    ) -> List[Union[ipaddress.IPv4Network, ipaddress.IPv6Network]]:
        """
        Parse trusted proxy CIDR ranges for XFF validation.
        Returns empty list if not configured (XFF will not be trusted).
        """
        if not configured_ranges:
            return []
        networks: List[Union[ipaddress.IPv4Network, ipaddress.IPv6Network]] = []
        for cidr in configured_ranges:
            try:
                networks.append(ipaddress.ip_network(cidr, strict=False))
            except ValueError:
                verbose_proxy_logger.warning(
                    "Invalid CIDR in mcp_trusted_proxy_ranges: %s, skipping", cidr
                )
        return networks

    @staticmethod
    def is_trusted_proxy(
        proxy_ip: Optional[str],
        trusted_networks: List[Union[ipaddress.IPv4Network, ipaddress.IPv6Network]],
    ) -> bool:
        """Check if the direct connection IP is from a trusted proxy."""
        if not proxy_ip or not trusted_networks:
            return False
        try:
            addr = ipaddress.ip_address(proxy_ip.strip())
            return any(addr in network for network in trusted_networks)
        except ValueError:
            return False

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
    def is_request_from_trusted_proxy(
        request: Request,
        general_settings: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Return True if X-Forwarded-* headers on this request should be trusted.

        Trusts the headers iff both:
          1. ``use_x_forwarded_for`` is enabled in proxy settings, AND
          2. ``mcp_trusted_proxy_ranges`` is configured AND the direct
             connection IP (``request.client.host``) falls inside one of
             those CIDRs.

        When ``use_x_forwarded_for`` is enabled but ``mcp_trusted_proxy_ranges``
        is missing, the headers are NOT trusted: there is no way to
        distinguish a trusted reverse proxy from a direct attacker, so callers
        that build URLs (OAuth issuer / redirect_uri / etc.) must fall back
        to the request's literal base URL instead of risking a poisoned host.
        """
        if general_settings is None:
            try:
                from litellm.proxy.proxy_server import (
                    general_settings as proxy_general_settings,
                )

                general_settings = proxy_general_settings
            except ImportError:
                general_settings = {}

        if general_settings is None:
            general_settings = {}

        if not general_settings.get("use_x_forwarded_for", False):
            return False

        trusted_ranges = general_settings.get("mcp_trusted_proxy_ranges")
        if not trusted_ranges:
            global _warned_xff_without_trusted_ranges
            if not _warned_xff_without_trusted_ranges:
                verbose_proxy_logger.warning(
                    "use_x_forwarded_for is enabled but mcp_trusted_proxy_ranges "
                    "is not configured. X-Forwarded-Host/Proto will NOT be "
                    "trusted when building MCP OAuth discovery URLs, which fall "
                    "back to the proxy's literal request URL. Set "
                    "mcp_trusted_proxy_ranges in general_settings to your "
                    "reverse-proxy CIDR(s) to allow X-Forwarded-* through."
                )
                _warned_xff_without_trusted_ranges = True
            return False

        direct_ip = request.client.host if request.client else None
        trusted_networks = IPAddressUtils.parse_trusted_proxy_networks(trusted_ranges)
        return IPAddressUtils.is_trusted_proxy(direct_ip, trusted_networks)

    @staticmethod
    def get_mcp_client_ip(
        request: Request,
        general_settings: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Extract client IP from a FastAPI request for MCP access control.

        Honors X-Forwarded-For whenever ``use_x_forwarded_for`` is enabled, so a
        client behind a reverse proxy is classified by its real (forwarded) IP
        rather than the proxy's peer address. When ``mcp_trusted_proxy_ranges``
        is also configured, the forwarded header is only honored if the direct
        peer falls inside a trusted range; an untrusted peer falls back to its
        literal IP so a spoofed header cannot grant internal access. With
        ``use_x_forwarded_for`` enabled but no trusted ranges configured the
        header is trusted as-is (matching the proxy's wider X-Forwarded-For
        contract), and a one-shot warning is logged so operators know
        access control rests on an unvalidated header.

        Args:
            request: FastAPI request object
            general_settings: Optional settings dict. If not provided, imports from proxy_server.
        """
        if general_settings is None:
            try:
                from litellm.proxy.proxy_server import (
                    general_settings as proxy_general_settings,
                )

                general_settings = proxy_general_settings
            except ImportError:
                general_settings = {}

        # Handle case where general_settings is still None after import
        if general_settings is None:
            general_settings = {}

        use_xff = general_settings.get("use_x_forwarded_for", False)
        xff_present = use_xff and "x-forwarded-for" in request.headers
        trusted_ranges_configured = bool(
            general_settings.get("mcp_trusted_proxy_ranges")
        )

        if (
            xff_present
            and trusted_ranges_configured
            and not IPAddressUtils.is_request_from_trusted_proxy(
                request, general_settings=general_settings
            )
        ):
            direct_ip = request.client.host if request.client else None
            verbose_proxy_logger.warning(
                "XFF header from untrusted IP %s, ignoring", direct_ip
            )
            return direct_ip

        if xff_present and not trusted_ranges_configured:
            IPAddressUtils._warn_xff_trusted_for_access_control()

        return _get_request_ip_address(request, use_x_forwarded_for=use_xff)

    @staticmethod
    def _warn_xff_trusted_for_access_control() -> None:
        global _warned_xff_access_control_without_trusted_ranges
        if _warned_xff_access_control_without_trusted_ranges:
            return
        verbose_proxy_logger.warning(
            "use_x_forwarded_for is enabled but mcp_trusted_proxy_ranges is not "
            "configured. X-Forwarded-For is trusted as-is for MCP IP-based "
            "access control, so a caller with direct network access to the proxy "
            "could spoof an internal IP and reach available_on_public_internet="
            "false servers. Set mcp_trusted_proxy_ranges in general_settings to "
            "your reverse-proxy CIDR(s) to validate the proxy before trusting "
            "X-Forwarded-For."
        )
        _warned_xff_access_control_without_trusted_ranges = True
