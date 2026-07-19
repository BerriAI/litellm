"""
IP address utilities for MCP public/private access control.

Internal callers (private IPs) see all MCP servers.
External callers (public IPs) only see servers with available_on_public_internet=True.
"""

import ipaddress
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from fastapi import Request
from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.proxy.auth.auth_utils import _get_request_ip_address

# One-shot warning so operators upgrading from the prior "always trust X-Forwarded-*"
# behaviour see an actionable message in their logs the first time it triggers.
_warned_xff_without_trusted_ranges = False

# Error for the inverse footgun: requests arrive with an X-Forwarded-For header
# but use_x_forwarded_for is off, so the real client IP is silently dropped and
# "internal network only" access control trusts the load balancer's IP instead.
# Logged once per misconfiguration window (not per-request) so a flood of crafted
# XFF headers can't spam the logs; re-arms whenever use_x_forwarded_for is observed
# enabled, so a later rollback to disabled warns again.
_warned_xff_present_but_disabled = False

_NUM_TRUSTED_HOPS_ADAPTER = TypeAdapter(int)


@dataclass(frozen=True, slots=True)
class _HopCountUnset:
    """mcp_xff_num_trusted_hops is absent: keep the legacy leftmost-XFF path."""


@dataclass(frozen=True, slots=True)
class _HopCountInvalid:
    """mcp_xff_num_trusted_hops is present but unusable: fail closed, never legacy."""


@dataclass(frozen=True, slots=True)
class _HopCount:
    value: int


_HopCountSetting = Union[_HopCountUnset, _HopCountInvalid, _HopCount]


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
                verbose_proxy_logger.warning("Invalid CIDR in mcp_internal_ip_ranges: %s, skipping", cidr)
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
                verbose_proxy_logger.warning("Invalid CIDR in mcp_trusted_proxy_ranges: %s, skipping", cidr)
        return networks

    @staticmethod
    def parse_cidrs(
        configured_ranges: list[str] | None,
    ) -> list[ipaddress.IPv4Network | ipaddress.IPv6Network]:
        """
        Parse a per-server ``allowed_cidrs`` allowlist into network objects.

        Invalid entries are skipped with a warning so one typo cannot break the
        whole list; callers must treat a configured-but-empty result as
        fail-closed rather than "no allowlist".
        """
        if not configured_ranges:
            return []
        networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
        for cidr in configured_ranges:
            try:
                networks.append(ipaddress.ip_network(cidr, strict=False))
            except ValueError:
                verbose_proxy_logger.warning("Invalid CIDR in MCP server allowed_cidrs: %s, skipping", cidr)
        return networks

    @staticmethod
    def is_ip_in_networks(
        client_ip: str | None,
        networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network],
    ) -> bool:
        """
        Return True if ``client_ip`` falls within any of ``networks``.

        Handles X-Forwarded-For comma chains (takes leftmost = original client)
        and IPv4/IPv6 uniformly. Fails closed: empty/invalid IPs never match.
        """
        if not client_ip:
            return False
        if "," in client_ip:
            client_ip = client_ip.split(",")[0].strip()
        try:
            addr = ipaddress.ip_address(client_ip.strip())
        except ValueError:
            return False
        return any(addr in network for network in networks)

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
        internal_networks: Optional[List[Union[ipaddress.IPv4Network, ipaddress.IPv6Network]]] = None,
    ) -> bool:
        """
        Check if a client IP is from an internal/private network.

        Handles X-Forwarded-For comma chains (takes leftmost = original client).
        Fails closed: empty/invalid IPs are treated as external.
        """
        networks = internal_networks or IPAddressUtils._DEFAULT_INTERNAL_NETWORKS
        return IPAddressUtils.is_ip_in_networks(client_ip, networks)

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
                    "is not configured, so X-Forwarded-* headers will NOT be trusted. "
                    "MCP OAuth discovery URLs fall back to the proxy's literal request "
                    "URL, and MCP access-control client-IP resolution fails closed "
                    "(callers are treated as external). Set mcp_trusted_proxy_ranges in "
                    "general_settings to your reverse-proxy CIDR(s) to trust "
                    "X-Forwarded-*."
                )
                _warned_xff_without_trusted_ranges = True
            return False

        direct_ip = request.client.host if request.client else None
        trusted_networks = IPAddressUtils.parse_trusted_proxy_networks(trusted_ranges)
        return IPAddressUtils.is_trusted_proxy(direct_ip, trusted_networks)

    @staticmethod
    def extract_client_ip_from_xff_hops(
        xff_header: str,
        num_trusted_hops: int,
    ) -> Optional[str]:
        """
        Resolve the originating client IP from an X-Forwarded-For chain by
        counting ``num_trusted_hops`` entries from the right.

        Each trusted proxy appends the address it received the connection from,
        so the right end of the chain is written by infrastructure while the
        left end is attacker-controllable. Selecting the Nth entry from the
        right, where N is the number of trusted appending proxies in front of
        the gateway, yields the real client IP and discards any values a client
        prepended to spoof an allowed address.

        Returns None when the chain has fewer than ``num_trusted_hops`` entries
        or the selected entry is not a valid IP, so callers can fail closed.
        """
        entries = tuple(part.strip() for part in xff_header.split(",") if part.strip())
        if num_trusted_hops < 1 or len(entries) < num_trusted_hops:
            return None
        candidate = entries[-num_trusted_hops]
        try:
            ipaddress.ip_address(candidate)
        except ValueError:
            return None
        return candidate

    @staticmethod
    def _resolve_num_trusted_hops(raw_num_trusted_hops: object) -> _HopCountSetting:
        if raw_num_trusted_hops is None:
            return _HopCountUnset()
        try:
            num_hops = _NUM_TRUSTED_HOPS_ADAPTER.validate_python(raw_num_trusted_hops)
        except ValidationError:
            verbose_proxy_logger.warning(
                "Invalid mcp_xff_num_trusted_hops value %r; failing closed for "
                "MCP client IP resolution. Set it to a positive integer, or "
                "remove the setting to restore the legacy X-Forwarded-For path",
                raw_num_trusted_hops,
            )
            return _HopCountInvalid()
        if num_hops < 1:
            verbose_proxy_logger.warning(
                "mcp_xff_num_trusted_hops=%s is below the minimum of 1; failing "
                "closed for MCP client IP resolution. Set it to a positive "
                "integer, or remove the setting to restore the legacy "
                "X-Forwarded-For path",
                num_hops,
            )
            return _HopCountInvalid()
        return _HopCount(num_hops)

    @staticmethod
    def get_mcp_client_ip(
        request: Request,
        general_settings: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Extract client IP from a FastAPI request for MCP access control.

        Security: Only trusts X-Forwarded-For if:
        1. use_x_forwarded_for is enabled in settings
        2. The direct connection is from a trusted proxy (if mcp_trusted_proxy_ranges configured)

        When ``mcp_xff_num_trusted_hops`` is set, the client IP is read that many
        entries from the right of the chain instead of the spoofable leftmost
        value, defeating append-style X-Forwarded-For forgery. A present-but-invalid
        value (non-integer or below 1) fails closed rather than silently reverting
        to the legacy path, so a config typo cannot quietly weaken access control.

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

        global _warned_xff_present_but_disabled
        if use_xff:
            _warned_xff_present_but_disabled = False
        elif "x-forwarded-for" in request.headers:
            if not _warned_xff_present_but_disabled:
                verbose_proxy_logger.error(
                    "Received a request with an X-Forwarded-For header but "
                    "use_x_forwarded_for is not enabled. The real client IP is "
                    "being ignored and the direct peer's IP (typically your load "
                    "balancer / reverse proxy) is used for MCP access control. "
                    "Because that peer almost always falls within "
                    "general_settings.mcp_internal_ip_ranges, every external caller "
                    "is treated as internal and 'available_on_public_internet: "
                    "false' MCP servers are effectively exposed. Set "
                    "use_x_forwarded_for: true (and mcp_trusted_proxy_ranges to "
                    "your proxy CIDRs) in general_settings to honor the real "
                    "client IP. Not failing the request: if there is no load "
                    "balancer, a crafted X-Forwarded-For header must not be able "
                    "to take down the service."
                )
                _warned_xff_present_but_disabled = True

        # If XFF is enabled, validate the request comes from a trusted proxy
        if use_xff and "x-forwarded-for" in request.headers:
            if not IPAddressUtils.is_request_from_trusted_proxy(request, general_settings=general_settings):
                direct_ip = request.client.host if request.client else None
                if general_settings.get("mcp_trusted_proxy_ranges"):
                    # Direct connection isn't in any configured trusted CIDR.
                    verbose_proxy_logger.warning("XFF header from untrusted IP %s, ignoring", direct_ip)
                    return direct_ip
                # XFF enabled but no trusted proxy ranges configured: the direct
                # peer is typically the reverse proxy's own (private) IP, so
                # returning it would mis-classify external callers as internal.
                # Fail closed for access control.
                return ""
            match IPAddressUtils._resolve_num_trusted_hops(general_settings.get("mcp_xff_num_trusted_hops")):
                case _HopCountInvalid():
                    return ""
                case _HopCount(value=num_trusted_hops):
                    client_ip = IPAddressUtils.extract_client_ip_from_xff_hops(
                        request.headers["x-forwarded-for"], num_trusted_hops
                    )
                    if client_ip is None:
                        verbose_proxy_logger.warning(
                            "X-Forwarded-For chain has fewer than "
                            "mcp_xff_num_trusted_hops=%s entries or an invalid "
                            "address; failing closed",
                            num_trusted_hops,
                        )
                        return ""
                    return client_ip
                case _HopCountUnset():
                    pass
        return _get_request_ip_address(request, use_x_forwarded_for=use_xff)
