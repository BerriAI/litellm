from __future__ import annotations

import ipaddress
from typing import Any, Union

from fastapi import Request
from pydantic import BaseModel, Field

from litellm._logging import verbose_proxy_logger

TrustedProxyNetwork = Union[ipaddress.IPv4Network, ipaddress.IPv6Network]


class NetworkContext(BaseModel):
    client_ip: str | None = None
    host: str | None = None
    via_trusted_proxy: bool = False


class TrustedProxyConfig(BaseModel):
    use_forwarded_for: bool = False
    trusted_proxy_cidrs: list[str] = Field(default_factory=list)


def normalize_cidr_ranges(configured_ranges: Any, *, setting_name: str = "trusted_proxy_cidrs") -> list[str]:
    if not configured_ranges:
        return []
    if isinstance(configured_ranges, str):
        return [r.strip() for r in configured_ranges.split(",") if r.strip()]
    if isinstance(configured_ranges, (list, tuple, set)):
        return [str(r).strip() for r in configured_ranges if str(r).strip()]
    verbose_proxy_logger.warning(
        "Invalid %s value: expected a list of CIDR ranges, got %s",
        setting_name,
        type(configured_ranges).__name__,
    )
    return []


def parse_trusted_proxy_ranges(
    configured_ranges: Any, *, setting_name: str = "trusted_proxy_cidrs"
) -> list[TrustedProxyNetwork]:
    networks: list[TrustedProxyNetwork] = []
    for cidr in normalize_cidr_ranges(configured_ranges, setting_name=setting_name):
        try:
            networks.append(ipaddress.ip_network(cidr, strict=False))
        except ValueError:
            verbose_proxy_logger.warning("Invalid CIDR in %s: %s, skipping", setting_name, cidr)
    return networks


def ip_in_networks(client_ip: str | None, networks: list[TrustedProxyNetwork]) -> bool:
    if not client_ip or not networks:
        return False
    try:
        addr = ipaddress.ip_address(client_ip.strip())
    except ValueError:
        return False
    return any(addr in network for network in networks)


def _is_valid_ip(value: str) -> bool:
    try:
        ipaddress.ip_address(value)
        return True
    except ValueError:
        return False


def resolve_client_ip(request: Request, config: TrustedProxyConfig) -> tuple[str | None, bool]:
    """Resolve the real client IP, trusting X-Forwarded-For only when the direct
    peer is itself a configured trusted proxy. Walks the header right-to-left and
    returns the first hop that is not a trusted proxy, so a forged left-most entry
    cannot spoof the client."""
    peer = request.client.host if request.client else None
    networks = parse_trusted_proxy_ranges(config.trusted_proxy_cidrs)
    if not config.use_forwarded_for or not ip_in_networks(peer, networks):
        return peer, False
    forwarded = request.headers.get("x-forwarded-for", "")
    hops = [h.strip() for h in forwarded.split(",") if h.strip()]
    for hop in reversed(hops):
        if _is_valid_ip(hop) and not ip_in_networks(hop, networks):
            return hop, True
    return peer, True


def resolve_network_context(request: Request, config: TrustedProxyConfig) -> NetworkContext:
    ip, via_proxy = resolve_client_ip(request, config)
    return NetworkContext(
        client_ip=ip,
        host=request.headers.get("host"),
        via_trusted_proxy=via_proxy,
    )
