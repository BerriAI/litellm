from __future__ import annotations

import ipaddress
from typing import Optional, Sequence, Tuple

from fastapi import Request

from litellm.proxy.auth_v2.config import TrustedProxyConfig
from litellm.proxy.auth_v2.models import NetworkContext


def _is_valid_ip(value: str) -> bool:
    try:
        ipaddress.ip_address(value)
        return True
    except ValueError:
        return False


def ip_in_cidrs(ip: Optional[str], cidrs: Sequence[str]) -> bool:
    if not ip or not _is_valid_ip(ip):
        return False
    address = ipaddress.ip_address(ip)
    for cidr in cidrs:
        try:
            if address in ipaddress.ip_network(cidr, strict=False):
                return True
        except ValueError:
            continue
    return False


def resolve_client_ip(
    request: Request, config: TrustedProxyConfig
) -> Tuple[Optional[str], bool]:
    peer = request.client.host if request.client else None
    if not config.use_forwarded_for or not ip_in_cidrs(
        peer, config.trusted_proxy_cidrs
    ):
        return peer, False
    forwarded = request.headers.get("x-forwarded-for", "")
    hops = [h.strip() for h in forwarded.split(",") if h.strip()]
    for hop in reversed(hops):
        if not ip_in_cidrs(hop, config.trusted_proxy_cidrs) and _is_valid_ip(hop):
            return hop, True
    return peer, True


def resolve_network_context(
    request: Request, config: TrustedProxyConfig
) -> NetworkContext:
    ip, via_proxy = resolve_client_ip(request, config)
    return NetworkContext(
        client_ip=ip,
        host=request.headers.get("host"),
        via_trusted_proxy=via_proxy,
    )
