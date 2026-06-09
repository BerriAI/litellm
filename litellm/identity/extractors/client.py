"""Client/network identity extraction.

``X-Forwarded-For`` is honored only when the direct peer is a configured
trusted proxy (delegated to ``IPAddressUtils.is_request_from_trusted_proxy``).
"""

from typing import Any, Dict, List, Mapping, Optional

from litellm.identity.context import ClientInfo
from litellm.proxy.auth.ip_address_utils import IPAddressUtils


def _split_forwarded_chain(raw: Optional[str]) -> List[str]:
    if not raw or not isinstance(raw, str):
        return []
    return [hop.strip() for hop in raw.split(",") if hop.strip()]


def _direct_client_host(request: Any) -> Optional[str]:
    client = getattr(request, "client", None)
    host = getattr(client, "host", None)
    if isinstance(host, str) and host:
        return host
    return None


def extract_client_info(
    request: Any,
    general_settings: Optional[Dict[str, Any]] = None,
) -> ClientInfo:
    headers: Mapping[str, str] = getattr(request, "headers", {}) or {}
    forwarded_chain: List[str] = []

    xff = headers.get("x-forwarded-for")
    if xff:
        forwarded_chain = _split_forwarded_chain(xff)

    ip: Optional[str]
    if forwarded_chain and IPAddressUtils.is_request_from_trusted_proxy(
        request=request, general_settings=general_settings
    ):
        ip = forwarded_chain[0]
    else:
        ip = _direct_client_host(request)

    user_agent = headers.get("user-agent")

    return ClientInfo(
        ip=ip,
        user_agent=user_agent if isinstance(user_agent, str) else None,
        forwarded_chain=forwarded_chain,
    )
