"""Client/network identity extraction.

Builds a ``ClientInfo`` from a FastAPI request. ``X-Forwarded-For`` is
only honored when the direct peer is in a configured trusted-proxy CIDR.
The trust logic is delegated to ``IPAddressUtils.is_request_from_trusted_proxy``
so we stay in sync with the rest of the proxy.
"""

from typing import Any, Dict, List, Optional

from litellm.identity.context import ClientInfo


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
    headers = getattr(request, "headers", {}) or {}
    forwarded_chain: List[str] = []
    ip: Optional[str] = None

    # Headers in FastAPI are case-insensitive; normalize defensively for dicts.
    def _get_header(name: str) -> Optional[str]:
        try:
            value = headers.get(name)
        except AttributeError:
            return None
        if value is not None:
            return value
        try:
            for key, val in headers.items():
                if isinstance(key, str) and key.lower() == name:
                    return val
        except Exception:
            return None
        return None

    xff = _get_header("x-forwarded-for")
    if xff:
        forwarded_chain = _split_forwarded_chain(xff)

    from litellm.proxy.auth.ip_address_utils import IPAddressUtils

    if forwarded_chain and IPAddressUtils.is_request_from_trusted_proxy(
        request=request, general_settings=general_settings
    ):
        ip = forwarded_chain[0]
    else:
        ip = _direct_client_host(request)

    user_agent = _get_header("user-agent")

    return ClientInfo(
        ip=ip,
        user_agent=user_agent if isinstance(user_agent, str) else None,
        forwarded_chain=forwarded_chain,
    )
