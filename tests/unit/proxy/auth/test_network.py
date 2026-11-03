from typing import Final

from fastapi import Request

from litellm.proxy.auth.network import (
    PROXY_HEADER_CONTEXT,
    ProxyHeaderContext,
    get_direct_client_ip,
    get_direct_request,
    resolve_network_context,
)


def test_network_context_reads_the_resolved_scope_without_reparsing_headers() -> None:
    request: Final = Request(
        {
            "type": "http",
            "client": ("203.0.113.9", 0),
            "scheme": "https",
            "headers": [(b"host", b"proxy.example"), (b"x-forwarded-for", b"198.51.100.4")],
            PROXY_HEADER_CONTEXT: ProxyHeaderContext(("10.0.0.1", 5555), "http", True, True),
        }
    )
    assert resolve_network_context(request).model_dump() == {
        "client_ip": "203.0.113.9",
        "host": "proxy.example",
        "via_trusted_proxy": True,
    }
    assert get_direct_client_ip(request) == "10.0.0.1"
    direct: Final = get_direct_request(request)
    assert direct.client == ("10.0.0.1", 5555)
    assert direct.scope["scheme"] == "http"
    assert request.client == ("203.0.113.9", 0)
    assert request.scope["scheme"] == "https"


def test_missing_peer_stays_missing_after_forwarding() -> None:
    request: Final = Request(
        {
            "type": "http",
            "client": None,
            "headers": [],
            "scheme": "http",
            PROXY_HEADER_CONTEXT: ProxyHeaderContext(None, "http", True, False),
        }
    )
    assert get_direct_client_ip(request) is None
    assert resolve_network_context(request).client_ip is None


def test_request_without_middleware_uses_the_peer() -> None:
    request: Final = Request(
        {
            "type": "http",
            "client": ("203.0.113.7", 5555),
            "headers": [(b"x-forwarded-for", b"198.51.100.4")],
        }
    )
    assert resolve_network_context(request).model_dump() == {
        "client_ip": "203.0.113.7",
        "host": None,
        "via_trusted_proxy": False,
    }
    assert get_direct_request(request) is request
