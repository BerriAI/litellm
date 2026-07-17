from __future__ import annotations

from litellm.proxy.auth_v2.config import TrustedProxyConfig
from litellm.proxy.auth_v2.network import resolve_client_ip, resolve_network_context

from auth_v2_helpers import make_request

TRUSTED = TrustedProxyConfig(use_forwarded_for=True, trusted_proxy_cidrs=["10.0.0.0/8"])


def test_xff_ignored_when_forwarding_disabled():
    config = TrustedProxyConfig(
        use_forwarded_for=False, trusted_proxy_cidrs=["10.0.0.0/8"]
    )
    request = make_request(
        headers={"x-forwarded-for": "203.0.113.9"}, client=("10.0.0.1", 1)
    )
    ip, via_proxy = resolve_client_ip(request, config)
    assert ip == "10.0.0.1"
    assert via_proxy is False


def test_xff_honored_from_trusted_peer():
    request = make_request(
        headers={"x-forwarded-for": "203.0.113.9, 10.0.0.5"}, client=("10.0.0.1", 1)
    )
    ip, via_proxy = resolve_client_ip(request, TRUSTED)
    assert ip == "203.0.113.9"
    assert via_proxy is True


def test_spoofed_xff_from_untrusted_peer_is_ignored():
    request = make_request(
        headers={"x-forwarded-for": "203.0.113.9"}, client=("8.8.8.8", 1)
    )
    ip, via_proxy = resolve_client_ip(request, TRUSTED)
    assert ip == "8.8.8.8"
    assert via_proxy is False


def test_right_to_left_parse_skips_chained_trusted_proxies():
    request = make_request(
        headers={"x-forwarded-for": "198.51.100.4, 10.1.1.1, 10.0.0.9"},
        client=("10.0.0.1", 1),
    )
    ip, via_proxy = resolve_client_ip(request, TRUSTED)
    assert ip == "198.51.100.4"
    assert via_proxy is True


def test_all_trusted_hops_fall_back_to_peer():
    request = make_request(
        headers={"x-forwarded-for": "10.1.1.1, 10.0.0.9"}, client=("10.0.0.1", 1)
    )
    ip, via_proxy = resolve_client_ip(request, TRUSTED)
    assert ip == "10.0.0.1"
    assert via_proxy is True


def test_invalid_xff_token_is_skipped():
    request = make_request(
        headers={"x-forwarded-for": "not-an-ip, 203.0.113.50"}, client=("10.0.0.1", 1)
    )
    ip, _ = resolve_client_ip(request, TRUSTED)
    assert ip == "203.0.113.50"


def test_network_context_captures_host_and_proxy_flag():
    request = make_request(
        headers={"x-forwarded-for": "203.0.113.9", "host": "proxy.litellm.ai"},
        client=("10.0.0.1", 1),
    )
    ctx = resolve_network_context(request, TRUSTED)
    assert ctx.client_ip == "203.0.113.9"
    assert ctx.host == "proxy.litellm.ai"
    assert ctx.via_trusted_proxy is True
