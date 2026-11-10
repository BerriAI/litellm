"""Which address the proxy treats as the client's, and what a network hop in front of it changes.

By default the client address is `request.client.host`: the TCP peer, or the address the web server derived from
`X-Forwarded-For` when the peer is a trusted proxy (loopback unless `FORWARDED_ALLOW_IPS` is set). With `general_settings.use_x_forwarded_for: true` the whole `X-Forwarded-For` header value, taken
verbatim with no splitting on commas, becomes the client address (`get_request_ip_address`,
`litellm/proxy/auth/auth_utils.py:97`). That address is compared against `general_settings.allowed_ips`
(403 `Access forbidden: IP address <ip> not allowed.`, `auth_utils.py:890`) and written to the spend row's
`requester_ip_address` (`litellm_pre_call_utils.py:3062`).

This matters for a Rust gateway in front of Python: every request then arrives from the gateway's own address, so
without extra work an IP allowlist sees `127.0.0.1` for everybody and spend rows lose the real client. The
forwarder passes `X-Forwarded-For` through unchanged and adds `x-litellm-rust-peer: <ip>:<port>`, which no Python
code reads in this checkout. One proxy is booted here with `use_x_forwarded_for: true` and
`allowed_ips: [203.0.113.9, 127.0.0.1]`. The loopback entry keeps the harness, which sends no forwarding header,
able to manage the proxy.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, object_value
from integration._support.process import graceful_stop_seconds
from integration.compatibility.dec1_http_support import (
    CHAT,
    OWNED_CHAT_MODEL,
    OwnedDoor,
    chat_body,
    front_door,
    owned_front_door,
    spend_rows,
)
from pydantic import JsonValue

ALLOWED: Final = "203.0.113.9"
OTHER: Final = "198.51.100.7"
OWNED_PROXY_TIMEOUT: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

pytestmark: Final = pytest.mark.xdist_group("dec1-http-client-ip")


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[OwnedDoor]:
    with owned_front_door(
        tmp_path_factory.mktemp("dec1-http-client-ip"),
        general_settings={"use_x_forwarded_for": True, "allowed_ips": [ALLOWED, "127.0.0.1"]},
    ) as door:
        yield door


def _chat(rig: OwnedDoor, headers: dict[str, str], key: str | None = None) -> httpx.Response:
    return rig.gateway.request("POST", CHAT, chat_body(OWNED_CHAT_MODEL), key=key, headers=headers)


@OWNED_PROXY_TIMEOUT
def test_a_forwarded_address_on_the_allowlist_is_served(rig: OwnedDoor) -> None:
    """The proxy trusts `X-Forwarded-For` and allows `203.0.113.9`. A chat call carrying
    `X-Forwarded-For: 203.0.113.9` must be served with 200."""
    assert _chat(rig, {"X-Forwarded-For": ALLOWED}).status_code == 200


@OWNED_PROXY_TIMEOUT
def test_a_forwarded_address_off_the_allowlist_is_refused_with_403_naming_it(rig: OwnedDoor) -> None:
    """A chat call carries `X-Forwarded-For: 198.51.100.7`, which is not on the allowlist, although the TCP peer
    (loopback) is. The proxy must answer 403 and the body must contain
    `Access forbidden: IP address 198.51.100.7 not allowed.`: once the header is trusted it replaces the peer
    address. The provider must receive nothing."""
    rig.upstream.drain()
    response: Final = _chat(rig, {"X-Forwarded-For": OTHER})
    assert response.status_code == 403, response.text
    assert f"Access forbidden: IP address {OTHER} not allowed." in response.text
    assert rig.calls() == ()


@OWNED_PROXY_TIMEOUT
def test_a_forwarded_chain_is_compared_as_one_string_and_refused(rig: OwnedDoor) -> None:
    """A chat call carries `X-Forwarded-For: 203.0.113.9, 10.0.0.5`, the form a second hop produces by appending
    its own peer. The proxy compares the whole header value to the allowlist, so it must answer 403 and name
    `203.0.113.9, 10.0.0.5` as the address. A forwarder that appends to `X-Forwarded-For` would therefore lock
    every allowlisted client out."""
    chain: Final = f"{ALLOWED}, 10.0.0.5"
    response: Final = _chat(rig, {"X-Forwarded-For": chain})
    assert response.status_code == 403, response.text
    assert f"Access forbidden: IP address {chain} not allowed." in response.text


@OWNED_PROXY_TIMEOUT
def test_without_the_header_the_tcp_peer_address_is_used(rig: OwnedDoor) -> None:
    """A chat call carries no `X-Forwarded-For`. The proxy must fall back to the TCP peer, here loopback, which is
    on the allowlist, and serve the call with 200. Behind a Rust forwarder that peer is always the forwarder."""
    assert _chat(rig, {}).status_code == 200


@OWNED_PROXY_TIMEOUT
def test_the_rust_peer_header_does_not_decide_the_client_address(rig: OwnedDoor) -> None:
    """A chat call carries `x-litellm-rust-peer: 198.51.100.7:40000` and no `X-Forwarded-For`. The call must be
    served with 200: the allowlist decision uses the TCP peer (loopback), and the peer header naming an address
    off the allowlist is ignored. This pins that the header is not an input to access control today. If Python
    starts reading it, it must only be trusted from the local forwarder, since any client can send it."""
    assert _chat(rig, {"x-litellm-rust-peer": f"{OTHER}:40000"}).status_code == 200


@OWNED_PROXY_TIMEOUT
def test_the_allowlist_check_applies_to_management_routes_too(rig: OwnedDoor) -> None:
    """The master key lists keys with `X-Forwarded-For: 198.51.100.7`. The proxy must answer 403 with the
    `Access forbidden` message: the address check runs before the key is evaluated, on every authenticated route
    and not only on model calls."""
    response: Final = rig.gateway.request("GET", "/key/list", headers={"X-Forwarded-For": OTHER})
    assert response.status_code == 403, response.text
    assert f"Access forbidden: IP address {OTHER} not allowed." in response.text


@OWNED_PROXY_TIMEOUT
def test_the_liveliness_probe_is_not_subject_to_the_allowlist(rig: OwnedDoor) -> None:
    """`GET /health/liveliness` is called with `X-Forwarded-For: 198.51.100.7`, an address off the allowlist. The
    probe must still answer 200: it has no auth dependency, so load balancer health checks keep working whatever
    the allowlist says."""
    response: Final = rig.gateway.client.get("/health/liveliness", headers={"X-Forwarded-For": OTHER})
    assert response.status_code == 200


@OWNED_PROXY_TIMEOUT
def test_a_refused_address_gets_the_403_even_with_an_invalid_key(rig: OwnedDoor) -> None:
    """A chat call carries an unknown key and `X-Forwarded-For: 198.51.100.7`. The proxy must answer 403 for the
    address and not 401 for the key. UNVERIFIED: read from the order of checks in `pre_db_read_auth_checks`
    (`auth_utils.py:880`), where the address check comes before any key lookup."""
    response: Final = _chat(rig, {"X-Forwarded-For": OTHER}, key="sk-dec1-unknown-key")
    assert response.status_code == 403, response.text


@OWNED_PROXY_TIMEOUT
def test_the_forwarded_address_is_recorded_on_the_spend_row(rig: OwnedDoor) -> None:
    """A fresh key makes a chat call with `X-Forwarded-For: 203.0.113.9` on the proxy that trusts the header. The
    spend row's `requester_ip_address` column and its `metadata.requester_ip_address` must both be
    `203.0.113.9`."""
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key()
        assert _chat(rig, {"X-Forwarded-For": ALLOWED}, key=key).status_code == 200
        (row,) = spend_rows(rig.gateway, key)
    assert row["requester_ip_address"] == ALLOWED
    assert object_value(row["metadata"])["requester_ip_address"] == ALLOWED


@OWNED_PROXY_TIMEOUT
def test_without_the_header_the_spend_row_records_the_tcp_peer(rig: OwnedDoor) -> None:
    """A fresh key makes a chat call with no `X-Forwarded-For` on the proxy that trusts the header. The spend
    row's `requester_ip_address` must be `127.0.0.1`, the TCP peer. Behind a Rust forwarder that sends no
    `X-Forwarded-For` of its own, every row would carry the forwarder's address."""
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key()
        assert _chat(rig, {}, key=key).status_code == 200
        (row,) = spend_rows(rig.gateway, key)
    assert row["requester_ip_address"] == "127.0.0.1"


def _default_row_address(gateway: Gateway, headers: dict[str, str]) -> JsonValue:
    with front_door(gateway) as door:
        key: Final = door.scenario.key()
        response: Final = gateway.request("POST", CHAT, chat_body(door.chat_model()), key=key, headers=headers)
        assert response.status_code == 200, response.text
        (row,) = spend_rows(gateway, key)
        return row["requester_ip_address"]


def test_a_loopback_peer_is_trusted_to_set_the_client_address_by_default(gateway: Gateway) -> None:
    """On the shared proxy, where `use_x_forwarded_for` is unset, a fresh key makes a chat call over loopback
    with `X-Forwarded-For: 203.0.113.9`. The spend row's `requester_ip_address` is `203.0.113.9`: the web server
    itself rewrites the peer address from that header when the TCP peer is `127.0.0.1` (uvicorn runs with proxy
    headers on and trusts loopback unless `FORWARDED_ALLOW_IPS` says otherwise). UNVERIFIED: read from uvicorn's
    defaults, not from proxy code. A Rust forwarder on loopback that passes a client's `X-Forwarded-For` through
    makes every client trusted in this way."""
    assert _default_row_address(gateway, {"X-Forwarded-For": ALLOWED}) == ALLOWED


def test_a_forwarded_chain_from_loopback_resolves_to_its_last_untrusted_hop(gateway: Gateway) -> None:
    """On the shared proxy a fresh key makes a chat call over loopback with
    `X-Forwarded-For: 203.0.113.9, 198.51.100.7`. The spend row's `requester_ip_address` is `198.51.100.7`: the
    web server walks the list from the right and stops at the first address it does not trust. UNVERIFIED: this
    is uvicorn's behaviour since 0.32 and older versions took the first entry."""
    assert _default_row_address(gateway, {"X-Forwarded-For": f"{ALLOWED}, {OTHER}"}) == OTHER


def test_the_forwarded_and_x_real_ip_headers_never_set_the_logged_address(gateway: Gateway) -> None:
    """On the shared proxy a fresh key makes a chat call over loopback with `Forwarded: for=203.0.113.9` and
    `X-Real-IP: 203.0.113.9` and no `X-Forwarded-For`. The spend row's `requester_ip_address` must be
    `127.0.0.1`, the TCP peer: neither the web server nor the proxy reads those two headers."""
    assert _default_row_address(gateway, {"Forwarded": f"for={ALLOWED}", "X-Real-IP": ALLOWED}) == "127.0.0.1"


def test_the_rust_peer_header_never_sets_the_logged_address(gateway: Gateway) -> None:
    """On the shared proxy a fresh key makes a chat call over loopback with
    `x-litellm-rust-peer: 203.0.113.9:40000`. The spend row's `requester_ip_address` must be `127.0.0.1`. This
    documents the gap a forwarder opens: the real client address travels in this header and nothing records it."""
    assert _default_row_address(gateway, {"x-litellm-rust-peer": f"{ALLOWED}:40000"}) == "127.0.0.1"
