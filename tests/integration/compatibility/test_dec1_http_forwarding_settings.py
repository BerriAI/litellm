"""Opt-in settings that change what crosses the proxy: header forwarding, disconnect cancellation, CORS origins.

One proxy is booted for this module with `general_settings.forward_client_headers_to_llm_api: true`,
`forward_openai_org_id: true`, `cancel_on_disconnect: true` and the environment variable
`LITELLM_CORS_ORIGINS=https://app.example`. With forwarding on, every client header whose name starts with `x-`
(except `x-stainless-*`) and every `anthropic-beta*` header is copied to the provider call
(`_get_forwardable_headers`, `litellm/proxy/litellm_pre_call_utils.py:1743`). That rule is name based, so it also
copies headers a network hop adds, such as `X-Forwarded-For` and the Rust forwarder's `x-litellm-rust-peer`.
The tests pin what is forwarded today and mark the internal peer header leak as a bug to settle before a Rust
forwarder is put in front of a proxy that has forwarding switched on.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final

import pytest
from integration._support.client import Gateway, eventually
from integration._support.process import graceful_stop_seconds
from integration._support.wire import Reply, Request, Wire
from integration.compatibility.dec1_http_support import (
    CHAT,
    OWNED_CHAT_MODEL,
    OWNED_MESSAGES_MODEL,
    chat_body,
    chat_completion,
    connection,
    json_post,
    owned_front_door,
    posts,
    read_response,
    scripted_reply,
)

ALLOWED_ORIGIN: Final = "https://app.example"
OTHER_ORIGIN: Final = "https://other.example"
HELD_MARKER: Final = "dec1-hold-the-answer"

pytestmark: Final = [
    pytest.mark.timeout(2 * graceful_stop_seconds() + 120),
    pytest.mark.xdist_group("dec1-http-forwarding-settings"),
]


@dataclass(frozen=True, slots=True)
class Rig:
    gateway: Gateway
    upstream: Wire
    gate: threading.Event


def _respond(gate: threading.Event, request: Request) -> Reply:
    if HELD_MARKER.encode() in request.body:
        return Reply(chunks=(b" ", chat_completion()), gate_after_first=gate, heartbeat=b" ", pause_between_chunks=0.2)
    return scripted_reply(request)


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    gate: Final = threading.Event()
    with owned_front_door(
        tmp_path_factory.mktemp("dec1-http-forwarding"),
        general_settings={
            "forward_client_headers_to_llm_api": True,
            "forward_openai_org_id": True,
            "cancel_on_disconnect": True,
        },
        environment={"LITELLM_CORS_ORIGINS": ALLOWED_ORIGIN},
        respond=lambda request: _respond(gate, request),
    ) as door:
        yield Rig(door.gateway, door.upstream, gate)
        gate.set()


@pytest.fixture(autouse=True)
def empty_upstream_log(rig: Rig) -> None:
    rig.upstream.drain()


def _provider_headers(rig: Rig, sent: dict[str, str], key: str | None = None) -> dict[str, str]:
    response: Final = rig.gateway.request("POST", CHAT, chat_body(OWNED_CHAT_MODEL), key=key, headers=sent)
    assert response.status_code == 200, response.text
    (call,) = posts(rig.upstream)
    return dict(call.headers)


def test_a_custom_x_header_is_forwarded_to_the_provider_when_forwarding_is_on(rig: Rig) -> None:
    """The caller sends `x-dec1-trace: trace-123` on a chat call to the proxy with
    `forward_client_headers_to_llm_api: true`. The provider must receive `x-dec1-trace: trace-123` unchanged."""
    assert _provider_headers(rig, {"x-dec1-trace": "trace-123"})["x-dec1-trace"] == "trace-123"


def test_a_mixed_case_x_header_is_forwarded_with_its_value_intact(rig: Rig) -> None:
    """The caller sends `X-Dec1-Tenant: Tenant-ABC` with forwarding on. The provider must receive the header under
    the lower-cased name `x-dec1-tenant` with the value `Tenant-ABC`, case of the value preserved."""
    assert _provider_headers(rig, {"X-Dec1-Tenant": "Tenant-ABC"})["x-dec1-tenant"] == "Tenant-ABC"


def test_the_clients_x_stainless_headers_are_not_forwarded(rig: Rig) -> None:
    """The caller sends `x-stainless-lang: dec1-lang` and `x-stainless-retry-count: 7` with forwarding on. The
    provider must not see either of those two values: `x-stainless*` is excluded from forwarding because it makes
    the proxy's own OpenAI SDK call fail (`litellm_pre_call_utils.py:1757`). The proxy's SDK may send its own
    `x-stainless-*` headers, so the test compares values."""
    received: Final = _provider_headers(rig, {"x-stainless-lang": "dec1-lang", "x-stainless-retry-count": "7"})
    assert received.get("x-stainless-lang") != "dec1-lang"
    assert received.get("x-stainless-retry-count") != "7"


def test_anthropic_beta_is_forwarded_on_a_messages_call(rig: Rig) -> None:
    """The caller sends `anthropic-beta: dec1-beta-2026` on `/v1/messages` with forwarding on. The Anthropic
    provider must receive an `anthropic-beta` header that contains `dec1-beta-2026`. UNVERIFIED: the proxy may
    merge further beta flags into the same header, so the test checks membership and not equality."""
    body: Final = {"model": OWNED_MESSAGES_MODEL, "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}
    response: Final = rig.gateway.request("POST", "/v1/messages", body, headers={"anthropic-beta": "dec1-beta-2026"})
    assert response.status_code == 200, response.text
    (call,) = posts(rig.upstream)
    assert "dec1-beta-2026" in call.headers.get("anthropic-beta", "")


def test_openai_organization_is_forwarded_when_forward_openai_org_id_is_on(rig: Rig) -> None:
    """The caller sends `OpenAI-Organization: org-dec1` to the proxy with `forward_openai_org_id: true`. The
    provider must receive `openai-organization: org-dec1` (`get_openai_org_id_from_headers`,
    `litellm_pre_call_utils.py:1822`)."""
    assert _provider_headers(rig, {"OpenAI-Organization": "org-dec1"})["openai-organization"] == "org-dec1"


def test_the_callers_key_is_never_forwarded_even_with_forwarding_on(rig: Rig) -> None:
    """A virtual key calls a chat model with forwarding on. The provider must receive
    `Authorization: Bearer sk-fixture` (the deployment's key) and the virtual key must not appear in any header
    value. Forwarding is limited to `x-` and `anthropic-beta` names, so `Authorization` is never copied."""
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key()
        received: Final = _provider_headers(rig, {}, key=key)
    assert received["authorization"] == "Bearer sk-fixture"
    assert [name for name, value in received.items() if key in value] == []


def test_a_key_in_x_litellm_api_key_is_not_forwarded_with_forwarding_on(rig: Rig) -> None:
    """A virtual key is sent as `x-litellm-api-key: Bearer <key>` with forwarding on. The name starts with `x-`,
    so the forwarding rule would match it, but the proxy must strip it first: the provider must receive no
    `x-litellm-api-key` header and the key must not appear in any header value (`clean_headers`,
    `litellm_pre_call_utils.py:1585`)."""
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key()
        response: Final = rig.gateway.client.post(
            CHAT, json=chat_body(OWNED_CHAT_MODEL), headers={"x-litellm-api-key": f"Bearer {key}"}
        )
        assert response.status_code == 200, response.text
        (call,) = posts(rig.upstream)
    assert "x-litellm-api-key" not in call.headers
    assert [name for name, value in call.headers.items() if key in value] == []


def test_cookie_and_user_agent_are_not_forwarded_even_with_forwarding_on(rig: Rig) -> None:
    """The caller sends `Cookie: session=dec1` and `User-Agent: dec1-client/9.9` with forwarding on. The provider
    must receive no `cookie` header and a `user-agent` other than `dec1-client/9.9`: neither name starts with
    `x-`."""
    received: Final = _provider_headers(rig, {"Cookie": "session=dec1", "User-Agent": "dec1-client/9.9"})
    assert "cookie" not in received
    assert received.get("user-agent") != "dec1-client/9.9"


def test_x_forwarded_for_is_currently_forwarded_to_the_provider_with_forwarding_on(rig: Rig) -> None:
    """The caller sends `X-Forwarded-For: 203.0.113.9` with forwarding on. Today the provider receives
    `x-forwarded-for: 203.0.113.9`, because the rule forwards every `x-` name. This pins current behaviour so a
    change is noticed: a forwarder that appends its own hop to `X-Forwarded-For` would change what providers see,
    and the client's IP address reaching a third party may itself be unwanted."""
    assert _provider_headers(rig, {"X-Forwarded-For": "203.0.113.9"})["x-forwarded-for"] == "203.0.113.9"


def test_the_rust_peer_header_does_not_reach_the_provider_with_forwarding_on(rig: Rig) -> None:
    """A request arrives with `x-litellm-rust-peer: 203.0.113.9:40000`, the header the Rust forwarder sets on
    every request to carry the client's TCP address (`litellm-rust/crates/gateway-forward/src/rewrite.rs`). The
    provider must not receive it: it is an internal hop header and it exposes the client's address. The `x-`
    forwarding rule matches it and no Python code strips it in this checkout."""
    pytest.skip(
        "BUG: x-litellm-rust-peer is forwarded to the LLM provider when forward_client_headers_to_llm_api is on"
    )
    assert "x-litellm-rust-peer" not in _provider_headers(rig, {"x-litellm-rust-peer": "203.0.113.9:40000"})


def test_litellm_control_headers_are_currently_forwarded_with_forwarding_on(rig: Rig) -> None:
    """The caller sends `x-litellm-tags: dec1-alpha` with forwarding on. Today the provider receives that header
    too, since its name starts with `x-`. UNVERIFIED: read from `_get_forwardable_headers`, which has no
    exclusion for `x-litellm-*`. Pinned so that a deliberate change to stop leaking control headers is a visible
    decision."""
    assert _provider_headers(rig, {"x-litellm-tags": "dec1-alpha"}).get("x-litellm-tags") == "dec1-alpha"


def test_hop_by_hop_headers_are_not_forwarded_with_forwarding_on(rig: Rig) -> None:
    """The caller sends `TE: trailers` and `Proxy-Authorization: Basic ZGVjMQ==` with forwarding on. The provider
    must receive neither value. Neither name starts with `x-`, and the Rust forwarder drops both as hop-by-hop
    headers before the proxy sees them, so the two layers agree."""
    received: Final = _provider_headers(rig, {"TE": "trailers", "Proxy-Authorization": "Basic ZGVjMQ=="})
    assert "proxy-authorization" not in received
    assert received.get("te") != "trailers"


def test_a_client_hang_up_cancels_the_provider_call_when_cancel_on_disconnect_is_on(rig: Rig) -> None:
    """The provider holds the answer to a non-streaming chat call. The client closes its socket while waiting.
    With `cancel_on_disconnect: true` the proxy must close its connection to the provider within 30 seconds
    instead of waiting for the answer. The provider notices because its whitespace keep-alive writes start failing (`_await_llm_call_cancelling_on_disconnect`,
    `common_request_processing.py:2186`). A forwarder that keeps the client leg half-open would hide the
    disconnect and this cancellation would never fire."""
    body: Final = json.dumps(chat_body(OWNED_CHAT_MODEL, text=f"{HELD_MARKER} {uuid.uuid4().hex}")).encode()
    hung_up_before: Final = rig.upstream.disconnected.qsize()
    with connection(rig.gateway) as (client, _):
        client.sendall(json_post(rig.gateway, CHAT, body))
        eventually(rig.upstream.received.qsize, lambda waiting: waiting >= 1, seconds=30)
    try:
        hung_up: Final = eventually(rig.upstream.disconnected.qsize, lambda count: count > hung_up_before, seconds=30)
        assert hung_up > hung_up_before
    finally:
        rig.gate.set()


def test_a_request_that_stays_connected_is_answered_normally_with_cancel_on_disconnect_on(rig: Rig) -> None:
    """A client sends a non-streaming chat call on a raw socket and keeps the socket open until the answer
    arrives. With `cancel_on_disconnect: true` the call must still succeed with 200: the disconnect monitor reads
    from the request stream, and it must not mistake the end of the request body for a hang up."""
    body: Final = json.dumps(chat_body(OWNED_CHAT_MODEL)).encode()
    with connection(rig.gateway) as (client, stream):
        client.sendall(json_post(rig.gateway, CHAT, body))
        response: Final = read_response(stream)
    assert response.status == 200, response.body


def test_an_allowed_origin_is_echoed_with_credentials(rig: Rig) -> None:
    """The proxy runs with `LITELLM_CORS_ORIGINS=https://app.example`. A GET of `/health/liveliness` with
    `Origin: https://app.example` must return `Access-Control-Allow-Origin: https://app.example` and
    `Access-Control-Allow-Credentials: true`: credentials default to on once origins are an explicit list
    (`_get_cors_config`, `proxy_server.py:2648`)."""
    response: Final = rig.gateway.client.get("/health/liveliness", headers={"Origin": ALLOWED_ORIGIN})
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == ALLOWED_ORIGIN
    assert response.headers["access-control-allow-credentials"] == "true"


def test_another_origin_gets_no_cors_headers(rig: Rig) -> None:
    """The proxy allows only `https://app.example`. A GET of `/health/liveliness` with
    `Origin: https://other.example` must still be served with 200, but without any
    `Access-Control-Allow-Origin` header, so a browser on that origin cannot read the answer."""
    response: Final = rig.gateway.client.get("/health/liveliness", headers={"Origin": OTHER_ORIGIN})
    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


def test_a_preflight_from_another_origin_is_refused_with_400(rig: Rig) -> None:
    """The proxy allows only `https://app.example`. A CORS preflight (`OPTIONS` with `Origin` and
    `Access-Control-Request-Method: POST`) for the chat route from `https://other.example` must be answered 400
    with the text `Disallowed CORS origin` and no `Access-Control-Allow-Origin` header. This is the web
    framework's fixed behaviour for a disallowed preflight."""
    response: Final = rig.gateway.client.options(
        CHAT, headers={"Origin": OTHER_ORIGIN, "Access-Control-Request-Method": "POST"}
    )
    assert response.status_code == 400
    assert response.text == "Disallowed CORS origin"
    assert "access-control-allow-origin" not in response.headers


def test_a_preflight_from_the_allowed_origin_is_accepted(rig: Rig) -> None:
    """A CORS preflight for the chat route from `https://app.example` asking for the `authorization` and
    `content-type` headers must be answered 200 with `Access-Control-Allow-Origin: https://app.example` and an
    `Access-Control-Allow-Headers` value that covers both requested headers."""
    response: Final = rig.gateway.client.options(
        CHAT,
        headers={
            "Origin": ALLOWED_ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization, content-type",
        },
    )
    allowed: Final = response.headers.get("access-control-allow-headers", "").lower()
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == ALLOWED_ORIGIN
    assert "authorization" in allowed
    assert "content-type" in allowed
