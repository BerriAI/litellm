"""The proxy started with `LITELLM_RUST_FIRST=true`: the Rust gateway owns the public port and forwards to Python.

Contract (litellm/proxy/rust_forward/AGENTS.md, peer.py, supervisor/AGENTS.md):

* The probes an orchestrator uses answer through Rust exactly as Python alone answers them: liveness, the two
  field readiness body, and the authenticated readiness details.
* `x-litellm-rust-peer` is private to the Rust to Python hop. Rust replaces any copy a client sends with the
  real TCP peer, and Python removes it from the request before anything else sees it
  (`_restore_peer`, peer.py:28). It must never reach a provider, never come back to the client, and a client
  can never use it to claim another address.

The security note behind the last point: the header carries the client's real address, so leaking it to a
provider discloses user IPs to a third party, and trusting a client supplied copy would let anyone pass an IP
allowlist. On the Python only path the header is forwarded when `forward_client_headers_to_llm_api` is on
(`compatibility/test_dec1_http_forwarding_settings.py`, skipped as a bug); behind Rust it must be stripped.
Signals and exit rules of the supervisor are in `test_rust_forward_lifecycle.py`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import pytest
from pydantic import JsonValue

from tests.integration._support.rust_forward import MASTER_KEY, EntryPoint, forwarded_proxy
from tests.integration.configuration.dec1_deploy_support import (
    ALIVE,
    CHAT,
    NOT_CONNECTED,
    Provider,
    deployment,
    marker,
    provider,
)

pytestmark: Final = pytest.mark.timeout(900)

PEER: Final = "x-litellm-rust-peer"
CLAIMED: Final = "203.0.113.50"
DETAIL_FIELDS: Final = frozenset(
    {
        "status",
        "db",
        "cache",
        "litellm_version",
        "success_callbacks",
        "use_aiohttp_transport",
        "log_level",
        "is_detailed_debug",
        "show_no_redis_warning",
        "show_env_credential_login_warning",
    }
)


@dataclass(frozen=True, slots=True)
class _Front:
    client: httpx.Client
    upstream: Provider

    def chat(self, headers: dict[str, str] | None = None) -> httpx.Response:
        return self.client.post(
            "/v1/chat/completions",
            json={"model": CHAT, "messages": [{"role": "user", "content": marker()}]},
            headers={"Authorization": f"Bearer {MASTER_KEY}", **(headers or {})},
        )


def _front(tmp_path: Path, general_settings: dict[str, JsonValue]) -> Iterator[_Front]:
    with provider() as upstream:
        config: Final[dict[str, JsonValue]] = {
            "model_list": [deployment(CHAT, upstream.api_base)],
            "general_settings": {"master_key": "os.environ/LITELLM_MASTER_KEY", **general_settings},
            "router_settings": {"num_retries": 0},
        }
        with (
            forwarded_proxy(EntryPoint.LITELLM, config, tmp_path) as proxy,
            httpx.Client(base_url=proxy.url, timeout=60, trust_env=False) as client,
        ):
            yield _Front(client, upstream)


@pytest.fixture(scope="module")
def forwarding(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Front]:
    yield from _front(
        tmp_path_factory.mktemp("dec1-deploy-rust-forwarding"), {"forward_client_headers_to_llm_api": True}
    )


@pytest.fixture(scope="module")
def allowlisted(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Front]:
    yield from _front(tmp_path_factory.mktemp("dec1-deploy-rust-allowlist"), {"allowed_ips": [CLAIMED]})


def _posted_headers(front: _Front) -> list[dict[str, str]]:
    return [dict(request.headers) for request in front.upstream.posted() if request.method == "POST"]


def test_liveness_answers_through_the_rust_gateway(forwarding: _Front) -> None:
    """With Rust on the public port, `GET /health/liveliness` with no key must answer 200 with the JSON string
    `"I'm alive!"`, exactly as Python alone: liveness probes do not change with the process model."""
    response: Final = forwarding.client.get("/health/liveliness")
    assert (response.status_code, response.json()) == (200, ALIVE)


def test_readiness_answers_through_the_rust_gateway_with_the_two_field_body(forwarding: _Front) -> None:
    """With Rust in front and no database configured, `GET /health/readiness` with no key must answer 200 with
    exactly `{"status": "healthy", "db": "Not connected"}`."""
    response: Final = forwarding.client.get("/health/readiness")
    assert (response.status_code, response.json()) == (200, dict(NOT_CONNECTED))


def test_readiness_details_through_the_rust_gateway_still_needs_a_key(forwarding: _Front) -> None:
    """With Rust in front, `GET /health/readiness/details` with no key must answer 401: the forwarder must not
    turn an authenticated Python route into a public one."""
    assert forwarding.client.get("/health/readiness/details").status_code == 401


def test_readiness_details_through_the_rust_gateway_has_the_documented_fields(forwarding: _Front) -> None:
    """With Rust in front, the master key's `GET /health/readiness/details` must answer 200 with exactly the ten
    documented fields and `db: Not connected`."""
    response: Final = forwarding.client.get(
        "/health/readiness/details", headers={"Authorization": f"Bearer {MASTER_KEY}"}
    )
    body: Final = response.json()
    assert response.status_code == 200, response.text
    assert frozenset(body) == DETAIL_FIELDS, sorted(body)
    assert body["db"] == "Not connected"


def test_a_chat_call_through_the_rust_gateway_reaches_the_provider_once(forwarding: _Front) -> None:
    """With Rust in front, one master key chat completion must answer 200 and the provider must receive exactly
    one `POST`: the forwarder adds no retry of its own."""
    forwarding.upstream.posted()
    assert forwarding.chat().status_code == 200
    assert len(_posted_headers(forwarding)) == 1


def test_the_rust_peer_header_never_reaches_the_provider_even_with_header_forwarding_on(forwarding: _Front) -> None:
    """`forward_client_headers_to_llm_api: true` forwards every `x-` header, and the client sends
    `x-litellm-rust-peer: 203.0.113.50:40000`. The provider's request must carry no `x-litellm-rust-peer` at all:
    Rust replaces the client's copy and Python removes the real one before routing (peer.py:28)."""
    forwarding.upstream.posted()
    assert forwarding.chat({PEER: f"{CLAIMED}:40000"}).status_code == 200
    received: Final = _posted_headers(forwarding)
    assert len(received) == 1 and PEER not in {name.lower() for name in received[0]}, received


def test_without_a_client_copy_the_rust_peer_header_still_never_reaches_the_provider(forwarding: _Front) -> None:
    """The client sends no peer header, with header forwarding on. The provider's request must still carry no
    `x-litellm-rust-peer`: the header Rust adds on every request is internal to the hop."""
    forwarding.upstream.posted()
    assert forwarding.chat().status_code == 200
    received: Final = _posted_headers(forwarding)
    assert len(received) == 1 and PEER not in {name.lower() for name in received[0]}, received


def test_the_rust_peer_header_is_never_echoed_back_to_the_client(forwarding: _Front) -> None:
    """The client sends `x-litellm-rust-peer: 203.0.113.50:40000` on a chat completion. The response must answer
    200 and carry no `x-litellm-rust-peer` header: the client's own address must not be reflected either way."""
    response: Final = forwarding.chat({PEER: f"{CLAIMED}:40000"})
    assert response.status_code == 200, response.text
    assert PEER not in {name.lower() for name in response.headers}


def test_a_client_cannot_pass_an_ip_allowlist_by_sending_the_peer_header(allowlisted: _Front) -> None:
    """`general_settings.allowed_ips: [203.0.113.50]`, and a loopback client sends
    `x-litellm-rust-peer: 203.0.113.50:40000`. The call must be refused with 403 and the message
    `Access forbidden: IP address 127.0.0.1 not allowed.` (auth_utils.py:1293), and the provider must not be
    called: the address comes from the real TCP peer, never from a client supplied header."""
    allowlisted.upstream.posted()
    refused: Final = allowlisted.chat({PEER: f"{CLAIMED}:40000"})
    assert refused.status_code == 403, refused.text
    assert "Access forbidden: IP address 127.0.0.1 not allowed." in refused.text
    assert _posted_headers(allowlisted) == []


def test_behind_rust_the_allowlist_sees_the_real_loopback_client(allowlisted: _Front) -> None:
    """Same allowlist, and the loopback client sends no forwarding or peer header. The call must be refused with
    403 naming `127.0.0.1`: Python restores the real client address that Rust saw, instead of treating every
    request as coming from Rust's own hop or from nobody."""
    refused: Final = allowlisted.chat()
    assert refused.status_code == 403, refused.text
    assert "Access forbidden: IP address 127.0.0.1 not allowed." in refused.text
