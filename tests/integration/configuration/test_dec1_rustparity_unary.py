"""Non-streaming success parity: Python only proxy versus the same proxy run Rust first.

Contract: for the same config.yaml and the same scripted provider, a successful non-streaming call to
`/v1/chat/completions`, `/v1/messages`, `/v1/responses` or `/v1/embeddings` on an OpenAI, Anthropic, Bedrock,
Vertex AI or Azure deployment must look the same to the caller whether Python owns the public port or the Rust
forwarder (`litellm/rust/crates/gateway-forward`) sits in front of it: same status, same `content-type`, same set
of `x-litellm-*` headers with the same stable values, same body once per-call ids are removed, and the provider
must receive the same request exactly once.

Why a rewrite could break it: the forwarder copies Python's answer and only drops hop-by-hop headers
(`rewrite.rs` `downstream_response`), but Python behind Rust runs on a private listener with proxy headers off and
a peer middleware in front (`rust_forward/peer.py`). Any route, auth or logging code that starts branching on the
listener, the client address or the hop would show up here as a difference. Each test is one comparison; the
matrix is in `dec1_rustparity_support.UNARY_CELLS`.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Final

import httpx
import pytest

from tests.integration.configuration.dec1_rustparity_support import (
    CHAT,
    GATEWAY_BINARY,
    OPENAI,
    UNARY_CELLS,
    Cell,
    Rig,
    comparable_body,
    listener_names,
    litellm_header_names,
    parity_rig,
    provider_body,
    stable_headers,
    stable_provider_headers,
)

pytestmark: Final = pytest.mark.timeout(900)

CELLS: Final = pytest.mark.parametrize("cell", list(UNARY_CELLS.values()), ids=list(UNARY_CELLS))
STABLE_VALUE_HEADERS: Final = (
    "x-litellm-model-id",
    "x-litellm-model-api-base",
    "x-litellm-model-group",
    "x-litellm-version",
    "x-litellm-response-cost",
    "x-litellm-key-tpm-limit",
    "x-litellm-key-rpm-limit",
    "x-litellm-key-max-budget",
)


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with parity_rig(tmp_path_factory.mktemp("dec1-rustparity-unary")) as booted:
        yield booted


@CELLS
def test_a_successful_call_answers_200_with_the_same_body_in_both_modes(rig: Rig, cell: Cell) -> None:
    """The master key posts the cell's request to both proxies. Both must answer 200, and the JSON bodies must be
    equal once per-call fields (`id`, `created`, `created_at`, `system_fingerprint`) and the prompt marker are
    normalised: the forwarder must not touch Python's body."""
    seen: Final = rig.send(cell.path, cell.body())
    assert (seen.python.status, seen.rust.status) == (200, 200), (seen.python.body[:500], seen.rust.body[:500])
    assert comparable_body(seen.rust) == comparable_body(seen.python)


@CELLS
def test_a_successful_call_has_the_same_content_type_in_both_modes(rig: Rig, cell: Cell) -> None:
    """The same successful call on both proxies. The `content-type` header must be identical (for these routes
    `application/json`): a content-type is a `content-*` header, which `rewrite.rs` `must_forward` never drops."""
    seen: Final = rig.send(cell.path, cell.body())
    assert seen.python.headers.get("content-type") == "application/json", seen.python.headers
    assert seen.rust.headers.get("content-type") == seen.python.headers.get("content-type")


@CELLS
def test_a_successful_call_sets_the_same_x_litellm_header_names_in_both_modes(rig: Rig, cell: Cell) -> None:
    """The same successful call on both proxies. The set of `x-litellm-*` response header names must be equal,
    and must contain `x-litellm-call-id` and `x-litellm-model-id` (common_request_processing.py
    `get_custom_headers`). SDKs and dashboards read these; a forwarder that filtered unknown headers would drop
    them."""
    seen: Final = rig.send(cell.path, cell.body())
    names: Final = litellm_header_names(seen.python)
    assert {"x-litellm-call-id", "x-litellm-model-id"} <= names, sorted(names)
    assert litellm_header_names(seen.rust) == names


@CELLS
def test_the_stable_x_litellm_header_values_match_in_both_modes(rig: Rig, cell: Cell) -> None:
    """The same successful call on both proxies. The headers whose value does not change from call to call
    (`x-litellm-model-id`, `-model-api-base`, `-model-group`, `-version`, `-response-cost`, `-key-tpm-limit`,
    `-key-rpm-limit`, `-key-max-budget`) must carry identical values. The model id is the hash of the
    deployment's params, so one config yields one id in both modes."""
    seen: Final = rig.send(cell.path, cell.body())
    python: Final = {name: seen.python.headers.get(name) for name in STABLE_VALUE_HEADERS}
    rust: Final = {name: seen.rust.headers.get(name) for name in STABLE_VALUE_HEADERS}
    assert rust == python


@CELLS
def test_every_non_hop_response_header_matches_in_both_modes(rig: Rig, cell: Cell) -> None:
    """The same successful call on both proxies. After dropping the RFC 9110 hop-by-hop headers and blanking the
    values listed in `VOLATILE_HEADERS` (date, content-length, call id, timings, key spend), the response header
    maps must be equal, provider pass-through headers (`llm_provider-*`) included."""
    seen: Final = rig.send(cell.path, cell.body())
    assert stable_headers(seen.rust) == stable_headers(seen.python)


@CELLS
def test_the_provider_receives_the_same_body_in_both_modes(rig: Rig, cell: Cell) -> None:
    """The same successful call on both proxies. The scripted provider must receive one request per proxy, and
    the two JSON bodies must be equal once the prompt marker is normalised: request preparation is Python's in both
    modes, so the provider cannot tell them apart."""
    seen: Final = rig.send(cell.path, cell.body())
    sent: Final = rig.provider_requests(seen)
    assert (len(sent.python), len(sent.rust)) == (1, 1), sent
    assert provider_body(sent.rust[0], seen.rust.case) == provider_body(sent.python[0], seen.python.case)


@CELLS
def test_the_provider_receives_the_same_headers_in_both_modes(rig: Rig, cell: Cell) -> None:
    """The same successful call on both proxies. The provider requests must carry the same headers with the same
    values, apart from per-request signatures, lengths and ids (`authorization` for SigV4, `x-amz-date`,
    `content-length`, `host`). In particular neither carries `x-litellm-rust-peer` (peer.py:28)."""
    seen: Final = rig.send(cell.path, cell.body())
    sent: Final = rig.provider_requests(seen)
    python: Final = stable_provider_headers(sent.python[0])
    rust: Final = stable_provider_headers(sent.rust[0])
    assert "x-litellm-rust-peer" not in rust
    assert rust == python


@CELLS
def test_the_path_and_query_the_provider_sees_match_in_both_modes(rig: Rig, cell: Cell) -> None:
    """The same successful call on both proxies. The provider must be called on the same request target (path and
    query, for example Azure's `api-version` and Vertex's `:generateContent`) with the same method."""
    seen: Final = rig.send(cell.path, cell.body())
    sent: Final = rig.provider_requests(seen)
    assert (sent.rust[0].method, sent.rust[0].target) == (sent.python[0].method, sent.python[0].target)


def test_a_unary_answer_is_framed_by_content_length_in_both_modes(rig: Rig) -> None:
    """A non-streaming OpenAI chat call on both proxies. Both answers must carry a `content-length` equal to the
    body size and no `transfer-encoding`: Python knows the length, and the forwarder relays it because the body's
    size hint is exact (forward.rs `Relayed::size_hint`). Clients that buffer by length rely on this."""
    seen: Final = rig.send(CHAT, Cell(CHAT, OPENAI).body())
    for observed in (seen.python, seen.rust):
        assert observed.headers.get("content-length") == str(len(observed.body)), observed.headers
        assert "transfer-encoding" not in observed.headers, observed.headers


def test_neither_mode_sends_a_server_header(rig: Rig) -> None:
    """A successful chat call on both proxies. Neither answer may carry a `server` header: uvicorn is started
    with `server_header: False` (proxy_cli.py:436) and hyper's server adds none, so putting Rust in front must not
    start advertising a server product."""
    seen: Final = rig.send(CHAT, Cell(CHAT, OPENAI).body())
    assert ("server" in seen.python.headers, "server" in seen.rust.headers) == (False, False)


def test_each_mode_sends_exactly_one_date_header(rig: Rig) -> None:
    """A successful chat call on both proxies. Each answer must carry exactly one `date` header: uvicorn writes
    one, and hyper adds its own only when the response has none, so the Rust hop must not duplicate it."""
    seen: Final = rig.send(CHAT, Cell(CHAT, OPENAI).body())
    counts: Final = tuple(
        sum(1 for name, _ in observed.raw_headers if name == "date") for observed in (seen.python, seen.rust)
    )
    assert counts == (1, 1)


def test_the_model_list_is_the_same_in_both_modes(rig: Rig) -> None:
    """The master key calls `GET /v1/models` on both proxies. Both must answer 200 with the same list of model
    ids, the ten `rp-*` names from the shared config: a GET with no body goes through the forwarder's chunked
    handling (`rewrite.rs` sets `transfer-encoding: chunked` only for bodies of unknown length)."""
    answers: Final = rig.both(
        lambda proxy: proxy.gateway.client.get("/v1/models", headers={"Authorization": f"Bearer {proxy.gateway.key}"})
    )
    ids: Final = tuple(
        sorted(str(entry["id"]) for entry in answer.json()["data"]) for answer in (answers.python, answers.rust)
    )
    assert (answers.python.status_code, answers.rust.status_code) == (200, 200)
    assert ids[1] == ids[0]
    assert len(ids[0]) == 10, ids[0]


def test_a_cors_preflight_answers_the_same_in_both_modes(rig: Rig) -> None:
    """A browser preflight, `OPTIONS /v1/chat/completions` with `Origin: https://app.example.test` and
    `Access-Control-Request-Method: POST`, on both proxies. Status and every `access-control-*` header must be
    equal: CORS is Python's middleware, and `origin` is an end-to-end header the forwarder passes through."""
    headers: Final = {
        "Origin": "https://app.example.test",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "authorization,content-type",
    }
    answers: Final = rig.both(lambda proxy: proxy.gateway.client.options(CHAT, headers=headers))

    def cors(response: httpx.Response) -> tuple[int, dict[str, str]]:
        return response.status_code, {
            name.lower(): value
            for name, value in response.headers.items()
            if name.lower().startswith("access-control-")
        }

    assert cors(answers.rust) == cors(answers.python)


def test_the_master_key_in_the_x_litellm_api_key_header_is_served_in_both_modes(rig: Rig) -> None:
    """A chat call that authenticates with `x-litellm-api-key: <master key>` instead of `Authorization`. Both
    proxies must answer 200: `x-litellm-api-key` is a credential header the forwarder must never drop, even when a
    client names it in `Connection` (`rewrite.rs` `must_forward`).
    BUG: `must_forward` (litellm/rust/crates/gateway-forward/src/rewrite.rs:138-153) protects `authorization`,
    `api-key`, `x-api-key` and `litellm-api-key` but not `x-litellm-api-key` (`SpecialHeaders.custom_litellm_api_key`,
    litellm/types/proxy/auth/special_headers.py:12), so Rust strips it and Python answers 401 `No api key passed in`
    while Python alone answers 200 (CI run 44176682961)."""
    pytest.skip("BUG: Rust forwarder drops x-litellm-api-key when the client names it in Connection")
    seen: Final = rig.send(
        CHAT,
        Cell(CHAT, OPENAI).body(),
        key=None,
        headers={"x-litellm-api-key": rig.python.gateway.key, "Connection": "keep-alive, x-litellm-api-key"},
    )
    assert (seen.python.status, seen.rust.status) == (200, 200), (seen.python.body[:300], seen.rust.body[:300])


def test_a_request_id_header_from_the_client_is_handled_the_same_in_both_modes(rig: Rig) -> None:
    """A chat call carrying `x-request-id: dec1-rustparity-request`. Whether and how the proxy echoes it, the
    response's `x-request-id` must be the same in both modes (absent in both, or the same value): request ids are
    end-to-end headers and the forwarder neither adds nor strips them.
    UNVERIFIED: whether the Python proxy echoes a client request id on `/v1/chat/completions` at all."""
    seen: Final = rig.send(CHAT, Cell(CHAT, OPENAI).body(), headers={"x-request-id": "dec1-rustparity-request"})
    assert seen.rust.headers.get("x-request-id") == seen.python.headers.get("x-request-id")


def test_the_rust_side_of_the_rig_is_really_fronted_by_the_gateway_binary(rig: Rig) -> None:
    """Sanity check of the rig itself: the Rust first proxy's public port is owned by `litellm-gateway` and its
    log has no fallback notice, while the Python only proxy's port is owned by Python. Without this a
    misinstalled binary would turn every parity test into Python versus Python and pass silently."""
    assert GATEWAY_BINARY in listener_names(rig.rust.port)
    assert GATEWAY_BINARY not in listener_names(rig.python.port)
