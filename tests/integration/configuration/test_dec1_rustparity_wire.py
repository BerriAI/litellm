"""HTTP wire parity: what the Rust forwarder changes on the way in and out, and what it must not.

Contract (gateway-forward/AGENTS.md, FORWARDING.md, src/rewrite.rs, src/error.rs): the forwarder sends Python the
client's method, target, headers and body byte for byte, except that it drops hop-by-hop headers (RFC 9110 7.6.1:
`connection`, `keep-alive`, `te`, `trailer`, `transfer-encoding`, `upgrade`, `proxy-*` and every header the client
names in `Connection`, apart from the protected credential, `content-*`, `x-forwarded-*` and websocket headers)
and replaces `x-litellm-rust-peer`. A request Python could not have received unchanged is answered by Rust itself,
with a bare status and an empty body (`Rejection::into_response`).

So this file holds two kinds of tests. Parity tests send the same bytes to both proxies and require the same answer
(unknown routes, method not allowed, HEAD, pipelining, `Expect: 100-continue`, `Connection: close`, encoded and
fragment targets, protected headers named in `Connection`). Pinned differences assert both sides literally and say
why they differ: a client header named in `Connection` reaches the provider only without Rust, and the framing
cases FORWARDING.md lists as "Known differences from uvicorn with h11" (obsolete line folding, a repeated
`Content-Length` list, HTTP/1.0 chunked uploads, transfer codings other than chunked, HTTP/1.1 without `Host`) get
Rust's own empty 400. The rig turns on `forward_client_headers_to_llm_api` so request headers become visible at the
provider.

UNVERIFIED for every Python only expectation on malformed framing: CI installs `httptools` (pyproject.toml
`proxy-dev`), so uvicorn's `auto` HTTP implementation is httptools/llhttp, not the h11 parser FORWARDING.md was
written against. Where the two parsers disagree the Python side is asserted loosely and the docstring says so.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from typing import Final

import pytest

from tests.integration.configuration.dec1_rustparity_support import (
    CHAT,
    MASTER_KEY,
    OPENAI,
    Booted,
    Cell,
    Pair,
    RawAnswer,
    Rig,
    chat_bytes,
    marker,
    parity_rig,
    raw_exchange,
    raw_request,
    read_answer,
)

pytestmark: Final = pytest.mark.timeout(900)

HOP_HEADER: Final = "x-dec1-rustparity-hop"
END_TO_END: Final = "x-dec1-rustparity-end-to-end"


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with parity_rig(
        tmp_path_factory.mktemp("dec1-rustparity-wire"), general_settings={"forward_client_headers_to_llm_api": True}
    ) as booted:
        yield booted


def _host(proxy: Booted) -> str:
    return f"127.0.0.1:{proxy.port}"


def _raw_chat(proxy: Booted, extra: tuple[tuple[str, str], ...] = (), version: str = "HTTP/1.1") -> RawAnswer:
    body: Final = chat_bytes(OPENAI, marker())
    return raw_exchange(
        proxy,
        raw_request(
            "POST",
            CHAT,
            (
                ("Host", _host(proxy)),
                ("Authorization", f"Bearer {MASTER_KEY}"),
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(body))),
                ("Connection", "close"),
                *extra,
            ),
            body,
            version=version,
        ),
    )


# ------------------------------------------------------------------------------------------- headers in


def test_a_header_the_client_names_in_connection_reaches_the_provider_only_without_rust(rig: Rig) -> None:
    """PINNED DIFFERENCE. With header forwarding on, the client sends `x-dec1-rustparity-hop: 1` and
    `Connection: keep-alive, x-dec1-rustparity-hop`. Python alone forwards every `x-` header, so its provider sees
    `x-dec1-rustparity-hop`; through Rust the header is hop-by-hop by RFC 9110 and is removed before Python
    (`rewrite.rs` `strip_hop_by_hop`), so that provider does not. Reason it is accepted: a proxy must drop headers
    the client scoped to its own connection, and uvicorn alone simply does not implement that rule."""
    seen: Final = rig.send(
        CHAT, Cell(CHAT, OPENAI).body(), headers={HOP_HEADER: "1", "Connection": f"keep-alive, {HOP_HEADER}"}
    )
    sent: Final = rig.provider_requests(seen)
    assert (HOP_HEADER in sent.python[0].headers, HOP_HEADER in sent.rust[0].headers) == (True, False)


def test_an_end_to_end_x_header_reaches_the_provider_in_both_modes(rig: Rig) -> None:
    """With header forwarding on, the client sends `x-dec1-rustparity-end-to-end: kept` without naming it in
    `Connection`. Both providers must receive it with the value `kept`
    (`LiteLLMProxyRequestSetup._get_forwardable_headers`)."""
    seen: Final = rig.send(CHAT, Cell(CHAT, OPENAI).body(), headers={END_TO_END: "kept"})
    sent: Final = rig.provider_requests(seen)
    assert (sent.python[0].headers.get(END_TO_END), sent.rust[0].headers.get(END_TO_END)) == ("kept", "kept")


def test_x_forwarded_for_named_in_connection_still_reaches_python_in_both_modes(rig: Rig) -> None:
    """The client sends `X-Forwarded-For: 198.51.100.7` and lists `x-forwarded-for` in `Connection`. The forwarder
    must keep it (`must_forward` protects `x-forwarded-*`), so with header forwarding on both providers receive the
    same `x-forwarded-for` value (present in both or absent in both)."""
    seen: Final = rig.send(
        CHAT,
        Cell(CHAT, OPENAI).body(),
        headers={"X-Forwarded-For": "198.51.100.7", "Connection": "keep-alive, x-forwarded-for"},
    )
    sent: Final = rig.provider_requests(seen)
    assert sent.rust[0].headers.get("x-forwarded-for") == sent.python[0].headers.get("x-forwarded-for")


def test_authorization_named_in_connection_still_authenticates_in_both_modes(rig: Rig) -> None:
    """The client lists `authorization` in `Connection`. Both proxies must still authenticate the master key and
    answer 200: a client must not be able to make the forwarder strip its own credentials (`must_forward`)."""
    seen: Final = rig.send(CHAT, Cell(CHAT, OPENAI).body(), headers={"Connection": "keep-alive, authorization"})
    assert (seen.python.status, seen.rust.status) == (200, 200), (seen.python.body[:200], seen.rust.body[:200])


def test_te_and_keep_alive_request_headers_do_not_change_the_answer(rig: Rig) -> None:
    """The client adds `TE: trailers` and `Keep-Alive: timeout=5`. Rust drops both before Python; Python alone
    ignores them. Both proxies must answer 200 with the same `content-type`."""
    seen: Final = rig.send(CHAT, Cell(CHAT, OPENAI).body(), headers={"TE": "trailers", "Keep-Alive": "timeout=5"})
    assert (seen.python.status, seen.rust.status) == (200, 200)
    assert seen.rust.headers.get("content-type") == seen.python.headers.get("content-type")


# ------------------------------------------------------------------------------------------- routes and methods


def test_an_unknown_route_answers_the_same_404_in_both_modes(rig: Rig) -> None:
    """`GET /dec1/rustparity/no-such-route` with the master key. Both must answer 404 with the same body: the
    forwarder never branches on a path (gateway-forward/AGENTS.md)."""
    answers: Final = rig.both(
        lambda proxy: proxy.gateway.client.get(
            "/dec1/rustparity/no-such-route", headers={"Authorization": f"Bearer {MASTER_KEY}"}
        )
    )
    assert (answers.python.status_code, answers.rust.status_code) == (404, 404)
    assert answers.rust.content == answers.python.content


def test_a_wrong_method_answers_the_same_405_in_both_modes(rig: Rig) -> None:
    """`DELETE /v1/chat/completions` with the master key. Both must give the same status, the same `allow` header
    and the same body. UNVERIFIED: 405 is expected from Starlette routing; a catch-all route would change it."""
    answers: Final = rig.both(
        lambda proxy: proxy.gateway.client.delete(CHAT, headers={"Authorization": f"Bearer {MASTER_KEY}"})
    )
    assert answers.rust.status_code == answers.python.status_code
    assert answers.rust.headers.get("allow") == answers.python.headers.get("allow")
    assert answers.rust.content == answers.python.content


def test_a_head_probe_answers_the_same_in_both_modes(rig: Rig) -> None:
    """`HEAD /health/liveliness`. Both must give the same status and an empty body; the forwarder must not wait
    for a body Python never sends for HEAD."""
    answers: Final = rig.both(lambda proxy: proxy.gateway.client.head("/health/liveliness"))
    assert answers.rust.status_code == answers.python.status_code
    assert (answers.python.content, answers.rust.content) == (b"", b"")


def test_a_percent_encoded_hash_in_the_target_reaches_python_in_both_modes(rig: Rig) -> None:
    """`GET /health/liveliness%23x`. `%23` is data, which the forwarder preserves (FORWARDING.md "URL fragments
    and literal hashes"), so both proxies route the same path and must give the same status (404 is expected:
    no route has that name)."""
    answers: Final = rig.both(lambda proxy: proxy.gateway.client.get("/health/liveliness%23x"))
    assert answers.rust.status_code == answers.python.status_code


def test_a_raw_fragment_in_the_target_is_dropped_the_same_way_in_both_modes(rig: Rig) -> None:
    """A raw client sends `GET /health/liveliness#probe HTTP/1.1`. Rust forwards only `/health/liveliness`
    (FORWARDING.md); uvicorn with httptools parses the URL and drops the fragment too, so both should answer 200.
    UNVERIFIED for Python alone: with the h11 parser the fragment stays in the path and the answer would be 404,
    which FORWARDING.md records as a difference."""
    answers: Final = rig.both(
        lambda proxy: raw_exchange(
            proxy,
            raw_request("GET", "/health/liveliness#probe", (("Host", _host(proxy)), ("Connection", "close"))),
        )
    )
    assert (answers.python.status, answers.rust.status) == (200, 200)


# ------------------------------------------------------------------------------------------- connection behaviour


def _pipelined(proxy: Booted) -> tuple[int, int]:
    one: Final = raw_request("GET", "/health/liveliness", (("Host", _host(proxy)),))
    two: Final = raw_request("GET", "/health/readiness", (("Host", _host(proxy)), ("Connection", "close")))
    with socket.create_connection(("127.0.0.1", proxy.port), timeout=30) as client, client.makefile("rb") as stream:
        client.sendall(one + two)
        return read_answer(stream).status, read_answer(stream).status


def test_two_pipelined_requests_are_answered_in_order_in_both_modes(rig: Rig) -> None:
    """A raw client writes two GETs (liveliness, then readiness) back to back on one connection before reading.
    Both proxies must answer both, in order, with 200 and 200: HTTP/1.1 pipelining is legal and the forwarder must
    neither drop the second request nor reorder the answers."""
    assert rig.both(_pipelined) == Pair((200, 200), (200, 200))


def _expect_continue(proxy: Booted) -> tuple[int, int]:
    body: Final = chat_bytes(OPENAI, marker())
    head: Final = raw_request(
        "POST",
        CHAT,
        (
            ("Host", _host(proxy)),
            ("Authorization", f"Bearer {MASTER_KEY}"),
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(body))),
            ("Expect", "100-continue"),
            ("Connection", "close"),
        ),
    )
    with socket.create_connection(("127.0.0.1", proxy.port), timeout=30) as client, client.makefile("rb") as stream:
        client.sendall(head)
        interim: Final = read_answer(stream).status
        client.sendall(body)
        return interim, read_answer(stream).status


def test_expect_100_continue_gets_an_interim_100_then_200_in_both_modes(rig: Rig) -> None:
    """A raw client sends a chat head with `Expect: 100-continue` and waits before sending the body. Both proxies
    must first write `100 Continue`, then answer 200 once the body arrives. Through Rust the interim answer comes
    from hyper's server as soon as the forwarder starts reading the body; clients such as curl stall for a second
    without it."""
    assert rig.both(_expect_continue) == Pair((100, 200), (100, 200))


def test_connection_close_from_the_client_closes_after_one_answer_in_both_modes(rig: Rig) -> None:
    """A raw chat request with `Connection: close`. Both proxies must answer 200 and say `connection: close` in the
    answer head: Python's own `connection` header is hop-by-hop and dropped by Rust, but hyper's server writes its
    own for a connection it is closing. UNVERIFIED: that hyper adds the header rather than just closing."""
    answers: Final = rig.both(_raw_chat)
    assert (answers.python.status, answers.rust.status) == (200, 200)
    assert (answers.python.header("connection"), answers.rust.header("connection")) == ("close", "close")


# ------------------------------------------------------------------------------------------- pinned framing differences


def test_an_obsolete_folded_header_is_refused_by_rust_with_an_empty_400(rig: Rig) -> None:
    """PINNED DIFFERENCE (FORWARDING.md "Known differences", `a_folded_header_reaches_python_unfolded`). A raw chat
    request carries `x-dec1-fold: a` followed by the continuation line ` b`. Rust's parser refuses obsolete line
    folding: 400 with an empty body. Python alone unfolds it and serves the call (h11). UNVERIFIED for Python
    alone under httptools, so only "not refused by Rust's empty 400" is asserted for it."""
    answers: Final = rig.both(lambda proxy: _raw_chat(proxy, (("x-dec1-fold", "a\r\n b"),)))
    assert (answers.rust.status, answers.rust.body) == (400, b"")
    assert (answers.python.status, answers.python.body) != (400, b"")


def test_a_repeated_content_length_list_is_refused_by_rust_with_an_empty_400(rig: Rig) -> None:
    """PINNED DIFFERENCE (FORWARDING.md, `a_content_length_list_of_one_value_reaches_python`). A raw chat request
    whose head says `Content-Length: <n>, <n>` with the same value twice. Rust refuses it with 400 and an empty
    body; h11 accepts the list as length n. UNVERIFIED for httptools, so the Python side is only required not to
    be Rust's empty 400."""

    def send(proxy: Booted) -> RawAnswer:
        body = chat_bytes(OPENAI, marker())  # rebind-ok: one per proxy
        return raw_exchange(
            proxy,
            raw_request(
                "POST",
                CHAT,
                (
                    ("Host", _host(proxy)),
                    ("Authorization", f"Bearer {MASTER_KEY}"),
                    ("Content-Type", "application/json"),
                    ("Content-Length", f"{len(body)}, {len(body)}"),
                    ("Connection", "close"),
                ),
                body,
            ),
        )

    answers: Final = rig.both(send)
    assert (answers.rust.status, answers.rust.body) == (400, b"")
    assert (answers.python.status, answers.python.body) != (400, b"")


def test_a_transfer_coding_other_than_chunked_is_refused_by_rust_with_an_empty_400(rig: Rig) -> None:
    """PINNED DIFFERENCE (`rewrite.rs` `only_chunked`, `Rejection::UnsupportedTransferCoding`). A raw chat request
    with `Transfer-Encoding: gzip, chunked`. Rust answers 400 with an empty body and never reaches Python. Python
    alone also refuses it, but with its own answer: uvicorn's `Invalid HTTP request received.` text, or, with
    httptools, a 400 JSON envelope after the still-gzip body fails to parse. Both are 400; only the bodies differ."""

    def send(proxy: Booted) -> RawAnswer:
        body = chat_bytes(OPENAI, marker())  # rebind-ok: one per proxy
        framed = b"%x\r\n%s\r\n0\r\n\r\n" % (len(body), body)  # rebind-ok: one per proxy
        return raw_exchange(
            proxy,
            raw_request(
                "POST",
                CHAT,
                (
                    ("Host", _host(proxy)),
                    ("Authorization", f"Bearer {MASTER_KEY}"),
                    ("Content-Type", "application/json"),
                    ("Transfer-Encoding", "gzip, chunked"),
                    ("Connection", "close"),
                ),
                framed,
            ),
        )

    answers: Final = rig.both(send)
    assert (answers.rust.status, answers.rust.body) == (400, b"")
    assert answers.python.status in (400, 501) and answers.python.body != b""


def test_http_1_1_without_a_host_header_is_refused_by_rust_with_an_empty_400(rig: Rig) -> None:
    """PINNED DIFFERENCE (`rewrite.rs` `public_host`, `Rejection::MissingHost`). A raw `GET /health/liveliness
    HTTP/1.1` with no `Host`. Rust answers 400 with an empty body because Python builds absolute URLs from Host
    and must never see the private upstream address instead. h11 refuses it too, with
    `Invalid HTTP request received.`; httptools does not require Host and would serve 200. UNVERIFIED which of
    those Python alone does in CI, so only Rust's side is pinned exactly."""
    answers: Final = rig.both(
        lambda proxy: raw_exchange(proxy, raw_request("GET", "/health/liveliness", (("Connection", "close"),)))
    )
    assert (answers.rust.status, answers.rust.body) == (400, b"")
    assert answers.python.status in (200, 400)


def test_an_http_1_0_chunked_upload_is_refused_by_rust_with_an_empty_400(rig: Rig) -> None:
    """PINNED DIFFERENCE (FORWARDING.md, `an_http10_chunked_upload_reaches_python`). A raw `HTTP/1.0` chat POST with
    `Transfer-Encoding: chunked`. Rust refuses it with 400 and an empty body; h11 decodes it and serves the call.
    UNVERIFIED for httptools, so the Python side is only required not to be Rust's empty 400."""

    def send(proxy: Booted) -> RawAnswer:
        body = chat_bytes(OPENAI, marker())  # rebind-ok: one per proxy
        framed = b"%x\r\n%s\r\n0\r\n\r\n" % (len(body), body)  # rebind-ok: one per proxy
        return raw_exchange(
            proxy,
            raw_request(
                "POST",
                CHAT,
                (
                    ("Host", _host(proxy)),
                    ("Authorization", f"Bearer {MASTER_KEY}"),
                    ("Content-Type", "application/json"),
                    ("Transfer-Encoding", "chunked"),
                ),
                framed,
                version="HTTP/1.0",
            ),
        )

    answers: Final = rig.both(send)
    assert (answers.rust.status, answers.rust.body) == (400, b"")
    assert (answers.python.status, answers.python.body) != (400, b"")
