"""Spend parity: the `LiteLLM_SpendLogs` row and key spend written for the same call in both process models.

Contract: the row a call writes depends only on what Python saw, so with the same config, the same scripted
provider (5 prompt and 3 completion tokens at 0.001 and 0.002 per token) and the same raw virtual key, the row must
have the same `call_type`, hashed `api_key`, `spend`, token counts, `model`, `model_id`, `model_group`,
`custom_llm_provider`, `api_base`, `user`, `end_user`, `team_id`, `request_tags`, `requester_ip_address`, `status`
and `cache_hit` in both modes (`SPEND_COLUMNS` in the support module).

The address is the part Rust can break: Python's TCP peer behind Rust is always the forwarder on loopback, and only
`RustPeerMiddleware` (rust_forward/peer.py) puts the real client back, then applies uvicorn's
`ProxyHeadersMiddleware` with the same trust Python alone uses (uvicorn's default `forwarded_allow_ips`,
`127.0.0.1`). `requester_ip_address` comes from `request.client.host` unless `use_x_forwarded_for` is set
(litellm_pre_call_utils.py:3319), so it records what that middleware chain produced.

Each proxy writes to its own scratch Postgres; rows are found by `litellm_call_id`, the `x-litellm-call-id` the
caller saw.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Final

import pytest

from tests.integration._support.client import eventually, held
from tests.integration.configuration.dec1_rustparity_support import (
    CHAT,
    EXPECTED_SPEND,
    MASTER_KEY,
    OPENAI,
    STREAM_CELLS,
    UNARY_CELLS,
    Booted,
    Cell,
    Pair,
    Rig,
    directive,
    marker,
    parity_rig,
    read_until_failure,
    spend_row,
    spend_rows_for,
)

pytestmark: Final = pytest.mark.timeout(900)


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with parity_rig(tmp_path_factory.mktemp("dec1-rustparity-spend")) as booted:
        yield booted


def _rows(rig: Rig, seen_call_ids: Pair[str]) -> Pair[dict[str, object]]:
    return Pair(spend_row(rig.python, seen_call_ids.python), spend_row(rig.rust, seen_call_ids.rust))


@pytest.mark.parametrize("cell", list(UNARY_CELLS.values()), ids=list(UNARY_CELLS))
def test_a_successful_call_writes_the_same_spend_row_in_both_modes(rig: Rig, cell: Cell) -> None:
    """The master key makes the cell's non-streaming call on both proxies. Each proxy must write exactly one spend
    row for the call id it returned, and the rows must be equal on every compared column, including the master
    key's hash in `api_key` and `requester_ip_address`."""
    seen: Final = rig.send(cell.path, cell.body())
    assert (seen.python.status, seen.rust.status) == (200, 200)
    rows: Final = _rows(rig, Pair(seen.python.call_id, seen.rust.call_id))
    assert rows.rust == rows.python


@pytest.mark.parametrize("cell", list(STREAM_CELLS.values()), ids=list(STREAM_CELLS))
def test_a_streamed_call_writes_the_same_spend_row_in_both_modes(rig: Rig, cell: Cell) -> None:
    """The same cell streamed on both proxies. Each must write one spend row once the stream ends, equal on every
    compared column: the usage Python assembles from the chunks is the same because the chunks are."""
    seen: Final = rig.send(cell.path, cell.body(stream=True), stream=True)
    rows: Final = _rows(rig, Pair(seen.python.call_id, seen.rust.call_id))
    assert rows.rust == rows.python


def test_the_spend_row_records_the_loopback_caller_in_both_modes(rig: Rig) -> None:
    """A loopback caller with no forwarding headers makes a chat call on both proxies. Both rows must record
    `requester_ip_address: "127.0.0.1"`: behind Rust that is the address `RustPeerMiddleware` restored from
    `x-litellm-rust-peer`, not an empty value and not the forwarder's own hop."""
    seen: Final = rig.send(CHAT, Cell(CHAT, OPENAI).body())
    rows: Final = _rows(rig, Pair(seen.python.call_id, seen.rust.call_id))
    assert (rows.python["requester_ip_address"], rows.rust["requester_ip_address"]) == ("127.0.0.1", "127.0.0.1")


def test_x_forwarded_for_from_loopback_sets_the_same_requester_ip_in_both_modes(rig: Rig) -> None:
    """A loopback caller sends `X-Forwarded-For: 198.51.100.7`. uvicorn trusts forwarding headers from
    `127.0.0.1` by default, so Python alone records `198.51.100.7`; behind Rust the peer middleware restores
    `127.0.0.1` and then runs the same `ProxyHeadersMiddleware`, so it must record `198.51.100.7` too. A different
    value would silently change IP allowlists and per-IP analytics when Rust is switched on."""
    seen: Final = rig.send(CHAT, Cell(CHAT, OPENAI).body(), headers={"X-Forwarded-For": "198.51.100.7"})
    rows: Final = _rows(rig, Pair(seen.python.call_id, seen.rust.call_id))
    assert (rows.python["requester_ip_address"], rows.rust["requester_ip_address"]) == (
        "198.51.100.7",
        "198.51.100.7",
    )


def test_a_virtual_key_call_records_the_same_key_hash_and_spend_in_both_modes(rig: Rig) -> None:
    """A virtual key created on both proxies with one raw value makes one chat call on each. Both rows must carry
    the same `api_key` hash and `spend` equal to `5 * 0.001 + 3 * 0.002 = 0.011`."""
    key: Final = rig.same_key()
    seen: Final = rig.send(CHAT, Cell(CHAT, OPENAI).body(), key=key)
    rows: Final = _rows(rig, Pair(seen.python.call_id, seen.rust.call_id))
    assert rows.rust["api_key"] == rows.python["api_key"]
    assert (float(str(rows.python["spend"])), float(str(rows.rust["spend"]))) == pytest.approx(
        (EXPECTED_SPEND, EXPECTED_SPEND)
    )


def _key_spend(proxy: Booted, key: str) -> float:
    response: Final = proxy.gateway.client.get(
        "/key/info", params={"key": key}, headers={"Authorization": f"Bearer {MASTER_KEY}"}
    )
    return float(response.json()["info"]["spend"])


def test_the_key_spend_counter_advances_by_the_same_amount_in_both_modes(rig: Rig) -> None:
    """A fresh key with one raw value makes one chat call on each proxy. `GET /key/info` must eventually report
    `spend` 0.011 on both: the spend update queue runs in Python in both modes and must see the call once, not
    zero times and not twice."""
    key: Final = rig.same_key()
    rig.send(CHAT, Cell(CHAT, OPENAI).body(), key=key)
    spends: Final = rig.both(
        lambda proxy: eventually(lambda: _key_spend(proxy, key), lambda spend: spend > 0, seconds=90)
    )
    assert (spends.python, spends.rust) == pytest.approx((EXPECTED_SPEND, EXPECTED_SPEND))


def test_request_tags_and_end_user_are_recorded_the_same_in_both_modes(rig: Rig) -> None:
    """A chat call with `"user": "dec1-rustparity-end-user"` and `metadata.tags: ["dec1-rustparity-tag"]` on both
    proxies. Both rows must record the same `end_user` and `request_tags`, `dec1-rustparity-end-user` and
    `["dec1-rustparity-tag"]`."""
    seen: Final = rig.send(
        CHAT,
        Cell(CHAT, OPENAI).body(user="dec1-rustparity-end-user", metadata={"tags": ["dec1-rustparity-tag"]}),
    )
    rows: Final = _rows(rig, Pair(seen.python.call_id, seen.rust.call_id))
    picked: Final = tuple((row["end_user"], row["request_tags"]) for row in (rows.python, rows.rust))
    assert picked == (("dec1-rustparity-end-user", ["dec1-rustparity-tag"]),) * 2


def test_a_provider_failure_writes_the_same_failure_row_in_both_modes(rig: Rig) -> None:
    """The provider answers 500 on a chat call on both proxies. Each proxy must write one row for the call id it
    returned, with `status: "failure"`, and the two rows must be equal on every compared column.
    UNVERIFIED: that failure rows are written by default (spend_tracking_utils.py builds them from the failure
    callback payload)."""
    seen: Final = rig.send(CHAT, directive(Cell(CHAT, OPENAI).body(), "fail=500"))
    rows: Final = _rows(rig, Pair(seen.python.call_id, seen.rust.call_id))
    assert rows.python["status"] == "failure", rows.python
    assert rows.rust == rows.python


def test_a_call_refused_at_auth_writes_no_spend_row_in_either_mode(rig: Rig) -> None:
    """A chat call with a wrong key on both proxies. For ten seconds neither database may hold a row that mentions
    the call's prompt marker: a refusal before routing is not billed, and Rust must not make Python log the hop."""
    seen: Final = rig.send(CHAT, Cell(CHAT, OPENAI).body(), key="sk-dec1-rustparity-wrong")
    counts: Final = rig.both(
        lambda proxy: len(
            held(
                lambda: spend_rows_for(proxy, seen.python.case if proxy is rig.python else seen.rust.case),
                lambda rows: rows == [],
                holding=10,
                seconds=30,
            )
        )
    )
    assert counts == Pair(0, 0)


def _disconnected_call_id(proxy: Booted) -> str:
    body: Final = Cell(CHAT, OPENAI).body(stream=True)(f"{marker()} endless-stream")
    with proxy.gateway.client.stream(
        "POST", CHAT, json=dict(body), headers={"Authorization": f"Bearer {MASTER_KEY}"}, timeout=60
    ) as response:
        call_id: Final = response.headers.get("x-litellm-call-id", "")
        next(response.iter_raw())
    return call_id


def test_a_stream_the_caller_abandons_is_logged_the_same_way_in_both_modes(rig: Rig) -> None:
    """The provider streams for a minute; the caller reads the first piece and hangs up. Each proxy must write one
    row for the call id it returned, and the rows must be equal on every compared column (status and token counts
    included): Python decides how an abandoned stream is billed, and through Rust it must learn about the hang-up
    the same way, from the closed connection.
    UNVERIFIED: the exact status and whether partial completion tokens are counted."""
    call_ids: Final = rig.both(_disconnected_call_id)
    rows: Final = _rows(rig, call_ids)
    assert rows.rust == rows.python


def test_a_body_read_in_full_by_both_modes_is_billed_once(rig: Rig) -> None:
    """A streamed chat call read to the end on each proxy, then the key info. Each proxy's database must hold
    exactly one row for the returned call id (not one for Rust's hop and one for Python's), with spend 0.011."""
    key: Final = rig.same_key()

    def one(proxy: Booted) -> tuple[int, float]:
        body = Cell(CHAT, OPENAI).body(stream=True)(marker())  # rebind-ok: one per proxy
        with proxy.gateway.client.stream(
            "POST", CHAT, json=dict(body), headers={"Authorization": f"Bearer {key}"}, timeout=60
        ) as response:
            call_id = response.headers.get("x-litellm-call-id", "")  # rebind-ok: one per proxy
            read_until_failure(response)
        rows = eventually(  # rebind-ok: one per proxy
            lambda: proxy.gateway.rows('SELECT spend FROM "LiteLLM_SpendLogs" WHERE litellm_call_id = %s', (call_id,)),
            lambda found: len(found) >= 1,
            seconds=90,
        )
        return len(rows), float(str(rows[0]["spend"]))

    counted: Final = rig.both(one)
    assert (counted.python[0], counted.rust[0]) == (1, 1)
    assert (counted.python[1], counted.rust[1]) == pytest.approx((EXPECTED_SPEND, EXPECTED_SPEND))


def test_the_spend_row_status_is_success_for_a_completed_call_in_both_modes(rig: Rig) -> None:
    """A plain chat call on both proxies. Both rows must say `status: "success"` and `call_type: "acompletion"`:
    the two anchors every spend dashboard filters on."""
    seen: Final = rig.send(CHAT, Cell(CHAT, OPENAI).body())
    rows: Final = _rows(rig, Pair(seen.python.call_id, seen.rust.call_id))
    assert (
        tuple((row["status"], row["call_type"]) for row in (rows.python, rows.rust))
        == (("success", "acompletion"),) * 2
    )
