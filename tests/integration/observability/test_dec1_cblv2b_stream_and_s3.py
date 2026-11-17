"""Per-chunk stream callbacks and the S3 object key layout, two contracts the CustomLoggerV2 ports must keep.

Streams. V2 has no per-chunk hook yet (`Hook.stream_chunk` is not subscribable, hooks.py:118-128, and the
StreamTransformer of the design is not in the code, user-guide.md:12), so chunk logic stays on the legacy
`async_post_call_streaming_hook`. The proxy calls it once per chunk with the text so far, not the chunk, and with
only the caller's auth, not the request (proxy/utils.py:5125-5212, common_request_processing.py:5574-5582). It is a
proxy hook, so `no-log` does not skip it. The probe's `chunk_ledger` records that text with the key alias.

S3. The Rust `s3_v2` logger (#51539, #51893; port spec custom-logger-v2/step1/s3-log-v3.md) writes each payload to
`<s3_path without trailing slashes>/<team alias>/<key alias>/<YYYY-MM-DD>/time-<HH-MM-SS-micros>_<id>.json`
(callbacks-s3-log/src/key.rs:94-128, sink.rs:244-272). Keys longer than 1024 bytes are bounded with SHA-256
digests (key.rs:113-127). Round 1 (`test_dec1_cblv2_s3_objects.py`) pins the default layout, the team prefix and
payload fields; this file pins the trailing slash, the key-only prefix and the 1024 byte bound.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterator
from itertools import pairwise
from typing import Final

import pytest
from integration._support.client import Scenario, eventually
from integration.observability.dec1_cblv2_support import (
    CHAT_REJECTED,
    REPLY_TEXT,
    Rig,
    S3Put,
    cblv2_rig,
    chat,
    marker,
    stream_chat,
    streamed_text,
)
from integration.observability.dec1_cblv2b_support import instances, new_key, records
from pydantic import JsonValue

S3B_PATH: Final = "cblv2b-logs"
DATE: Final = r"\d{4}-\d{2}-\d{2}"
CLOCK: Final = r"\d{2}-\d{2}-\d{2}-\d{6}"
CHUNK_HOOK: Final = "async_post_call_streaming_hook"


@pytest.fixture(scope="module")
def streams(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    directory: Final = tmp_path_factory.mktemp("cblv2b-streams")
    with cblv2_rig(directory, {"callbacks": instances("chunk_ledger", "required_reader")}) as booted:
        yield booted


@pytest.fixture(scope="module")
def stream_scenario(streams: Rig) -> Iterator[Scenario]:
    with streams.proxy.scenario() as opened:
        yield opened


@pytest.fixture(scope="module")
def bucket(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    """`s3_v2` with `s3_path: "cblv2b-logs/"` (trailing slash), the key prefix on and the team prefix off."""
    directory: Final = tmp_path_factory.mktemp("cblv2b-s3")
    with cblv2_rig(
        directory,
        {"callbacks": ["s3_v2"]},
        s3_params={"s3_path": f"{S3B_PATH}/", "s3_use_key_prefix": True, "s3_use_team_prefix": False},
    ) as booted:
        yield booted


@pytest.fixture(scope="module")
def bucket_scenario(bucket: Rig) -> Iterator[Scenario]:
    with bucket.proxy.scenario() as opened:
        yield opened


def _chunks(rig: Rig, alias: str) -> tuple[str, ...]:
    """The text-so-far values `chunk_ledger` recorded for the key alias, in call order, once they stop growing."""
    found: Final = eventually(
        lambda: tuple(str(record["so_far"]) for record in rig.mentioning(alias, "chunk_ledger")),
        lambda texts: bool(texts) and texts[-1] == REPLY_TEXT,
        seconds=30,
    )
    return found


def _object(rig: Rig, response_id: str) -> S3Put:
    return eventually(lambda: rig.s3.objects_for(response_id), lambda found: len(found) == 1, seconds=60)[0]


def _object_for_call(rig: Rig, call_id: str) -> S3Put:
    def matching() -> tuple[S3Put, ...]:
        return tuple(put for put in rig.s3.recorded() if any(line.get("id") == call_id for line in put.lines()))

    return eventually(matching, lambda found: len(found) == 1, seconds=60)[0]


def test_chunk_hook_ends_with_the_whole_reply(streams: Rig, stream_scenario: Scenario) -> None:
    """A streamed chat on a fresh key must call the per-chunk hook several times, the last call carrying the whole
    scripted reply "cblv2 scripted reply" as text so far."""
    key, alias, _ = new_key(stream_scenario)
    status, _, _ = stream_chat(streams, marker(), key=key)
    assert status == 200
    texts: Final = _chunks(streams, alias)
    assert len(texts) >= 2
    assert texts[-1] == REPLY_TEXT


def test_chunk_hook_gets_the_text_so_far_not_the_chunk(streams: Rig, stream_scenario: Scenario) -> None:
    """Each recorded text must be a prefix of the next one, and the first non-empty one must be the first scripted
    chunk "cblv2 scri" (the provider splits the reply in two halves): the hook sees `str_so_far + chunk`
    (proxy/utils.py:5197-5201). A port that hands it only the delta breaks every chunk-scanning callback."""
    key, alias, _ = new_key(stream_scenario)
    assert stream_chat(streams, marker(), key=key)[0] == 200
    texts: Final = _chunks(streams, alias)
    assert all(later.startswith(earlier) for earlier, later in pairwise(texts))
    assert next(text for text in texts if text) == REPLY_TEXT[: len(REPLY_TEXT) // 2]


def test_chunk_hook_returning_nothing_leaves_the_stream_unchanged(streams: Rig, stream_scenario: Scenario) -> None:
    """`chunk_ledger` returns None, so the caller's assembled stream text must be exactly the scripted reply."""
    key, _, _ = new_key(stream_scenario)
    status, _, sse = stream_chat(streams, marker(), key=key)
    assert status == 200
    assert streamed_text(sse) == REPLY_TEXT


def test_chunk_hook_is_not_called_for_a_plain_chat(streams: Rig, stream_scenario: Scenario) -> None:
    """A non-streamed chat on a fresh key must never call the per-chunk hook, once the reader has seen the call."""
    key, alias, _ = new_key(stream_scenario)
    text: Final = marker()
    assert chat(streams, text, key=key).status_code == 200
    records(streams, text, "required_reader", "attempt_ended", 1)
    assert records(streams, alias, "chunk_ledger", CHUNK_HOOK, 0) == ()


def test_chunk_hook_still_runs_for_a_no_log_stream(streams: Rig, stream_scenario: Scenario) -> None:
    """The per-chunk hook is a proxy hook, not logging, so a `no-log: true` streamed chat must still reach it and end
    with the whole reply. UNVERIFIED: `stream_chat` sends no extra body fields, so the request is sent directly."""
    key, alias, _ = new_key(stream_scenario)
    body: Final[dict[str, JsonValue]] = {
        "model": "cblv2-chat",
        "messages": [{"role": "user", "content": marker()}],
        "stream": True,
        "no-log": True,
    }
    with streams.proxy.client.stream(
        "POST", "/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {key}"}
    ) as response:
        assert response.status_code == 200
        response.read()
    assert _chunks(streams, alias)[-1] == REPLY_TEXT


def test_s3_path_trailing_slash_is_not_doubled(bucket: Rig, bucket_scenario: Scenario) -> None:
    """With `s3_path: "cblv2b-logs/"` and the key prefix on, a chat's object key must be
    `cblv2b-logs/<key alias>/<date>/time-<clock>_<response id>.json` with a single slash after the path
    (key.rs:104-106 trims trailing slashes)."""
    key, alias, _ = new_key(bucket_scenario)
    response: Final = chat(bucket, marker(), key=key)
    assert response.status_code == 200
    response_id: Final = str(response.json()["id"])
    put: Final = _object(bucket, response_id)
    assert re.fullmatch(
        rf"{S3B_PATH}/{re.escape(alias)}/{DATE}/time-{CLOCK}_{re.escape(response_id)}\.json", put.key
    ), put.key


def test_team_alias_is_left_out_when_only_the_key_prefix_is_on(bucket: Rig, bucket_scenario: Scenario) -> None:
    """A key in a team with an alias, under `s3_use_team_prefix: false` (a boolean, unlike round 1's string) and
    the key prefix on, must be written under the key alias only, with no team alias segment."""
    team_alias: Final = f"cblv2b-team-{uuid.uuid4().hex[:8]}"
    key, alias, _ = new_key(bucket_scenario, team_id=bucket_scenario.team(team_alias=team_alias))
    response: Final = chat(bucket, marker(), key=key)
    assert response.status_code == 200
    put: Final = _object(bucket, str(response.json()["id"]))
    assert put.key.startswith(f"{S3B_PATH}/{alias}/"), put.key
    assert team_alias not in put.key


def test_failure_object_uses_the_same_prefixed_layout(bucket: Rig, bucket_scenario: Scenario) -> None:
    """A provider 400 writes one failure object whose payload id is the `x-litellm-call-id` of the response. Its key
    must follow the same `cblv2b-logs/<key alias>/<date>/time-<clock>_<call id>.json` layout."""
    key, alias, _ = new_key(bucket_scenario)
    response: Final = chat(bucket, marker(), model=CHAT_REJECTED, key=key)
    assert response.status_code == 400
    call_id: Final = response.headers["x-litellm-call-id"]
    put: Final = _object_for_call(bucket, call_id)
    assert re.fullmatch(rf"{S3B_PATH}/{re.escape(alias)}/{DATE}/time-{CLOCK}_{re.escape(call_id)}\.json", put.key), (
        put.key
    )


def test_object_key_is_bounded_to_1024_bytes_for_a_very_long_key_alias(bucket: Rig, bucket_scenario: Scenario) -> None:
    """A key alias of 1100 characters makes the plain key longer than S3's 1024 byte limit. The key must be at most
    1024 bytes, keep the path and the date folder, and end in the 64 hex SHA-256 of the file name
    (`<head>/<16 hex>/<date>/<64 hex>.json`, key.rs:124-127, 145-168). UNVERIFIED: assumes /key/generate accepts an
    alias this long."""
    alias: Final = "cblv2b-" + "a" * 1093
    key: Final = bucket_scenario.key(key_alias=alias)
    response: Final = chat(bucket, marker(), key=key)
    assert response.status_code == 200
    put: Final = _object(bucket, str(response.json()["id"]))
    assert len(put.key.encode()) <= 1024, len(put.key.encode())
    assert put.key.startswith(f"{S3B_PATH}/cblv2b-aaaa"), put.key[:40]
    assert re.search(rf"/[0-9a-f]{{16}}/{DATE}/[0-9a-f]{{64}}\.json$", put.key), put.key[-120:]
