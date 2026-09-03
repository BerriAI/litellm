"""Tests for litellm/llms/a2a/chat/streaming_iterator.py."""

from collections.abc import Iterable

import pytest

from litellm.llms.a2a.chat.streaming_iterator import A2AModelResponseIterator
from litellm.llms.a2a.common_utils import A2AError


def _iterator(lines: Iterable[str] = ()) -> A2AModelResponseIterator:
    return A2AModelResponseIterator(streaming_response=iter(lines), sync_stream=True)


def _status_update(
    *,
    text: str | None = None,
    role: str = "agent",
    state: str = "working",
    final: bool = False,
) -> dict:
    status: dict = {"state": state}
    if text is not None:
        status["message"] = {
            "kind": "message",
            "role": role,
            "parts": [{"kind": "text", "text": text}],
        }
    return {
        "jsonrpc": "2.0",
        "id": "1",
        "result": {"kind": "status-update", "final": final, "status": status},
    }


def _artifact_update(*texts: str, append: bool = False) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": "1",
        "result": {
            "kind": "artifact-update",
            "append": append,
            "lastChunk": True,
            "artifact": {"parts": [{"kind": "text", "text": text} for text in texts]},
        },
    }


def test_a_jsonrpc_error_in_the_stream_fails_the_call():
    """An agent that answers message/stream with a JSON-RPC error (Microsoft Foundry replies -32004
    "operation not supported") must fail the call with that message instead of ending an empty stream."""
    iterator = _iterator(
        ['{"jsonrpc":"2.0","id":"1","error":{"code":-32004,"message":"This operation is not supported"}}']
    )

    with pytest.raises(A2AError, match="This operation is not supported"):
        next(iterator)


def test_a_completed_task_chunk_yields_its_text_and_stops():
    iterator = _iterator(
        [
            '{"jsonrpc":"2.0","id":"1","result":{"kind":"task","status":{"state":"completed"},'
            '"artifacts":[{"parts":[{"kind":"text","text":"7"}]}]}}'
        ]
    )

    chunk = next(iterator)

    assert chunk["text"] == "7"
    assert chunk["is_finished"] is True
    assert chunk["finish_reason"] == "stop"


# Event sequence captured from a real kagent A2A agent replying "OK": the submitted
# status-update echoes the caller's own message, then two true deltas are followed by two
# cumulative snapshots of the whole reply.
KAGENT_OK_STREAM = [
    _status_update(text="Reply with exactly: OK", role="user", state="submitted"),
    _status_update(),
    _status_update(text="O"),
    _status_update(text="K"),
    _status_update(text="OK"),
    _artifact_update("OK"),
    _status_update(state="completed", final=True),
]


def test_kagent_stream_yields_reply_exactly_once():
    """
    Regression: the caller's echoed message must not be emitted as assistant output, and
    cumulative snapshots must not repeat the reply.

    Previously this stream rendered as "user: Reply with exactly: OKOKOKOK".
    """
    iterator = _iterator()
    assert "".join(iterator.chunk_parser(e)["text"] for e in KAGENT_OK_STREAM) == "OK"


def test_kagent_stream_finishes_on_completed_state():
    iterator = _iterator()
    chunks = [iterator.chunk_parser(e) for e in KAGENT_OK_STREAM]
    assert [c["finish_reason"] for c in chunks if c["is_finished"]] == ["stop"]


@pytest.mark.parametrize(
    "texts, expected",
    [
        pytest.param(["Hello", " world"], "Hello world", id="incremental_deltas"),
        pytest.param(["O", "OK", "OKAY"], "OKAY", id="cumulative_snapshots"),
        pytest.param(["O", "K", "OK"], "OK", id="deltas_then_final_snapshot"),
        pytest.param(["O", "K", "OK", "OK"], "OK", id="deltas_then_repeated_snapshots"),
        pytest.param(["a", "a", "a"], "aaa", id="genuinely_repeated_deltas"),
        pytest.param(["Hello", "world", "Hello world"], "Helloworld", id="multipart_snapshot_respaced"),
        pytest.param(
            ["Hello", "world", "Hello world again"],
            "Helloworld again",
            id="multipart_snapshot_extends",
        ),
        pytest.param(["", "OK", ""], "OK", id="empty_events_ignored"),
    ],
)
def test_incremental_text_reduction(texts, expected):
    """Delta-style and snapshot-style servers must collapse to the same output."""
    iterator = _iterator()
    assert "".join(iterator._to_incremental_text(t) for t in texts) == expected


# A server that chunks its reply into separate delta events but sends the final artifact
# as one multi-part message: A2A joins those parts with a space, so the snapshot reads
# "Hello world" while the deltas accumulated to "Helloworld".
MULTIPART_SNAPSHOT_STREAM = [
    _status_update(text="Hello"),
    _status_update(text="world"),
    _artifact_update("Hello", "world"),
    _status_update(state="completed", final=True),
]


def test_multipart_snapshot_is_not_re_emitted():
    """Regression: whitespace introduced by part-joining must not defeat snapshot detection."""
    iterator = _iterator()
    rendered = "".join(iterator.chunk_parser(e)["text"] for e in MULTIPART_SNAPSHOT_STREAM)
    assert rendered == "Helloworld"
