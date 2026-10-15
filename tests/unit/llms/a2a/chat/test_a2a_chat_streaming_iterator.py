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
        pytest.param(["OK", "OK"], "OK", id="one_delta_then_equal_snapshot"),
        pytest.param(["Hello ", "Hello world"], "Hello world", id="snapshot_keeps_emitted_space_once"),
        pytest.param(["OK\n", "OK\n"], "OK\n", id="snapshot_keeps_emitted_newline_once"),
        pytest.param(
            ["Hello ", "Hello \n\nworld"],
            "Hello \n\nworld",
            id="snapshot_keeps_its_own_new_whitespace",
        ),
        # Known limitation: A2A marks no event as delta-or-snapshot, so a delta that
        # exactly reproduces the accumulated text is indistinguishable from a snapshot
        # and collapses. Duplicating a whole reply is the worse failure of the two.
        pytest.param(["a", "a", "a"], "a", id="identical_deltas_collapse"),
        pytest.param(["Hello", "world", "Hello world"], "Helloworld", id="multipart_snapshot_respaced"),
        pytest.param(
            ["Hello", "world", "Hello world again"],
            "Helloworld again",
            id="multipart_snapshot_extends",
        ),
        pytest.param(["", "OK", ""], "OK", id="empty_events_ignored"),
        # A whitespace-only delta carries no non-whitespace text, so it must not be
        # mistaken for a snapshot repeating the tail of the stream.
        pytest.param(["Hello", " "], "Hello ", id="whitespace_only_delta_kept"),
        pytest.param(["chatter", "answer", "answer"], "chatteranswer", id="snapshot_repeats_only_tail"),
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


# A one-token reply: a single delta followed by the terminal cumulative snapshot. This is
# the ordinary shape of a short A2A answer, so the snapshot must not be forwarded again.
SINGLE_DELTA_STREAM = [
    _status_update(text="Reply with exactly: OK", role="user", state="submitted"),
    _status_update(text="OK"),
    _artifact_update("OK"),
    _status_update(state="completed", final=True),
]


def test_single_delta_then_snapshot_is_not_duplicated():
    """Regression: snapshot suppression must not depend on how many deltas preceded it."""
    iterator = _iterator()
    assert "".join(iterator.chunk_parser(e)["text"] for e in SINGLE_DELTA_STREAM) == "OK"


# A delegating agent reports sub-agent progress into the same task before producing its own
# answer. Modelled on kagent's ADK executor, which enqueues a status-update for every ADK event
# and then re-sends the aggregated answer as BOTH a terminal status-update and a final artifact
# (kagent-adk/src/kagent/adk/_agent_executor.py: run loop, then task result publication).
ANSWER = "Diagnosis: optic degraded on emm001a-jnx-01."

DELEGATING_AGENT_STREAM = [
    _status_update(text="Diagnose the optical alarm on emm001a-jnx-01.", role="user", state="submitted"),
    _status_update(text="Calling telemetry-agent..."),
    _status_update(text="telemetry-agent: no anomalies found."),
    _status_update(text="Diagnosis: optic degraded"),
    _status_update(text=" on emm001a-jnx-01."),
    _status_update(text=ANSWER),
    _artifact_update(ANSWER),
    _status_update(state="completed", final=True),
]


def test_delegating_agent_answer_is_not_repeated_after_tool_chatter():
    """
    Regression: a terminal snapshot repeats only the agent's answer, not the progress text
    streamed ahead of it, so snapshot suppression cannot depend on the snapshot extending
    everything emitted so far.

    Without this, a delegating agent renders its answer three times: once from the deltas,
    once from the terminal status-update and once from the final artifact.
    """
    iterator = _iterator()
    rendered = "".join(iterator.chunk_parser(e)["text"] for e in DELEGATING_AGENT_STREAM)

    assert rendered.count(ANSWER) == 1
    assert rendered == f"Calling telemetry-agent...telemetry-agent: no anomalies found.{ANSWER}"
