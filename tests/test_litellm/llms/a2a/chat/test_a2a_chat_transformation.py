"""Tests for the A2A chat bridge: litellm/llms/a2a/chat/ response transform and streaming."""

from unittest.mock import MagicMock

import pytest

from litellm.llms.a2a.chat.streaming_iterator import A2AModelResponseIterator
from litellm.llms.a2a.common_utils import extract_text_from_a2a_response
from litellm.llms.a2a.chat.transformation import A2AConfig
from litellm.types.utils import ModelResponse


def _raw_response(text: str) -> MagicMock:
    raw = MagicMock()
    raw.status_code = 200
    raw.headers = {}
    raw.json.return_value = {
        "jsonrpc": "2.0",
        "id": "resp-1",
        "result": {
            "kind": "message",
            "parts": [{"kind": "text", "text": text}],
        },
    }
    return raw


def test_transform_response_sets_usage():
    """Regression: A2AConfig.transform_response must populate usage so per-token
    pricing computes real cost and callers don't get usage 0/0/0."""
    result = A2AConfig().transform_response(
        model="a2a/test-agent",
        raw_response=_raw_response("hello from the agent"),
        model_response=ModelResponse(),
        logging_obj=MagicMock(),
        request_data={},
        messages=[{"role": "user", "content": "hi there agent"}],
        optional_params={},
        litellm_params={},
        encoding=None,
    )

    assert result.usage is not None
    assert result.usage.prompt_tokens > 0
    assert result.usage.completion_tokens > 0
    assert result.usage.total_tokens == (result.usage.prompt_tokens + result.usage.completion_tokens)


def _iterator() -> A2AModelResponseIterator:
    return A2AModelResponseIterator(streaming_response=iter(()), sync_stream=True)


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


def _task(*, status_role: str | None = None, status_text: str = "", artifact_text: str | None = None) -> dict:
    result: dict = {"kind": "task"}
    if status_role is not None:
        result["status"] = {
            "state": "completed",
            "message": {"role": status_role, "parts": [{"kind": "text", "text": status_text}]},
        }
    if artifact_text is not None:
        result["artifacts"] = [{"parts": [{"kind": "text", "text": artifact_text}]}]
    return {"jsonrpc": "2.0", "id": "1", "result": result}


@pytest.mark.parametrize(
    "response, expected",
    [
        # Regression: a task echoing the caller in status.message while carrying the real
        # answer in artifacts must not come back empty. Skipping a caller-authored message
        # has to fall through to the agent's own output, not short-circuit extraction.
        pytest.param(
            _task(status_role="user", status_text="check active alarms", artifact_text="THE ANSWER"),
            "THE ANSWER",
            id="user_echo_falls_through_to_artifacts",
        ),
        pytest.param(
            _task(status_role="agent", status_text="agent status", artifact_text="THE ANSWER"),
            "agent status",
            id="agent_status_message_preferred",
        ),
        pytest.param(_task(artifact_text="THE ANSWER"), "THE ANSWER", id="artifacts_only"),
        pytest.param(_task(status_role="user", status_text="echo"), "", id="user_echo_alone_is_empty"),
        pytest.param(
            {"result": {"kind": "message", "role": "user", "parts": [{"kind": "text", "text": "echo"}]}},
            "",
            id="direct_user_message_is_empty",
        ),
        pytest.param(
            {"result": {"kind": "message", "role": "agent", "parts": [{"kind": "text", "text": "hi"}]}},
            "hi",
            id="direct_agent_message",
        ),
    ],
)
def test_extract_text_skips_caller_authored_messages(response, expected):
    """Caller-authored parts are skipped, never returned, and never hide the agent's reply."""
    assert extract_text_from_a2a_response(response) == expected
