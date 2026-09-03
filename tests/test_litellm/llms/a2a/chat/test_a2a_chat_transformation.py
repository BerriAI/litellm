"""Tests for the A2A chat bridge: litellm/llms/a2a/chat/ response transform and streaming."""

from unittest.mock import MagicMock

import pytest

from litellm.llms.a2a.chat.streaming_iterator import A2AModelResponseIterator
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
