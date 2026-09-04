"""Harness coverage for realtime_client.RealtimeSession.collect_until.

Not an e2e test (no live proxy): it drives the collect loop with a scripted
connection injected in place of the websocket, so it runs regardless of whether a
proxy is up. The behavior under test is the diagnostic contract: when the provider
emits an `error` event (OpenAI realtime quota rejection is the motivating case)
while a test waits for a lifecycle or response event, the wait must fail fast and
surface the provider's reason, not time out with the cause discarded.
"""

from collections.abc import Iterator
from dataclasses import dataclass

import pytest

from realtime_client import RealtimeError, RealtimeSession, error_summary

QUOTA_ERROR = (
    '{"type":"error","event_id":"event_1","error":{"type":"insufficient_quota",'
    '"code":"insufficient_quota","message":"You exceeded your current quota, '
    'please check your plan and billing details."}}'
)
SESSION_CREATED = '{"type":"session.created","session":{"id":"sess_1"}}'


@dataclass(frozen=True, slots=True)
class _ScriptedConnection:
    """A WebSocketConnection that replays a fixed list of frames, then behaves like
    a silent socket (recv times out). send is a no-op; the collect loop only recvs."""

    frames: Iterator[str]

    def send(self, message: str) -> None:
        return None

    def recv(self, timeout: float | None = None) -> str:
        try:
            return next(self.frames)
        except StopIteration:
            raise TimeoutError("no more frames") from None


def _session(*frames: str) -> RealtimeSession:
    return RealtimeSession(connection=_ScriptedConnection(iter(frames)))


def test_collect_until_raises_realtime_error_on_error_event() -> None:
    session = _session(QUOTA_ERROR)

    with pytest.raises(RealtimeError) as excinfo:
        session.collect_until("session.created", timeout=20)

    message = str(excinfo.value)
    assert "insufficient_quota" in message
    assert "exceeded your current quota" in message
    assert "session.created" in message
    assert excinfo.value.payload == QUOTA_ERROR


def test_collect_until_reports_events_seen_before_the_error() -> None:
    session = _session(SESSION_CREATED, QUOTA_ERROR)

    with pytest.raises(RealtimeError) as excinfo:
        session.collect_until("response.done", timeout=20)

    assert "session.created" in str(excinfo.value)


def test_collect_until_returns_on_stop_event_without_raising() -> None:
    session = _session(SESSION_CREATED)

    events = session.collect_until("session.created", timeout=20)

    assert [e.type for e in events] == ["session.created"]


def test_collect_until_times_out_when_no_stop_or_error_arrives() -> None:
    session = _session(SESSION_CREATED)

    with pytest.raises(TimeoutError) as excinfo:
        session.collect_until("session.updated", timeout=5)

    assert "session.updated" in str(excinfo.value)


def test_error_summary_extracts_openai_quota_reason() -> None:
    assert error_summary(QUOTA_ERROR) == (
        "insufficient_quota: You exceeded your current quota, "
        "please check your plan and billing details."
    )


def test_error_summary_falls_back_to_raw_payload_when_not_normalized() -> None:
    payload = '{"type":"error","message":"boom"}'
    assert error_summary(payload) == payload
