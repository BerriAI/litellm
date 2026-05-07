"""Unit tests for the opencode normalization helpers.

Covers `normalize_opencode_message` and `normalize_opencode_event` per the
v2 contract §7 (`.claude/v2_api_contract.md`):

  - Message: text-only / text+tool / completedAt absent/present / error part
    / our_session_id preserved / id pass-through.
  - Event: message.updated -> message.started; message.part.updated text ->
    message.text.delta; message.part.updated tool start/finish ->
    message.tool.started/completed; session.idle -> message.completed;
    session.error -> error; unknown types -> None.

These tests are pure: no HTTP, no DB, no async. They exercise dict-in /
object-out behavior of the normalization layer only.
"""

from datetime import datetime
from typing import Any, Dict

import pytest

# The normalization module is being built in parallel by the adapter agent.
# Skip cleanly if it isn't here yet so the rest of the test suite still runs.
normalization = pytest.importorskip(
    "litellm.managed_agents.adapters.normalization"
)

normalize_opencode_message = normalization.normalize_opencode_message
normalize_opencode_event = normalization.normalize_opencode_event


# ---------------------------------------------------------------------------
# normalize_opencode_message
# ---------------------------------------------------------------------------


class TestNormalizeOpencodeMessage:
    """Message normalization (opencode -> our MessageRow)."""

    def test_text_only_message_joins_content_and_omits_tools(self) -> None:
        oc_msg: Dict[str, Any] = {
            "id": "msg_oc_abc",
            "sessionID": "oc_sid_xxx",
            "role": "assistant",
            "parts": [
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "world."},
            ],
            "completedAt": "2026-05-07T15:04:05.123Z",
        }

        row = normalize_opencode_message(oc_msg, our_session_id="ses_test")

        assert row.content == "Hello world."
        # tools field should be omitted (None) when there are no tool parts.
        assert row.tools is None

    def test_mixed_text_and_tool_parts_extracts_both(self) -> None:
        oc_msg: Dict[str, Any] = {
            "id": "msg_oc_abc",
            "sessionID": "oc_sid_xxx",
            "role": "assistant",
            "parts": [
                {"type": "text", "text": "I'll read the file..."},
                {
                    "type": "tool",
                    "name": "read",
                    "input": {"path": "src/auth.py"},
                    "output": "...file contents...",
                },
            ],
            "completedAt": "2026-05-07T15:04:05.123Z",
        }

        row = normalize_opencode_message(oc_msg, our_session_id="ses_test")

        assert row.content == "I'll read the file..."
        assert row.tools == [
            {
                "name": "read",
                "input": {"path": "src/auth.py"},
                "output": "...file contents...",
            }
        ]

    def test_completed_at_null_yields_in_progress_status(self) -> None:
        oc_msg: Dict[str, Any] = {
            "id": "msg_oc_abc",
            "sessionID": "oc_sid_xxx",
            "role": "assistant",
            "parts": [{"type": "text", "text": "thinking..."}],
            "completedAt": None,
        }

        row = normalize_opencode_message(oc_msg, our_session_id="ses_test")

        assert row.status == "in_progress"
        assert row.completed_at is None

    def test_completed_at_set_yields_completed_status(self) -> None:
        oc_msg: Dict[str, Any] = {
            "id": "msg_oc_abc",
            "sessionID": "oc_sid_xxx",
            "role": "assistant",
            "parts": [{"type": "text", "text": "Done."}],
            "completedAt": "2026-05-07T15:04:05.123Z",
        }

        row = normalize_opencode_message(oc_msg, our_session_id="ses_test")

        assert row.status == "completed"
        assert row.completed_at is not None
        assert isinstance(row.completed_at, datetime)

    def test_error_part_yields_failed_status(self) -> None:
        oc_msg: Dict[str, Any] = {
            "id": "msg_oc_abc",
            "sessionID": "oc_sid_xxx",
            "role": "assistant",
            "parts": [
                {"type": "text", "text": "tried..."},
                {"type": "error", "error": "boom"},
            ],
            # Even with completedAt set, an error part should win.
            "completedAt": "2026-05-07T15:04:05.123Z",
        }

        row = normalize_opencode_message(oc_msg, our_session_id="ses_test")

        assert row.status == "failed"

    def test_session_id_is_our_session_not_opencode_session(self) -> None:
        """The opencode `sessionID` MUST NOT leak into our MessageRow."""
        oc_msg: Dict[str, Any] = {
            "id": "msg_oc_abc",
            "sessionID": "oc_sid_NEVER_LEAK",
            "role": "assistant",
            "parts": [{"type": "text", "text": "hi"}],
            "completedAt": "2026-05-07T15:04:05.123Z",
        }

        row = normalize_opencode_message(oc_msg, our_session_id="ses_ours")

        assert row.session_id == "ses_ours"
        # Defensive: the opencode session id must not appear in serialization.
        dumped = row.model_dump()
        assert dumped["session_id"] == "ses_ours"
        assert "oc_sid_NEVER_LEAK" not in str(dumped.get("session_id", ""))

    def test_id_is_passed_through_from_opencode(self) -> None:
        oc_msg: Dict[str, Any] = {
            "id": "msg_oc_specific_id_123",
            "sessionID": "oc_sid_xxx",
            "role": "assistant",
            "parts": [{"type": "text", "text": "hi"}],
            "completedAt": "2026-05-07T15:04:05.123Z",
        }

        row = normalize_opencode_message(oc_msg, our_session_id="ses_test")

        assert row.id == "msg_oc_specific_id_123"


# ---------------------------------------------------------------------------
# normalize_opencode_event
# ---------------------------------------------------------------------------


class TestNormalizeOpencodeEvent:
    """SSE event normalization (opencode bus -> our event tuples)."""

    def test_message_updated_yields_message_started(self) -> None:
        raw = {
            "type": "message.updated",
            "properties": {
                "info": {"id": "msg_a4b5c6", "role": "assistant"},
            },
        }

        out = normalize_opencode_event(raw)

        assert out is not None
        event_type, data = out
        assert event_type == "message.started"
        assert data == {"message_id": "msg_a4b5c6", "role": "assistant"}

    def test_message_part_updated_text_yields_text_delta(self) -> None:
        raw = {
            "type": "message.part.updated",
            "properties": {
                "part": {
                    "type": "text",
                    "messageID": "msg_a4b5c6",
                    "text": "Hello",
                },
            },
        }

        out = normalize_opencode_event(raw)

        assert out is not None
        event_type, data = out
        assert event_type == "message.text.delta"
        assert data["message_id"] == "msg_a4b5c6"
        assert data["delta"] == "Hello"

    def test_message_part_updated_tool_start_yields_tool_started(self) -> None:
        raw = {
            "type": "message.part.updated",
            "properties": {
                "part": {
                    "type": "tool",
                    "messageID": "msg_a4b5c6",
                    "name": "read",
                    "input": {"path": "src/auth.py"},
                    # No `output` key yet — tool is in flight.
                },
            },
        }

        out = normalize_opencode_event(raw)

        assert out is not None
        event_type, data = out
        assert event_type == "message.tool.started"
        assert data["message_id"] == "msg_a4b5c6"
        assert data["tool"] == "read"
        assert data["input"] == {"path": "src/auth.py"}

    def test_message_part_updated_tool_finish_yields_tool_completed(
        self,
    ) -> None:
        raw = {
            "type": "message.part.updated",
            "properties": {
                "part": {
                    "type": "tool",
                    "messageID": "msg_a4b5c6",
                    "name": "read",
                    "input": {"path": "src/auth.py"},
                    "output": "file contents here",
                },
            },
        }

        out = normalize_opencode_event(raw)

        assert out is not None
        event_type, data = out
        assert event_type == "message.tool.completed"
        assert data["message_id"] == "msg_a4b5c6"
        assert data["tool"] == "read"
        assert data["output"] == "file contents here"

    def test_session_idle_yields_message_completed(self) -> None:
        raw = {
            "type": "session.idle",
            "properties": {
                "messageID": "msg_a4b5c6",
                "content": "I'm done.",
                "completedAt": "2026-05-07T15:04:05.123Z",
            },
        }

        out = normalize_opencode_event(raw)

        assert out is not None
        event_type, data = out
        assert event_type == "message.completed"
        assert data["message_id"] == "msg_a4b5c6"
        assert data["content"] == "I'm done."
        assert data["completed_at"] == "2026-05-07T15:04:05.123Z"

    def test_session_error_yields_error(self) -> None:
        raw = {
            "type": "session.error",
            "properties": {
                "messageID": "msg_a4b5c6",
                "error": "model unavailable",
            },
        }

        out = normalize_opencode_event(raw)

        assert out is not None
        event_type, data = out
        assert event_type == "error"
        assert data["message_id"] == "msg_a4b5c6"
        assert data["error"] == "model unavailable"

    @pytest.mark.parametrize(
        "raw",
        [
            {"type": "permission.updated", "properties": {}},
            {"type": "lsp.client.diagnostics", "properties": {}},
            {"type": "totally.unknown.event", "properties": {}},
            {"type": "", "properties": {}},
        ],
    )
    def test_unknown_event_types_return_none(self, raw: Dict[str, Any]) -> None:
        assert normalize_opencode_event(raw) is None
