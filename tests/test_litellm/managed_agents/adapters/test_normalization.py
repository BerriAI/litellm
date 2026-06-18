"""Unit tests for the opencode normalization helpers.

Covers `normalize_opencode_message` and `normalize_opencode_event` per
the v2 contract §7 (`.claude/v2_api_contract.md`) and against the REAL
opencode 1.14.41 wire shapes documented in
`.claude/v2_opencode_real_responses.md`.

Coverage:
  - Message: nested ``{info, parts}`` envelope; epoch-ms timestamps;
    camelCase ``providerID``/``modelID`` combined into our ``model``;
    text-only / mixed text+tool / completed absent-vs-present /
    tool error → failed; opencode session id never leaks; id passthrough.
  - Event: ``message.updated`` → ``message.started`` (new) /
    ``message.completed`` (with ``info.time.completed``);
    ``message.part.updated`` populates the ``part_types`` map and emits
    ``message.tool.started`` / ``.completed`` based on ``state.status``;
    ``message.part.delta`` routes via ``part_types`` to
    ``message.text.delta`` (or drops for reasoning); ``session.idle``
    emits a ``message.completed``; ``session.error`` extracts the nested
    ``error.data.message``; unknown types return None.

These tests are pure: no HTTP, no DB, no async. They exercise dict-in /
object-out behavior of the normalization layer only.
"""

from datetime import datetime
from typing import Any, Dict

import pytest

normalization = pytest.importorskip("litellm.managed_agents.adapters.normalization")

normalize_opencode_message = normalization.normalize_opencode_message
normalize_opencode_event = normalization.normalize_opencode_event


# Real opencode timestamp from the preflight (epoch ms).
_TS_CREATED_MS = 1778172682972
_TS_COMPLETED_MS = 1778172689905


# ---------------------------------------------------------------------------
# normalize_opencode_message
# ---------------------------------------------------------------------------


class TestNormalizeOpencodeMessage:
    """Message normalization (opencode {info, parts} -> our MessageRow)."""

    def test_text_only_message_joins_content_and_omits_tools(self) -> None:
        oc_msg: Dict[str, Any] = {
            "info": {
                "id": "msg_oc_abc",
                "sessionID": "ses_oc_xxx",
                "role": "assistant",
                "providerID": "anthropic",
                "modelID": "claude-3-haiku",
                "time": {
                    "created": _TS_CREATED_MS,
                    "completed": _TS_COMPLETED_MS,
                },
            },
            "parts": [
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "world."},
            ],
        }

        row = normalize_opencode_message(oc_msg, our_session_id="ses_test")

        assert row.content == "Hello world."
        # tools field should be omitted (None) when there are no tool parts.
        assert row.tools is None
        assert row.model == "anthropic/claude-3-haiku"

    def test_filters_out_step_and_reasoning_parts(self) -> None:
        """Only ``text`` parts make it into ``content``. ``reasoning``,
        ``step-start``, ``step-finish`` are excluded for MVP."""
        oc_msg: Dict[str, Any] = {
            "info": {
                "id": "msg_oc_abc",
                "sessionID": "ses_oc_xxx",
                "role": "assistant",
                "providerID": "opencode",
                "modelID": "minimax-m2.5-free",
                "time": {
                    "created": _TS_CREATED_MS,
                    "completed": _TS_COMPLETED_MS,
                },
            },
            "parts": [
                {"type": "step-start", "snapshot": "..."},
                {"type": "reasoning", "text": "thinking out loud..."},
                {"type": "text", "text": "\n\n4"},
                {"type": "step-finish", "reason": "stop"},
            ],
        }

        row = normalize_opencode_message(oc_msg, our_session_id="ses_test")

        assert row.content == "\n\n4"
        assert row.tools is None

    def test_mixed_text_and_tool_parts_extracts_both(self) -> None:
        oc_msg: Dict[str, Any] = {
            "info": {
                "id": "msg_oc_abc",
                "sessionID": "ses_oc_xxx",
                "role": "assistant",
                "providerID": "anthropic",
                "modelID": "claude-3-haiku",
                "time": {
                    "created": _TS_CREATED_MS,
                    "completed": _TS_COMPLETED_MS,
                },
            },
            "parts": [
                {"type": "text", "text": "I'll read the file..."},
                {
                    "type": "tool",
                    "tool": "read",
                    "callID": "call_xxx",
                    "state": {
                        "status": "completed",
                        "input": {"filePath": "src/auth.py"},
                        "output": "...file contents...",
                    },
                },
            ],
        }

        row = normalize_opencode_message(oc_msg, our_session_id="ses_test")

        assert row.content == "I'll read the file..."
        assert row.tools == [
            {
                "name": "read",
                "input": {"filePath": "src/auth.py"},
                "output": "...file contents...",
            }
        ]

    def test_completed_absent_yields_in_progress_status(self) -> None:
        oc_msg: Dict[str, Any] = {
            "info": {
                "id": "msg_oc_abc",
                "sessionID": "ses_oc_xxx",
                "role": "assistant",
                "providerID": "anthropic",
                "modelID": "claude-3-haiku",
                "time": {"created": _TS_CREATED_MS},
            },
            "parts": [{"type": "text", "text": "thinking..."}],
        }

        row = normalize_opencode_message(oc_msg, our_session_id="ses_test")

        assert row.status == "in_progress"
        assert row.completed_at is None

    def test_completed_set_yields_completed_status(self) -> None:
        oc_msg: Dict[str, Any] = {
            "info": {
                "id": "msg_oc_abc",
                "sessionID": "ses_oc_xxx",
                "role": "assistant",
                "providerID": "anthropic",
                "modelID": "claude-3-haiku",
                "time": {
                    "created": _TS_CREATED_MS,
                    "completed": _TS_COMPLETED_MS,
                },
            },
            "parts": [{"type": "text", "text": "Done."}],
        }

        row = normalize_opencode_message(oc_msg, our_session_id="ses_test")

        assert row.status == "completed"
        assert row.completed_at is not None
        assert isinstance(row.completed_at, datetime)

    def test_tool_state_error_yields_failed_status(self) -> None:
        """A tool part with ``state.status == "error"`` overrides ``completed``."""
        oc_msg: Dict[str, Any] = {
            "info": {
                "id": "msg_oc_abc",
                "sessionID": "ses_oc_xxx",
                "role": "assistant",
                "providerID": "anthropic",
                "modelID": "claude-3-haiku",
                "time": {
                    "created": _TS_CREATED_MS,
                    "completed": _TS_COMPLETED_MS,
                },
            },
            "parts": [
                {"type": "text", "text": "tried..."},
                {
                    "type": "tool",
                    "tool": "read",
                    "state": {
                        "status": "error",
                        "input": {"filePath": "/etc/hostname"},
                        "error": "File not found",
                    },
                },
            ],
        }

        row = normalize_opencode_message(oc_msg, our_session_id="ses_test")

        assert row.status == "failed"
        assert row.tools == [
            {
                "name": "read",
                "input": {"filePath": "/etc/hostname"},
                "output": "File not found",
            }
        ]

    def test_session_id_is_our_session_not_opencode_session(self) -> None:
        """The opencode ``sessionID`` MUST NOT leak into our MessageRow."""
        oc_msg: Dict[str, Any] = {
            "info": {
                "id": "msg_oc_abc",
                "sessionID": "ses_oc_NEVER_LEAK",
                "role": "assistant",
                "providerID": "anthropic",
                "modelID": "claude-3-haiku",
                "time": {
                    "created": _TS_CREATED_MS,
                    "completed": _TS_COMPLETED_MS,
                },
            },
            "parts": [{"type": "text", "text": "hi"}],
        }

        row = normalize_opencode_message(oc_msg, our_session_id="ses_ours")

        assert row.session_id == "ses_ours"
        dumped = row.model_dump()
        assert dumped["session_id"] == "ses_ours"
        assert "ses_oc_NEVER_LEAK" not in str(dumped.get("session_id", ""))

    def test_id_is_passed_through_from_info(self) -> None:
        oc_msg: Dict[str, Any] = {
            "info": {
                "id": "msg_oc_specific_id_123",
                "sessionID": "ses_oc_xxx",
                "role": "assistant",
                "providerID": "anthropic",
                "modelID": "claude-3-haiku",
                "time": {
                    "created": _TS_CREATED_MS,
                    "completed": _TS_COMPLETED_MS,
                },
            },
            "parts": [{"type": "text", "text": "hi"}],
        }

        row = normalize_opencode_message(oc_msg, our_session_id="ses_test")

        assert row.id == "msg_oc_specific_id_123"

    def test_user_message_with_no_provider_or_model_yields_none_model(
        self,
    ) -> None:
        """User messages on opencode have ``info.model`` (object) but no
        flat ``providerID``/``modelID``; the normalizer falls back to
        None when neither is present."""
        oc_msg: Dict[str, Any] = {
            "info": {
                "id": "msg_user_1",
                "sessionID": "ses_oc_xxx",
                "role": "user",
                "time": {"created": _TS_CREATED_MS},
            },
            "parts": [{"type": "text", "text": "hi"}],
        }

        row = normalize_opencode_message(oc_msg, our_session_id="ses_test")

        assert row.role == "user"
        assert row.model is None


# ---------------------------------------------------------------------------
# normalize_opencode_event
# ---------------------------------------------------------------------------


class TestNormalizeOpencodeEvent:
    """SSE event normalization (opencode bus -> our event tuples)."""

    def test_message_updated_for_new_assistant_yields_message_started(
        self,
    ) -> None:
        """A ``message.updated`` event without ``time.completed`` =
        in-flight message; emit ``message.started``."""
        raw = {
            "type": "message.updated",
            "properties": {
                "sessionID": "ses_oc_xxx",
                "info": {
                    "id": "msg_a4b5c6",
                    "sessionID": "ses_oc_xxx",
                    "role": "assistant",
                    "time": {"created": _TS_CREATED_MS},
                },
            },
        }

        out = normalize_opencode_event(raw)

        assert out is not None
        event_type, data = out
        assert event_type == "message.started"
        assert data == {"message_id": "msg_a4b5c6", "role": "assistant"}

    def test_message_updated_with_completed_yields_message_completed(self) -> None:
        """When ``info.time.completed`` is set on an assistant message,
        we emit ``message.completed`` with the per-message id."""
        raw = {
            "type": "message.updated",
            "properties": {
                "sessionID": "ses_oc_xxx",
                "info": {
                    "id": "msg_a4b5c6",
                    "sessionID": "ses_oc_xxx",
                    "role": "assistant",
                    "time": {
                        "created": _TS_CREATED_MS,
                        "completed": _TS_COMPLETED_MS,
                    },
                },
            },
        }

        out = normalize_opencode_event(raw)

        assert out is not None
        event_type, data = out
        assert event_type == "message.completed"
        assert data["message_id"] == "msg_a4b5c6"
        assert data["completed_at"] is not None
        # ISO 8601 UTC string with 'Z' suffix.
        assert data["completed_at"].endswith("Z")

    def test_message_part_updated_text_populates_part_types_map(self) -> None:
        """A text ``message.part.updated`` doesn't emit a normalized
        event itself — but it MUST register the ``partID -> "text"``
        mapping for subsequent deltas."""
        raw = {
            "type": "message.part.updated",
            "properties": {
                "sessionID": "ses_oc_xxx",
                "part": {
                    "id": "prt_text_1",
                    "type": "text",
                    "messageID": "msg_a4b5c6",
                    "sessionID": "ses_oc_xxx",
                    "text": "",
                },
            },
        }

        part_types: Dict[str, str] = {}
        out = normalize_opencode_event(raw, part_types=part_types)

        assert out is None
        assert part_types == {"prt_text_1": "text"}

    def test_message_part_delta_routes_text_to_text_delta(self) -> None:
        """``message.part.delta`` looks up the partID in ``part_types``
        and emits ``message.text.delta`` only for text parts."""
        part_types = {"prt_text_1": "text"}
        raw = {
            "type": "message.part.delta",
            "properties": {
                "sessionID": "ses_oc_xxx",
                "messageID": "msg_a4b5c6",
                "partID": "prt_text_1",
                "field": "text",
                "delta": "Hello",
            },
        }

        out = normalize_opencode_event(raw, part_types=part_types)

        assert out is not None
        event_type, data = out
        assert event_type == "message.text.delta"
        assert data["message_id"] == "msg_a4b5c6"
        assert data["delta"] == "Hello"

    def test_message_part_delta_drops_reasoning_chunks(self) -> None:
        """Reasoning deltas are intentionally dropped for MVP."""
        part_types = {"prt_reasoning_1": "reasoning"}
        raw = {
            "type": "message.part.delta",
            "properties": {
                "sessionID": "ses_oc_xxx",
                "messageID": "msg_a4b5c6",
                "partID": "prt_reasoning_1",
                "field": "text",
                "delta": "the user is asking...",
            },
        }

        out = normalize_opencode_event(raw, part_types=part_types)

        assert out is None

    def test_message_part_delta_with_unknown_partid_drops(self) -> None:
        """A delta whose partID was never seen in a prior
        ``message.part.updated`` cannot be routed and must be dropped
        rather than guessed."""
        raw = {
            "type": "message.part.delta",
            "properties": {
                "sessionID": "ses_oc_xxx",
                "messageID": "msg_a4b5c6",
                "partID": "prt_unknown",
                "field": "text",
                "delta": "??",
            },
        }

        out = normalize_opencode_event(raw, part_types={})

        assert out is None

    def test_message_part_updated_tool_running_yields_tool_started(self) -> None:
        raw = {
            "type": "message.part.updated",
            "properties": {
                "sessionID": "ses_oc_xxx",
                "part": {
                    "id": "prt_tool_1",
                    "type": "tool",
                    "messageID": "msg_a4b5c6",
                    "sessionID": "ses_oc_xxx",
                    "tool": "read",
                    "callID": "call_xxx",
                    "state": {
                        "status": "running",
                        "input": {"filePath": "src/auth.py"},
                    },
                },
            },
        }

        part_types: Dict[str, str] = {}
        out = normalize_opencode_event(raw, part_types=part_types)

        assert out is not None
        event_type, data = out
        assert event_type == "message.tool.started"
        assert data["message_id"] == "msg_a4b5c6"
        assert data["tool"] == "read"
        assert data["input"] == {"filePath": "src/auth.py"}
        assert part_types == {"prt_tool_1": "tool"}

    def test_message_part_updated_tool_completed_yields_tool_completed(
        self,
    ) -> None:
        raw = {
            "type": "message.part.updated",
            "properties": {
                "sessionID": "ses_oc_xxx",
                "part": {
                    "id": "prt_tool_1",
                    "type": "tool",
                    "messageID": "msg_a4b5c6",
                    "sessionID": "ses_oc_xxx",
                    "tool": "read",
                    "callID": "call_xxx",
                    "state": {
                        "status": "completed",
                        "input": {"filePath": "src/auth.py"},
                        "output": "file contents here",
                    },
                },
            },
        }

        out = normalize_opencode_event(raw, part_types={})

        assert out is not None
        event_type, data = out
        assert event_type == "message.tool.completed"
        assert data["message_id"] == "msg_a4b5c6"
        assert data["tool"] == "read"
        assert data["output"] == "file contents here"
        assert "error" not in data

    def test_message_part_updated_tool_error_yields_tool_completed_with_error_flag(
        self,
    ) -> None:
        raw = {
            "type": "message.part.updated",
            "properties": {
                "sessionID": "ses_oc_xxx",
                "part": {
                    "id": "prt_tool_1",
                    "type": "tool",
                    "messageID": "msg_a4b5c6",
                    "sessionID": "ses_oc_xxx",
                    "tool": "read",
                    "callID": "call_xxx",
                    "state": {
                        "status": "error",
                        "input": {"filePath": "/etc/hostname"},
                        "error": "File not found: /etc/hostname",
                    },
                },
            },
        }

        out = normalize_opencode_event(raw, part_types={})

        assert out is not None
        event_type, data = out
        assert event_type == "message.tool.completed"
        assert data["error"] is True
        assert data["output"] == "File not found: /etc/hostname"

    def test_message_part_updated_tool_pending_drops(self) -> None:
        """``pending`` is the brief state before opencode populates the
        tool input; we wait for ``running`` (which has the input) before
        emitting our ``tool.started`` event."""
        raw = {
            "type": "message.part.updated",
            "properties": {
                "sessionID": "ses_oc_xxx",
                "part": {
                    "id": "prt_tool_1",
                    "type": "tool",
                    "messageID": "msg_a4b5c6",
                    "tool": "read",
                    "state": {"status": "pending", "input": {}},
                },
            },
        }

        out = normalize_opencode_event(raw, part_types={})

        assert out is None

    def test_session_idle_yields_message_completed(self) -> None:
        """``session.idle`` payload only carries ``sessionID`` — the
        adapter is expected to enrich ``message_id`` from tracked state.
        Bare normalizer output has ``message_id == None``."""
        raw = {
            "type": "session.idle",
            "properties": {"sessionID": "ses_oc_xxx"},
        }

        out = normalize_opencode_event(raw)

        assert out is not None
        event_type, data = out
        assert event_type == "message.completed"
        assert data["message_id"] is None
        assert data["completed_at"] is None

    def test_session_error_extracts_nested_message(self) -> None:
        """Real opencode shape: ``properties.error.data.message``."""
        raw = {
            "type": "session.error",
            "properties": {
                "sessionID": "ses_oc_xxx",
                "error": {
                    "name": "UnknownError",
                    "data": {"message": "Model not found: anthropic/claude-3-haiku."},
                },
            },
        }

        out = normalize_opencode_event(raw)

        assert out is not None
        event_type, data = out
        assert event_type == "error"
        assert data["error"] == "Model not found: anthropic/claude-3-haiku."

    @pytest.mark.parametrize(
        "raw",
        [
            {"type": "permission.updated", "properties": {}},
            {"type": "lsp.client.diagnostics", "properties": {}},
            {"type": "totally.unknown.event", "properties": {}},
            {"type": "", "properties": {}},
            {"type": "server.heartbeat", "properties": {}},
            {"type": "session.created", "properties": {}},
        ],
    )
    def test_unknown_event_types_return_none(self, raw: Dict[str, Any]) -> None:
        assert normalize_opencode_event(raw) is None
