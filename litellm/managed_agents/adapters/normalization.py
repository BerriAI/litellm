"""Pure normalization helpers for the opencode adapter.

These functions translate opencode wire shapes into our normalized
`MessageRow` and SSE event tuples per contract §7. They have NO side
effects, NO HTTP, NO DB calls — pure dict-in / object-out.

The opencode wire shapes documented in the contract are best-effort. The
preflight agent (running in parallel) will document the real payload
shapes; defensive `.get()` access is used everywhere so missing keys
degrade gracefully instead of raising. Branches that depend on shape
details we are not 100% sure of are tagged with
`# TODO: verify against preflight findings`.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from litellm.managed_agents.types import MessageRow


# ---------------------------------------------------------------------------
# Message normalization
# ---------------------------------------------------------------------------


def normalize_opencode_message(
    opencode_msg: Dict[str, Any],
    our_session_id: str,
) -> MessageRow:
    """Translate one opencode message dict into a MessageRow.

    Rules (contract §7):
      - `id` passes through from opencode.
      - `session_id` is OUR `ses_*` id, NEVER opencode's `oc_sid`.
      - `content` = concat of `text` parts in order.
      - `tools` = list of `tool` parts (omit if zero).
      - `status` = "completed" if `completedAt` set, else "in_progress";
        "failed" if any part has type="error".
      - `model` from `modelID` or `model` (whichever opencode emits).
      - timestamps: parse ISO if string, pass through if already datetime.
    """
    parts = opencode_msg.get("parts") or []

    text_chunks: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    has_error_part = False

    for part in parts:
        if not isinstance(part, dict):
            # TODO: verify against preflight findings — opencode SHOULD always
            # emit dict parts, but be defensive about non-dicts in the array.
            continue
        ptype = part.get("type")
        if ptype == "text":
            text = part.get("text")
            if isinstance(text, str):
                text_chunks.append(text)
        elif ptype == "tool":
            tool_calls.append(
                {
                    "name": part.get("name"),
                    "input": part.get("input"),
                    "output": part.get("output"),
                }
            )
        elif ptype == "error":
            has_error_part = True
        # All other part types (e.g. permission, lsp diagnostics) are ignored.
        # TODO: verify against preflight findings.

    content = "".join(text_chunks)
    tools: Optional[List[Dict[str, Any]]] = tool_calls if tool_calls else None

    completed_at_raw = opencode_msg.get("completedAt")
    if has_error_part:
        status = "failed"
    elif completed_at_raw:
        status = "completed"
    else:
        status = "in_progress"

    # opencode field name: prefer `modelID` (camelCase, opencode convention),
    # fall back to `model` for forward-compat.
    # TODO: verify against preflight findings.
    model = opencode_msg.get("modelID") or opencode_msg.get("model")

    created_at_raw = opencode_msg.get("createdAt")

    return MessageRow(
        id=opencode_msg.get("id", ""),
        session_id=our_session_id,
        role=opencode_msg.get("role", "assistant"),
        content=content,
        model=model,
        tools=tools,
        status=status,  # type: ignore[arg-type]
        created_at=_parse_ts(created_at_raw) or datetime.utcnow(),
        completed_at=_parse_ts(completed_at_raw),
    )


# ---------------------------------------------------------------------------
# Event normalization
# ---------------------------------------------------------------------------


def normalize_opencode_event(
    raw_event: Dict[str, Any],
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Translate one opencode SSE event dict into our (type, data) tuple.

    Returns None for events we drop (per contract §7 — `permission.updated`,
    `lsp.client.diagnostics`, etc.). The caller is responsible for filtering
    by `sessionID` BEFORE invoking this function — this helper only does
    type-and-shape mapping.

    opencode events come in shapes like:
      {"type": "message.updated", "properties": {...}}
      {"type": "message.part.updated", "properties": {"part": {...}, ...}}

    We access `properties` defensively because the exact wrapper shape
    differs across opencode versions.
    """
    if not isinstance(raw_event, dict):
        return None

    event_type = raw_event.get("type")
    if not isinstance(event_type, str):
        return None

    # opencode wraps payload data in `properties` per its event-bus convention.
    # TODO: verify against preflight findings — some versions may inline the
    # payload at the top level instead.
    props = raw_event.get("properties") or {}
    if not isinstance(props, dict):
        props = {}

    if event_type == "message.updated":
        message = props.get("info") or props.get("message") or {}
        message_id = (
            message.get("id") if isinstance(message, dict) else None
        ) or props.get("messageID")
        role = (
            message.get("role") if isinstance(message, dict) else None
        ) or props.get("role")
        return (
            "message.started",
            {"message_id": message_id, "role": role},
        )

    if event_type == "message.part.updated":
        part = props.get("part") or {}
        if not isinstance(part, dict):
            return None
        message_id = part.get("messageID") or props.get("messageID")
        ptype = part.get("type")

        if ptype == "text":
            # TODO: verify against preflight findings — opencode may emit
            # cumulative text or only the new delta. We pass through what
            # the part contains; the SDK handles either case.
            delta = part.get("text") or part.get("delta") or ""
            return (
                "message.text.delta",
                {"message_id": message_id, "delta": delta},
            )

        if ptype == "tool":
            tool_name = part.get("name")
            output = part.get("output")
            if output is not None:
                # Tool finished (has output).
                return (
                    "message.tool.completed",
                    {
                        "message_id": message_id,
                        "tool": tool_name,
                        "output": output,
                    },
                )
            # Tool started (no output yet).
            # TODO: verify against preflight findings — some opencode
            # versions use a `state`/`status` field on the tool part to
            # distinguish start from completion instead of presence of
            # `output`. We use presence-of-output as the default.
            return (
                "message.tool.started",
                {
                    "message_id": message_id,
                    "tool": tool_name,
                    "input": part.get("input"),
                },
            )

        # Other part types (e.g. step-start, file) are dropped.
        return None

    if event_type == "session.idle":
        # Turn done — emit `message.completed`. opencode's idle payload may
        # not always include the message id/content directly; best-effort.
        # TODO: verify against preflight findings.
        message_id = props.get("messageID") or props.get("message_id")
        return (
            "message.completed",
            {
                "message_id": message_id,
                "content": props.get("content"),
                "completed_at": props.get("completedAt") or props.get("completed_at"),
            },
        )

    if event_type == "session.error":
        message_id = props.get("messageID") or props.get("message_id")
        error = props.get("error") or props.get("message")
        return (
            "error",
            {"message_id": message_id, "error": error},
        )

    return None


# ---------------------------------------------------------------------------
# Helpers — opencode session id matching
# ---------------------------------------------------------------------------


def event_matches_session(
    raw_event: Dict[str, Any],
    opencode_session_id: str,
) -> bool:
    """Return True if `raw_event` references `opencode_session_id`.

    opencode events include the session id under `sessionID` at the top
    level OR nested under `properties.sessionID`. We check both. Events
    without a session id (rare — typically global health/heartbeat) are
    treated as non-matching.
    """
    if not isinstance(raw_event, dict):
        return False
    if raw_event.get("sessionID") == opencode_session_id:
        return True
    props = raw_event.get("properties") or {}
    if isinstance(props, dict):
        if props.get("sessionID") == opencode_session_id:
            return True
        # The session id may also live nested on a sub-object (e.g. on the
        # `info` message for `message.updated`).
        info = props.get("info") or props.get("message") or {}
        if isinstance(info, dict) and info.get("sessionID") == opencode_session_id:
            return True
    return False


# ---------------------------------------------------------------------------
# Internal — timestamp parsing
# ---------------------------------------------------------------------------


def _parse_ts(value: Any) -> Optional[datetime]:
    """Parse an opencode timestamp into a datetime, or None.

    opencode timestamps are sometimes ISO 8601 strings, sometimes
    ms-since-epoch ints (per its TS codebase). Be defensive about both.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        # opencode timestamps are ms-since-epoch in its TS code.
        try:
            return datetime.utcfromtimestamp(value / 1000.0)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        # Accept "2026-05-07T15:04:05.123Z" by swapping trailing Z for +00:00,
        # which fromisoformat understands on 3.10+.
        s = value.replace("Z", "+00:00") if value.endswith("Z") else value
        try:
            return datetime.fromisoformat(s)
        except ValueError:
            return None
    return None
