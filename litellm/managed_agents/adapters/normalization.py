"""Pure normalization helpers for the opencode adapter.

These functions translate opencode wire shapes into our normalized
`MessageRow` and SSE event tuples per contract §7.

The shapes encoded here are the REAL shapes verified against opencode
1.14.41 in the preflight (`.claude/v2_opencode_real_responses.md`). Key
deviations from the contract document:

  - opencode messages are nested as ``{info, parts}``, NOT flat.
  - Field naming on the wire is camelCase with capital-ID suffix
    (``sessionID``, ``messageID``, ``partID``, ``callID``, ``providerID``,
    ``modelID``). Our normalized output is snake_case.
  - Timestamps are epoch milliseconds (int), not ISO 8601.
  - Part types include ``text``, ``reasoning``, ``tool``, ``step-start``,
    ``step-finish``. Only ``text`` parts go into ``content``; only
    ``tool`` parts go into ``tools``. ``reasoning`` / ``step-start`` /
    ``step-finish`` are filtered out for MVP.
  - Tool I/O lives under ``part.state.input`` / ``part.state.output``
    (or ``part.state.error``), not directly on ``part``.
  - Streaming text deltas come via ``message.part.delta`` events with
    ``{messageID, partID, field, delta}`` — the part ``type`` is NOT on
    the delta event, so the caller must maintain a per-stream
    ``partID -> type`` map sourced from ``message.part.updated`` events
    (see `route_part_delta`).

These helpers have NO side effects, NO HTTP, NO DB calls — pure
dict-in / object-out.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from litellm.managed_agents.types import MessageRow


# ---------------------------------------------------------------------------
# Message normalization
# ---------------------------------------------------------------------------


def normalize_opencode_message(
    opencode_msg: Dict[str, Any],
    our_session_id: str,
) -> MessageRow:
    """Translate one opencode message envelope into a MessageRow.

    opencode returns each message as ``{info: {...}, parts: [...]}`` where
    ``info`` carries the message metadata (id, role, model, time, finish)
    and ``parts`` is the per-part body array.

    Rules (contract §7, adjusted to the real shapes):
      - ``id`` passes through from ``info.id``.
      - ``session_id`` is OUR ``ses_*`` id, NEVER opencode's ``sessionID``.
      - ``content`` = concat of ``text`` parts in order. ``reasoning``,
        ``step-start``, ``step-finish`` are excluded.
      - ``tools`` = list of normalized tool parts (omit field if zero).
      - ``model`` = ``"<providerID>/<modelID>"`` if both present; falls
        back to whichever is present, else None.
      - ``status`` = ``"completed"`` if ``info.time.completed`` is set,
        else ``"in_progress"``. ``"failed"`` if any tool part has
        ``state.status == "error"`` OR any part has ``type == "error"``.
      - timestamps come from ``info.time.created`` / ``info.time.completed``
        (epoch ms ints).
    """
    info = opencode_msg.get("info") or {}
    if not isinstance(info, dict):
        info = {}
    parts = opencode_msg.get("parts") or []
    if not isinstance(parts, list):
        parts = []

    text_chunks: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    has_error_part = False

    for part in parts:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype == "text":
            text = part.get("text")
            if isinstance(text, str):
                text_chunks.append(text)
        elif ptype == "tool":
            tool_calls.append(_normalize_tool_part(part))
            state = part.get("state") or {}
            if isinstance(state, dict) and state.get("status") == "error":
                has_error_part = True
        elif ptype == "error":
            # Defensive: opencode does not currently emit a top-level "error"
            # part type (errors live under tool ``state.error`` or as
            # ``session.error`` events), but we treat one as a failure if
            # it ever appears.
            has_error_part = True
        # All other part types (``reasoning``, ``step-start``, ``step-finish``,
        # plus any future additions) are deliberately dropped from
        # ``content``/``tools`` for MVP.

    content = "".join(text_chunks)
    tools: Optional[List[Dict[str, Any]]] = tool_calls if tool_calls else None

    time_obj = info.get("time") or {}
    if not isinstance(time_obj, dict):
        time_obj = {}
    completed_at_raw = time_obj.get("completed")
    created_at_raw = time_obj.get("created")

    if has_error_part:
        status = "failed"
    elif completed_at_raw is not None:
        status = "completed"
    else:
        status = "in_progress"

    model = _format_model(info.get("providerID"), info.get("modelID"))

    return MessageRow(
        id=info.get("id", ""),
        session_id=our_session_id,
        role=info.get("role", "assistant"),
        content=content,
        model=model,
        tools=tools,
        status=status,  # type: ignore[arg-type]
        created_at=_parse_ts(created_at_raw) or datetime.now(timezone.utc),
        completed_at=_parse_ts(completed_at_raw),
    )


def _normalize_tool_part(part: Dict[str, Any]) -> Dict[str, Any]:
    """Extract ``{name, input, output}`` from an opencode tool part.

    Real opencode shape:
        {
          "type": "tool",
          "tool": "read",                       # tool name (top-level)
          "callID": "call_...",                 # opencode call id
          "state": {
            "status": "completed",              # pending|running|completed|error
            "input": {...},
            "output": "...",                    # OR
            "error": "...",                     # for status=="error"
          },
          ...
        }
    """
    state = part.get("state") or {}
    if not isinstance(state, dict):
        state = {}
    output: Any = state.get("output")
    if output is None and state.get("status") == "error":
        output = state.get("error")
    return {
        "name": part.get("tool"),
        "input": state.get("input"),
        "output": output,
    }


def _format_model(provider_id: Any, model_id: Any) -> Optional[str]:
    """Combine opencode's ``providerID`` + ``modelID`` into our model string.

    Returns ``"<provider>/<model>"`` when both are present, else whichever
    is present, else None.
    """
    p = provider_id if isinstance(provider_id, str) and provider_id else None
    m = model_id if isinstance(model_id, str) and model_id else None
    if p and m:
        return f"{p}/{m}"
    return p or m


# ---------------------------------------------------------------------------
# Event normalization
# ---------------------------------------------------------------------------


def normalize_opencode_event(
    raw_event: Dict[str, Any],
    part_types: Optional[Dict[str, str]] = None,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Translate one opencode SSE event dict into our (type, data) tuple.

    The ``part_types`` argument is a mutable map of ``partID -> part_type``
    maintained by the caller across the lifetime of one stream. It's
    required to route ``message.part.delta`` events because those events
    do not carry the part type — only the ``partID``.

    Returns None for events we drop (heartbeats, unknown types,
    reasoning deltas, etc.).
    """
    if not isinstance(raw_event, dict):
        return None

    event_type = raw_event.get("type")
    if not isinstance(event_type, str) or not event_type:
        return None

    props = raw_event.get("properties") or {}
    if not isinstance(props, dict):
        props = {}

    if event_type == "message.updated":
        return _normalize_message_updated(props)

    if event_type == "message.part.updated":
        return _normalize_message_part_updated(props, part_types)

    if event_type == "message.part.delta":
        return _normalize_message_part_delta(props, part_types)

    if event_type == "session.idle":
        return _normalize_session_idle(props)

    if event_type == "session.error":
        return _normalize_session_error(props)

    return None


def _normalize_message_updated(
    props: Dict[str, Any],
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """``message.updated`` — emit ``message.started`` (new msg) or
    ``message.completed`` (existing msg with ``time.completed`` set)."""
    info = props.get("info") or {}
    if not isinstance(info, dict):
        return None

    message_id = info.get("id")
    role = info.get("role")
    time_obj = info.get("time") or {}
    if not isinstance(time_obj, dict):
        time_obj = {}
    completed_ms = time_obj.get("completed")

    if completed_ms is not None and role == "assistant":
        # The assistant turn has finished. opencode emits this just before
        # ``session.idle`` and it carries enough info to attach to a
        # specific message. We prefer this over ``session.idle`` because
        # it's per-message rather than per-session.
        return (
            "message.completed",
            {
                "message_id": message_id,
                "completed_at": _ms_to_iso_str(completed_ms),
            },
        )

    return (
        "message.started",
        {"message_id": message_id, "role": role},
    )


def _normalize_message_part_updated(
    props: Dict[str, Any],
    part_types: Optional[Dict[str, str]],
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """``message.part.updated`` — covers part lifecycle (creation, status
    transitions for tools, finalization). Streaming text deltas do NOT come
    through this event — see ``_normalize_message_part_delta``.
    """
    part = props.get("part") or {}
    if not isinstance(part, dict):
        return None

    part_id = part.get("id")
    ptype = part.get("type")
    message_id = part.get("messageID")

    # Track the part's type for later ``message.part.delta`` routing.
    if part_types is not None and isinstance(part_id, str) and isinstance(ptype, str):
        part_types[part_id] = ptype

    if ptype == "tool":
        state = part.get("state") or {}
        if not isinstance(state, dict):
            state = {}
        status = state.get("status")
        tool_name = part.get("tool")

        if status == "running":
            return (
                "message.tool.started",
                {
                    "message_id": message_id,
                    "tool": tool_name,
                    "input": state.get("input"),
                },
            )
        if status == "completed":
            return (
                "message.tool.completed",
                {
                    "message_id": message_id,
                    "tool": tool_name,
                    "output": state.get("output"),
                },
            )
        if status == "error":
            return (
                "message.tool.completed",
                {
                    "message_id": message_id,
                    "tool": tool_name,
                    "output": state.get("error"),
                    "error": True,
                },
            )
        # ``pending`` or unknown statuses: drop. The ``running`` transition
        # carries the populated ``input`` we want.
        return None

    # Text/reasoning/step-* part lifecycle events don't translate to our
    # event surface — only their delta chunks do. We've already updated
    # ``part_types`` above so subsequent deltas can be routed.
    return None


def _normalize_message_part_delta(
    props: Dict[str, Any],
    part_types: Optional[Dict[str, str]],
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """``message.part.delta`` — incremental chunk for a streaming part.

    Real shape: ``{sessionID, messageID, partID, field, delta}``. The part
    ``type`` is NOT on this event. We look it up in ``part_types`` (which
    must have been populated by a prior ``message.part.updated`` event for
    this ``partID``).
    """
    part_id = props.get("partID")
    field = props.get("field")
    delta = props.get("delta")
    message_id = props.get("messageID")

    if not isinstance(part_id, str) or not isinstance(delta, str):
        return None

    ptype = (part_types or {}).get(part_id)

    if field == "text" and ptype == "text":
        return (
            "message.text.delta",
            {"message_id": message_id, "delta": delta},
        )

    # Reasoning deltas are intentionally dropped for MVP. Other field
    # values (e.g. tool ``raw``) are also dropped.
    return None


def _normalize_session_idle(
    props: Dict[str, Any],
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """``session.idle`` — turn done.

    The real opencode payload is ``{sessionID}`` only — it does NOT carry
    a ``messageID`` or ``content``. The adapter is expected to track the
    "current assistant message id" from prior ``message.updated`` events
    and attach it before yielding to the SSE consumer (see
    ``OpencodeAdapter.stream_events``).

    We still emit a ``message.completed`` here as a safety net for cases
    where the per-message ``message.updated`` with ``time.completed`` was
    missed. The adapter overrides ``message_id`` if it has tracked one.
    """
    return (
        "message.completed",
        {
            "message_id": props.get("messageID"),
            "completed_at": None,
        },
    )


def _normalize_session_error(
    props: Dict[str, Any],
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """``session.error`` — opencode runtime error.

    Real shape: ``{sessionID, error: {name, data: {message}}}``. We unwrap
    to the human-readable message string when possible.
    """
    error_obj = props.get("error")
    error_message: Any = error_obj
    if isinstance(error_obj, dict):
        data = error_obj.get("data")
        if isinstance(data, dict) and isinstance(data.get("message"), str):
            error_message = data["message"]
        elif isinstance(error_obj.get("message"), str):
            error_message = error_obj["message"]
    return (
        "error",
        {
            "message_id": props.get("messageID"),
            "error": error_message,
        },
    )


# ---------------------------------------------------------------------------
# Helpers — opencode session id matching
# ---------------------------------------------------------------------------


def event_matches_session(
    raw_event: Dict[str, Any],
    opencode_session_id: str,
) -> bool:
    """Return True if ``raw_event`` references ``opencode_session_id``.

    opencode events carry the session id as ``properties.sessionID``
    (camelCase) — and sometimes also at the top level. We check both, plus
    the nested ``info.sessionID`` for ``message.updated`` events.

    Events without any session id are treated as non-matching, which means
    global heartbeats (``server.heartbeat``, ``server.connected``) are
    correctly filtered out.
    """
    if not isinstance(raw_event, dict):
        return False
    if raw_event.get("sessionID") == opencode_session_id:
        return True
    props = raw_event.get("properties") or {}
    if isinstance(props, dict):
        if props.get("sessionID") == opencode_session_id:
            return True
        info = props.get("info")
        if isinstance(info, dict) and info.get("sessionID") == opencode_session_id:
            return True
        part = props.get("part")
        if isinstance(part, dict) and part.get("sessionID") == opencode_session_id:
            return True
    return False


# ---------------------------------------------------------------------------
# Internal — timestamp parsing
# ---------------------------------------------------------------------------


def _parse_ts(value: Any) -> Optional[datetime]:
    """Parse an opencode timestamp into a tz-aware UTC datetime, or None.

    opencode timestamps on the wire are epoch milliseconds (int). Strings
    are accepted defensively (e.g. for tests that still use ISO strings),
    and existing datetime values pass through.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, bool):
        # ``True``/``False`` are technically ``int`` subclasses — guard explicitly.
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        s = value.replace("Z", "+00:00") if value.endswith("Z") else value
        try:
            return datetime.fromisoformat(s)
        except ValueError:
            return None
    return None


def _ms_to_iso(ms: Any) -> Optional[datetime]:
    """Convert opencode ``epoch ms`` into a tz-aware UTC datetime.

    Thin wrapper around ``_parse_ts`` for callers that want the explicit
    "this is opencode-style ms" intent.
    """
    return _parse_ts(ms)


def _ms_to_iso_str(ms: Any) -> Optional[str]:
    """Convert opencode ``epoch ms`` into an ISO 8601 UTC string.

    Used for SSE event payloads where the wire format is JSON (string),
    not a Python datetime.
    """
    dt = _parse_ts(ms)
    if dt is None:
        return None
    return dt.isoformat().replace("+00:00", "Z")
