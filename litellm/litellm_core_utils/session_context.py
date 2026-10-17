"""Where a request's session and trace ids come from, shared by proxy request setup, every raw-header
caller (MCP, A2A, pass-through, guardrails, auth) and the loggers, with no proxy or FastAPI dependency."""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Final,
    Literal,
    TypeAlias,
    cast,  # noqa: TID251  # casts moved verbatim from litellm_pre_call_utils
)

from litellm.litellm_core_utils.core_helpers import is_codex_user_agent

if TYPE_CHECKING:
    from opentelemetry.trace import Span as OtelSpan


# Matches any header of the form x-<something>-session-id (case-insensitive).
# Excludes the two explicit litellm headers which are handled with higher priority.
_GENERIC_SESSION_ID_HEADER_RE: Final = re.compile(r"^x-.+-session-id$", re.IGNORECASE)


_EXPLICIT_SESSION_HEADERS: Final = frozenset({"x-litellm-trace-id", "x-litellm-session-id"})


# Codex carries its conversation uuid in unprefixed headers, so the
# x-<vendor>-session-id convention above never matches it. Current builds send
# ``session-id``/``thread-id``; builds before the codex-api split sent
# ``session_id``/``conversation_id``. Ordered session before thread.
_CODEX_SESSION_ID_HEADERS: Final = ("session-id", "session_id", "thread-id", "conversation_id")


# Session-id values must be non-empty strings of alphanumerics, hyphens, or underscores
# (covers UUIDs and most common session-id formats).
_SESSION_ID_VALUE_RE: Final = re.compile(r"^[a-zA-Z0-9_\-]{8,}$")


_SESSION_AFFINITY_HEADER: Final = "x-session-affinity"


_SESSION_AFFINITY_VALUE_RE: Final = re.compile(r"^[a-zA-Z0-9_\-.:]{8,256}$")


# W3C Trace Context traceparent header: https://www.w3.org/TR/trace-context/
# e.g. "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
_TRACEPARENT_RE: Final = re.compile(r"^[0-9a-f]{2}-([0-9a-f]{32})-[0-9a-f]{16}-[0-9a-f]{2}$", re.IGNORECASE)


_ANTHROPIC_SESSION_ID_VALUE_RE: Final = re.compile(r"^[a-zA-Z0-9_\-]+$")


LITELLM_TRACE_CONTROL_METADATA_FIELDS: Final = frozenset(
    {
        "mask_input",
        "mask_output",
        "session_id",
        "trace_id",
        "trace_metadata",
        "trace_name",
        "trace_release",
        "trace_user_id",
        "trace_version",
    }
)


SessionSource: TypeAlias = Literal["trace_header", "session_header", "anthropic_metadata", "baggage", "otel_span"]


_NO_BODY: Final[Mapping[str, object]] = MappingProxyType({})


def _trace_id_from_traceparent(traceparent: str) -> str | None:
    """Extract the trace-id from a W3C Trace Context traceparent header, e.g.
    "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01" -> the 32-hex
    trace-id in the middle. An all-zero trace-id is invalid per spec and is
    rejected, matching how the OpenTelemetry SDK itself treats it."""
    match: Final = _TRACEPARENT_RE.match(traceparent.strip())
    if not match:
        return None
    trace_id: Final = match.group(1).lower()
    return trace_id if trace_id != "0" * 32 else None


def _trace_id_from_otel_span(span: "OtelSpan | None") -> str | None:
    if span is None:
        return None
    try:
        span_context: Final = span.get_span_context()
        is_valid: Final = span_context.is_valid
        trace_id: Final = cast(object, span_context.trace_id)  # cast-ok: trace_id is int at runtime; tests mock it
    except AttributeError:
        return None
    if not is_valid or not isinstance(trace_id, int):
        return None
    return format(trace_id, "032x")


def _session_id_from_baggage(baggage: str) -> str | None:
    """Extract a session.id entry from a W3C Baggage header
    (https://www.w3.org/TR/baggage/), e.g. "session.id=abc-123,user.id=42"."""
    for pair in baggage.split(","):
        key, _, value = pair.strip().partition("=")
        if key.strip() == "session.id" and value.strip():
            return value.strip()
    return None


def caller_trace_field(data: Mapping[str, object], metadata_variable_name: str, field: str) -> str | None:
    """The caller's value for a trace-control field, counted only when it is a
    usable id: a non-empty string. An explicitly empty/unusable value on the
    active metadata container still shadows the promoted requester value, but
    neither ever counts as "the caller supplied this field" on its own, so a
    numeric session id or an empty string cannot suppress the W3C header
    fallback or satisfy a missing-session-id policy."""
    active: Final = data.get(metadata_variable_name)
    if isinstance(active, Mapping) and field in active:
        active_map: Final = cast(Mapping[str, object], active)  # cast-ok: isinstance above, free-form JSON values
        active_value: Final = active_map[field]
        return active_value if isinstance(active_value, str) and active_value else None
    promoted: Final = metadata_variable_name == "litellm_metadata" and field in LITELLM_TRACE_CONTROL_METADATA_FIELDS
    requester: Final = data.get("metadata")
    if not promoted or not isinstance(requester, Mapping):
        return None
    requester_map: Final = cast(Mapping[str, object], requester)  # cast-ok: isinstance above, free-form JSON values
    requester_value: Final = requester_map.get(field)
    return requester_value if isinstance(requester_value, str) and requester_value else None


def _caller_body_trace_id(data: Mapping[str, object], metadata_variable_name: str) -> str | None:
    top_level: Final = data.get("litellm_trace_id")
    if isinstance(top_level, str) and top_level:
        return top_level
    return caller_trace_field(data, metadata_variable_name, "trace_id")


def _first_agreeing_source(
    caller_value: str | None, candidates: Sequence[tuple[str | None, "SessionSource"]]
) -> tuple[str | None, "SessionSource | None"]:
    """The first candidate id, in precedence order, that the caller's own body does not contradict.
    A body that already carries a different usable id keeps it, so no source ever overrides the caller."""
    for value, source in candidates:
        if value and (caller_value is None or caller_value == value):
            return value, source
    return None, None


def caller_body_session_id(data: Mapping[str, object], metadata_variable_name: str) -> str | None:
    top_level: Final = data.get("litellm_session_id")
    if isinstance(top_level, str) and top_level:
        return top_level
    return caller_trace_field(data, metadata_variable_name, "session_id")


def _extract_generic_session_id_from_headers(
    normalized: Mapping[str, str],
) -> str | None:
    """
    Scan a normalised (lower-cased keys) header dict for any header that looks
    like ``x-<vendor>-session-id`` and whose value is a plausible session/trace
    identifier (alphanumeric + hyphens/underscores, at least 8 chars).

    The two explicit LiteLLM headers (``x-litellm-trace-id`` /
    ``x-litellm-session-id``) are excluded here because they are handled with
    higher priority by the caller.

    Example: ``x-claude-code-session-id: e96634a3-fa28-4083-b354-55542e2dca01``
    """
    for key, value in normalized.items():
        if (
            key not in _EXPLICIT_SESSION_HEADERS
            and _GENERIC_SESSION_ID_HEADER_RE.match(key)
            and isinstance(value, str)
            and _SESSION_ID_VALUE_RE.match(value)
        ):
            return value
    return None


def _extract_codex_session_id_from_headers(
    normalized: Mapping[str, str],
) -> str | None:
    """
    Read Codex's conversation uuid off one of ``_CODEX_SESSION_ID_HEADERS``.

    Codex sends no request metadata the Anthropic path could parse and no
    ``x-``-prefixed session header, so without this every turn of a Codex session
    falls through to a freshly generated per-call trace id and lands as its own
    row in the logs instead of grouping.

    Unprefixed names like ``session-id`` are generic enough that another client
    could send one meaning something unrelated, and colliding values across
    callers would merge their traces, so this only applies to callers that
    identify as Codex.
    """
    user_agent: Final = normalized.get("user-agent")
    if not isinstance(user_agent, str) or not is_codex_user_agent(user_agent):
        return None
    return next(
        (
            value
            for value in (normalized.get(header) for header in _CODEX_SESSION_ID_HEADERS)
            if isinstance(value, str) and _SESSION_ID_VALUE_RE.match(value)
        ),
        None,
    )


def _extract_bare_session_id_from_headers(
    normalized: Mapping[str, str],
) -> str | None:
    """
    Read a vendor-less ``x-session-id`` header (opencode sends ``X-Session-Id``
    alongside ``x-session-affinity`` on every turn of a session). Checked after
    the ``x-<vendor>-session-id`` scan so a more specific header such as
    opencode's ``x-parent-session-id`` on subagent calls keeps winning.
    """
    value: Final = normalized.get("x-session-id")
    if isinstance(value, str) and _SESSION_ID_VALUE_RE.match(value):
        return value
    return None


def _extract_session_affinity_from_headers(
    normalized: Mapping[str, str],
) -> str | None:
    """
    Read ``x-session-affinity``, the only session header some agent clients
    (pi-ai, opencode) send. Their ids look like ``web:<principal>:<uuid>`` or
    ``<ref>:task:<n>``, so this header alone also accepts colons and dots, with
    a length cap. Checked after every other session source so a client that
    also sends a more specific header keeps that one.
    """
    value: Final = normalized.get(_SESSION_AFFINITY_HEADER)
    if isinstance(value, str) and _SESSION_AFFINITY_VALUE_RE.fullmatch(value):
        return value
    return None


def normalize_header_keys(headers: Mapping[str, str]) -> Mapping[str, str]:
    return MappingProxyType({k.lower(): v for k, v in headers.items() if isinstance(k, str)})


def _extract_session_header_chain_id(normalized: Mapping[str, str]) -> str | None:
    return (
        normalized.get("x-litellm-trace-id")
        or normalized.get("x-litellm-session-id")
        or _extract_generic_session_id_from_headers(normalized)
        or _extract_codex_session_id_from_headers(normalized)
        or _extract_bare_session_id_from_headers(normalized)
    )


def session_header_id(normalized: Mapping[str, str]) -> str | None:
    return _extract_session_header_chain_id(normalized) or _extract_session_affinity_from_headers(normalized)


@dataclass(frozen=True, slots=True)
class SessionContext:
    """What a request's session and trace ids resolve to, and which source set each, so a
    downstream logger can tell a trace id that is only a session header's value from a real trace."""

    session_id: str | None = None
    trace_id: str | None = None
    session_source: SessionSource | None = None
    trace_source: SessionSource | None = None


def resolve_session_context(
    headers: Mapping[str, str],
    body: Mapping[str, object] = _NO_BODY,
    metadata_variable_name: str = "metadata",
    parent_otel_span: "OtelSpan | None" = None,
) -> SessionContext:
    """
    The one precedence order for session and trace ids, shared by request setup and every
    raw-header caller (MCP, A2A, pass-through, guardrails, auth). In order: the litellm trace
    and session headers, vendor and Codex session headers, bare ``x-session-id``, Anthropic
    ``metadata.user_id``, W3C ``traceparent`` (trace) and ``baggage`` (session), the parent
    OTel span (trace), and last ``x-session-affinity``. A usable id the caller's body already
    carries is kept unless the source agrees with it.
    """
    normalized: Final = normalize_header_keys(headers)
    chain_id: Final = _extract_session_header_chain_id(normalized)
    if chain_id:
        source: Final = "trace_header" if chain_id == normalized.get("x-litellm-trace-id") else "session_header"
        return SessionContext(session_id=chain_id, trace_id=chain_id, session_source=source, trace_source=source)
    anthropic_session: Final = _get_anthropic_session_id_from_metadata(body.get("metadata"))
    traceparent: Final = normalized.get("traceparent")
    baggage: Final = normalized.get("baggage")
    affinity: Final = _extract_session_affinity_from_headers(normalized)
    session_id, session_source = (
        (anthropic_session, "anthropic_metadata")
        if anthropic_session
        else _first_agreeing_source(
            caller_body_session_id(body, metadata_variable_name),
            ((_session_id_from_baggage(baggage) if baggage else None, "baggage"), (affinity, "session_header")),
        )
    )
    trace_id, trace_source = _first_agreeing_source(
        _caller_body_trace_id(body, metadata_variable_name),
        (
            (_trace_id_from_traceparent(traceparent) if traceparent else None, "trace_header"),
            (_trace_id_from_otel_span(parent_otel_span), "otel_span"),
            (affinity if session_id == affinity else None, "session_header"),
        ),
    )
    return SessionContext(
        session_id=session_id, trace_id=trace_id, session_source=session_source, trace_source=trace_source
    )


def _get_anthropic_session_id_from_metadata(metadata: object) -> str | None:
    if not isinstance(metadata, dict):
        return None

    user_id: Final = metadata.get("user_id")
    if isinstance(user_id, dict):
        session_id = user_id.get("session_id")
        if isinstance(session_id, str) and _ANTHROPIC_SESSION_ID_VALUE_RE.fullmatch(session_id):
            return session_id
        return None
    if not isinstance(user_id, str):
        return None

    session_marker: Final = "_session_"
    session_marker_index: Final = user_id.rfind(session_marker)
    if session_marker_index == -1:
        return None

    session_id = user_id[session_marker_index + len(session_marker) :]
    if not session_id or not _ANTHROPIC_SESSION_ID_VALUE_RE.fullmatch(session_id):
        return None
    return session_id
