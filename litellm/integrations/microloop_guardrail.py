"""
Microloop Guardrail — Deterministic loop detection for LiteLLM.

Hooks into LiteLLM's :code:`async_pre_call_hook` to examine tool call history
in every request. If the same tool + arguments repeat within a configurable
window, the call is blocked BEFORE it reaches the LLM — saving API costs.

Usage::

    from litellm import completion
    from litellm.integrations.microloop_guardrail import MicroloopGuardrail

    litellm.callbacks = [MicroloopGuardrail(max_repeats=3)]
"""

import hashlib
import json
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone

from litellm.caching import DualCache
from litellm.exceptions import GuardrailRaisedException
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    get_session_id_from_request_data,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import CallTypes

# ---------------------------------------------------------------------------
# In-memory call history store
# ---------------------------------------------------------------------------


class _CallHistory:
    """Tracks tool call trajectories per session with bounded memory."""

    MAX_SESSIONS = 10000

    def __init__(self) -> None:
        self._store: dict[str, list[tuple[str, str, datetime]]] = {}

    def append(self, session_id: str, tool_name: str, args_json: str) -> None:
        if len(self._store) >= self.MAX_SESSIONS and session_id not in self._store:
            oldest_key = next(iter(self._store))
            self._store.pop(oldest_key, None)
        if session_id not in self._store:
            self._store[session_id] = []
        self._store[session_id].append((tool_name, args_json, datetime.now(timezone.utc)))

    def get_recent(self, session_id: str, window: int) -> list[tuple[str, str]]:
        entries = self._store.get(session_id, [])
        return [(t, a) for t, a, _ in entries[-window:]]

    def trim(self, session_id: str, max_len: int) -> None:
        entries = self._store.get(session_id, [])
        if len(entries) > max_len:
            self._store[session_id] = entries[-max_len:]

    def clear(self, session_id: str) -> None:
        self._store.pop(session_id, None)


# ---------------------------------------------------------------------------
# Volatile field utilities
# ---------------------------------------------------------------------------


def _parse_json_dict(raw: str) -> dict[str, object] | None:
    """Parse a JSON string into a dict, returning None on failure."""
    try:
        result: object = json.loads(raw)  # pyright: ignore[reportAny]
    except (json.JSONDecodeError, TypeError):
        return None
    else:
        if isinstance(result, dict):
            return dict(result)
        return None


def _canonicalize_args(args_json: str, volatile_fields: set[str]) -> str:
    """Parse, canonicalize (sort keys), and optionally strip volatile fields.

    Always parses and re-serializes with ``sort_keys=True`` so that
    logically identical JSON objects produce the same hash regardless of
    key ordering.

    Args:
        args_json: A JSON-encoded string of tool arguments.
        volatile_fields: Field names to strip from the JSON object.

    Returns:
        Canonical JSON string with sorted keys, or the original
        string if it cannot be parsed or is not a JSON object.
    """
    obj = _parse_json_dict(args_json)
    if obj is None:
        return args_json
    if volatile_fields:
        for field in volatile_fields:
            obj.pop(field, None)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _auto_infer_volatile(
    recent_calls: Sequence[tuple[str, str]],
    current_args: str,
    min_occurrences: int = 2,
) -> set[str]:
    """Detect fields that consistently differ across consecutive calls.

    A field is considered volatile only if:

    1. It is the *only* differing field between calls
    2. AND at least one other field stays constant
    3. AND the pattern appears >= ``min_occurrences`` times

    This prevents inferring the only changing parameter as volatile.

    Args:
        recent_calls: Sequence of past ``(tool_name, args_json)`` pairs.
        current_args: JSON-encoded string of the current call arguments.
        min_occurrences: Minimum number of diff occurrences to infer.

    Returns:
        Set of field names inferred as volatile.
    """
    inferred: set[str] = set()
    diff_counts: dict[str, int] = {}
    const_fields: set[str] = set()

    current = _parse_json_dict(current_args)
    if current is None:
        return inferred

    for _past_tool, past_args_str in recent_calls:
        past = _parse_json_dict(past_args_str)
        if past is None:
            continue

        diffs: list[str] = []
        consts: list[str] = []
        all_keys = set(current.keys()) | set(past.keys())
        for k in all_keys:
            if current.get(k) != past.get(k):
                diffs.append(k)
            else:
                consts.append(k)

        if len(diffs) == 1:
            diff_counts[diffs[0]] = diff_counts.get(diffs[0], 0) + 1
            const_fields.update(consts)

    for field, count in diff_counts.items():
        if count >= min_occurrences and const_fields:
            inferred.add(field)

    return inferred


# ---------------------------------------------------------------------------
# Main guardrail class
# ---------------------------------------------------------------------------


class MicroloopGuardrail(CustomGuardrail):
    """LiteLLM guardrail that detects deterministic tool call loops.

    Intercepts *before* the LLM API call by examining the message history
    included in every request. If the same tool with the same arguments has
    repeated ``max_repeats`` times within the ``history_window``, the call
    is blocked and a :code:`GuardrailRaisedException` is raised.

    Example:

        .. code-block:: python

            import litellm
            from litellm.integrations.microloop_guardrail import MicroloopGuardrail

            litellm.callbacks = [MicroloopGuardrail(max_repeats=3)]
    """

    def __init__(
        self,
        max_repeats: int = 3,
        history_window: int | None = None,
        volatile_fields: list[str] | None = None,
        auto_infer_volatile: bool = True,
        guardrail_name: str = "microloop",
        default_on: bool = True,
    ) -> None:
        """
        Args:
            max_repeats: Number of identical tool calls before blocking.
            history_window: How many recent calls to compare against.
                Defaults to ``2 * max_repeats``.
            volatile_fields: Fields to ignore during comparison
                (e.g. request IDs, timestamps).
            auto_infer_volatile: When True, automatically detect volatile
                fields from call history.
            guardrail_name: Name passed to parent guardrail.
            default_on: Whether the guardrail is active by default.
        """
        if max_repeats < 2:
            raise ValueError(
                "max_repeats must be at least 2. A value of 1 would incorrectly block the first tool call."
            )
        super().__init__(
            guardrail_name=guardrail_name,
            default_on=default_on,
        )
        self._max_repeats = max_repeats
        self._history_window = history_window or (max_repeats * 2)
        self._volatile_fields: set[str] = set(volatile_fields or [])
        self._auto_infer_volatile = auto_infer_volatile
        self._history = _CallHistory()

    # ---- LiteLLM hooks ---------------------------------------------------

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: CallTypes,
    ) -> dict | None:
        """LiteLLM hook called before each LLM API request.

        Examines ``data["messages"]`` for tool call history and raises
        :code:`GuardrailRaisedException` if a loop is detected.

        Args:
            user_api_key_dict: The user API key authentication object.
            cache: The LiteLLM cache instance.
            data: The full request data dictionary containing ``messages``.
            call_type: The type of LLM call (completion, etc.).

        Returns:
            The request data dict (unchanged) or None if no modification.
            A :code:`GuardrailRaisedException` is raised when a loop is
            detected.

        Raises:
            GuardrailRaisedException: If a tool call loop is detected.
        """
        session_id = self._get_session_id(data, user_api_key_dict)
        messages = data.get("messages", [])
        tool_pairs = self._extract_tool_calls(messages)

        if not tool_pairs:
            return None

        for tool_name, tool_args in tool_pairs:
            self._check_and_record(session_id, tool_name, tool_args)

        return None

    # ---- Core loop detection ---------------------------------------------

    def _check_and_record(self, session_id: str, tool_name: str, raw_args: str) -> None:
        """Check if ``tool_name(raw_args)`` is a loop, then record it.

        Args:
            session_id: The session identifier.
            tool_name: Name of the tool being called.
            raw_args: JSON-encoded arguments string.

        Raises:
            GuardrailRaisedException: When the same tool+arguments repeat
                beyond the configured limit.
        """
        recent = self._history.get_recent(session_id, self._history_window)

        # 1. Canonicalize (parse + sort keys + strip volatile fields)
        canonical = _canonicalize_args(raw_args, self._volatile_fields)

        # 2. Count exact matches in recent history
        match_count = self._count_matches(recent, tool_name, canonical)

        # 3. Auto-inference (if enabled and no match found)
        if self._auto_infer_volatile and match_count == 0:
            inferred = _auto_infer_volatile(recent, raw_args)
            if inferred:
                combined = self._volatile_fields | inferred
                canonical = _canonicalize_args(raw_args, combined)
                match_count = 0
                for past_tool, past_args in recent:
                    stripped = _canonicalize_args(past_args, combined)
                    if past_tool == tool_name and stripped == canonical:
                        match_count += 1

        # 4. Block or record
        if match_count + 1 >= self._max_repeats:
            raise GuardrailRaisedException(
                guardrail_name="microloop",
                message=(
                    f"Microloop: Loop detected on tool '{tool_name}' -- "
                    f"seen {match_count + 1}x (limit: {self._max_repeats})"
                    f"{' in session ' + session_id if session_id else ''}"
                ),
            )

        self._history.append(session_id, tool_name, canonical)
        self._history.trim(session_id, self._history_window * 2)

    # ---- Helpers ---------------------------------------------------------

    def _get_session_id(self, data: dict, user_api_key_dict: object = None) -> str:
        """Extract or derive a session identifier to prevent cross-tenant contamination.

        1. Try explicit session ID from request metadata.
        2. Fall back to a hash of the API key to isolate tenants.
        3. Ultimate fallback: per-request UUID (disables loop detection for
           stateless requests, preventing false positives).

        Args:
            data: The request data dictionary.
            user_api_key_dict: The user API key auth object, if available.

        Returns:
            A session identifier string.
        """
        result = get_session_id_from_request_data(data)
        if result:
            return result
        api_key = getattr(user_api_key_dict, "api_key", None) if user_api_key_dict is not None else None
        if isinstance(api_key, str) and api_key:
            hashed = hashlib.md5(api_key.encode()).hexdigest()[:16]
            return f"api_key_{hashed}"
        return f"req_{uuid.uuid4().hex}"

    @staticmethod
    def _count_matches(
        recent: list[tuple[str, str]],
        tool_name: str,
        canonical: str,
    ) -> int:
        """Count how many entries in ``recent`` match ``tool_name`` + ``canonical``."""
        return sum(1 for past_tool, past_args in recent if past_tool == tool_name and past_args == canonical)

    @staticmethod
    def _extract_tool_calls(
        messages: object,
    ) -> list[tuple[str, str]]:
        """Extract ``(tool_name, arguments_json)`` pairs from assistant messages.

        Supports both OpenAI ``tool_calls`` and Anthropic ``tool_use`` formats.

        Args:
            messages: The messages list from the request data.

        Returns:
            List of ``(tool_name, arguments_json)`` tuples from the most
            recent assistant message containing tool calls.
        """
        if not isinstance(messages, list):
            return []

        pairs: list[tuple[str, str]] = []
        for msg in reversed(messages):
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            if role != "assistant":
                continue

            # OpenAI format
            tool_calls = msg.get("tool_calls")
            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    func = tc.get("function", {})
                    name = func.get("name", "") if isinstance(func, dict) else ""
                    args_raw = func.get("arguments", "{}") if isinstance(func, dict) else "{}"
                    if name:
                        pairs.append((name, args_raw))
                if pairs:
                    break

            # Anthropic format
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        name = block.get("name", "")
                        args_raw = json.dumps(block.get("input", {}), sort_keys=True)
                        if name:
                            pairs.append((name, args_raw))
                if pairs:
                    break

        return pairs
