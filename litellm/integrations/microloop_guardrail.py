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

import json
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
    """Tracks tool call trajectories per session."""

    def __init__(self) -> None:
        self._store: dict[str, list[tuple[str, str, datetime]]] = {}

    def append(self, session_id: str, tool_name: str, args_json: str) -> None:
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


def _strip_volatile_fields(args_json: str, volatile_fields: set[str]) -> str:
    """Remove known volatile fields from a JSON arguments string.

    Args:
        args_json: A JSON-encoded string of tool arguments.
        volatile_fields: Field names to strip from the JSON object.

    Returns:
        The JSON string with volatile fields removed, or the original
        string if it cannot be parsed or is not a JSON object.
    """
    if not volatile_fields:
        return args_json
    try:
        obj = json.loads(args_json)
        if not isinstance(obj, dict):
            return args_json
        for field in volatile_fields:
            obj.pop(field, None)
        return json.dumps(obj, sort_keys=True, separators=(",", ":"))
    except (json.JSONDecodeError, TypeError):
        return args_json


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

    try:
        current = json.loads(current_args)
        if not isinstance(current, dict):
            return inferred
    except (json.JSONDecodeError, TypeError):
        return inferred

    for _past_tool, past_args_str in recent_calls:
        try:
            past = json.loads(past_args_str)
            if not isinstance(past, dict):
                continue
        except (json.JSONDecodeError, TypeError):
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
        **kwargs: object,
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
            **kwargs: Forwarded to :class:`litellm.integrations.custom_guardrail.CustomGuardrail`.
        """
        guardrail_name = kwargs.pop("guardrail_name", "microloop")
        supported_event_hooks = kwargs.pop(
            "supported_event_hooks",
            None,  # will register for pre_call by default
        )
        default_on = kwargs.pop("default_on", True)
        super().__init__(
            guardrail_name=guardrail_name,
            supported_event_hooks=supported_event_hooks,
            default_on=default_on,
            **kwargs,
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
        session_id = self._get_session_id(data)
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

        # 1. Strip configured volatile fields
        canonical = _strip_volatile_fields(raw_args, self._volatile_fields)

        # 2. Count exact matches in recent history
        match_count = self._count_matches(recent, tool_name, canonical)

        # 3. Auto-inference (if enabled and no match found)
        if self._auto_infer_volatile and match_count == 0:
            inferred = _auto_infer_volatile(recent, raw_args)
            if inferred:
                combined = self._volatile_fields | inferred
                canonical = _strip_volatile_fields(raw_args, combined)
                match_count = 0
                for past_tool, past_args in recent:
                    stripped = _strip_volatile_fields(past_args, combined)
                    if past_tool == tool_name and stripped == canonical:
                        match_count += 1

        # 4. Block or record
        if match_count + 1 >= self._max_repeats:
            raise GuardrailRaisedException(
                guardrail_name="microloop",
                message=(
                    f"Microloop: Loop detected on tool '{tool_name}' -- "
                    f"seen {match_count + 1}x (limit: {self._max_repeats})"
                    f"{' in session ' + session_id if session_id != 'default' else ''}"
                ),
            )

        self._history.append(session_id, tool_name, canonical)
        self._history.trim(session_id, self._history_window * 2)

    # ---- Helpers ---------------------------------------------------------

    def _get_session_id(self, data: dict) -> str:
        """Extract or derive a session identifier from request data.

        Args:
            data: The request data dictionary.

        Returns:
            A session identifier string, defaulting to ``"default"``.
        """
        result = get_session_id_from_request_data(data)
        return result or "default"

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
