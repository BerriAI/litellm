"""
Microloop Guardrail for LiteLLM.
=================================
This integration uses the official Microloop PyPI package to detect 
infinite agent loops using the high-performance Rust engine.

If the user has not installed `microloop`, this guardrail silently 
disables itself to prevent breaking the LiteLLM deployment.
"""
import json
<<<<<<< HEAD
import logging
from typing import Any, Dict, Optional, List
=======
import uuid

from typing import Any
>>>>>>> f26cdbdbb0609b8d13fd74a0694ca02a56d2990c

# 1. Graceful import of the Microloop Rust engine
try:
    from microloop import Microloop
    MICROLOOP_AVAILABLE = True
except ImportError:
    MICROLOOP_AVAILABLE = False

# LiteLLM native imports
try:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.exceptions import GuardrailRaisedException
except ImportError:  # pragma: no cover - defensive fallback
    CustomGuardrail = object  # type: ignore[misc,assignment]
    GuardrailRaisedException = Exception  # type: ignore[assignment,misc]

<<<<<<< HEAD
logger = logging.getLogger(__name__)
=======

class _CallHistory:
    """Tracks tool call trajectory digests per session with bounded memory.

    Stores only fixed-size SHA-256 digests (64 bytes each), never raw JSON.
    This prevents CWE-400 (Uncontrolled Resource Consumption): a malicious
    client sending 10 MB payloads cannot inflate memory beyond the per-entry
    digest cost.
    """

    MAX_SESSIONS = 10000

    def __init__(self) -> None:
        self._store: dict[str, list[tuple[str, str]]] = {}

    def append(self, session_id: str, tool_name: str, trajectory_hash: str) -> None:
        if len(self._store) >= self.MAX_SESSIONS and session_id not in self._store:
            oldest_key = next(iter(self._store))
            self._store.pop(oldest_key, None)
        if session_id not in self._store:
            self._store[session_id] = []
        self._store[session_id].append((tool_name, trajectory_hash))

    def get_recent(self, session_id: str, window: int) -> list[tuple[str, str]]:
        entries = self._store.get(session_id, [])
        return entries[-window:]

    def trim(self, session_id: str, max_len: int) -> None:
        entries = self._store.get(session_id, [])
        if len(entries) > max_len:
            self._store[session_id] = entries[-max_len:]

    def correct_last(self, session_id: str, tool_name: str, trajectory_hash: str) -> None:
        """Replace the hash of the most recent entry for *tool_name*.

        Used when auto-inference discovers new volatile fields: the previous
        call's hash must be retroactively fixed so consistency with the new
        canonical form is maintained.
        """
        entries = self._store.get(session_id)
        if not entries:
            return
        for i in range(len(entries) - 1, -1, -1):
            if entries[i][0] == tool_name:
                entries[i] = (tool_name, trajectory_hash)
                return

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


def _compute_trajectory_hash(tool_name: str, canonical_args: str) -> str:
    """Deterministic SHA-256 digest of a tool call trajectory.

    Returns a hex string (64 bytes) — fixed-size regardless of payload size.
    This is the only value stored in ``_CallHistory``, preventing unbounded
    memory growth from large JSON payloads (CWE-400 mitigation).
    """
    raw = f"{tool_name}|{canonical_args}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _auto_infer_volatile(
    last_raw_args: dict[str, Any] | None,
    current_args: str,
) -> set[str]:
    """Detect fields that differ between the last call and current call.

    Unlike the previous implementation, this only compares the immediately
    preceding raw args (O(1) memory) rather than scanning the full history
    window (O(n)), eliminating the unbounded raw-JSON storage vector.

    A field is considered volatile only if:
    1. It is the *only* differing field between the two calls
    2. AND at least one other field stays constant

    This prevents inferring the only changing parameter as volatile.

    Args:
        last_raw_args: The ``dict`` of the immediately preceding call (or ``None``).
        current_args: JSON-encoded string of the current call arguments.

    Returns:
        Set of field names inferred as volatile.
    """
    inferred: set[str] = set()

    if last_raw_args is None:
        return inferred

    current = _parse_json_dict(current_args)
    if current is None:
        return inferred

    diffs: list[str] = []
    consts: list[str] = []
    all_keys = set(current.keys()) | set(last_raw_args.keys())
    for k in all_keys:
        if current.get(k) != last_raw_args.get(k):
            diffs.append(k)
        else:
            consts.append(k)

    if len(diffs) == 1 and consts:
        inferred.add(diffs[0])

    return inferred


# ---------------------------------------------------------------------------
# Main guardrail class
# ---------------------------------------------------------------------------
>>>>>>> f26cdbdbb0609b8d13fd74a0694ca02a56d2990c


class MicroloopGuardrail(CustomGuardrail):
    """
    Deterministic loop detection powered by the Microloop Rust engine.
    """
    def __init__(
        self,
        max_repeats: int = 3,
        history_window: int = 10,
        volatile_fields: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(
            guardrail_name="microloop",
            supported_event_hooks=["pre_call"],
            default_on=True,
            **kwargs
        )
<<<<<<< HEAD
=======
        self._max_repeats = max_repeats
        self._history_window = history_window or (max_repeats * 2)
        self._volatile_fields: set[str] = set(volatile_fields or [])
        self._auto_infer_volatile = auto_infer_volatile
        self._history = _CallHistory()
        # O(1) raw-args cache for auto_infer_volatile — only the immediately
        # preceding call per (session, tool), never the full history window.
        self._last_raw_args: dict[str, dict[str, Any]] = {}
>>>>>>> f26cdbdbb0609b8d13fd74a0694ca02a56d2990c

        self.max_repeats = max_repeats
        self.history_window = history_window
        self.volatile_fields = volatile_fields or []

        # Per-session engine isolation.
        # Because the PyO3 binding holds state internally, we instantiate
        # one Rust engine per session_id to prevent cross-tenant contamination.
        self._engines: Dict[str, Any] = {}

        if not MICROLOOP_AVAILABLE:
            logger.warning(
                "Microloop package not found. Guardrail is disabled. "
                "Install it via: pip install microloop"
            )

    def _get_engine(self, session_id: str) -> Optional[Any]:
        """Lazily initialize the Rust engine for a specific session."""
        if not MICROLOOP_AVAILABLE:
            return None

        if session_id not in self._engines:
            # Construct YAML config for the Rust engine
            yaml_config = (
                f"max_repeats: {self.max_repeats}\n"
                f"history_window: {self.history_window}\n"
                f"volatile_fields: {json.dumps(self.volatile_fields)}\n"
            )
            self._engines[session_id] = Microloop(yaml_config)

        return self._engines[session_id]

    def _extract_all_tool_calls(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extracts all tool calls, supporting both OpenAI and Anthropic formats."""
        messages = data.get("messages", [])
        if not messages:
            return []

        last_msg = messages[-1]
        tool_calls = []

        # OpenAI format
        for call in last_msg.get("tool_calls") or []:
            fn = call.get("function")
            if fn and fn.get("name"):
                tool_calls.append({
                    "name": fn["name"],
                    "arguments": fn.get("arguments", "{}")
                })

        # Anthropic format
        for block in last_msg.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tool_calls.append({
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {}))
                })

        return tool_calls

    def _get_session_id(self, data: Dict[str, Any]) -> str:
        metadata = data.get("metadata", {}) or {}
        return str(metadata.get("session_id") or data.get("litellm_session_id", "default"))

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Dict[str, Any],
        data: Dict[str, Any],
        call_type: Any,
    ) -> Optional[Dict[str, Any]]:
        """LiteLLM pre-call hook. Intercepts tool calls to check for loops."""
        if not MICROLOOP_AVAILABLE:
            return  # Fail open if package isn't installed

        session_id = self._get_session_id(data)
        engine = self._get_engine(session_id)

        if not engine:
            return

        # Check ALL tool calls in the current turn
        tool_calls = self._extract_all_tool_calls(data)

        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            raw_args = tool_call["arguments"]

            # Ensure args is a string for the Rust engine
            if not isinstance(raw_args, str):
                raw_args = json.dumps(raw_args)

            # Call the Microloop Rust Engine (Returns u8: 0 = Allow, >0 = Block)
            verdict = engine.verify(tool_name, raw_args)

<<<<<<< HEAD
            if verdict != 0:
                # Use LiteLLM's native exception to cleanly block the call
                error_msg = (
                    f"Microloop: Infinite loop detected on tool '{tool_name}' "
                    f"(verdict: {verdict}). Blocked to prevent runaway API costs."
                )
                raise GuardrailRaisedException(exception=Exception(error_msg))
=======
        return None

    # ---- Core loop detection ---------------------------------------------

    def _check_and_record(self, session_id: str, tool_name: str, raw_args: str) -> None:
        """Check if ``tool_name(raw_args)`` is a loop, then record it.

        Memory safety: raw_args is canonicalized and hashed *before* any
        history interaction. The raw payload is never stored in ``_CallHistory``
        — only the fixed-size SHA-256 digest enters the history store,
        preventing CWE-400 (Uncontrolled Resource Consumption) via oversized
        payloads.

        Args:
            session_id: The session identifier.
            tool_name: Name of the tool being called.
            raw_args: JSON-encoded arguments string.

        Raises:
            GuardrailRaisedException: When the same tool+arguments repeat
                beyond the configured limit.
        """
        # 1. Canonicalize (parse + sort keys + strip configured volatile fields)
        canonical = _canonicalize_args(raw_args, self._volatile_fields)

        # 2. Auto-inference: compare current raw args to the immediately
        #    preceding raw args (O(1) per-session per-tool cache).
        inferred: set[str] = set()
        if self._auto_infer_volatile:
            cache_key = f"{session_id}:{tool_name}"
            prev_raw = self._last_raw_args.get(cache_key)
            inferred = _auto_infer_volatile(prev_raw, raw_args)
            if inferred:
                combined = self._volatile_fields | inferred
                canonical = _canonicalize_args(raw_args, combined)

        # 3. Compute fixed-size digest — raw JSON is NEVER stored after this point
        trajectory_hash = _compute_trajectory_hash(tool_name, canonical)

        # 4. If auto-inference discovered new volatile fields, retroactively fix
        #    the last history entry for this tool so previous hashes match the
        #    new canonical form. This ensures loop detection works across the
        #    boundary where the volatile field was first learned.
        if inferred:
            self._history.correct_last(session_id, tool_name, trajectory_hash)

        # 5. Check digest-only history for repeats
        recent = self._history.get_recent(session_id, self._history_window)
        match_count = sum(1 for t, h in recent if t == tool_name and h == trajectory_hash)

        # 6. Block or record
        if match_count + 1 >= self._max_repeats:
            raise GuardrailRaisedException(
                guardrail_name="microloop",
                message=(
                    f"Microloop: Loop detected on tool '{tool_name}' -- "
                    f"seen {match_count + 1}x (limit: {self._max_repeats})"
                    f"{' in session ' + session_id if session_id else ''}"
                ),
            )

        self._history.append(session_id, tool_name, trajectory_hash)
        self._history.trim(session_id, self._history_window * 2)

        # 7. Update O(1) raw-args cache for next auto-inference
        parsed = _parse_json_dict(raw_args)
        if parsed is not None:
            cache_key = f"{session_id}:{tool_name}"
            self._last_raw_args[cache_key] = parsed
            if len(self._last_raw_args) > _CallHistory.MAX_SESSIONS:
                self._last_raw_args.pop(next(iter(self._last_raw_args)), None)

    # ---- Helpers ---------------------------------------------------------

    def _get_session_id(self, data: dict, user_api_key_dict: object = None) -> str:
        """Extract or derive a session identifier.

        1. Try explicit session ID from request metadata.
        2. Fall back to hashed API key for best-effort per-tenant isolation.
           Note: all conversations using the same key without a session_id
           share one history bucket, which can cause false positives across
           independent agent conversations.
        3. Ultimate fallback: per-request UUID (disables loop detection for
           truly anonymous requests).

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
            hashed = hashlib.md5(api_key.encode(), usedforsecurity=False).hexdigest()[:16]
            return f"api_key_{hashed}"
        return f"req_{uuid.uuid4().hex}"

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
>>>>>>> f26cdbdbb0609b8d13fd74a0694ca02a56d2990c
