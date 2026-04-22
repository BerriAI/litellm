"""
Incremental signal detection for the adaptive router.

Each session maintains a SessionState. On every turn, we call apply_turn(state, turn)
which mutates the state in place and returns a SignalDelta listing which signals
fired on THIS turn. The router then queues the delta to be flushed to DB.

Design constraint: O(1) work per turn. No re-scanning the full session history.
We keep small bounded windows: last_user_content, last_assistant_content, and a
bounded list of recent tool call signatures.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from litellm.router_strategy.adaptive_router.config import (
    LOOP_REPEAT_THRESHOLD,
    MIN_TURNS_FOR_CLEAN_CREDIT,
    MISALIGNMENT_JACCARD_THRESHOLD,
    STAGNATION_JACCARD_NEAR_DUP,
    TOOL_CALL_HISTORY_MAX,
)


# ---- Public types ---------------------------------------------------------


@dataclass
class SignalDelta:
    """Which signals fired on a single turn. Counts are 0 or 1 (one delta per turn)."""

    misalignment: int = 0
    stagnation: int = 0
    disengagement: int = 0
    satisfaction: int = 0
    failure: int = 0
    loop: int = 0
    exhaustion: int = 0

    def any_fired(self) -> bool:
        return any(
            [
                self.misalignment,
                self.stagnation,
                self.disengagement,
                self.satisfaction,
                self.failure,
                self.loop,
                self.exhaustion,
            ]
        )


@dataclass
class SessionState:
    """In-memory rolling state for one session.

    Mirrors the LiteLLM_AdaptiveRouterSession DB row (Wave 0 schema). The flusher
    later persists this. We keep this as a plain dataclass — no DB coupling.
    """

    session_id: str
    router_name: str
    model_name: str
    classified_type: str

    misalignment_count: int = 0
    stagnation_count: int = 0
    disengagement_count: int = 0
    satisfaction_count: int = 0
    failure_count: int = 0
    loop_count: int = 0
    exhaustion_count: int = 0

    last_user_content: Optional[str] = None
    last_assistant_content: Optional[str] = None
    tool_call_history: List[str] = field(default_factory=list)
    pending_tool_calls: Dict[str, str] = field(default_factory=dict)

    turn_count: int = 0
    last_processed_turn: int = -1
    clean_credit_awarded: bool = False
    terminal_status: Optional[int] = None


@dataclass
class Turn:
    """One turn of input. Caller assembles this from the request/response."""

    user_content: Optional[str] = None
    assistant_content: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    response_status: Optional[int] = None


# ---- Detection helpers ----------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokens(text: Optional[str]) -> Set[str]:
    if not text:
        return set()
    return {t.lower() for t in _TOKEN_RE.findall(text)}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


_DISENGAGEMENT_PATTERNS = [
    re.compile(
        r"\b(forget it|never mind|give up|talk to (?:a )?human|cancel)\b", re.IGNORECASE
    ),
    re.compile(r"\b(this (?:isn'?t|is not) working|stop|abort)\b", re.IGNORECASE),
    re.compile(r"\bi'?ll do it (?:myself|manually)\b", re.IGNORECASE),
]

_SATISFACTION_PATTERNS = [
    re.compile(
        r"\b(that worked|that did it|works now|fixed it|solved it|nice)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(thanks|thank you|thx|appreciated|appreciate it)\b", re.IGNORECASE),
    re.compile(r"\b(perfect|great|excellent|exactly)\b", re.IGNORECASE),
]


def _detect_misalignment(prev_user: Optional[str], curr_user: Optional[str]) -> bool:
    """Fires when consecutive user messages share *some* topic (jaccard > 0)
    but are sufficiently different (jaccard < threshold) — i.e. user is
    rephrasing, not changing topic, not repeating."""
    if not prev_user or not curr_user:
        return False
    j = _jaccard(_tokens(prev_user), _tokens(curr_user))
    return 0.0 < j < MISALIGNMENT_JACCARD_THRESHOLD


def _detect_stagnation(prev_asst: Optional[str], curr_asst: Optional[str]) -> bool:
    """Fires when consecutive assistant messages are near-duplicates."""
    if not prev_asst or not curr_asst:
        return False
    j = _jaccard(_tokens(prev_asst), _tokens(curr_asst))
    return j >= STAGNATION_JACCARD_NEAR_DUP


def _detect_disengagement(curr_user: Optional[str]) -> bool:
    if not curr_user:
        return False
    return any(p.search(curr_user) for p in _DISENGAGEMENT_PATTERNS)


def _detect_satisfaction(curr_user: Optional[str]) -> bool:
    if not curr_user:
        return False
    return any(p.search(curr_user) for p in _SATISFACTION_PATTERNS)


def _detect_failure(tool_results: List[Dict[str, Any]]) -> bool:
    """Any tool result explicitly flagged as an error.

    We do NOT treat empty content as failure — many tools legitimately return
    empty output (zero-result searches, silent bash commands, void writes) and
    penalizing the model for those would corrupt the bandit posterior.
    """
    for r in tool_results:
        if r.get("is_error"):
            return True
    return False


def _signature(call: Dict[str, Any]) -> str:
    """Stable signature for loop detection: name + sorted JSON-ish args."""
    name = call.get("name") or call.get("function", {}).get("name", "")
    call_args = call.get("arguments")
    if call_args is None:
        call_args = call.get("function", {}).get("arguments", "")
    if isinstance(call_args, dict):
        call_args = ",".join(f"{k}={call_args[k]}" for k in sorted(call_args.keys()))
    return f"{name}({call_args})"


def _detect_loop(history: List[str], new_calls: List[Dict[str, Any]]) -> bool:
    """Fires if any new call's signature appears >= LOOP_REPEAT_THRESHOLD-1 times
    in recent history (so this call would be the Nth)."""
    if not new_calls:
        return False
    for call in new_calls:
        sig = _signature(call)
        recent_count = history.count(sig)
        if recent_count >= LOOP_REPEAT_THRESHOLD - 1:
            return True
    return False


_EXHAUSTION_STATUSES = {408, 413, 429, 503, 504}

_EXHAUSTION_KEYWORDS = (
    "context length",
    "context window",
    "token limit",
    "rate limit",
    "too many requests",
    "timeout",
)


def _detect_exhaustion(
    status: Optional[int], tool_results: List[Dict[str, Any]]
) -> bool:
    if status is not None and status in _EXHAUSTION_STATUSES:
        return True
    for r in tool_results:
        content = str(r.get("content", "")).lower()
        if any(kw in content for kw in _EXHAUSTION_KEYWORDS):
            return True
    return False


# ---- Public entrypoint ----------------------------------------------------


def apply_turn(state: SessionState, turn: Turn) -> SignalDelta:
    """
    Detect signals on this turn, mutate state, return the delta.

    O(1) per turn (no full-history rescan). Only inspects last_*, recent tool history
    (which is bounded at TOOL_CALL_HISTORY_MAX), and the new turn payload.
    """
    delta = SignalDelta()

    if _detect_misalignment(state.last_user_content, turn.user_content):
        delta.misalignment = 1
    if _detect_stagnation(state.last_assistant_content, turn.assistant_content):
        delta.stagnation = 1
    if _detect_disengagement(turn.user_content):
        delta.disengagement = 1
    if _detect_satisfaction(turn.user_content):
        # Gate: only award satisfaction credit once per session, and only
        # after MIN_TURNS_FOR_CLEAN_CREDIT turns of context. Early "thanks"
        # on turn 1-2 is noise, not a validated quality signal.
        current_turn_index = state.turn_count + 1
        if (
            not state.clean_credit_awarded
            and current_turn_index >= MIN_TURNS_FOR_CLEAN_CREDIT
        ):
            delta.satisfaction = 1
            state.clean_credit_awarded = True
    if _detect_failure(turn.tool_results):
        delta.failure = 1
    if _detect_loop(state.tool_call_history, turn.tool_calls):
        delta.loop = 1
    if _detect_exhaustion(turn.response_status, turn.tool_results):
        delta.exhaustion = 1

    state.misalignment_count += delta.misalignment
    state.stagnation_count += delta.stagnation
    state.disengagement_count += delta.disengagement
    state.satisfaction_count += delta.satisfaction
    state.failure_count += delta.failure
    state.loop_count += delta.loop
    state.exhaustion_count += delta.exhaustion

    if turn.user_content:
        state.last_user_content = turn.user_content
    if turn.assistant_content:
        state.last_assistant_content = turn.assistant_content

    for call in turn.tool_calls:
        state.tool_call_history.append(_signature(call))
    if len(state.tool_call_history) > TOOL_CALL_HISTORY_MAX:
        state.tool_call_history = state.tool_call_history[-TOOL_CALL_HISTORY_MAX:]

    if turn.response_status is not None:
        state.terminal_status = turn.response_status

    state.turn_count += 1
    state.last_processed_turn = state.turn_count

    return delta
