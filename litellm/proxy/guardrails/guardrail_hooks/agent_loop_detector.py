"""
agent_loop_detector — a LiteLLM guardrail that detects and halts runaway
agent loops before each LLM call.

Signals detected
----------------
1. **Fingerprint repetition**: the same (tool_name, normalized_args) pair
   appears >= ``fingerprint_threshold`` times in the last ``window_size``
   assistant turns.
2. **Sequence repetition**: a sub-sequence of tool names (length
   ``sequence_min_len``-``sequence_max_len``) repeats >= 2 times
   consecutively in recent history.

Both OpenAI-format messages (``tool_calls`` on assistant turns) and
Anthropic-format messages (``tool_use`` content blocks) are handled.

Usage — Proxy config.yaml (recommended)
--------------------------
::

    guardrails:
      - guardrail_name: "agent-loop-detector"
        litellm_params:
          guardrail: litellm.proxy.guardrails.guardrail_hooks.agent_loop_detector.AgentLoopDetector
          mode: pre_call
          fingerprint_threshold: 3
          window_size: 20

Usage — SDK
-----------
SDK dispatch only runs a guardrail for requests that carry a ``guardrails`` list.
Register the detector as a callback and pass ``guardrails=[...]`` on every
``acompletion`` call that should be checked::

    import litellm
    from litellm.proxy.guardrails.guardrail_hooks.agent_loop_detector import (
        AgentLoopDetector,
    )

    litellm.callbacks = [
        AgentLoopDetector(
            guardrail_name="agent-loop-detector",
            fingerprint_threshold=3,
            window_size=20,
        )
    ]

    response = await litellm.acompletion(
        model="claude-sonnet-5-5",
        messages=messages,
        guardrails=["agent-loop-detector"],
    )
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

from litellm._logging import verbose_proxy_logger
from litellm.exceptions import GuardrailRaisedException
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.types.guardrails import GuardrailEventHooks

if TYPE_CHECKING:
    from litellm.caching import DualCache
    from litellm.proxy._types import UserAPIKeyAuth


class AgentLoopDetector(CustomGuardrail):
    """Pre-call guardrail that halts agent loops before the next LLM call.

    Parameters
    ----------
    window_size:
        How many of the most-recent assistant turns to inspect. Default: 20.
    fingerprint_threshold:
        How many times the identical (tool, args) pair must appear within
        *window_size* turns to trigger a halt. Default: 3.
    sequence_min_len / sequence_max_len:
        Range of tool-name sub-sequence lengths checked for consecutive
        repetition. Defaults: 2 / 5.
    """

    def __init__(
        self,
        window_size: int = 20,
        fingerprint_threshold: int = 3,
        sequence_min_len: int = 2,
        sequence_max_len: int = 5,
        **kwargs: Any,
    ) -> None:
        self.window_size = window_size
        self.fingerprint_threshold = fingerprint_threshold
        self.sequence_min_len = sequence_min_len
        self.sequence_max_len = sequence_max_len
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize(name: str) -> str:
        """Replace control characters in tool names to prevent log injection."""
        return "".join(c if c.isprintable() else f"\\x{ord(c):02x}" for c in name)

    @staticmethod
    def _fingerprint(tool_name: str, args: Any) -> str:
        """Return a 16-hex-char SHA-256 of ``(tool_name, sorted_args_json)``."""
        try:
            args_str = json.dumps(args, sort_keys=True, default=str)
        except Exception:
            args_str = str(args)
        digest = hashlib.sha256(f"{tool_name}:{args_str}".encode()).hexdigest()
        return digest[:16]

    def _extract_tool_fingerprints(self, messages: list[dict[str, Any]]) -> list[tuple[str, str]]:
        """Extract ``(tool_name, fingerprint)`` pairs from recent assistant turns.

        Handles:
        - OpenAI format: ``{"role": "assistant", "tool_calls": [...]}``
        - Anthropic format: ``{"role": "assistant", "content": [{"type": "tool_use", ...}]}``
        """
        pairs: list[tuple[str, str]] = []
        assistant_turns = [m for m in messages if m.get("role") == "assistant"]
        for turn in assistant_turns[-self.window_size :]:
            # OpenAI shape -----------------------------------------------
            tool_calls = turn.get("tool_calls")
            if tool_calls and isinstance(tool_calls, list):
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function") or {}
                    name = fn.get("name") or ""
                    raw_args = fn.get("arguments") or "{}"
                    try:
                        args = json.loads(raw_args)
                    except (json.JSONDecodeError, TypeError):
                        args = raw_args
                    if name:
                        pairs.append((name, self._fingerprint(name, args)))
                continue  # don't double-count with Anthropic branch

            # Anthropic shape --------------------------------------------
            content = turn.get("content")
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "tool_use":
                        name = block.get("name") or ""
                        inp = block.get("input") or {}
                        if name:
                            pairs.append((name, self._fingerprint(name, inp)))

        return pairs

    def _check_fingerprint_repetition(self, pairs: list[tuple[str, str]]) -> str | None:
        """Return a human-readable reason if any fingerprint repeats too often."""
        counts: dict[str, tuple[str, int]] = {}  # fp -> (name, count)
        for name, fp in pairs:
            prev_name, count = counts.get(fp, (name, 0))
            count += 1
            counts[fp] = (prev_name, count)
            if count >= self.fingerprint_threshold:
                return (
                    f"tool '{self._sanitize(name)}' called with identical arguments "
                    f"{count} times in the last {len(pairs)} assistant turns"
                )
        return None

    def _check_sequence_repetition(self, pairs: list[tuple[str, str]]) -> str | None:
        """Return a reason if any tool-name sub-sequence repeats consecutively."""
        names = [name for name, _ in pairs]
        n = len(names)
        for seq_len in range(self.sequence_min_len, self.sequence_max_len + 1):
            if n < seq_len * 2:
                continue
            for start in range(n - seq_len * 2 + 1):
                window = names[start : start + 2 * seq_len]
                seq = tuple(window[:seq_len])
                # Count consecutive repetitions of *seq* from position 0 in window
                reps, idx = 0, 0
                while idx + seq_len <= len(window):
                    if tuple(window[idx : idx + seq_len]) == seq:
                        reps += 1
                        idx += seq_len
                    else:
                        break
                if reps >= 2:
                    readable = " -> ".join(self._sanitize(s) for s in seq)
                    return f"tool sequence [{readable}] repeated {reps} times consecutively in the last {n} turns"
        return None

    # ------------------------------------------------------------------
    # Guardrail interface
    # ------------------------------------------------------------------

    def should_run_guardrail(self, data: dict[str, Any], event_type: GuardrailEventHooks) -> bool:
        return event_type == GuardrailEventHooks.pre_call

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict[str, Any],
        call_type: str,
    ) -> dict[str, Any]:
        """Inspect ``data["messages"]`` and raise if a loop is detected."""
        messages: list[dict[str, Any]] = data.get("messages") or []
        if not messages:
            return data

        pairs = self._extract_tool_fingerprints(messages)
        if len(pairs) < 2:
            return data  # not enough tool-call history to make a judgement

        reason = self._check_fingerprint_repetition(pairs) or self._check_sequence_repetition(pairs)

        if reason is not None:
            verbose_proxy_logger.warning("AgentLoopDetector: loop detected — %s. Halting session.", reason)
            raise GuardrailRaisedException(
                guardrail_name="AgentLoopDetector",
                message=(
                    f"loop detected — {reason}. "
                    "The agent session has been halted to prevent runaway costs. "
                    "Inspect your agent's tool-calling logic or raise "
                    "`fingerprint_threshold` / `window_size` if this is a "
                    "false positive."
                ),
                blocked_content=True,
            )

        return data
