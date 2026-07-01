"""
Microloop Guardrail for LiteLLM.
=================================
This integration uses the official Microloop PyPI package to detect 
infinite agent loops using the high-performance Rust engine.

If the user has not installed `microloop`, this guardrail silently 
disables itself to prevent breaking the LiteLLM deployment.
"""
import json
import logging
from typing import Any, Dict, Optional, List

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

logger = logging.getLogger(__name__)


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

            if verdict != 0:
                # Use LiteLLM's native exception to cleanly block the call
                error_msg = (
                    f"Microloop: Infinite loop detected on tool '{tool_name}' "
                    f"(verdict: {verdict}). Blocked to prevent runaway API costs."
                )
                raise GuardrailRaisedException(exception=Exception(error_msg))
