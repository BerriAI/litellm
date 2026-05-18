"""
ATR (Agent Threat Rules) guardrail integration for LiteLLM.

Scans LLM input and output against the open-source ATR detection rule
set, an MIT-licensed YAML-based format for AI-agent security threats
(prompt injection, tool poisoning, credential exfiltration, context
manipulation, and other categories).

Detection runs locally via the ``pyatr`` reference engine -- no network
call is required and no data leaves the proxy. ATR rules are evaluated
against ``llm_input`` events on the request hook and ``llm_output``
events on the response hook.

Configuration::

    guardrails:
      - guardrail_name: "atr-pre-call"
        litellm_params:
          guardrail: atr
          mode: "pre_call"
          rules_path: "./rules"            # optional, falls back to ATR_RULES_PATH
          severity_threshold: "high"        # critical | high | medium | low

Install::

    pip install pyatr

Rules and documentation: https://github.com/Agent-Threat-Rule/agent-threat-rules
"""

import os
from typing import (
    TYPE_CHECKING,
    Any,
    List,
    Literal,
    Optional,
    Type,
    Union,
)

from fastapi.exceptions import HTTPException

from litellm import DualCache
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.guardrails import GuardrailEventHooks

if TYPE_CHECKING:
    from litellm.types.proxy.guardrails.guardrail_hooks.base import (
        GuardrailConfigModel,
    )


_DEFAULT_SEVERITY_THRESHOLD = "high"
_SEVERITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}


class ATRGuardrailImportError(Exception):
    """Raised when the optional ``pyatr`` dependency is not installed."""


class ATRGuardrailRulesError(Exception):
    """Raised when ATR rules cannot be loaded from the configured path."""


class ATRGuardrail(CustomGuardrail):
    """Local ATR rule scanner for LiteLLM proxy."""

    def __init__(
        self,
        rules_path: Optional[str] = None,
        severity_threshold: Optional[str] = None,
        include_tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        try:
            from pyatr import ATREngine
        except ImportError as exc:
            raise ATRGuardrailImportError(
                "ATRGuardrail requires the `pyatr` package. "
                "Install it with: pip install pyatr"
            ) from exc

        threshold = (
            severity_threshold
            or os.environ.get("ATR_SEVERITY_THRESHOLD")
            or _DEFAULT_SEVERITY_THRESHOLD
        )
        threshold = threshold.lower()
        if threshold not in _SEVERITY_RANK:
            raise ATRGuardrailRulesError(
                f"Invalid severity_threshold '{threshold}'. "
                f"Must be one of: {sorted(_SEVERITY_RANK)}"
            )
        self.severity_threshold = threshold
        self.include_tags: Optional[List[str]] = include_tags or None

        self.engine = ATREngine()
        resolved_path = rules_path or os.environ.get("ATR_RULES_PATH")
        if resolved_path:
            if not os.path.isdir(resolved_path):
                raise ATRGuardrailRulesError(
                    f"ATR rules_path '{resolved_path}' is not a directory."
                )
            loaded = self.engine.load_rules_from_directory(resolved_path)
            verbose_proxy_logger.debug(
                "ATR guardrail loaded %d rules from %s", loaded, resolved_path
            )
        else:
            # Fall back to the rules directory bundled alongside pyatr.
            try:
                import pyatr as _pyatr

                bundled = (
                    _pyatr._DEFAULT_RULES_DIR
                    if hasattr(_pyatr, "_DEFAULT_RULES_DIR")
                    else None
                )
            except Exception:
                bundled = None
            if bundled and os.path.isdir(bundled):
                loaded = self.engine.load_rules_from_directory(bundled)
                verbose_proxy_logger.debug(
                    "ATR guardrail loaded %d bundled rules from %s",
                    loaded,
                    bundled,
                )
            else:
                raise ATRGuardrailRulesError(
                    "No ATR rules directory found. Set `rules_path` in the "
                    "guardrail config or the ATR_RULES_PATH environment "
                    "variable to a directory of ATR rule YAML files."
                )

        if "supported_event_hooks" not in kwargs:
            kwargs["supported_event_hooks"] = [
                GuardrailEventHooks.pre_call,
                GuardrailEventHooks.post_call,
            ]
        super().__init__(**kwargs)

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.atr import (
            ATRGuardrailConfigModel,
        )

        return ATRGuardrailConfigModel

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "pass_through_endpoint",
            "rerank",
            "mcp_call",
            "anthropic_messages",
        ],
    ) -> Union[Exception, str, dict, None]:
        event_type: GuardrailEventHooks = GuardrailEventHooks.pre_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return data

        content = self._extract_request_content(data)
        if not content:
            return data

        matches = self._scan(content, event_type="llm_input")
        if matches:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Request blocked by ATR guardrail",
                    "matched_rules": [self._summarize_match(m) for m in matches],
                },
            )
        return data

    @log_guardrail_information
    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response,
    ):
        event_type: GuardrailEventHooks = GuardrailEventHooks.post_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return response

        content = self._extract_response_content(response)
        if not content:
            return response

        matches = self._scan(content, event_type="llm_output")
        if matches:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Response blocked by ATR guardrail",
                    "matched_rules": [self._summarize_match(m) for m in matches],
                },
            )
        return response

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _extract_request_content(self, data: dict) -> str:
        parts: List[str] = []

        # Chat completions: messages[].content (str or content-part list)
        for msg in data.get("messages") or []:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for chunk in content:
                    if isinstance(chunk, dict):
                        text = chunk.get("text")
                        if isinstance(text, str):
                            parts.append(text)

        # Text completions (/v1/completions): prompt is str or list[str]
        prompt = data.get("prompt")
        if isinstance(prompt, str):
            parts.append(prompt)
        elif isinstance(prompt, list):
            for p in prompt:
                if isinstance(p, str):
                    parts.append(p)

        return "\n".join(p for p in parts if p)

    def _extract_response_content(self, response: Any) -> str:
        choices = getattr(response, "choices", None)
        if choices is None and isinstance(response, dict):
            choices = response.get("choices", [])
        parts: List[str] = []
        for choice in choices or []:
            # Chat completions: choice.message.content
            message = getattr(choice, "message", None)
            if message is None and isinstance(choice, dict):
                message = choice.get("message", {})
            if message is not None:
                content: Optional[str] = getattr(message, "content", None)
                if content is None and isinstance(message, dict):
                    content = message.get("content")
                if isinstance(content, str) and content:
                    parts.append(content)
                    continue

            # Text completions (/v1/completions): choice.text
            text = getattr(choice, "text", None)
            if text is None and isinstance(choice, dict):
                text = choice.get("text")
            if isinstance(text, str) and text:
                parts.append(text)

        return "\n".join(parts)

    def _scan(self, content: str, event_type: str) -> List[Any]:
        from pyatr import AgentEvent

        default_field = "user_input" if event_type == "llm_input" else "agent_output"
        event = AgentEvent(
            content=content,
            event_type=event_type,
            fields={default_field: content},
        )
        matches = self.engine.evaluate(event)
        threshold_rank = _SEVERITY_RANK[self.severity_threshold]

        result = []
        for m in matches:
            # include_tags filter: skip rules whose tags don't intersect the allow-list
            if self.include_tags is not None:
                tags = getattr(m, "tags", {}) or {}
                tag_values: set = (
                    set(tags.values()) if isinstance(tags, dict) else set()
                )
                if not tag_values.intersection(self.include_tags):
                    continue

            # Treat None or unrecognised severity conservatively (rank 0 = critical)
            raw_severity = getattr(m, "severity", None)
            severity_str = (
                (raw_severity or "").lower() if raw_severity is not None else ""
            )
            rank = _SEVERITY_RANK.get(severity_str, 0)
            if rank <= threshold_rank:
                result.append(m)

        return result

    @staticmethod
    def _summarize_match(match: Any) -> dict:
        return {
            "rule_id": getattr(match, "rule_id", ""),
            "title": getattr(match, "title", ""),
            "severity": getattr(match, "severity", ""),
        }
