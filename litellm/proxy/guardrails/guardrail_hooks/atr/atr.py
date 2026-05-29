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

import json
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

    @log_guardrail_information
    async def async_post_call_streaming_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: str,
    ) -> Any:
        """
        Scan the aggregated streamed response after stream completion.

        ATR rules match against complete content (a regex over a full
        response). Per-chunk scanning would emit false negatives (the
        attack pattern split across two chunks never appears in either)
        and inconsistent false positives. LiteLLM aggregates the streamed
        response before this hook fires, so we get a uniform policy
        whether the caller opts into streaming or not.

        Known limitation (documented for honesty rather than fixed): an
        attacker who streams a long-running response specifically to
        inject content that is acted on mid-stream is out of scope. That
        requires per-chunk inspection with a stateful aggregator and a
        semantic gate, not a regex catalog.
        """
        if response is None or len(response) == 0:
            return response

        matches = self._scan(response, event_type="llm_output")
        if matches:
            import json

            error_detail = {
                "error": "Streamed response blocked by ATR guardrail",
                "matched_rules": [self._summarize_match(m) for m in matches],
            }
            return f"data: {json.dumps({'error': error_detail})}\n\n"
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

        # Responses API (/v1/responses): data["input"] is str or content-part list.
        # OpenAI Responses API uses `input` instead of `messages` and the same
        # part-list shape applies (per veria-ai #28050 review medium 2026-05-27).
        responses_input = data.get("input")
        if isinstance(responses_input, str):
            parts.append(responses_input)
        elif isinstance(responses_input, list):
            for item in responses_input:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                    # Responses API also nests content parts under "content"
                    nested_content = item.get("content")
                    if isinstance(nested_content, str):
                        parts.append(nested_content)
                    elif isinstance(nested_content, list):
                        for chunk in nested_content:
                            if isinstance(chunk, dict):
                                ctext = chunk.get("text")
                                if isinstance(ctext, str):
                                    parts.append(ctext)

        # Tool / function definitions can carry prompt injection in
        # function.description or function.parameters (per veria-ai #28050
        # review medium 2026-05-27). A malicious client can inject hidden
        # instructions in the tool catalog that the LLM treats as system text.
        for tool in data.get("tools") or []:
            if not isinstance(tool, dict):
                continue
            # OpenAI tool function shape: tool.type == "function" with tool.function
            if tool.get("type") == "function":
                fn = tool.get("function") or {}
                if isinstance(fn, dict):
                    for key in ("name", "description"):
                        val = fn.get(key)
                        if isinstance(val, str):
                            parts.append(val)
                    params = fn.get("parameters")
                    if params is not None:
                        try:
                            parts.append(json.dumps(params, ensure_ascii=False))
                        except (TypeError, ValueError):
                            pass
            # Anthropic / Claude tool shape: tool.name + tool.description direct
            for key in ("name", "description"):
                val = tool.get(key)
                if isinstance(val, str):
                    parts.append(val)

        # tool_choice can carry a function definition when the client wants to
        # force a specific tool. Scan its description too.
        tool_choice = data.get("tool_choice")
        if isinstance(tool_choice, dict):
            fn = tool_choice.get("function") or {}
            if isinstance(fn, dict):
                desc = fn.get("description")
                if isinstance(desc, str):
                    parts.append(desc)

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

        # Responses API (/v1/responses): response.output is a list of message
        # objects each with content parts (per veria-ai #28050 review medium
        # 2026-05-27). Shape: response.output[i].content[j].text
        output = getattr(response, "output", None)
        if output is None and isinstance(response, dict):
            output = response.get("output")
        if isinstance(output, list):
            for item in output:
                # message objects with nested content parts
                content = getattr(item, "content", None)
                if content is None and isinstance(item, dict):
                    content = item.get("content")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    for chunk in content:
                        if isinstance(chunk, dict):
                            t = chunk.get("text")
                            if isinstance(t, str):
                                parts.append(t)
                        else:
                            t = getattr(chunk, "text", None)
                            if isinstance(t, str):
                                parts.append(t)
                # Some Responses API shapes put text directly on the item
                if isinstance(item, dict):
                    direct = item.get("text")
                    if isinstance(direct, str):
                        parts.append(direct)

        # Responses API top-level output_text convenience field
        output_text = getattr(response, "output_text", None)
        if output_text is None and isinstance(response, dict):
            output_text = response.get("output_text")
        if isinstance(output_text, str) and output_text:
            parts.append(output_text)

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
