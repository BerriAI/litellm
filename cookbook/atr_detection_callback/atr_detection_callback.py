"""
ATR detection callback for LiteLLM.

A minimal CustomGuardrail that screens user input against a small set of
ATR-inspired regex patterns covering common AI agent threats: prompt
injection overrides, role-play jailbreaks, base64-wrapped instructions,
MCP tool override, and file:// SSRF. The patterns below are illustrative
copies; the full open detection set lives in Agent Threat Rules:
https://github.com/Agent-Threat-Rule/agent-threat-rules (Apache-2.0).

Wire it up via proxy_config.yaml:

    guardrails:
      - guardrail_name: "atr-input-screen"
        litellm_params:
          guardrail: cookbook.atr_detection_callback.atr_detection_callback.ATRDetectionGuardrail
          mode: "pre_call"
          default_on: true
"""

import re
from typing import Optional, Union

from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import CallTypesLiteral

# (rule_id, label, compiled_pattern). rule_id mirrors the ATR namespace.
ATR_INSPIRED_PATTERNS = [
    (
        "ATR-PI-001",
        "instruction override",
        re.compile(
            r"\b(ignore|disregard|forget)\s+(all\s+)?(previous|prior|above)\s+"
            r"(instructions?|prompts?|rules?)",
            re.IGNORECASE,
        ),
    ),
    (
        "ATR-PI-002",
        "system prompt exfiltration",
        re.compile(
            r"(reveal|print|repeat|show)\s+(your\s+)?"
            r"(system\s+prompt|initial\s+instructions)",
            re.IGNORECASE,
        ),
    ),
    (
        "ATR-PI-003",
        "role-play jailbreak",
        re.compile(
            r"\b(you\s+are\s+now|act\s+as|pretend\s+to\s+be)\s+"
            r"(DAN|developer\s+mode|jailbroken|an?\s+unrestricted)",
            re.IGNORECASE,
        ),
    ),
    (
        "ATR-PI-004",
        "base64-wrapped payload hint",
        re.compile(
            r"(decode|run|execute)\s+(this\s+)?base64[:\s]+[A-Za-z0-9+/=]{40,}",
            re.IGNORECASE,
        ),
    ),
    (
        "ATR-MCP-001",
        "mcp tool override",
        re.compile(r"<\s*(tool_override|mcp_override|new_tool_definition)\s*>", re.IGNORECASE),
    ),
    (
        "ATR-SSRF-001",
        "file:// scheme reference",
        re.compile(r"file://[^\s\"'<>]+", re.IGNORECASE),
    ),
]


class ATRDetectionGuardrail(CustomGuardrail):
    """Block requests that hit any ATR-inspired threat pattern."""

    def __init__(self, **kwargs):
        self.optional_params = kwargs
        super().__init__(**kwargs)

    @staticmethod
    def _scan(text: str):
        for rule_id, label, pattern in ATR_INSPIRED_PATTERNS:
            if pattern.search(text):
                return rule_id, label
        return None

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: CallTypesLiteral,
    ) -> Optional[Union[Exception, str, dict]]:
        for message in data.get("messages") or []:
            content = message.get("content")
            if not isinstance(content, str):
                continue
            hit = self._scan(content)
            if hit is None:
                continue
            rule_id, label = hit
            verbose_proxy_logger.warning(
                "ATR threat pattern matched: rule_id=%s label=%s", rule_id, label
            )
            raise ValueError(f"Request blocked by ATR guardrail: {rule_id} ({label}).")
        return data
