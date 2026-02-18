"""
Block Hosted Tools Guardrail

Provider-agnostic guardrail that blocks platform-executed hosted tools from
Anthropic, OpenAI, and Google Gemini.  Tool definitions are loaded from
per-provider YAML files in this directory so the list can be extended without
touching Python code.

Detection strategy per provider:
  Anthropic — prefix match on tool 'type' field (versioned: bash_20250124, etc.)
  OpenAI    — exact match on tool 'type' field (code_interpreter, file_search, …)
  Gemini    — exact match on tool 'type' field OR presence of a top-level key
               (googleSearch, codeExecution, …) in the tool dict
"""

import os
from typing import TYPE_CHECKING, Dict, FrozenSet, List, Literal, Optional, Tuple

import yaml
from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

_DIR = os.path.dirname(__file__)


# ---------------------------------------------------------------------------
# Config loading — runs once at import time
# ---------------------------------------------------------------------------

def _load_provider_configs() -> Dict[str, dict]:
    """Load all *.yaml files in this directory, keyed by provider name."""
    configs: Dict[str, dict] = {}
    for fname in os.listdir(_DIR):
        if not fname.endswith(".yaml"):
            continue
        with open(os.path.join(_DIR, fname)) as f:
            cfg = yaml.safe_load(f)
        provider = cfg.get("provider")
        if provider:
            configs[provider] = cfg
    return configs


def _build_match_sets(
    configs: Dict[str, dict],
) -> Tuple[tuple, FrozenSet[str], FrozenSet[str]]:
    """
    Derive three matching structures from all provider configs:

    Returns:
        type_prefixes  — tuple of prefix strings (Anthropic-style versioned types)
        exact_types    — frozenset of exact type strings (OpenAI / Gemini-compat)
        top_level_keys — frozenset of Gemini native top-level dict keys
    """
    prefixes: List[str] = []
    exact: List[str] = []
    keys: List[str] = []

    for cfg in configs.values():
        for entry in cfg.get("tool_type_prefixes", []):
            if isinstance(entry, dict) and "prefix" in entry:
                prefixes.append(entry["prefix"])
        for entry in cfg.get("tool_type_exact", []):
            if isinstance(entry, dict) and "type" in entry:
                exact.append(entry["type"])
        for entry in cfg.get("tool_top_level_keys", []):
            if isinstance(entry, dict) and "key" in entry:
                keys.append(entry["key"])

    return tuple(prefixes), frozenset(exact), frozenset(keys)


_PROVIDER_CONFIGS = _load_provider_configs()
_TYPE_PREFIXES, _EXACT_TYPES, _TOP_LEVEL_KEYS = _build_match_sets(_PROVIDER_CONFIGS)


# ---------------------------------------------------------------------------
# Tool-matching helpers
# ---------------------------------------------------------------------------

def _match_tool(tool: dict) -> Optional[str]:
    """
    Return a human-readable description if the tool is a known hosted tool,
    or None if it is a user-defined tool (pass through).
    """
    if not isinstance(tool, dict):
        return None

    raw_type: Optional[str] = tool.get("type")

    # Skip generic OpenAI function wrapper — never a hosted tool
    if raw_type == "function":
        return None

    if raw_type:
        # Prefix match (Anthropic versioned types)
        if raw_type.startswith(_TYPE_PREFIXES):
            name = tool.get("name") or raw_type
            return f"{name} ({raw_type})"
        # Exact match (OpenAI / Gemini-compat)
        if raw_type in _EXACT_TYPES:
            name = tool.get("name") or raw_type
            return f"{name} ({raw_type})"

    # Gemini native top-level keys ({"googleSearch": {}, ...})
    for key in _TOP_LEVEL_KEYS:
        if key in tool:
            return f"{key}"

    return None


# ---------------------------------------------------------------------------
# Guardrail class
# ---------------------------------------------------------------------------

class BlockHostedToolsGuardrail(CustomGuardrail):
    """
    Guardrail that blocks platform-hosted tools from Anthropic, OpenAI, and Gemini.

    Raises HTTP 403 if any tool in the request matches a known hosted tool
    from any supported provider.  Provider tool lists are defined in YAML
    files alongside this module and loaded at import time.
    """

    def __init__(self, **kwargs):
        if "supported_event_hooks" not in kwargs:
            kwargs["supported_event_hooks"] = [GuardrailEventHooks.pre_call]
        super().__init__(**kwargs)
        verbose_proxy_logger.debug(
            f"BlockHostedToolsGuardrail initialized "
            f"(providers: {list(_PROVIDER_CONFIGS.keys())})"
        )

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        if input_type != "request":
            return inputs

        tools: List[dict] = list(inputs.get("tools") or [])  # type: ignore[assignment]
        blocked: List[str] = []

        for tool in tools:
            desc = _match_tool(tool)
            if desc:
                blocked.append(desc)

        if blocked:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": (
                        "the following platform-hosted tools are disabled by "
                        "your organization's policy: " + ", ".join(blocked)
                    ),
                    "guardrail": self.guardrail_name,
                    "blocked_tools": blocked,
                },
            )

        return inputs
