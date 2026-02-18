"""
Claude Code - Block Expensive API Flags Guardrail

Blocks Anthropic API parameters that trigger feature-specific pricing surcharges
(fast mode, inference_geo, extended thinking).  Also inherits the hosted tool
type prefixes from hosted_tool_types.yaml so hosted tools are blocked here too.

Blocked params are driven by expensive_api_flags.yaml which references
hosted_tool_types.yaml via `inherit_from`, following the same pattern as
harmful_child_safety.yaml inherits from harm_toxic_abuse.json.
"""

import os
from typing import TYPE_CHECKING, Any, List, Literal, Optional

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
_FLAGS_YAML = os.path.join(_DIR, "expensive_api_flags.yaml")
_TOOLS_YAML = os.path.join(_DIR, "hosted_tool_types.yaml")


def _load_config() -> dict:
    """Load expensive_api_flags.yaml, merging any inherited config."""
    with open(_FLAGS_YAML) as f:
        config: dict = yaml.safe_load(f)

    inherited_prefixes: List[str] = []
    inherit_from = config.get("inherit_from")
    if inherit_from:
        inherit_path = os.path.join(_DIR, inherit_from)
        with open(inherit_path) as f:
            inherited: dict = yaml.safe_load(f)
        inherited_prefixes = [
            entry["prefix"]
            for entry in inherited.get("tool_type_prefixes", [])
            if isinstance(entry, dict) and "prefix" in entry
        ]

    config["_inherited_tool_type_prefixes"] = inherited_prefixes
    return config


_CONFIG: dict = _load_config()
_BLOCKED_PARAMS: List[dict] = _CONFIG.get("blocked_params", [])
_INHERITED_TOOL_TYPE_PREFIXES: tuple = tuple(_CONFIG.get("_inherited_tool_type_prefixes", []))


def _tool_type(tool: dict) -> Optional[str]:
    t = tool.get("type")
    if t and t != "function":
        return t
    return None


def _is_hosted_tool(tool: dict) -> bool:
    if not _INHERITED_TOOL_TYPE_PREFIXES:
        return False
    t = _tool_type(tool)
    if not t:
        return False
    return t.startswith(_INHERITED_TOOL_TYPE_PREFIXES)


def _check_param(
    request_data: dict, param_cfg: dict
) -> Optional[str]:
    """
    Return an error message if the param in request_data matches a blocked value.
    Returns None when the param is not blocked.
    """
    param = param_cfg.get("param")
    if not param:
        return None

    value: Any = request_data.get(param)
    if value is None:
        return None

    nested_key = param_cfg.get("nested_key")
    if nested_key:
        # e.g. thinking.type — value must be a dict
        if not isinstance(value, dict):
            return None
        value = value.get(nested_key)
        if value is None:
            return None

    blocked_values: List[str] = param_cfg.get("blocked_values", [])
    if "*" in blocked_values or str(value) in blocked_values:
        return param_cfg.get("error_message", f"{param} is disabled by your organization's policy")

    return None


class ClaudeCodeBlockExpensiveFlagsGuardrail(CustomGuardrail):
    """
    Guardrail that blocks expensive Anthropic API flags.

    Checks request_data for feature-specific pricing flags (fast mode,
    inference_geo, extended thinking) and Anthropic-hosted tools inherited
    from hosted_tool_types.yaml.  Raises HTTP 403 on the first violation.
    """

    def __init__(self, **kwargs):
        if "supported_event_hooks" not in kwargs:
            kwargs["supported_event_hooks"] = [GuardrailEventHooks.pre_call]
        super().__init__(**kwargs)
        verbose_proxy_logger.debug("ClaudeCodeBlockExpensiveFlagsGuardrail initialized")

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

        # ------------------------------------------------------------------ #
        # 1. Check request-level blocked params (speed, inference_geo, etc.) #
        # ------------------------------------------------------------------ #
        for param_cfg in _BLOCKED_PARAMS:
            error_msg = _check_param(request_data, param_cfg)
            if error_msg:
                param = param_cfg.get("param", "unknown")
                raise HTTPException(
                    status_code=403,
                    detail={
                        "error": error_msg,
                        "guardrail": self.guardrail_name,
                        "blocked_param": param,
                    },
                )

        # ------------------------------------------------------------------ #
        # 2. Check for inherited hosted tools (from hosted_tool_types.yaml)  #
        # ------------------------------------------------------------------ #
        if _INHERITED_TOOL_TYPE_PREFIXES:
            tools: List[dict] = list(inputs.get("tools") or [])  # type: ignore[assignment]
            blocked_tools: List[str] = []
            for tool in tools:
                if _is_hosted_tool(tool):
                    tool_type = _tool_type(tool) or "unknown"
                    tool_name = tool.get("name") or tool_type
                    blocked_tools.append(f"{tool_name} ({tool_type})")

            if blocked_tools:
                raise HTTPException(
                    status_code=403,
                    detail={
                        "error": (
                            "the following Anthropic-hosted tools are disabled by "
                            "your organization's policy: " + ", ".join(blocked_tools)
                        ),
                        "guardrail": self.guardrail_name,
                        "blocked_tools": blocked_tools,
                    },
                )

        return inputs
