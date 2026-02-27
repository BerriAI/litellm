"""
Tool Policy Guardrail

Reads call_policy from LiteLLM_ToolTable and enforces it on LLM requests/responses.

Policy values:
  "trusted"   - allow through (no action)
  "untrusted" - allow through (no action; default for newly discovered tools)
  "blocked"   - raise HTTPException, preventing the tool call
  "dual_llm"  - (Phase 3) send to second LLM for verification; currently treated as allowed

Configuration in proxy config YAML:
  guardrails:
    - guardrail_name: "tool_policy"
      litellm_params:
        guardrail: tool_policy
        mode: post_call

or both pre and post call:
    - guardrail_name: "tool_policy"
      litellm_params:
        guardrail: tool_policy
        mode: during_call  # runs before LLM and on response
"""

from typing import TYPE_CHECKING, Any, Literal, Optional, Tuple

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.proxy.guardrails.tool_name_extraction import extract_request_tool_names
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

GUARDRAIL_NAME = "tool_policy"


def _get_request_object_permission_ids(
    request_data: dict,
) -> Tuple[Optional[str], Optional[str]]:
    """Extract object_permission_id and team_object_permission_id from request_data."""
    if not request_data:
        return None, None
    for key in ("litellm_metadata", "metadata"):
        meta = request_data.get(key)
        if not isinstance(meta, dict):
            continue
        auth = meta.get("user_api_key_auth")
        if auth is not None and hasattr(auth, "object_permission_id"):
            key_op = getattr(auth, "object_permission_id", None)
            team_op = getattr(auth, "team_object_permission_id", None)
            if key_op is not None or team_op is not None:
                return (
                    str(key_op).strip() if key_op else None,
                    str(team_op).strip() if team_op else None,
                )
        key_op = meta.get("user_api_key_object_permission_id")
        team_op = meta.get("user_api_key_team_object_permission_id")
        if key_op is not None or team_op is not None:
            return (
                str(key_op).strip() if key_op else None,
                str(team_op).strip() if team_op else None,
            )
    return None, None


def _get_request_route_from_data(request_data: dict) -> Optional[str]:
    """Get request route from request_data (metadata or top-level)."""
    route = request_data.get("user_api_key_request_route")
    if route:
        return route
    meta = request_data.get("metadata") or request_data.get("litellm_metadata") or {}
    return meta.get("user_api_key_request_route")


class ToolPolicyGuardrail(CustomGuardrail):
    """
    Guardrail that enforces per-tool call policies from the in-memory
    ToolPolicyRegistry (synced from DB). Key/team allowed_tools (allowlist) is
    enforced in the auth layer (check_tools_allowlist). No DB or cache in hot
    path — registry lookups only.
    """

    def __init__(self, **kwargs: Any) -> None:
        if "supported_event_hooks" not in kwargs:
            kwargs["supported_event_hooks"] = [
                GuardrailEventHooks.pre_call,
                GuardrailEventHooks.post_call,
                GuardrailEventHooks.during_call,
            ]
        super().__init__(**kwargs)

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        """
        Enforce DB call_policy on request tools / response tool_calls.
        """
        if input_type == "request":
            tools = inputs.get("tools") or []
            tool_names = [
                t["function"]["name"]
                for t in tools
                if isinstance(t, dict)
                and isinstance(t.get("function"), dict)
                and t["function"].get("name")
            ]
            if not tool_names:
                route = _get_request_route_from_data(request_data)
                if route:

                    tool_names = extract_request_tool_names(route, request_data)
        else:  # response
            tool_calls = inputs.get("tool_calls") or []
            tool_names = []
            for tc in tool_calls:
                fn = None
                if isinstance(tc, dict):
                    fn = (tc.get("function") or {}).get("name")
                elif hasattr(tc, "function"):
                    fn = getattr(tc.function, "name", None)
                if fn:
                    tool_names.append(fn)

        if not tool_names:
            return inputs

        object_permission_id, team_object_permission_id = (
            _get_request_object_permission_ids(request_data)
        )
        from litellm.proxy.db.tool_registry_writer import get_tool_policy_registry

        registry = get_tool_policy_registry()
        if not registry.is_initialized():
            policy_map = {}
        else:
            policy_map = registry.get_effective_policies(
                tool_names,
                object_permission_id=object_permission_id,
                team_object_permission_id=team_object_permission_id,
            )
        blocked = [name for name in tool_names if policy_map.get(name) == "blocked"]
        if blocked:
            verbose_proxy_logger.warning(
                "ToolPolicyGuardrail: blocking tool(s) %s (policy=blocked)", blocked
            )
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Violated tool policy",
                    "blocked_tools": blocked,
                    "message": f"Tool(s) {blocked} are blocked by policy.",
                },
            )

        return inputs
